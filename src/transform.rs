use std::simd::num::SimdFloat;
use std::simd::prelude::*;

use glam::{Affine3A, Mat3A, Quat, Vec3, Vec3A};

use crate::bvh::aabb::Aabb;
use crate::intersect::interaction::{Hit, MaterialHit};
use crate::intersect::{Bounded, Intersectable};
use crate::light::{LightSample, Sampleable};
use crate::math::interval::Interval;
use crate::math::vec3::{Direction3, Point3};
#[cfg(test)]
use crate::ray::{Ray, RayDifferentials};
use crate::ray::{RayDifferentialsPacked, RayPacked};

/// A spatial transform that can be applied to intersectable objects.
/// Implementations may be static (one matrix) or animated (interpolated over time).
pub trait Transform: Send + Sync + Clone + 'static {
    /// Evaluate the forward transform at a given time.
    /// For static transforms, `time` is ignored.
    fn eval(&self, time: f32) -> Affine3A;

    /// Is this transform time-dependent?
    fn is_animated(&self) -> bool;

    /// Transform a ray from world space to object space, transforming all N lanes.
    /// Uses the inverse of the forward transform at `ray.time`.
    fn ray<const N: usize>(&self, ray: &RayPacked<N>) -> RayPacked<N>;

    /// Transform a Hit from object space to world space (point, normals, mapping_point).
    /// Called after the inner object produces a local-space hit.
    fn hit(&self, hit: &mut Hit);

    /// Transform an AABB from object space to world space (8-corner method).
    fn bbox(&self, bbox: Aabb) -> Aabb;

    /// Transform a direction vector (no translation, may apply rotation + scale).
    fn transform_direction(&self, dir: Direction3, time: f32) -> Direction3;

    /// Transform a point (applies full affine: rotation + translation + scale).
    fn transform_point(&self, point: Point3, time: f32) -> Point3;

    /// Transform a normal vector using the inverse-transpose.
    /// Correct for non-rigid transforms (non-uniform scale, shear).
    fn transform_normal(&self, normal: Direction3, time: f32) -> Direction3;

    /// Transform UV gradients (dx, dy) using the inverse-transpose.
    fn transform_gradients(
        &self,
        dx: Direction3,
        dy: Direction3,
        time: f32,
    ) -> (Direction3, Direction3) {
        let dx_transformed = self.transform_normal(dx, time);
        let dy_transformed = self.transform_normal(dy, time);
        (dx_transformed, dy_transformed)
    }
}

// ── SoA affine kernels ──────────────────────────────────────────────────────
// The static path applies one lane-invariant matrix to all N lanes with
// `std::simd` registers; the animated path falls back to a per-lane scalar
// loop (per-lane matrices prevent splatting). Both use the same left-
// associative op order as glam's `mul_vec3a` (+ translation), so the results
// are bit-exact with the scalar reference.

/// out[a] = ((m[0][a]·x + m[1][a]·y) + m[2][a]·z) + t[a], all lanes.
#[inline]
fn affine_point_axes<const N: usize>(
    m: &Mat3A,
    x: Simd<f32, N>,
    y: Simd<f32, N>,
    z: Simd<f32, N>,
    t: [f32; 3],
) -> [Simd<f32, N>; 3] {
    [
        Simd::splat(m.col(0)[0]) * x
            + Simd::splat(m.col(1)[0]) * y
            + Simd::splat(m.col(2)[0]) * z
            + Simd::splat(t[0]),
        Simd::splat(m.col(0)[1]) * x
            + Simd::splat(m.col(1)[1]) * y
            + Simd::splat(m.col(2)[1]) * z
            + Simd::splat(t[1]),
        Simd::splat(m.col(0)[2]) * x
            + Simd::splat(m.col(1)[2]) * y
            + Simd::splat(m.col(2)[2]) * z
            + Simd::splat(t[2]),
    ]
}

/// out[a] = ((m[0][a]·x + m[1][a]·y) + m[2][a]·z), all lanes (no translation).
#[inline]
fn affine_vector_axes<const N: usize>(
    m: &Mat3A,
    x: Simd<f32, N>,
    y: Simd<f32, N>,
    z: Simd<f32, N>,
) -> [Simd<f32, N>; 3] {
    [
        Simd::splat(m.col(0)[0]) * x + Simd::splat(m.col(1)[0]) * y + Simd::splat(m.col(2)[0]) * z,
        Simd::splat(m.col(0)[1]) * x + Simd::splat(m.col(1)[1]) * y + Simd::splat(m.col(2)[1]) * z,
        Simd::splat(m.col(0)[2]) * x + Simd::splat(m.col(1)[2]) * y + Simd::splat(m.col(2)[2]) * z,
    ]
}

/// out[a] = ((m[0][a]·v[0] + m[1][a]·v[1]) + m[2][a]·v[2]) + t[a], lane `i`.
#[inline]
fn affine_point_lane(m: &Mat3A, v: [f32; 3], t: [f32; 3]) -> [f32; 3] {
    [
        m.col(0)[0] * v[0] + m.col(1)[0] * v[1] + m.col(2)[0] * v[2] + t[0],
        m.col(0)[1] * v[0] + m.col(1)[1] * v[1] + m.col(2)[1] * v[2] + t[1],
        m.col(0)[2] * v[0] + m.col(1)[2] * v[1] + m.col(2)[2] * v[2] + t[2],
    ]
}

/// out[a] = ((m[0][a]·v[0] + m[1][a]·v[1]) + m[2][a]·v[2]), lane `i` (no translation).
#[inline]
fn affine_vector_lane(m: &Mat3A, v: [f32; 3]) -> [f32; 3] {
    [
        m.col(0)[0] * v[0] + m.col(1)[0] * v[1] + m.col(2)[0] * v[2],
        m.col(0)[1] * v[0] + m.col(1)[1] * v[1] + m.col(2)[1] * v[2],
        m.col(0)[2] * v[0] + m.col(1)[2] * v[1] + m.col(2)[2] * v[2],
    ]
}

/// Transform lane `i` of a packed ray with the world → object affine `inv`,
/// writing origin, direction, and inverse direction directly into the packed
/// output arrays (no gather/scatter round-trip through `RayPacked<1>`).
#[inline]
fn transform_ray_lane<const N: usize>(
    inv: &Affine3A,
    i: usize,
    src: &RayPacked<N>,
    origin: &mut [[f32; N]; 3],
    direction: &mut [[f32; N]; 3],
    inverse_direction: &mut [[f32; N]; 3],
) {
    let m = inv.matrix3;
    let t = [inv.translation.x, inv.translation.y, inv.translation.z];
    let o = affine_point_lane(
        &m,
        [src.origin[0][i], src.origin[1][i], src.origin[2][i]],
        t,
    );
    let d = affine_vector_lane(
        &m,
        [
            src.direction[0][i],
            src.direction[1][i],
            src.direction[2][i],
        ],
    );
    origin[0][i] = o[0];
    origin[1][i] = o[1];
    origin[2][i] = o[2];
    direction[0][i] = d[0];
    direction[1][i] = d[1];
    direction[2][i] = d[2];
    inverse_direction[0][i] = d[0].recip();
    inverse_direction[1][i] = d[1].recip();
    inverse_direction[2][i] = d[2].recip();
}

/// Transform lane `i` of packed ray differentials with the world → object
/// affine `inv`, writing into `out`.
#[inline]
fn transform_differentials_lane<const N: usize>(
    inv: &Affine3A,
    i: usize,
    src: &RayDifferentialsPacked<N>,
    out: &mut RayDifferentialsPacked<N>,
) {
    let m = inv.matrix3;
    let t = [inv.translation.x, inv.translation.y, inv.translation.z];
    let rx_o = affine_point_lane(
        &m,
        [
            src.rx_origin[0][i],
            src.rx_origin[1][i],
            src.rx_origin[2][i],
        ],
        t,
    );
    let ry_o = affine_point_lane(
        &m,
        [
            src.ry_origin[0][i],
            src.ry_origin[1][i],
            src.ry_origin[2][i],
        ],
        t,
    );
    let rx_d = affine_vector_lane(
        &m,
        [
            src.rx_direction[0][i],
            src.rx_direction[1][i],
            src.rx_direction[2][i],
        ],
    );
    let ry_d = affine_vector_lane(
        &m,
        [
            src.ry_direction[0][i],
            src.ry_direction[1][i],
            src.ry_direction[2][i],
        ],
    );
    out.rx_origin[0][i] = rx_o[0];
    out.rx_origin[1][i] = rx_o[1];
    out.rx_origin[2][i] = rx_o[2];
    out.ry_origin[0][i] = ry_o[0];
    out.ry_origin[1][i] = ry_o[1];
    out.ry_origin[2][i] = ry_o[2];
    out.rx_direction[0][i] = rx_d[0];
    out.rx_direction[1][i] = rx_d[1];
    out.rx_direction[2][i] = rx_d[2];
    out.ry_direction[0][i] = ry_d[0];
    out.ry_direction[1][i] = ry_d[1];
    out.ry_direction[2][i] = ry_d[2];
}

/// A non-animated transform. Precomputes the inverse matrix at construction.
/// Used for the vast majority of scene objects.
#[derive(Clone)]
pub struct StaticTransform {
    forward: Affine3A,
    inverse: Affine3A,
}

impl StaticTransform {
    /// Create an identity transform (no-op).
    pub fn identity() -> Self {
        Self::from_affine3a(Affine3A::IDENTITY)
    }

    pub fn from_affine3a(forward: Affine3A) -> Self {
        Self {
            forward,
            inverse: forward.inverse(),
        }
    }

    /// Create a translation transform that moves points by the given offset.
    pub fn translation(offset: Vec3) -> Self {
        Self::from_affine3a(Affine3A::from_translation(offset))
    }

    /// Create a rotation transform around the Y axis by the given angle in degrees.
    pub fn rotation_y(degrees: f32) -> Self {
        Self::from_affine3a(Affine3A::from_rotation_y(degrees.to_radians()))
    }

    /// Create a rotation transform around the X axis by the given angle in degrees.
    pub fn rotation_x(degrees: f32) -> Self {
        Self::from_affine3a(Affine3A::from_rotation_x(degrees.to_radians()))
    }

    /// Create a rotation transform around the Z axis by the given angle in degrees.
    pub fn rotation_z(degrees: f32) -> Self {
        Self::from_affine3a(Affine3A::from_rotation_z(degrees.to_radians()))
    }

    /// Create a rotation transform around the Y axis by the given angle in radians.
    pub fn rotation_x_axis(radians: f32) -> Self {
        Self::from_affine3a(Affine3A::from_rotation_x(radians))
    }

    /// Create a rotation transform around the Y axis by the given angle in radians.
    pub fn scale(s: Vec3) -> Self {
        Self::from_affine3a(Affine3A::from_scale(s))
    }

    pub fn compose(parent: Affine3A, child: Affine3A) -> Self {
        Self::from_affine3a(parent * child)
    }

    /// Scalar world → object ray transform (lane-0 body). Kept as the scalar
    /// reference for the packet tests; the SIMD `ray` path is bit-exact with it.
    #[cfg(test)]
    fn ray_scalar(&self, ray: &Ray) -> Ray {
        // World → object: apply inverse
        let origin = Point3(
            self.inverse
                .transform_point3a(ray.origin().into_inner().into())
                .into(),
        );
        let direction = Direction3(
            self.inverse
                .transform_vector3a(ray.direction().into_inner().into())
                .into(),
        );
        let diffs = ray.differentials.map(|differentials| {
            RayDifferentials::new(
                Point3(
                    self.inverse
                        .transform_point3a(differentials.rx_origin().into_inner().into())
                        .into(),
                ),
                Point3(
                    self.inverse
                        .transform_point3a(differentials.ry_origin().into_inner().into())
                        .into(),
                ),
                Direction3(
                    self.inverse
                        .transform_vector3a(differentials.rx_direction().into_inner().into())
                        .into(),
                ),
                Direction3(
                    self.inverse
                        .transform_vector3a(differentials.ry_direction().into_inner().into())
                        .into(),
                ),
            )
        });

        Ray::new_with_differentials(origin, direction, ray.time(), diffs)
    }
}

impl Transform for StaticTransform {
    #[inline]
    fn eval(&self, _time: f32) -> Affine3A {
        self.forward // O(1), no computation
    }

    #[inline]
    fn is_animated(&self) -> bool {
        false
    }

    #[inline]
    fn ray<const N: usize>(&self, ray: &RayPacked<N>) -> RayPacked<N> {
        // World → object: apply the precomputed inverse affine directly on the
        // SoA axes. The matrix is lane-invariant, so each output axis is a
        // splat-mul-add over the three input axis registers (same left-
        // associative op order as glam's `mul_vec3a` + translation, so the
        // results are bit-exact with the scalar reference).
        let m = self.inverse.matrix3;
        let t = self.inverse.translation;

        let ox = Simd::from_array(ray.origin[0]);
        let oy = Simd::from_array(ray.origin[1]);
        let oz = Simd::from_array(ray.origin[2]);
        let dx = Simd::from_array(ray.direction[0]);
        let dy = Simd::from_array(ray.direction[1]);
        let dz = Simd::from_array(ray.direction[2]);

        let origin = affine_point_axes(&m, ox, oy, oz, [t.x, t.y, t.z]);
        let direction = affine_vector_axes(&m, dx, dy, dz);

        // Recompute the inverse direction once, from the transformed direction.
        let inverse_direction = [
            direction[0].recip().to_array(),
            direction[1].recip().to_array(),
            direction[2].recip().to_array(),
        ];

        // Differentials are all-or-nothing per pack: transform every field when
        // present, keep `None` otherwise.
        let differentials = ray.differentials.map(|rd| {
            let rx_origin = affine_point_axes(
                &m,
                Simd::from_array(rd.rx_origin[0]),
                Simd::from_array(rd.rx_origin[1]),
                Simd::from_array(rd.rx_origin[2]),
                [t.x, t.y, t.z],
            );
            let ry_origin = affine_point_axes(
                &m,
                Simd::from_array(rd.ry_origin[0]),
                Simd::from_array(rd.ry_origin[1]),
                Simd::from_array(rd.ry_origin[2]),
                [t.x, t.y, t.z],
            );
            let rx_direction = affine_vector_axes(
                &m,
                Simd::from_array(rd.rx_direction[0]),
                Simd::from_array(rd.rx_direction[1]),
                Simd::from_array(rd.rx_direction[2]),
            );
            let ry_direction = affine_vector_axes(
                &m,
                Simd::from_array(rd.ry_direction[0]),
                Simd::from_array(rd.ry_direction[1]),
                Simd::from_array(rd.ry_direction[2]),
            );
            RayDifferentialsPacked {
                rx_origin: [
                    rx_origin[0].to_array(),
                    rx_origin[1].to_array(),
                    rx_origin[2].to_array(),
                ],
                ry_origin: [
                    ry_origin[0].to_array(),
                    ry_origin[1].to_array(),
                    ry_origin[2].to_array(),
                ],
                rx_direction: [
                    rx_direction[0].to_array(),
                    rx_direction[1].to_array(),
                    rx_direction[2].to_array(),
                ],
                ry_direction: [
                    ry_direction[0].to_array(),
                    ry_direction[1].to_array(),
                    ry_direction[2].to_array(),
                ],
            }
        });

        RayPacked {
            origin: [
                origin[0].to_array(),
                origin[1].to_array(),
                origin[2].to_array(),
            ],
            direction: [
                direction[0].to_array(),
                direction[1].to_array(),
                direction[2].to_array(),
            ],
            time: ray.time,
            inverse_direction,
            differentials,
        }
    }

    fn hit(&self, hit: &mut Hit) {
        // Object → world: apply forward
        hit.point = Point3(
            self.forward
                .transform_point3a(hit.point.into_inner().into())
                .into(),
        );
        hit.mapping_point = Point3(
            self.forward
                .transform_point3a(hit.mapping_point.into_inner().into())
                .into(),
        );

        // Normals via inverse-transpose (correct for non-rigid transforms)
        hit.set_geometric_normal(self.transform_normal(hit.geometric_normal(), hit.time));

        // Transform UV gradients if present
        hit.uv_gradients = if let Some(grad) = hit.uv_gradients {
            let (dx, dy) = grad;
            // Transform the UV gradients using the transform_normal method, which applies the inverse-transpose
            let transformed_gradients = self.transform_gradients(dx, dy, hit.time);

            Some(transformed_gradients)
        } else {
            None
        };

        // Adjust curvature by the average scale factor. If the scale is very small, avoid division
        // by zero.
        let scale = self.forward.matrix3.determinant().abs().cbrt();
        if scale > 1e-10 {
            hit.curvature /= scale;
        }
    }

    fn bbox(&self, bbox: Aabb) -> Aabb {
        // Transform all 8 corners, take min/max
        let corners = [
            Vec3::new(bbox.min[0][0], bbox.min[1][0], bbox.min[2][0]),
            Vec3::new(bbox.max[0][0], bbox.min[1][0], bbox.min[2][0]),
            Vec3::new(bbox.min[0][0], bbox.max[1][0], bbox.min[2][0]),
            Vec3::new(bbox.max[0][0], bbox.max[1][0], bbox.min[2][0]),
            Vec3::new(bbox.min[0][0], bbox.min[1][0], bbox.max[2][0]),
            Vec3::new(bbox.max[0][0], bbox.min[1][0], bbox.max[2][0]),
            Vec3::new(bbox.min[0][0], bbox.max[1][0], bbox.max[2][0]),
            Vec3::new(bbox.max[0][0], bbox.max[1][0], bbox.max[2][0]),
        ];
        let mut min = Vec3A::splat(f32::INFINITY);
        let mut max = Vec3A::splat(f32::NEG_INFINITY);
        for c in corners {
            let p = self.forward.transform_point3a(c.into());
            min = min.min(p);
            max = max.max(p);
        }
        Aabb::from_corners(Point3(min.into()), Point3(max.into()))
    }

    fn transform_direction(&self, dir: Direction3, _time: f32) -> Direction3 {
        Direction3(
            self.forward
                .transform_vector3a(dir.into_inner().into())
                .into(),
        )
    }

    fn transform_point(&self, point: Point3, _time: f32) -> Point3 {
        Point3(
            self.forward
                .transform_point3a(point.into_inner().into())
                .into(),
        )
    }

    fn transform_normal(&self, normal: Direction3, _time: f32) -> Direction3 {
        // Inverse-transpose: correct for non-uniform scale
        let n = normal.into_inner();
        let inv = self.inverse;

        // (M⁻¹)ᵀ · n = transpose(M⁻¹) · n
        // With Affine3A, extract the 3x3 upper-left, invert, transpose, apply
        let inv_mat = inv.matrix3; // 3x3 matrix

        // Normalize the result to ensure it's a unit normal
        Direction3((inv_mat.transpose() * n).normalize())
    }
}

impl std::ops::Mul for StaticTransform {
    type Output = Self;
    /// Compose two static transforms: `self` is applied first, then `other`.
    /// Result = other.forward * self.forward, inverse = self.inverse * other.inverse
    fn mul(self, other: Self) -> Self {
        let forward = other.forward * self.forward;
        let inverse = self.inverse * other.inverse;
        Self { forward, inverse }
    }
}

/// An animated transform that linearly interpolates between two keyframes.
///
/// At construction, decomposes both matrices into translation/rotation/scale.
/// On `eval(t)`, lerps T and S, SLERPs R, recomposes into Affine3A.
///
/// Follows pbrt-v4's AnimatedTransform decomposition model.
#[derive(Clone)]
pub struct AnimatedTransform {
    start: Affine3A,
    end: Affine3A,
    /// Decomposed start: (translation, rotation quaternion, scale matrix)
    start_decomposed: Decomposed,
    /// Decomposed end
    end_decomposed: Decomposed,
}

impl AnimatedTransform {
    pub fn new(start: Affine3A, end: Affine3A) -> Self {
        // Handle quaternion hemisphere flip (pbrt-v4: ensure shortest SLERP path)
        let start_decomposed = Decomposed::from_affine(start);
        let mut end_decomposed = Decomposed::from_affine(end);

        // Flip end quaternion if dot product is negative (opposite hemisphere)
        if start_decomposed.rotation.dot(end_decomposed.rotation) < 0.0 {
            end_decomposed.rotation = -end_decomposed.rotation;
        }

        Self {
            start,
            end,
            start_decomposed,
            end_decomposed,
        }
    }

    /// Scalar world → object ray transform (lane-0 body). Kept as the scalar
    /// reference for the packet tests; the animated `ray` path is bit-exact
    /// with it.
    #[cfg(test)]
    fn ray_scalar(&self, ray: &Ray) -> Ray {
        let inv = self.eval(ray.time()).inverse();
        let origin = Point3(
            inv.transform_point3a(ray.origin().into_inner().into())
                .into(),
        );
        let direction = Direction3(
            inv.transform_vector3a(ray.direction().into_inner().into())
                .into(),
        );
        let time = ray.time();
        let differentials = ray.differentials.map(|diffs| {
            RayDifferentials::new(
                Point3(
                    inv.transform_point3a(diffs.rx_origin().into_inner().into())
                        .into(),
                ),
                Point3(
                    inv.transform_point3a(diffs.ry_origin().into_inner().into())
                        .into(),
                ),
                Direction3(
                    inv.transform_vector3a(diffs.rx_direction().into_inner().into())
                        .into(),
                ),
                Direction3(
                    inv.transform_vector3a(diffs.ry_direction().into_inner().into())
                        .into(),
                ),
            )
        });
        Ray::new_with_differentials(origin, direction, time, differentials)
    }
}

impl Transform for AnimatedTransform {
    #[inline]
    fn is_animated(&self) -> bool {
        true
    }

    /// Evaluate the transform at a given time in [0, 1]. Linearly interpolate between start and end.
    fn eval(&self, time: f32) -> Affine3A {
        // For single-sample scenes (time is always 0.0 or 1.0),
        // just return the exact keyframe
        if time <= 0.0 {
            return self.start;
        }
        if time >= 1.0 {
            return self.end;
        }

        self.start_decomposed.lerp(&self.end_decomposed, time)
    }

    /// Transform a ray from world space to object space using the inverse of the transform at
    /// `ray.time`.
    #[inline]
    fn ray<const N: usize>(&self, ray: &RayPacked<N>) -> RayPacked<N> {
        // Per-lane inverse transforms: lane times differ and the SLERP
        // recomposition can't be vectorized, so evaluate one inverse per lane
        // (once, reused for the differentials), then apply each directly to the
        // packed arrays — no gather/scatter round-trip through `RayPacked<1>`.
        let invs: [Affine3A; N] = core::array::from_fn(|i| self.eval(ray.time[i]).inverse());

        let mut origin = [[0.0; N]; 3];
        let mut direction = [[0.0; N]; 3];
        let mut inverse_direction = [[0.0; N]; 3];
        for i in 0..N {
            transform_ray_lane(
                &invs[i],
                i,
                ray,
                &mut origin,
                &mut direction,
                &mut inverse_direction,
            );
        }

        // Differentials are all-or-nothing per pack: transform every field when
        // present, keep `None` otherwise.
        let differentials = ray.differentials.map(|rd| {
            let mut out = RayDifferentialsPacked {
                rx_origin: [[0.0; N]; 3],
                ry_origin: [[0.0; N]; 3],
                rx_direction: [[0.0; N]; 3],
                ry_direction: [[0.0; N]; 3],
            };
            for i in 0..N {
                transform_differentials_lane(&invs[i], i, &rd, &mut out);
            }
            out
        });

        RayPacked {
            origin,
            direction,
            time: ray.time,
            inverse_direction,
            differentials,
        }
    }

    /// Transform a Hit from object space to world space (point, normals, mapping_point).
    fn hit(&self, hit: &mut Hit) {
        let m = self.eval(hit.time);
        hit.point = Point3(m.transform_point3a(hit.point.into_inner().into()).into());
        hit.mapping_point = Point3(
            m.transform_point3a(hit.mapping_point.into_inner().into())
                .into(),
        );

        // Inverse-transpose for normals
        let inv = m.inverse();
        let inv_mat = inv.matrix3;
        hit.set_geometric_normal(Direction3(
            inv_mat.transpose() * hit.geometric_normal().into_inner(),
        ));

        // Transform UV gradients if present
        hit.uv_gradients = if let Some(grad) = hit.uv_gradients {
            let (dx, dy) = grad;
            let dx_transformed = self.transform_direction(dx, hit.time);
            let dy_transformed = self.transform_direction(dy, hit.time);
            Some((dx_transformed, dy_transformed))
        } else {
            None
        };

        // Adjust curvature by the average scale factor. If the scale is very small, avoid division
        // by zero.
        let scale = m.matrix3.determinant().abs().cbrt();
        if scale > 1e-10 {
            hit.curvature /= scale;
        }
    }

    /// N-Sample Union Bounding Box: sample the animated transform at N time steps, compute the
    /// transformed AABB at each step, and merge them.
    ///
    /// TODO: pbrt-v4's motion-swept AABB is more precise but requires solving dp_i/dt = 0
    /// analytically for each axis of each corner. Implement that if we need more precise bounds for
    /// motion blur.
    fn bbox(&self, bbox: Aabb) -> Aabb {
        let n_samples = 8; // OpenMoonRay uses 64 for GPU baking
        let mut result = Aabb::empty();
        for i in 0..n_samples {
            let t = i as f32 / (n_samples - 1) as f32;
            let xform = StaticTransform::from_affine3a(self.eval(t));
            result = result.merge(&xform.bbox(bbox));
        }
        result
    }

    // transform_direction, transform_point, transform_normal: same pattern as StaticTransform but
    // using self.eval(time) instead of self.forward

    /// Transform a direction vector (no translation, may apply rotation + scale).
    fn transform_direction(&self, dir: Direction3, time: f32) -> Direction3 {
        Direction3(
            self.eval(time)
                .transform_vector3a(dir.into_inner().into())
                .into(),
        )
    }

    /// Transform a point (applies full affine: rotation + translation + scale).
    fn transform_point(&self, point: Point3, time: f32) -> Point3 {
        Point3(
            self.eval(time)
                .transform_point3a(point.into_inner().into())
                .into(),
        )
    }

    /// Transform a normal vector using the inverse-transpose.
    fn transform_normal(&self, normal: Direction3, time: f32) -> Direction3 {
        let m = self.eval(time);
        let inv = m.inverse();
        let inv_mat = Mat3A::from_cols(inv.matrix3.col(0), inv.matrix3.col(1), inv.matrix3.col(2));
        Direction3(inv_mat.transpose() * normal.into_inner())
    }
}

/// Decomposed representation of an Affine3A transform into translation, rotation, and scale
/// components.
#[derive(Clone, Copy)]
struct Decomposed {
    translation: Vec3A, // already SIMD-aligned (Vec3A)
    rotation: Quat,     // unit quaternion
    scale: Affine3A,    // upper-left 3×3 (scale + shear)
}

impl Decomposed {
    /// Decompose an Affine3A into translation, rotation, and scale components.
    fn from_affine(m: Affine3A) -> Self {
        // Translation = last column of the 4×3 matrix (already Vec3A)
        let translation = m.translation;

        // Extract 3×3 upper-left (rotation × scale)
        let cols = [m.matrix3.col(0), m.matrix3.col(1), m.matrix3.col(2)];

        // Gram-Schmidt to separate rotation from scale (pbrt-v4: `Orthogonalize()` — makes columns
        // unit-length, captures scale)
        let x = cols[0];
        let xs = x.length();
        if xs < 1e-10 {
            // Degenerate: zero X column — return identity rotation, zero scale
            return Self {
                translation,
                rotation: Quat::IDENTITY,
                scale: Affine3A {
                    matrix3: Mat3A::ZERO,
                    translation: Vec3::ZERO.into(),
                },
            };
        }
        let x_normalized = x / xs;

        let y = cols[1];
        let dot_xy = x_normalized.dot(y);
        let y_orth = y - x_normalized * dot_xy;
        let ys = y_orth.length();
        let y_normalized = if ys < 1e-10 {
            // Y column is degenerate (parallel to X) — generate orthogonal basis
            let fallback = if x_normalized.abs().cmplt(Vec3A::splat(0.9)).any() {
                Vec3A::Z
            } else {
                Vec3A::Y
            };
            let y_fallback = x_normalized.cross(fallback).normalize();
            // Re-orthogonalize against X
            y_fallback - x_normalized * x_normalized.dot(y_fallback)
        } else {
            y_orth / ys
        };

        let z = x_normalized.cross(y_normalized);
        let zs = cols[2].dot(z); // signed scale

        // 4. Reconstruct rotation matrix from orthonormal basis
        let rot_mat = Mat3A::from_cols(x_normalized, y_normalized, z);
        let rotation = Quat::from_mat3a(&rot_mat);

        // 5. Scale matrix
        let scale = Affine3A {
            matrix3: Mat3A::from_cols(
                Vec3A::new(xs, 0.0, 0.0),
                Vec3A::new(dot_xy, ys, 0.0),
                Vec3A::new(x_normalized.dot(cols[2]), y_normalized.dot(cols[2]), zs),
            ),
            translation: Vec3::ZERO.into(),
        };

        Self {
            translation,
            rotation,
            scale,
        }
    }

    fn lerp(&self, other: &Self, t: f32) -> Affine3A {
        let translation = self.translation.lerp(other.translation, t);
        let rotation = self.rotation.slerp(other.rotation, t);
        let scale_cols = [
            self.scale
                .matrix3
                .col(0)
                .lerp(other.scale.matrix3.col(0), t),
            self.scale
                .matrix3
                .col(1)
                .lerp(other.scale.matrix3.col(1), t),
            self.scale
                .matrix3
                .col(2)
                .lerp(other.scale.matrix3.col(2), t),
        ];
        // Recompose: translation + rotation * scale
        let rotated_scale = Mat3A::from_cols(
            rotation * scale_cols[0],
            rotation * scale_cols[1],
            rotation * scale_cols[2],
        );
        Affine3A {
            matrix3: rotated_scale,
            translation,
        }
    }
}

/// A zero-cost wrapper that applies a transform to any intersectable object.
/// The transform type is a generic parameter — `StaticTransform` for most objects,
/// `AnimatedTransform` for moving objects. Zero-cost monomorphization.
#[derive(Clone)]
pub struct TransformObject<O: Intersectable, T: Transform = StaticTransform> {
    xform: T,
    object: O,
    bbox: Aabb,
}

impl<O: Intersectable, T: Transform> TransformObject<O, T> {
    pub fn new(xform: T, object: O) -> Self {
        let bbox = xform.bbox(object.bounding_box());
        Self {
            xform,
            object,
            bbox,
        }
    }

    /// Access the transform
    pub fn xform(&self) -> &T {
        &self.xform
    }

    /// Access the inner object
    pub fn object(&self) -> &O {
        &self.object
    }
}

impl<O: Intersectable, T: Transform> Bounded for TransformObject<O, T> {
    fn bounding_box(&self) -> Aabb {
        self.bbox
    }
}

impl<O: Intersectable, T: Transform> Intersectable for TransformObject<O, T> {
    fn intersect_scalar<'a>(
        &'a self,
        ray: &RayPacked<1>,
        ray_t: Interval<1>,
    ) -> Option<MaterialHit<'a>> {
        let local_ray = self.xform.ray(ray);
        let mut hit = self.object.intersect_scalar(&local_ray, ray_t)?;
        self.xform.hit(&mut hit.hit);
        Some(hit)
    }

    fn intersect<'a, const N: usize>(
        &'a self,
        ray: &RayPacked<N>,
        ray_t: Interval<N>,
    ) -> [Option<MaterialHit<'a>>; N] {
        // 1. Transform ray from world → object space
        let local_ray = self.xform.ray(ray);

        // 2. Intersect the inner object in object space
        let mut hits = self.object.intersect(&local_ray, ray_t);

        // 3. Transform each hit from object → world space
        for mat_hit in hits.iter_mut().flatten() {
            self.xform.hit(&mut mat_hit.hit);
        }

        hits
    }
}

impl<O: Sampleable + Intersectable, T: Transform> Sampleable for TransformObject<O, T> {
    fn pdf_value(&self, origin: Point3, direction: Direction3, time: f32) -> f32 {
        // Transform origin and direction into object space, delegate
        let m = self.xform.eval(time).inverse();
        let obj_origin = Point3(m.transform_point3a(origin.into_inner().into()).into());
        let obj_dir = Direction3(m.transform_vector3a(direction.into_inner().into()).into());
        self.object.pdf_value(obj_origin, obj_dir, time)
    }

    fn random_direction(&self, origin: Point3, u: f32, v: f32, time: f32) -> Direction3 {
        // Sample in object space, transform result to world space
        let m = self.xform.eval(time).inverse();
        let obj_origin = Point3(m.transform_point3a(origin.into_inner().into()).into());
        let obj_dir = self.object.random_direction(obj_origin, u, v, time);
        // Transform direction back to world space
        self.xform.transform_direction(obj_dir, time)
    }

    fn sample_light(&self, origin: Point3, u: f32, v: f32, time: f32) -> LightSample {
        // Sample in object space
        let m = self.xform.eval(time).inverse();
        let obj_origin = Point3(m.transform_point3a(origin.into_inner().into()).into());
        let local_sample = self.object.sample_light(obj_origin, u, v, time);

        // Transform results to world space
        let world_point = self.xform.transform_point(
            obj_origin + local_sample.direction.normalize_or_zero() * local_sample.distance,
            time,
        );
        let world_dir = world_point - origin;

        LightSample {
            direction: world_dir,
            normal: self.xform.transform_normal(local_sample.normal, time),
            distance: world_dir.length(),
            pdf: local_sample.pdf, // Area PDF is invariant under rigid transforms
            emission: local_sample.emission,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ray with differentials, varying per lane.
    fn test_ray_at(i: usize, time: f32) -> RayPacked<1> {
        let f = i as f32;
        RayPacked::new_with_differentials(
            Point3::new(1.0 + f, 2.0, 3.0),
            Direction3(Vec3::new(0.1 + f * 0.1, 0.2, 0.3).normalize()),
            time,
            Some(RayDifferentialsPacked::new(
                Point3::new(1.1 + f, 2.1, 3.1),
                Point3::new(1.2 + f, 2.2, 3.2),
                Direction3(Vec3::new(0.4, 0.5, 0.6).normalize()),
                Direction3(Vec3::new(0.7, 0.8, 0.9).normalize()),
            )),
        )
    }

    fn test_ray(i: usize) -> RayPacked<1> {
        test_ray_at(i, 0.5 + i as f32)
    }

    /// Bit-exact comparison of every ray field, including differentials.
    fn assert_ray_eq(a: &RayPacked<1>, b: &RayPacked<1>) {
        assert_eq!(a.origin(), b.origin());
        assert_eq!(a.direction(), b.direction());
        assert_eq!(a.time(), b.time());
        assert_eq!(a.inverse_direction(), b.inverse_direction());
        match (a.differentials, b.differentials) {
            (Some(ad), Some(bd)) => {
                assert_eq!(ad.rx_origin(), bd.rx_origin());
                assert_eq!(ad.ry_origin(), bd.ry_origin());
                assert_eq!(ad.rx_direction(), bd.rx_direction());
                assert_eq!(ad.ry_direction(), bd.ry_direction());
            }
            (None, None) => {}
            _ => panic!("differential presence mismatch"),
        }
    }

    fn static_xform() -> StaticTransform {
        StaticTransform::from_affine3a(
            Affine3A::from_rotation_y(0.7)
                * Affine3A::from_scale(Vec3::new(1.5, 0.5, 2.0))
                * Affine3A::from_translation(Vec3::new(1.0, -2.0, 3.0)),
        )
    }

    fn animated_xform() -> AnimatedTransform {
        AnimatedTransform::new(
            Affine3A::from_rotation_y(0.3) * Affine3A::from_translation(Vec3::new(1.0, 0.0, 0.0)),
            Affine3A::from_rotation_y(1.2)
                * Affine3A::from_scale(Vec3::new(2.0, 2.0, 2.0))
                * Affine3A::from_translation(Vec3::new(0.0, 0.0, 1.0)),
        )
    }

    /// Every lane of a statically-transformed packet must match the scalar
    /// transform bit-for-bit, including differentials and inverse directions.
    #[test]
    fn static_ray_packet_matches_scalar() {
        let xform = static_xform();
        let rays: [RayPacked<1>; 4] = core::array::from_fn(test_ray);
        let packed: RayPacked<4> = rays.into();
        let transformed = xform.ray(&packed);
        let lanes: [RayPacked<1>; 4] = transformed.into();
        for (i, lane) in lanes.iter().enumerate() {
            assert_ray_eq(lane, &xform.ray_scalar(&rays[i]));
        }
    }

    /// A differential-less pack must stay `None` through the static transform.
    #[test]
    fn static_ray_packet_without_differentials_stays_none() {
        let xform = static_xform();
        let rays: [RayPacked<1>; 4] = core::array::from_fn(|i| {
            RayPacked::new_with_time(
                Point3::new(1.0 + i as f32, 2.0, 3.0),
                Direction3(Vec3::new(0.1, 0.2, 0.3).normalize()),
                0.25 * i as f32,
            )
        });
        let packed: RayPacked<4> = rays.into();
        assert!(packed.differentials.is_none());
        let transformed = xform.ray(&packed);
        assert!(transformed.differentials.is_none());
        let lanes: [RayPacked<1>; 4] = transformed.into();
        for (i, lane) in lanes.iter().enumerate() {
            assert_ray_eq(lane, &xform.ray_scalar(&rays[i]));
        }
    }

    /// N=1 (the scalar alias) must go through the same SIMD path and match the
    /// scalar reference.
    #[test]
    fn static_ray_n1_matches_scalar() {
        let xform = static_xform();
        let ray = test_ray(2);
        assert_ray_eq(&xform.ray(&ray), &xform.ray_scalar(&ray));
    }

    /// Non-power-of-two lane counts (e.g. the tail chunk of a wavefront batch)
    /// must go through the same SIMD path.
    #[test]
    fn static_ray_packet_n3_matches_scalar() {
        let xform = static_xform();
        let rays: [RayPacked<1>; 3] = core::array::from_fn(test_ray);
        let packed: RayPacked<3> = rays.into();
        let transformed = xform.ray(&packed);
        let lanes: [RayPacked<1>; 3] = transformed.into();
        for (i, lane) in lanes.iter().enumerate() {
            assert_ray_eq(lane, &xform.ray_scalar(&rays[i]));
        }
    }

    /// Animated packets at endpoint (0.0, 1.0) and interior (0.25, 0.5) times
    /// must match the scalar per-lane transform, including differentials.
    #[test]
    fn animated_ray_packet_matches_scalar() {
        let xform = animated_xform();
        let times = [0.0, 0.25, 0.5, 1.0];
        let rays: [RayPacked<1>; 4] = core::array::from_fn(|i| test_ray_at(i, times[i]));
        let packed: RayPacked<4> = rays.into();
        let transformed = xform.ray(&packed);
        let lanes: [RayPacked<1>; 4] = transformed.into();
        for (i, lane) in lanes.iter().enumerate() {
            assert_ray_eq(lane, &xform.ray_scalar(&rays[i]));
        }
    }

    /// N=1 animated ray at an interior time.
    #[test]
    fn animated_ray_n1_matches_scalar() {
        let xform = animated_xform();
        let ray = test_ray_at(1, 0.37);
        assert_ray_eq(&xform.ray(&ray), &xform.ray_scalar(&ray));
    }

    /// The inverse direction must be recomputed from the transformed direction
    /// (a non-uniform scale changes the direction magnitude), not carried over.
    #[test]
    fn static_inverse_direction_matches_recip_of_transformed_direction() {
        let xform = StaticTransform::from_affine3a(
            Affine3A::from_scale(Vec3::new(2.0, 0.5, 1.0)) * Affine3A::from_rotation_z(0.4),
        );
        let rays: [RayPacked<1>; 4] = core::array::from_fn(test_ray);
        let packed: RayPacked<4> = rays.into();
        let transformed = xform.ray(&packed);
        for axis in 0..3 {
            for i in 0..4 {
                let d = transformed.direction[axis][i];
                assert_eq!(transformed.inverse_direction[axis][i], d.recip());
            }
        }
    }

    /// Same invariant for the animated path (scale changes over time).
    #[test]
    fn animated_inverse_direction_matches_recip_of_transformed_direction() {
        let xform = AnimatedTransform::new(
            Affine3A::from_rotation_y(0.3),
            Affine3A::from_scale(Vec3::new(2.0, 2.0, 2.0)) * Affine3A::from_rotation_y(1.2),
        );
        let times = [0.0, 0.25, 0.5, 1.0];
        let rays: [RayPacked<1>; 4] = core::array::from_fn(|i| test_ray_at(i, times[i]));
        let packed: RayPacked<4> = rays.into();
        let transformed = xform.ray(&packed);
        for axis in 0..3 {
            for i in 0..4 {
                let d = transformed.direction[axis][i];
                assert_eq!(transformed.inverse_direction[axis][i], d.recip());
            }
        }
    }
}
