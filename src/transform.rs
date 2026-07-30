use glam::{Affine3A, Mat3A, Quat, Vec3, Vec3A};

use crate::bvh::aabb::Aabb;
use crate::hittable::{Bounded, Hit, Intersectable, LightSample, MaterialHit, Sampleable};
use crate::interval::Interval;
use crate::ray::Ray;
use crate::vec3::{Direction3, Point3};

/// A spatial transform that can be applied to intersectable objects.
/// Implementations may be static (one matrix) or animated (interpolated over time).
pub trait Transform: Send + Sync + Clone + 'static {
    /// Evaluate the forward transform at a given time.
    /// For static transforms, `time` is ignored.
    fn eval(&self, time: f32) -> Affine3A;

    /// Is this transform time-dependent?
    fn is_animated(&self) -> bool;

    /// Transform a ray from world space to object space.
    /// Uses the inverse of the forward transform at `ray.time`.
    fn ray(&self, ray: &Ray) -> Ray;

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

    fn ray(&self, ray: &Ray) -> Ray {
        // World → object: apply inverse
        let origin = self
            .inverse
            .transform_point3(ray.origin.into_inner())
            .into();
        let direction = self
            .inverse
            .transform_vector3(ray.direction.into_inner())
            .into();
        let diffs = if let Some(mut differentials) = ray.differentials {
            differentials.rx_origin = self
                .inverse
                .transform_point3(differentials.rx_origin.into_inner())
                .into();
            differentials.rx_direction = self
                .inverse
                .transform_vector3(differentials.rx_direction.into_inner())
                .into();
            differentials.ry_origin = self
                .inverse
                .transform_point3(differentials.ry_origin.into_inner())
                .into();
            differentials.ry_direction = self
                .inverse
                .transform_vector3(differentials.ry_direction.into_inner())
                .into();
            Some(differentials)
        } else {
            None
        };

        Ray::new_with_differentials(origin, direction, ray.time, diffs)
    }

    fn hit(&self, hit: &mut Hit) {
        // Object → world: apply forward
        hit.point = self.forward.transform_point3(hit.point.into_inner()).into();
        hit.mapping_point = self
            .forward
            .transform_point3(hit.mapping_point.into_inner())
            .into();

        // Normals via inverse-transpose (correct for non-rigid transforms)
        hit.set_geometric_normal(self.transform_normal(hit.geometric_normal(), hit.time));

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
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for c in corners {
            let p = self.forward.transform_point3(c);
            min = min.min(p);
            max = max.max(p);
        }
        Aabb::from_corners(min.into(), max.into())
    }

    fn transform_direction(&self, dir: Direction3, _time: f32) -> Direction3 {
        self.forward.transform_vector3(dir.into_inner()).into()
    }

    fn transform_point(&self, point: Point3, _time: f32) -> Point3 {
        self.forward.transform_point3(point.into_inner()).into()
    }

    fn transform_normal(&self, normal: Direction3, _time: f32) -> Direction3 {
        // Inverse-transpose: correct for non-uniform scale
        let n = normal.into_inner();
        let inv = self.inverse;

        // (M⁻¹)ᵀ · n = transpose(M⁻¹) · n
        // With Affine3A, extract the 3x3 upper-left, invert, transpose, apply
        let inv_mat = Mat3A::from_cols(inv.matrix3.col(0), inv.matrix3.col(1), inv.matrix3.col(2));

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
    fn ray(&self, ray: &Ray) -> Ray {
        let inv = self.eval(ray.time).inverse();
        let origin = inv.transform_point3(ray.origin.into_inner()).into();
        let direction = inv.transform_vector3(ray.direction.into_inner()).into();
        let time = ray.time;
        let differentials = if let Some(mut diffs) = ray.differentials {
            diffs.rx_origin = inv.transform_point3(diffs.rx_origin.into_inner()).into();
            diffs.rx_direction = inv
                .transform_vector3(diffs.rx_direction.into_inner())
                .into();
            diffs.ry_origin = inv.transform_point3(diffs.ry_origin.into_inner()).into();
            diffs.ry_direction = inv
                .transform_vector3(diffs.ry_direction.into_inner())
                .into();
            Some(diffs)
        } else {
            None
        };
        Ray::new_with_differentials(origin, direction, time, differentials)
    }

    /// Transform a Hit from object space to world space (point, normals, mapping_point).
    fn hit(&self, hit: &mut Hit) {
        let m = self.eval(hit.time);
        hit.point = m.transform_point3(hit.point.into_inner()).into();
        hit.mapping_point = m.transform_point3(hit.mapping_point.into_inner()).into();

        // Inverse-transpose for normals
        let inv = m.inverse();
        let inv_mat = Mat3A::from_cols(inv.matrix3.col(0), inv.matrix3.col(1), inv.matrix3.col(2));
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
        self.eval(time).transform_vector3(dir.into_inner()).into()
    }

    /// Transform a point (applies full affine: rotation + translation + scale).
    fn transform_point(&self, point: Point3, time: f32) -> Point3 {
        self.eval(time).transform_point3(point.into_inner()).into()
    }

    /// Transform a normal vector using the inverse-transpose.
    fn transform_normal(&self, normal: Direction3, time: f32) -> Direction3 {
        let m = self.eval(time);
        let inv = m.inverse();
        let inv_mat = Mat3A::from_cols(inv.matrix3.col(0), inv.matrix3.col(1), inv.matrix3.col(2));
        (inv_mat.transpose() * normal.into_inner()).into()
    }
}

/// Decomposed representation of an Affine3A transform into translation, rotation, and scale
/// components.
#[derive(Clone, Copy)]
struct Decomposed {
    translation: Vec3,
    rotation: Quat,  // unit quaternion
    scale: Affine3A, // upper-left 3×3 (scale + shear)
}

impl Decomposed {
    /// Decompose an Affine3A into translation, rotation, and scale components.
    fn from_affine(m: Affine3A) -> Self {
        // Translation = last column of the 4×3 matrix
        let translation = m.translation.into();

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
        let translation = self.translation.lerp(other.translation, t).into();
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
    fn intersect<'a>(&'a self, ray: &Ray, ray_t: Interval) -> Option<MaterialHit<'a>> {
        // 1. Transform ray from world → object space
        let local_ray = self.xform.ray(ray);

        // 2. Intersect the inner object in object space
        let mut mat_hit = self.object.intersect(&local_ray, ray_t)?;

        // 3. Transform the hit from object → world space
        self.xform.hit(&mut mat_hit.hit);

        Some(mat_hit)
    }
}

impl<O: Sampleable + Intersectable, T: Transform> Sampleable for TransformObject<O, T> {
    fn pdf_value(&self, origin: Point3, direction: Direction3, time: f32) -> f32 {
        // Transform origin and direction into object space, delegate
        let m = self.xform.eval(time).inverse();
        let obj_origin = Point3(m.transform_point3(origin.into_inner()));
        let obj_dir = Direction3(m.transform_vector3(direction.into_inner()));
        self.object.pdf_value(obj_origin, obj_dir, time)
    }

    fn random_direction(&self, origin: Point3, u: f32, v: f32, time: f32) -> Direction3 {
        // Sample in object space, transform result to world space
        let m = self.xform.eval(time).inverse();
        let obj_origin = Point3(m.transform_point3(origin.into_inner()));
        let obj_dir = self.object.random_direction(obj_origin, u, v, time);
        // Transform direction back to world space
        self.xform.transform_direction(obj_dir, time)
    }

    fn sample_light(&self, origin: Point3, u: f32, v: f32, time: f32) -> LightSample {
        // Sample in object space
        let m = self.xform.eval(time).inverse();
        let obj_origin = Point3(m.transform_point3(origin.into_inner()));
        let local_sample = self.object.sample_light(obj_origin, u, v, time);

        // Transform results to world space
        let world_point = self.xform.transform_point(
            Point3(
                local_sample.direction.into_inner().normalize_or_zero() * local_sample.distance
                    + obj_origin.into_inner(),
            ),
            time,
        );
        let world_dir = Direction3(world_point.into_inner() - origin.into_inner());

        LightSample {
            direction: world_dir,
            normal: self.xform.transform_normal(local_sample.normal, time),
            distance: world_dir.into_inner().length(),
            pdf: local_sample.pdf, // Area PDF is invariant under rigid transforms
            emission: local_sample.emission,
        }
    }
}
