use std::f32::consts::GOLDEN_RATIO;
use std::sync::Arc;

use glam::{Mat3, Vec3};

use crate::bvh::aabb::Aabb;
use crate::intersect::Bounded;
use crate::intersect::interaction::Hit;
use crate::math::interval::Interval;
use crate::math::vec3::{Direction3, Point3};
use crate::ray::RayPacked;
use crate::shape::Shape3D;
use crate::shape::ShapeSurfaceSampling;
use crate::texture::UVDifferentiable;

mod dispatch;
pub mod dual;
pub mod expr;
mod impls;
#[cfg(test)]
mod tests;

pub use dispatch::DynEval;
pub use expr::SdfExpr;
pub use impls::{
    BoxSdf, CapsuleSdf, CylinderSdf, MandelbulbSdf, RoundBoxSdf, SdfRepeat, SphereSdf, TorusSdf,
};

use dual::{Dual, Scalar};

/// Maximum number of sphere-tracing steps before giving up. This is a hard limit to prevent
/// infinite loops on degenerate SDFs (e.g., a "spike" with a very small SDF value that never
/// reaches the surface).
const MAX_MARCH_STEPS: usize = 64;

/// Minimum physical-distance step to prevent sphere-tracing stall when the SDF value is extremely
/// small (noise, numerical imprecision).
const MIN_PHYSICAL_STEP: f32 = 1e-3;

/// Threshold for considering a sphere-tracing step to have "hit" the surface. If the SDF value
/// at the current point is less than this threshold, we consider it a hit and stop marching.
const HIT_EPSILON: f32 = 1e-3;

/// Physical distance to advance past the ray origin for self-intersection guarding. Shadow/bounce
/// rays start at the SDF surface; without this warmup they immediately self-intersect, killing NEE
/// and indirect light.
const SELF_INTERSECTION_GUARD: f32 = 1e-2;

/// Over-relaxation multiplier for sphere tracing. Values > 1.0 accelerate convergence on near-flat
/// surfaces. From Keinert et al. "Enhanced Sphere Tracing" (2014). Typical range: 1.3–1.7.
const SOR_FACTOR: f32 = 1.3;

/// Maximum SDF distance for which SOR is applied. Beyond this threshold the standard sphere-tracing
/// step is already near-optimal and over-relaxation risks overshooting the surface entirely.
const SOR_DIST_THRESHOLD: f32 = 1.0;

/// An SDF evaluation function, generic over scalar type.
///
/// `eval::<f32>(...)` → value path (sphere tracing) `eval::<Dual<3>>(...)` → value + gradient (normal)
///
/// The `DynEval` bound enables the expression tree's `Custom` escape hatch to
/// dispatch to the correct dual evaluation method for the concrete scalar
/// type.  All scalar types the SDF pipeline uses (`f32`, `Dual<f32, 3>`,
/// `Dual<Dual<f32, 3>, 3>`) implement it.
pub trait SdfFn: Send + Sync {
    fn eval<T: Scalar + DynEval>(&self, x: T, y: T, z: T) -> T;
}

/// Convenience blanket: any Box<dyn SdfFn> is itself SdfFn
impl<T: SdfFn + ?Sized> SdfFn for Box<T> {
    fn eval<U: Scalar + DynEval>(&self, x: U, y: U, z: U) -> U {
        (**self).eval(x, y, z)
    }
}

/// Convenience blanket: Arc<dyn SdfFn> is itself SdfFn
impl<T: SdfFn + ?Sized> SdfFn for Arc<T> {
    fn eval<U: Scalar + DynEval>(&self, x: U, y: U, z: U) -> U {
        (**self).eval(x, y, z)
    }
}

/// An SDF-evaluated shape, generic over the evaluation function.
///
/// The generic `F` is erased at the scene boundary through
/// `ShapeObject<SdfShape<F>, M>` → `Arc<dyn Intersectable>`.
#[derive(Clone)]
pub struct SdfShape<F: SdfFn> {
    sdf: F,
    bbox: Aabb,
}

impl<F: SdfFn> SdfShape<F> {
    pub fn new(sdf: F, bbox: Aabb) -> Self {
        Self { sdf, bbox }
    }

    /// Gradient via forward-mode AD (Dual<3>), fallback to central finite differences.
    ///
    /// Dual AD requires the SDF to use a single expression for both interior and exterior (e.g.
    /// `0.5·r/dr` negated inside vs outside) so the derivative path is the same regardless of
    /// escape status — see `scene.rs` Mandelbulb eval for the pattern.
    fn gradient(&self, p: Point3) -> Direction3 {
        // Dual AD: 1 eval, exact gradient.  Works when the SDF unifies interior/exterior
        // expressions (so Dual differentiates the same path regardless of escape status).
        let x = Dual::<f32, 3>::variable(0, p.x());
        let y = Dual::<f32, 3>::variable(1, p.y());
        let z = Dual::<f32, 3>::variable(2, p.z());
        let r = self.sdf.eval(x, y, z);
        let n = Direction3::new(r.tangent(0), r.tangent(1), r.tangent(2));
        let len = n.length();
        if len.is_finite() && len > 1e-4 {
            return n / len;
        }

        // Dual AD failed (singularity at this evaluation point) — finite differences.
        // 3 axes × 2 evals = 6 evaluations, works regardless of branching.
        let eps = 1e-4_f32;
        let px: [f32; 3] = p.into_inner().to_array();
        let hi = Vec3::new(
            self.sdf.eval::<f32>(px[0] + eps, px[1], px[2]),
            self.sdf.eval::<f32>(px[0], px[1] + eps, px[2]),
            self.sdf.eval::<f32>(px[0], px[1], px[2] + eps),
        );
        let lo = Vec3::new(
            self.sdf.eval::<f32>(px[0] - eps, px[1], px[2]),
            self.sdf.eval::<f32>(px[0], px[1] - eps, px[2]),
            self.sdf.eval::<f32>(px[0], px[1], px[2] - eps),
        );
        let gradient = (hi - lo) / (2.0 * eps);
        let len = gradient.length();

        if !len.is_finite() || len <= 1e-4 {
            Direction3::Y
        } else {
            Direction3(gradient / len)
        }
    }

    /// Compute mean curvature at a world-space point using single-pass second-order AD.
    ///
    /// Evaluates the SDF once at `p` with nested Dual<N, Dual<N, f32>> to obtain:
    /// - first partials (gradient): `result.v.d[i]`
    /// - second partials (Hessian): `result.d[i].d[j]`
    ///
    /// Curvature is then κ = (∇fᵀ·H·∇f − |∇f|²·tr(H)) / (2·|∇f|³), which for unit-gradient SDFs
    /// (|∇f| ≈ 1 at the surface) simplifies to κ = (nᵀ·H·n − tr(H)) / 2.
    fn mean_curvature(&self, p: Point3) -> f32 {
        let result = self.sdf.eval::<Dual<Dual<f32, 3>, 3>>(
            Dual::<Dual<f32, 3>, 3>::variable(0, Dual::<f32, 3>::variable(0, p.x())),
            Dual::<Dual<f32, 3>, 3>::variable(1, Dual::<f32, 3>::variable(1, p.y())),
            Dual::<Dual<f32, 3>, 3>::variable(2, Dual::<f32, 3>::variable(2, p.z())),
        );

        // Gradient ∇f
        // The gradient is a 3D vector of first partial derivatives. We extract it from the nested
        let g = Vec3::from(result.v.d);
        let g_len_sq = g.length_squared();
        if !g_len_sq.is_finite() || g_len_sq <= f32::EPSILON {
            // Degenerate case: gradient is zero, cannot compute curvature. Return 0.0 as a fallback.
            return 0.0;
        }
        let g_len = g_len_sq.sqrt();
        let g_len_cu = g_len_sq * g_len;

        // Hessian H (symmetric, ∂²f/∂xᵢ∂xⱼ)
        // The Hessian is a 3x3 matrix of second partial derivatives. We extract it from the nested
        // Dual structure.
        let hessian = Mat3::from_cols_array_2d(&result.d.map(|row| row.d));

        // ∇fᵀ·H·∇f
        // This is a quadratic form: gᵀ·H·g = Σᵢ Σⱼ gᵢ Hᵢⱼ gⱼ
        let g_h_g = hessian.mul_vec3(g).dot(g);

        // tr(H) = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²
        // The trace is the sum of the diagonal elements of the Hessian matrix.
        let trace = hessian.diagonal().element_sum();

        // κ = (∇fᵀ·H·∇f − |∇f|²·tr(H)) / (2·|∇f|³)
        (g_h_g - g_len_sq * trace) / (2.0 * g_len_cu)
    }

    /// Sphere-tracing intersection test. Returns the first hit (t, p) if any.
    ///
    /// Handles several edge cases common in fractal DE rendering:
    /// **Non-unit ray direction** — steps are scaled by 1/|dir| because SDF distances are physical
    /// units but we march in ray-parameter space.
    /// **Self-intersection guard** — bounce rays start at the surface where |d| < HIT_EPSILON and
    /// would immediately self-intersect. A warmup loop advances past this zone before the main
    /// march.
    /// **Fractal DE overshoot** — the Mandelbulb's distance estimate is not a true SDF (|∇f| ≠ 1).
    /// The backward step from an interior point can re-overshoot the surface, creating a limit
    /// cycle. Bisection between the last outside `t` and the current inside `t` guarantees
    /// convergence.
    /// **Fractal DE overestimation** — camera rays from far away converge to a point where d
    /// approaches 0 from above (never crosses into negative). Without the `guard_broke_early` flag,
    /// these hits would be rejected.
    fn march<const N: usize>(
        &self,
        ray: &RayPacked<N>,
        ray_t: Interval<N>,
    ) -> Option<(f32, Point3)> {
        // SDF distances are physical; scale steps by 1/|dir| (ray-parameter space).
        let dir_len = ray.direction().length();
        if dir_len <= 0.0 {
            return None;
        }
        let inv_dir_len = dir_len.recip();

        let mut t = ray_t.min_value();

        // Ray was ever outside (d > 0). Surface-originating inward rays never escape the interior —
        // this flag distinguishes overshoot from inward.
        let mut was_outside = false;

        // Last outside t — bisection anchor for fractal DE overshoot (Mandelbulb's backward-step
        // oscillates on interior points).
        let mut t_outside = t;

        // Warmup: advance past the |d| < HIT_EPSILON zone near ray_t.min
        // (bounce rays start at the surface and would self-intersect otherwise).
        let guard_end = ray_t.min_value() + SELF_INTERSECTION_GUARD * inv_dir_len;
        while t < guard_end {
            let p = ray.at(t);
            let d = self.sdf.eval::<f32>(p.x(), p.y(), p.z());
            // Small d > 0 (DE underestimates) still counts as outside.
            if d > 0.0 {
                was_outside = true;
            }
            if d.abs() >= HIT_EPSILON {
                break;
            }
            t += MIN_PHYSICAL_STEP * inv_dir_len;
            if t > ray_t.max_value() {
                return None;
            }
        }
        // Camera rays break early (far start); surface rays run full course.
        let guard_broke_early = t < guard_end; // false → surface ray (full warmup)

        for _ in 0..MAX_MARCH_STEPS {
            let p = ray.at(t);
            let d = self.sdf.eval::<f32>(p.x(), p.y(), p.z());

            if d.abs() < HIT_EPSILON {
                if t <= guard_end {
                    t += MIN_PHYSICAL_STEP * inv_dir_len;
                    continue;
                }
                // Accept: crossed into set (d < 0) or camera ray (d ≥ 0 but far start).
                // Without the camera-ray case, fractal DE overestimation would reject
                // every hit — convergence approaches from above and never crosses d < 0.
                if d < 0.0 || guard_broke_early {
                    return Some((t, p));
                }
                // Outward surface ray still near start — not a real hit.
                t += MIN_PHYSICAL_STEP * inv_dir_len;
                continue;
            }

            if t > ray_t.max_value() {
                return None;
            }

            if d > 0.0 {
                t_outside = t;
                let std_step = d.max(MIN_PHYSICAL_STEP);
                // Over-relaxed step (SOR) accelerates grazing convergence when
                // close to the surface; far from the surface use standard step.
                let step = if d <= SOR_DIST_THRESHOLD {
                    std_step * SOR_FACTOR
                } else {
                    std_step
                };
                t += step * inv_dir_len;
            } else if d < 0.0 && !was_outside {
                // Surface-originating inward: step forward (standard would go backward)
                t += (-d).max(MIN_PHYSICAL_STEP) * inv_dir_len;
            } else {
                // Overshoot: bisect — fractal DE inside-distance oscillates,
                // halving the interval guarantees convergence.
                t = (t_outside + t) * 0.5;
            }
        }
        None
    }
}

impl<F: SdfFn> Shape3D for SdfShape<F> {
    /// Intersection test: returns the first hit if any.
    fn intersect_shape<const N: usize>(
        &self,
        ray: &RayPacked<N>,
        ray_t: Interval<N>,
    ) -> Option<Hit> {
        if let Some((t, p)) = self.march(ray, ray_t) {
            // Compute the normal at the hit point
            let mut normal = self.gradient(p);
            // For fractal DEs (Mandelbulb, etc.), bisection may converge to a point slightly inside
            // the set where the Dual gradient follows the inside branch and points inward.
            // Orient the normal to face against the incoming ray (standard ray tracing convention:
            // normal · ray_direction < 0).
            normal *= -normal.dot(ray.direction().into_inner()).signum();

            // Nudge the hit point slightly outward along the normal. Bisection converges to a point
            // that may be slightly inside the set (negative SDF). Without this, downstream rays
            // (NEE shadows, scatter bounces) start interior and immediately exit the far side,
            // causing self-occlusion that kills all illumination.
            let hit_point = p + normal.into_inner() * 1e-3;

            // Fill in the Hit record with the intersection details. SDFs do not have natural UV
            // coordinates, so we leave them as None.
            let mut hit = Hit::new(
                t, hit_point, hit_point, normal, None, // UV — SDFs have no natural UV
                None,
            );
            // Fill in curvature if requested (e.g., for bump mapping or shading effects)
            if ray.differentials.is_some() {
                let curvature = self.mean_curvature(p);
                hit.curvature = -curvature;
            }

            Some(hit)
        } else {
            None
        }
    }

    /// Occlusion test: returns true if the ray intersects the SDF surface within the interval.
    fn occluded_shape<const N: usize>(&self, ray: &RayPacked<N>, ray_t: Interval<N>) -> bool {
        self.march(ray, ray_t).is_some()
    }
}

impl<F: SdfFn> Bounded for SdfShape<F> {
    fn bounding_box(&self) -> Aabb {
        self.bbox
    }
}

impl<F: SdfFn> ShapeSurfaceSampling for SdfShape<F> {
    fn area(&self) -> f32 {
        self.bounding_box().surface_area()[0]
    }

    fn sample(&self, u: f32, v: f32, time: f32) -> (Point3, Direction3) {
        // Rejection sample within bounding box using SDF as inside test.
        // Inefficient for large boxes — user should provide a tight bbox.
        let bbox = self.bounding_box();
        let dx = bbox.max[0][0] - bbox.min[0][0];
        let dy = bbox.max[1][0] - bbox.min[1][0];
        let dz = bbox.max[2][0] - bbox.min[2][0];
        let mut ru = u;
        let mut rv = v;
        let mut rt = time;
        // Scramble constants (fractional parts of the golden ratio and its powers)
        // for pseudo-random jitter per rejection try.
        let scramble = |x: f32| (x * GOLDEN_RATIO).fract();
        for _ in 0..32 {
            let p = Point3::new(
                bbox.min[0][0] + ru * dx,
                bbox.min[1][0] + rv * dy,
                bbox.min[2][0] + rt * dz,
            );
            let d = self.sdf.eval::<f32>(p.x(), p.y(), p.z());
            if d <= 0.0 {
                // inside or on surface — compute normal at the hit point
                return (p, self.gradient(p));
            }
            ru = scramble(ru);
            rv = scramble(rv);
            rt = scramble(rt);
        }
        // Fallback: center of the bounding box (rejection budget exhausted)
        let center = bbox.centroid_point();
        (center, self.gradient(center))
    }
}

impl<F: SdfFn> UVDifferentiable for SdfShape<F> {
    fn uv_gradient(&self, _p: &Point3) -> (Direction3, Direction3) {
        (Direction3::ZERO, Direction3::ZERO) // no natural UV for SDFs
    }
}
