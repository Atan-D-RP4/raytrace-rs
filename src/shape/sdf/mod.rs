pub mod dual;

use std::sync::Arc;

use glam::Vec3;

use crate::bvh::aabb::Aabb;
use crate::hittable::Hit;
use crate::interval::Interval;
use crate::ray::Ray;
use crate::shape::Shape3D;
use crate::shape::ShapeSurfaceSampling;
use crate::shape::sdf::dual::{Dual, Scalar};
use crate::texture::UVDifferentiable;
use crate::vec3::{Direction3, Point3};

/// Maximum number of sphere-tracing steps before giving up. This is a hard limit to prevent
/// infinite loops on degenerate SDFs (e.g., a "spike" with a very small SDF value that never
/// reaches the surface).
const MAX_MARCH_STEPS: usize = 64;

/// Minimum physical-distance step to prevent sphere-tracing stall when the SDF value is extremely
/// small (noise, numerical imprecision).
const MIN_PHYSICAL_STEP: f32 = 1e-3;

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
/// `eval::<f64>(...)` → value path (sphere tracing) `eval::<Dual<3>>(...)` → value + gradient (normal)
pub trait SdfFn: Send + Sync {
    fn eval<T: Scalar>(&self, x: T, y: T, z: T) -> T;
}

/// Convenience blanket: any Box<dyn SdfFn> is itself SdfFn
impl<T: SdfFn + ?Sized> SdfFn for Box<T> {
    fn eval<U: Scalar>(&self, x: U, y: U, z: U) -> U {
        (**self).eval(x, y, z)
    }
}

/// Convenience blanket: Arc<dyn SdfFn> is itself SdfFn
impl<T: SdfFn + ?Sized> SdfFn for Arc<T> {
    fn eval<U: Scalar>(&self, x: U, y: U, z: U) -> U {
        (**self).eval(x, y, z)
    }
}

/// An SDF-evaluated shape, generic over the evaluation function.
///
/// The generic `F` is erased at the scene boundary through
/// `ShapeObject<SdfShape<F>, M>` → `Arc<dyn Intersectable>`.
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
        if len > 1e-4 {
            return n / len;
        }

        // Dual AD failed (singularity at this evaluation point) — finite differences.
        // 3 axes × 2 evals = 6 evaluations, works regardless of branching.
        let eps = 1e-4_f32;
        let px: [f32; 3] = [p.x(), p.y(), p.z()];
        let mut g = [0.0_f32; 3];
        for axis in 0..3 {
            let mut p_lo = px;
            p_lo[axis] -= eps;
            let mut p_hi = px;
            p_hi[axis] += eps;
            let d_lo = self.sdf.eval::<f32>(p_lo[0], p_lo[1], p_lo[2]);
            let d_hi = self.sdf.eval::<f32>(p_hi[0], p_hi[1], p_hi[2]);
            g[axis] = (d_hi - d_lo) / (2.0 * eps);
        }
        let len = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
        if len < 1e-10 {
            Vec3::from((0.0, 1.0, 0.0)).into() // still degenerate — arbitrary fallbac
        } else {
            (Vec3::from_slice(&g) / len).into()
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
        let gx = result.v.d[0];
        let gy = result.v.d[1];
        let gz = result.v.d[2];
        let g_len_sq = gx * gx + gy * gy + gz * gz;
        if g_len_sq <= f32::EPSILON {
            return 0.0; // degenerate — flat or singularity
        }
        let g_len = g_len_sq.sqrt();
        let g_len_cu = g_len_sq * g_len;

        // Hessian H (symmetric, ∂²f/∂xᵢ∂xⱼ)
        let h00 = result.d[0].d[0];
        let h01 = result.d[0].d[1];
        let h02 = result.d[0].d[2];
        let h11 = result.d[1].d[1];
        let h12 = result.d[1].d[2];
        let h22 = result.d[2].d[2];

        // ∇fᵀ·H·∇f
        let g_h_g = gx * (gx * h00 + gy * h01 + gz * h02)
            + gy * (gx * h01 + gy * h11 + gz * h12)
            + gz * (gx * h02 + gy * h12 + gz * h22);

        // tr(H)
        let trace = h00 + h11 + h22;

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
    fn march(&self, ray: &Ray, ray_t: Interval) -> Option<(f32, Point3)> {
        // SDF distances are physical; scale steps by 1/|dir| (ray-parameter space).
        let dir_len = ray.direction.length();
        if dir_len <= 0.0 {
            return None;
        }
        let inv_dir_len = 1.0 / dir_len;

        let mut t = ray_t.min;

        // Ray was ever outside (d > 0). Surface-originating inward rays never escape the interior —
        // this flag distinguishes overshoot from inward.
        let mut was_outside = false;

        // Last outside t — bisection anchor for fractal DE overshoot (Mandelbulb's backward-step
        // oscillates on interior points).
        let mut t_outside = t;

        // Warmup: advance past the |d| < HIT_EPSILON zone near ray_t.min
        // (bounce rays start at the surface and would self-intersect otherwise).
        let guard_end = ray_t.min + SELF_INTERSECTION_GUARD * inv_dir_len;
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
            if t > ray_t.max {
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

            if t > ray_t.max {
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
    fn intersect_shape(&self, ray: &Ray, ray_t: Interval) -> Option<Hit> {
        if let Some((t, p)) = self.march(ray, ray_t) {
            // Compute the normal at the hit point
            let mut normal = self.gradient(p);
            // For fractal DEs (Mandelbulb, etc.), bisection may converge to a point slightly inside
            // the set where the Dual gradient follows the inside branch and points inward.
            // Orient the normal to face against the incoming ray (standard ray tracing convention:
            // normal · ray_direction < 0).
            normal *= normal.dot(ray.direction.into_inner()).signum();
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
                hit.curvature = curvature;
            }

            Some(hit)
        } else {
            None
        }
    }

    /// Occlusion test: returns true if the ray intersects the SDF surface within the interval.
    fn occluded_shape(&self, ray: &Ray, ray_t: Interval) -> bool {
        self.march(ray, ray_t).is_some()
    }

    fn bounding_box(&self) -> Aabb {
        self.bbox
    }
}

impl<F: SdfFn> ShapeSurfaceSampling for SdfShape<F> {
    fn area(&self) -> f32 {
        let bbox = self.bounding_box();
        let dx = bbox.max[0][0] - bbox.min[0][0];
        let dy = bbox.max[1][0] - bbox.min[1][0];
        let dz = bbox.max[2][0] - bbox.min[2][0];

        // Surface area of the bounding box as a rough approximation for SDF surface area.
        2.0 * (dx * dy + dy * dz + dz * dx)
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
        let scramble = |x: f32| (x * 1.618_034).fract();
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
        let center = Point3::new(
            bbox.min[0][0] + 0.5 * dx,
            bbox.min[1][0] + 0.5 * dy,
            bbox.min[2][0] + 0.5 * dz,
        );
        (center, self.gradient(center))
    }
}

impl<F: SdfFn> UVDifferentiable for SdfShape<F> {
    fn uv_gradient(&self, _p: &Point3) -> (Direction3, Direction3) {
        (Direction3::ZERO, Direction3::ZERO) // no natural UV for SDFs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interval::Interval;
    use crate::ray::Ray;

    struct TestSphere {
        radius: f32,
    }

    impl SdfFn for TestSphere {
        fn eval<T: Scalar>(&self, x: T, y: T, z: T) -> T {
            (x * x + y * y + z * z).sqrt() - T::from_f32(self.radius)
        }
    }

    #[test]
    fn sphere_tracing_hits_sphere_from_outside() {
        let sdf = SdfShape::new(
            TestSphere { radius: 1.0 },
            Aabb::from_corners(Point3::new(-1.1, -1.1, -1.1), Point3::new(1.1, 1.1, 1.1)),
        );
        // Ray from (0, 0, 5) toward origin along -z
        let ray = Ray::new(Point3::new(0.0, 0.0, 5.0), Direction3::new(0.0, 0.0, -1.0));
        let ray_t = Interval::from(0.001, 100.0);
        let hit = sdf.intersect_shape(&ray, ray_t);
        assert!(hit.is_some(), "sphere should be hit");
        let hit = hit.unwrap();
        // Expected: hit at t ≈ 4 (from z=5, sphere surface at z=1)
        assert!((hit.time - 4.0).abs() < 0.01, "hit at t={}", hit.time);
        // Front face at z=1 (sphere center at origin, radius 1)
        assert!(
            (hit.point.z() - 1.0).abs() < 0.01,
            "hit point z={}",
            hit.point.z()
        );
    }

    #[test]
    fn sphere_tracing_misses() {
        let sdf = SdfShape::new(
            TestSphere { radius: 1.0 },
            Aabb::from_corners(Point3::new(-1.1, -1.1, -1.1), Point3::new(1.1, 1.1, 1.1)),
        );
        let ray = Ray::new(
            Point3::new(0.0, 0.0, 5.0),
            Direction3::new(0.0, 2.0, -1.0).normalize(),
        );
        let ray_t = Interval::from(0.001, 100.0);
        let hit = sdf.intersect_shape(&ray, ray_t);
        assert!(
            hit.is_none(),
            "should miss, but got hit at t={:?}",
            hit.map(|h| h.time)
        );
    }

    #[test]
    fn cylinder_sdf_hit_from_camera_angle() {
        struct Cylinder {
            r: f32,
            h: f32,
        }
        impl SdfFn for Cylinder {
            fn eval<T: Scalar>(&self, x: T, y: T, z: T) -> T {
                let d = (x * x + z * z).sqrt() - T::from_f32(self.r);
                let h = y.abs() - T::from_f32(self.h / 2.0);
                d.max(h)
            }
        }
        let sdf = SdfShape::new(
            Cylinder { r: 50.0, h: 100.0 },
            Aabb::from_corners(
                Point3::new(-50.0, -50.0, -50.0),
                Point3::new(50.0, 50.0, 50.0),
            ),
        );
        // Camera at (278, 278, -800), ray toward cylinder center (0, 0, 0)
        let origin = Point3::new(278.0, 278.0, -800.0);
        let dir = Direction3::new(-278.0, -278.0, 800.0).normalize();
        let ray = Ray::new(origin, dir);
        let ray_t = Interval::from(0.001, 1000.0);
        let hit = sdf.intersect_shape(&ray, ray_t);
        assert!(hit.is_some(), "camera ray should hit cylinder");
        let hit = hit.unwrap();
        // Hit should be within the cylinder's geometry
        assert!(
            hit.time > 800.0 && hit.time < 900.0,
            "unexpected t={}",
            hit.time
        );
        assert!(
            hit.point.x() > -50.0 && hit.point.x() < 50.0,
            "x out of range: {}",
            hit.point.x()
        );
        assert!(
            hit.point.z() > -50.0 && hit.point.z() < 50.0,
            "z out of range: {}",
            hit.point.z()
        );
    }

    #[test]
    fn cylinder_sdf_does_not_self_intersect_from_surface() {
        struct Cylinder {
            r: f32,
            h: f32,
        }
        impl SdfFn for Cylinder {
            fn eval<T: Scalar>(&self, x: T, y: T, z: T) -> T {
                let d = (x * x + z * z).sqrt() - T::from_f32(self.r);
                let h = y.abs() - T::from_f32(self.h / 2.0);
                d.max(h)
            }
        }
        let sdf = SdfShape::new(
            Cylinder { r: 50.0, h: 100.0 },
            Aabb::from_corners(
                Point3::new(-50.0, -50.0, -50.0),
                Point3::new(50.0, 50.0, 50.0),
            ),
        );
        // Surface-originating ray: starts at the cylinder surface pointing outward along +x.
        // Without the self-intersection guard, this would return a hit at t ≈ 0.001.
        // With the guard, it should escape the surface and find no hit within a short interval.
        let origin = Point3::new(50.0, 0.0, 0.0);
        let dir = Direction3::new(1.0, 0.0, 0.0); // outward from surface
        let ray = Ray::new(origin, dir);
        // Short interval: just enough to escape the self-intersection zone
        let ray_t = Interval::from(0.001, 0.1);
        let hit = sdf.intersect_shape(&ray, ray_t);
        assert!(
            hit.is_none(),
            "surface-originating ray should not self-intersect, but got hit at t={:?}",
            hit.map(|h| h.time)
        );
    }

    #[test]
    fn cylinder_sdf_non_normalized_camera_ray() {
        // Simulates the camera ray from the Cornell box sdf_test scene:
        //   camera at (278, 278, -800), direction toward cylinder (0, 0, 0)
        //   → direction = (-278, -278, 800) with length ≈ 891.
        // Before the inv_dir_len fix, the sphere-tracing step `t += d` would
        // overshoot the cylinder (first step jumps from t=0.001 to t=≈797),
        // skipping the surface at t≈0.941 entirely.
        struct Cylinder {
            r: f32,
            h: f32,
        }
        impl SdfFn for Cylinder {
            fn eval<T: Scalar>(&self, x: T, y: T, z: T) -> T {
                let d = (x * x + z * z).sqrt() - T::from_f32(self.r);
                let h = y.abs() - T::from_f32(self.h / 2.0);
                d.max(h)
            }
        }
        let sdf = SdfShape::new(
            Cylinder { r: 50.0, h: 100.0 },
            Aabb::from_corners(
                Point3::new(-50.0, -50.0, -50.0),
                Point3::new(50.0, 50.0, 50.0),
            ),
        );
        // Camera ray: non-normalized direction, |dir| ≈ 891
        let origin = Point3::new(278.0, 278.0, -800.0);
        let dir = Point3::new(0.0, 0.0, 0.0) - origin; // (-278, -278, 800)
        let ray = Ray::new(
            Point3::new(278.0, 278.0, -800.0),
            Direction3(dir.into_inner()),
        );
        let ray_t = Interval::from(0.001, 1000.0);
        let hit = sdf.intersect_shape(&ray, ray_t);
        assert!(hit.is_some(), "camera ray should hit cylinder");
        let hit = hit.unwrap();
        // Hit should be at t ≈ 0.941 (ray-parameter units), not t ≈ 797
        // which is what the unfixed sphere tracing would produce.
        assert!(
            hit.time > 0.9 && hit.time < 1.0,
            "expected t≈0.941, got t={}",
            hit.time
        );
        assert!(
            (hit.point.x() - (-50.0)).abs() > 40.0,
            "hit should be on front face (not opposite side), got x={}",
            hit.point.x()
        );
    }

    #[test]
    fn cylinder_sdf_self_intersection_guard_still_hits_real_surface() {
        struct Cylinder {
            r: f32,
            h: f32,
        }
        impl SdfFn for Cylinder {
            fn eval<T: Scalar>(&self, x: T, y: T, z: T) -> T {
                let d = (x * x + z * z).sqrt() - T::from_f32(self.r);
                let h = y.abs() - T::from_f32(self.h / 2.0);
                d.max(h)
            }
        }
        let sdf = SdfShape::new(
            Cylinder { r: 50.0, h: 100.0 },
            Aabb::from_corners(
                Point3::new(-50.0, -50.0, -50.0),
                Point3::new(50.0, 50.0, 50.0),
            ),
        );
        // Surface-originating ray with a LONG interval should still hit the
        // OPPOSITE side of the cylinder after escaping the start surface.
        let origin = Point3::new(50.0, 0.0, 0.0);
        let dir = Direction3::new(1.0, 0.0, 0.0); // through +x, should exit AABB
        let ray = Ray::new(origin, dir);
        // Long enough to pass through the cylinder and exit AABB
        let ray_t = Interval::from(0.001, 200.0);
        let hit = sdf.intersect_shape(&ray, ray_t);
        // No hit because the ray goes away from the cylinder (outward +x)
        // Actually, the cylinder extends from -50 to 50 in x, so at x=50 the
        // outward+1 ray goes away from the cylinder and never hits it again.
        // This is the correct behavior for a surface-originating outward ray.
        // For a ray going inward (through the cylinder), we should get a hit
        // on the opposite side.
        assert!(
            hit.is_none(),
            "outward surface ray should not re-hit cylinder, but got hit at t={:?}",
            hit.map(|h| h.time)
        );

        // Now test an inward ray: from the surface going INTO the cylinder.
        // With the was_outside=false forward-step strategy, the sphere tracing
        // traverses the interior and converges to the opposite surface.
        let dir = Direction3::new(-1.0, 0.0, 0.0); // inward through cylinder
        let ray = Ray::new(origin, dir);
        let hit = sdf.intersect_shape(&ray, ray_t);
        assert!(
            hit.is_some(),
            "inward surface ray should hit opposite side of cylinder"
        );
        let hit = hit.unwrap();
        // Hit should be at x ≈ -50 (the opposite side), t ≈ 100
        assert!(
            (hit.point.x() - (-50.0)).abs() < 1.0,
            "expected hit at x≈-50, got x={} at t={}",
            hit.point.x(),
            hit.time
        );
    }
}
