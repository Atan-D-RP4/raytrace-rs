use glam::Vec3;

use crate::math::vec3::Point3;
use crate::shape::sdf::SdfFn;
use crate::shape::sdf::dispatch::DynEval;
use crate::shape::sdf::dual::Scalar;

/// Space repetition — wraps any `SdfFn` into a periodic tiling.
///
/// Evaluates the inner SDF in the canonical cell `[-period/2, period/2]³`,
/// so the inner SDF should be centered at the origin.  Useful for creating
/// infinite or bounded-periodic structures without instantiating many copies.
///
/// # Example
/// ```ignore
/// let cell = SdfExpr::Sphere(SphereSdf::new(0.5)) - SdfExpr::Box(BoxSdf::new(Vec3::splat(0.25)));
/// let lattice = SdfRepeat::new(Vec3::splat(2.0), cell);
/// let shape = SdfShape::new(lattice, bbox);
/// ```
pub struct SdfRepeat<F: SdfFn> {
    /// Repeat period along each axis.  The inner SDF is evaluated in
    /// `[-period/2, period/2]` along each axis.
    pub period: Vec3,
    inner: F,
}

impl<F: SdfFn> SdfRepeat<F> {
    pub fn new(period: Vec3, inner: F) -> Self {
        Self { period, inner }
    }
}

impl<F: SdfFn> SdfFn for SdfRepeat<F> {
    /// Maps `(x,y,z)` into the nearest cell center using
    /// `q = p - period · round(p / period)`, then delegates to the inner SDF.
    fn eval<T: Scalar + DynEval>(&self, x: T, y: T, z: T) -> T {
        let px = T::from_f32(self.period.x);
        let py = T::from_f32(self.period.y);
        let pz = T::from_f32(self.period.z);

        let qx = x - px * (x / px).round();
        let qy = y - py * (y / py).round();
        let qz = z - pz * (z / pz).round();

        self.inner.eval(qx, qy, qz)
    }
}

/// Mandelbulb fractal SDF — IQ's `sdMandelbulb`.
#[derive(Clone)]
pub struct MandelbulbSdf {
    /// Exponent of the Mandelbulb iteration.  Common values are 8 or 10.
    power: i32,
    /// Maximum number of iterations to run before assuming the point is inside the set.
    iters: usize,
    /// Escape radius for the Mandelbulb iteration.  Common values are 2 or 4.
    bailout: f32,
}

impl MandelbulbSdf {
    pub fn new(power: i32, iters: usize, bailout: f32) -> Self {
        Self {
            power,
            iters,
            bailout,
        }
    }
}

impl SdfFn for MandelbulbSdf {
    fn eval<T: Scalar + DynEval>(&self, x: T, y: T, z: T) -> T {
        let mut zx = x;
        let mut zy = y;
        let mut zz = z;
        let mut dr = T::one();
        let mut escaped = false;
        let mut did_iterate = false;

        for _ in 0..self.iters {
            let r2 = zx * zx + zy * zy + zz * zz;
            if r2 > T::from_f32(self.bailout * self.bailout) {
                // Already outside the bailout sphere — point is outside the set.
                escaped = true;
                break;
            }

            did_iterate = true;
            let r = r2.sqrt();
            // Guard against polar singularity r ≈ 0.
            let r_safe = r.max(T::from_f32(1e-8));
            let inv_r = T::one() / r_safe;
            let r_xy = (zx * zx + zy * zy).sqrt();

            let ct = zz * inv_r;
            let st = r_xy * inv_r;

            let (cp, sp) = if r_xy >= T::from_f32(f32::EPSILON) {
                (zx / r_xy, zy / r_xy)
            } else {
                (T::one(), T::zero())
            };

            // Multiple-angle iteration for power n
            let mut cos_nt = ct;
            let mut sin_nt = st;
            for _ in 1..self.power {
                (cos_nt, sin_nt) = (cos_nt * ct - sin_nt * st, sin_nt * ct + cos_nt * st);
            }

            let mut cos_np = cp;
            let mut sin_np = sp;
            for _ in 1..self.power {
                (cos_np, sin_np) = (cos_np * cp - sin_np * sp, sin_np * cp + cos_np * sp);
            }

            dr = T::from_f32(self.power as f32) * r.powi(self.power - 1) * dr + T::one();

            let rn = r.powi(self.power);
            zx = rn * sin_nt * cos_np + x;
            zy = rn * sin_nt * sin_np + y;
            zz = rn * cos_nt + z;

            // Check bailout on the NEW z (after iteration) so dr is from the
            // iteration that caused escape, not the one before it.
            let r2_new = zx * zx + zy * zy + zz * zz;
            if r2_new > T::from_f32(self.bailout * self.bailout) {
                escaped = true;
                break;
            }
        }

        let r = (zx * zx + zy * zy + zz * zz).sqrt().max(T::from_f32(1e-8));
        // Exterior distance estimate (common for all paths so Dual AD
        // differentiates the same expression regardless of escape status).
        let mut de = T::from_f32(0.5) * r / dr;

        if did_iterate && !escaped {
            // Inside the set — negate the distance; the derivative of
            // -(0.5·r/dr) = -(0.5·dr⁻¹)·dr/dp which is continuous
            // across the surface boundary.
            de = -de;
        } else if !did_iterate {
            // No iteration ran — started outside the bailout sphere.
            // dr = 1, so de = 0.5·r.  Use Euclidean distance to bailout
            // sphere instead: same far-field limit, tighter near the
            // bailout sphere.  Negligible for gradient (never hit surface).
            de = (r - T::from_f32(self.bailout)).max(T::from_f32(1e-3));
        }
        de
    }
}

/// Cylinder aligned along the Y axis — IQ's `sdCylinder`.
#[derive(Clone)]
pub struct CylinderSdf {
    center: Point3,
    radius: f32,
    height: f32,
}

impl CylinderSdf {
    pub fn new(center: Point3, radius: f32, height: f32) -> Self {
        Self {
            center,
            radius,
            height,
        }
    }
}

impl SdfFn for CylinderSdf {
    fn eval<T: Scalar + DynEval>(&self, x: T, y: T, z: T) -> T {
        let cx = x - T::from_f32(self.center.x());
        let cy = y - T::from_f32(self.center.y());
        let cz = z - T::from_f32(self.center.z());
        let d = (cx * cx + cz * cz).sqrt() - T::from_f32(self.radius);
        let h = cy.abs() - T::from_f32(self.height / 2.0);
        d.max(h)
    }
}

/// Axis-aligned sphere centered at the origin — IQ's `sdSphere`.
///
/// `f(p) = |p| − r`. The standard building block for CSG compositing.
#[derive(Clone)]
pub struct SphereSdf {
    radius: f32,
}

impl SphereSdf {
    pub fn new(radius: f32) -> Self {
        Self { radius }
    }
}

impl SdfFn for SphereSdf {
    fn eval<T: Scalar + DynEval>(&self, x: T, y: T, z: T) -> T {
        (x * x + y * y + z * z).sqrt() - T::from_f32(self.radius)
    }
}

/// Axis-aligned box centered at the origin — IQ's `sdBox`.
///
/// `q = |p| − h`; `f(p) = |max(q, 0)| + min(max(q.x, max(q.y, q.z)), 0)`.
#[derive(Clone)]
pub struct BoxSdf {
    half: Vec3,
}

impl BoxSdf {
    pub fn new(half: Vec3) -> Self {
        Self { half }
    }
}

impl SdfFn for BoxSdf {
    fn eval<T: Scalar + DynEval>(&self, x: T, y: T, z: T) -> T {
        let zero = T::zero();
        let qx = x.abs() - T::from_f32(self.half.x);
        let qy = y.abs() - T::from_f32(self.half.y);
        let qz = z.abs() - T::from_f32(self.half.z);
        let ox = qx.max(zero);
        let oy = qy.max(zero);
        let oz = qz.max(zero);
        let outside = (ox * ox + oy * oy + oz * oz).sqrt();
        let inside = qx.max(qy).max(qz).min(zero);
        outside + inside
    }
}

/// Axis-aligned box with rounded edges — IQ's `sdRoundBox`.
///
/// `q = |p| − h + r`; `f(p) = |max(q, 0)| + min(max(q.x, max(q.y, q.z)), 0) − r`.
///
/// The flat faces sit at `|p| = h`; the edge rounding radius is `r`.
#[derive(Clone)]
pub struct RoundBoxSdf {
    half: Vec3,
    radius: f32,
}

impl RoundBoxSdf {
    pub fn new(half: Vec3, radius: f32) -> Self {
        Self { half, radius }
    }
}

impl SdfFn for RoundBoxSdf {
    fn eval<T: Scalar + DynEval>(&self, x: T, y: T, z: T) -> T {
        let zero = T::zero();
        let r = T::from_f32(self.radius);
        let qx = x.abs() - T::from_f32(self.half.x) + r;
        let qy = y.abs() - T::from_f32(self.half.y) + r;
        let qz = z.abs() - T::from_f32(self.half.z) + r;
        let ox = qx.max(zero);
        let oy = qy.max(zero);
        let oz = qz.max(zero);
        let outside = (ox * ox + oy * oy + oz * oz).sqrt();
        let inside = qx.max(qy).max(qz).min(zero);
        outside + inside - r
    }
}

/// Torus in the XZ-plane (Y axis of revolution) — IQ's `sdTorus`.
///
/// `q = (|(x, z)| − R, y)`; `f(p) = |q| − r`, with `R` the major radius
/// (distance from the origin to the tube center) and `r` the minor radius
/// (tube thickness).
#[derive(Clone)]
pub struct TorusSdf {
    major: f32,
    minor: f32,
}

impl TorusSdf {
    pub fn new(major: f32, minor: f32) -> Self {
        Self { major, minor }
    }
}

impl SdfFn for TorusSdf {
    fn eval<T: Scalar + DynEval>(&self, x: T, y: T, z: T) -> T {
        let qx = (x * x + z * z).sqrt() - T::from_f32(self.major);
        let qy = y;
        (qx * qx + qy * qy).sqrt() - T::from_f32(self.minor)
    }
}

/// Y-axis-aligned capsule (cylinder with hemispherical caps) — IQ's `sdCapsule`.
///
/// `q = (|(x, z)|, |y| − h)`; `f(p) = |q| − r`.
#[derive(Clone)]
pub struct CapsuleSdf {
    half_height: f32,
    radius: f32,
}

impl CapsuleSdf {
    pub fn new(half_height: f32, radius: f32) -> Self {
        Self {
            half_height,
            radius,
        }
    }
}

impl SdfFn for CapsuleSdf {
    fn eval<T: Scalar + DynEval>(&self, x: T, y: T, z: T) -> T {
        let qx = (x * x + z * z).sqrt();
        let qy = y.abs() - T::from_f32(self.half_height);
        (qx * qx + qy * qy).sqrt() - T::from_f32(self.radius)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn val<F: SdfFn>(sdf: &F, x: f32, y: f32, z: f32) -> f32 {
        sdf.eval::<f32>(x, y, z)
    }

    #[test]
    fn sphere_sign() {
        let s = SphereSdf::new(1.0);
        assert!(val(&s, 0.0, 0.0, 0.0) < 0.0);
        assert_eq!(val(&s, 1.0, 0.0, 0.0), 0.0);
        assert!(val(&s, 2.0, 0.0, 0.0) > 0.0);
    }

    #[test]
    fn box_sign() {
        let b = BoxSdf::new(Vec3::ONE);
        assert!(val(&b, 0.0, 0.0, 0.0) < 0.0);
        assert_eq!(val(&b, 1.0, 1.0, 1.0), 0.0);
        assert!(val(&b, 2.0, 0.0, 0.0) > 0.0);
        assert!(val(&b, 1.5, 1.5, 1.5) > 0.0); // outside the corner diagonal
    }

    #[test]
    fn round_box_sign() {
        let r = RoundBoxSdf::new(Vec3::ONE, 0.5);
        assert!(val(&r, 0.0, 0.0, 0.0) < 0.0);
        assert!(val(&r, 0.75, 0.75, 0.0) < 0.0); // inside the rounded edge region
        assert_eq!(val(&r, 1.0, 0.0, 0.0), 0.0); // face sits at |x| = half
        assert!(val(&r, 1.5, 1.5, 0.0) > 0.0); // outside the corner
    }

    #[test]
    fn torus_sign() {
        let t = TorusSdf::new(2.0, 0.5);
        assert!(val(&t, 2.0, 0.0, 0.0) < 0.0); // on the major circle, inside the tube
        assert_eq!(val(&t, 2.5, 0.0, 0.0), 0.0); // outer surface of the tube
        assert!(val(&t, 0.0, 0.0, 0.0) > 0.0); // hole in the middle
    }

    #[test]
    fn capsule_sign() {
        let c = CapsuleSdf::new(1.0, 0.5);
        assert!(val(&c, 0.0, 0.0, 0.0) > 0.0); // between the caps, off the surface
        assert!(val(&c, 0.0, 1.2, 0.0) < 0.0); // inside the cap
        assert!(val(&c, 0.0, 1.6, 0.0) > 0.0); // above the cap
    }
}
