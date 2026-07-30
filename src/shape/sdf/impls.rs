use glam::Vec3;

use super::SdfFn;
use crate::shape::sdf::dual::Scalar;

/// Space repetition — wraps any `SdfFn` into a periodic tiling.
///
/// Evaluates the inner SDF in the canonical cell `[-period/2, period/2]³`,
/// so the inner SDF should be centered at the origin.  Useful for creating
/// infinite or bounded-periodic structures without instantiating many copies.
///
/// # Example
/// ```ignore
/// let cell = SdfExpr::Sphere(0.5) - SdfExpr::Box(Vec3::splat(0.25));
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
    fn eval<T: Scalar>(&self, x: T, y: T, z: T) -> T {
        let px = T::from_f32(self.period.x);
        let py = T::from_f32(self.period.y);
        let pz = T::from_f32(self.period.z);

        let qx = x - px * (x / px).round();
        let qy = y - py * (y / py).round();
        let qz = z - pz * (z / pz).round();

        self.inner.eval(qx, qy, qz)
    }
}
