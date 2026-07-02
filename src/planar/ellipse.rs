use std::f64::consts::PI;

use crate::planar::Region2D;

/// Region type for a unit-disk ellipse in (a,b) space.
#[derive(Clone)]
pub struct EllipseRegion;

impl Region2D for EllipseRegion {
    fn contains(&self, a: f64, b: f64) -> bool {
        (a * a + b * b) <= 1.0
    }

    fn uv(&self, a: f64, b: f64) -> (f64, f64) {
        (a * 0.5 + 0.5, b * 0.5 + 0.5)
    }

    fn area(&self) -> f64 {
        PI
    }

    fn sample(&self, u: f64, v: f64) -> (f64, f64) {
        let r = u.sqrt();
        let theta = v * 2.0 * PI;
        let (sin_theta, cos_theta) = theta.sin_cos();
        (r * cos_theta, r * sin_theta)
    }
}
