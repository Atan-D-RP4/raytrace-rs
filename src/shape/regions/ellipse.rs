use std::f32::consts::PI;

use crate::shape::Region2D;

/// Region type for a unit-disk ellipse in (a,b) space.
#[derive(Clone)]
pub struct EllipseRegion;

impl Region2D for EllipseRegion {
    fn contains(&self, a: f32, b: f32) -> bool {
        (a * a + b * b) <= 1.0
    }

    fn uv(&self, a: f32, b: f32) -> (f32, f32) {
        (a * 0.5 + 0.5, b * 0.5 + 0.5)
    }

    fn area(&self) -> f32 {
        PI
    }

    fn sample(&self, u: f32, v: f32) -> (f32, f32) {
        let r = u.sqrt();
        let theta = v * 2.0 * PI;
        let (sin_theta, cos_theta) = theta.sin_cos();
        (r * cos_theta, r * sin_theta)
    }
}
