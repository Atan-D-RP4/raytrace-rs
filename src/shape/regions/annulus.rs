use std::f32::consts::PI;

use crate::shape::Region2D;

/// Region type for an annular (ring) region with configurable inner radius.
#[derive(Clone)]
pub struct AnnulusRegion {
    pub inner: f32,
}

impl Region2D for AnnulusRegion {
    fn contains(&self, a: f32, b: f32) -> bool {
        let radius = (a * a + b * b).sqrt();
        radius >= self.inner && radius <= 1.0
    }

    fn uv(&self, a: f32, b: f32) -> (f32, f32) {
        (a * 0.5 + 0.5, b * 0.5 + 0.5)
    }

    fn area(&self) -> f32 {
        PI * (1.0 - self.inner * self.inner)
    }

    fn sample(&self, u: f32, v: f32) -> (f32, f32) {
        let r = (self.inner * self.inner + u * (1.0 - self.inner * self.inner)).sqrt();
        let theta = v * 2.0 * PI;
        let (sin_theta, cos_theta) = theta.sin_cos();
        (r * cos_theta, r * sin_theta)
    }
}
