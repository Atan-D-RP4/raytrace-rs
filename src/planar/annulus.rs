use std::f64::consts::PI;

use crate::planar::Region2D;

/// Region type for an annular (ring) region with configurable inner radius.
#[derive(Clone)]
pub struct AnnulusRegion {
    pub inner: f64,
}

impl Region2D for AnnulusRegion {
    fn contains(&self, a: f64, b: f64) -> bool {
        let radius = (a * a + b * b).sqrt();
        radius >= self.inner && radius <= 1.0
    }

    fn uv(&self, a: f64, b: f64) -> (f64, f64) {
        (a * 0.5 + 0.5, b * 0.5 + 0.5)
    }

    fn area(&self) -> f64 {
        PI * (1.0 - self.inner * self.inner)
    }

    fn sample(&self, u: f64, v: f64) -> (f64, f64) {
        let r = (self.inner * self.inner + u * (1.0 - self.inner * self.inner)).sqrt();
        let theta = v * 2.0 * PI;
        (r * theta.cos(), r * theta.sin())
    }
}
