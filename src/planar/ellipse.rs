use std::f64::consts::PI;

use crate::planar::Region2D;
use crate::sampler::Sampler;

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

    fn sample(&self, sampler: &mut dyn Sampler) -> (f64, f64) {
        // Uniform sampling in the unit disk via polar coordinates.
        let r = sampler.get_next_1d().sqrt();
        let theta = sampler.get_next_1d() * 2.0 * PI;
        (r * theta.cos(), r * theta.sin())
    }
}
