use crate::planar::Region2D;
use crate::sampler::Sampler;

/// Region type for a triangle (a ≥ 0, b ≥ 0, a+b ≤ 1).
#[derive(Clone)]
pub struct TriRegion;

impl Region2D for TriRegion {
    fn contains(&self, a: f64, b: f64) -> bool {
        a >= 0.0 && b >= 0.0 && (a + b) <= 1.0
    }

    fn area(&self) -> f64 {
        0.5
    }

    fn sample(&self, sampler: &mut dyn Sampler) -> (f64, f64) {
        // Barycentric sampling for uniform triangle distribution.
        let r1: f64 = sampler.get_next_1d();
        let r2: f64 = sampler.get_next_1d();
        let sqrt_r1 = r1.sqrt();
        let a = 1.0 - sqrt_r1;
        let b = r2 * sqrt_r1;
        (a, b)
    }
}
