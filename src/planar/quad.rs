use crate::interval::Interval;

use crate::planar::Region2D;
use rand::RngExt;

/// Region type for a full parallelogram (unit square in (a,b) space).
#[derive(Clone)]
pub struct QuadRegion;

impl Region2D for QuadRegion {
    fn contains(&self, a: f64, b: f64) -> bool {
        let unit = Interval::from(0., 1.);
        unit.contains(a) && unit.contains(b)
    }

    fn area(&self) -> f64 {
        1.0
    }

    fn sample(&self, rng: &mut dyn rand::Rng) -> (f64, f64) {
        (rng.random::<f64>(), rng.random::<f64>())
    }
}
