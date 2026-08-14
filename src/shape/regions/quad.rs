use crate::math::interval::Interval;

use crate::shape::Region2D;

/// Region type for a full parallelogram (unit square in (a,b) space).
#[derive(Clone)]
pub struct QuadRegion;

impl Region2D for QuadRegion {
    fn contains(&self, a: f32, b: f32) -> bool {
        let unit = Interval::<1>::from(0., 1.);
        unit.contains_value(a) && unit.contains_value(b)
    }

    fn area(&self) -> f32 {
        1.0
    }

    fn sample(&self, u: f32, v: f32) -> (f32, f32) {
        (u, v)
    }
}
