use std::f32::consts::PI;

use crate::planar::Region2D;

/// Region type for a rounded rectangle in `[-1, 1] × [-1, 1]` (a, b) parametric space.
///
/// The corner radius `r` must satisfy `0 ≤ r ≤ 1`. At `r = 0` the shape is the
/// full unit square; at `r = 1` it degenerates to the inscribed unit circle.
#[derive(Clone)]
pub struct RoundedRectRegion {
    pub radius: f32,
}

impl Region2D for RoundedRectRegion {
    fn contains(&self, a: f32, b: f32) -> bool {
        if a.abs() > 1.0 || b.abs() > 1.0 {
            return false;
        }
        // Inside the central cross: always in.
        if a.abs() <= 1.0 - self.radius || b.abs() <= 1.0 - self.radius {
            return true;
        }
        // Corner region: distance from the inner-rect corner must be ≤ radius.
        let dx = a.abs() - (1.0 - self.radius);
        let dy = b.abs() - (1.0 - self.radius);
        dx * dx + dy * dy <= self.radius * self.radius
    }

    fn area(&self) -> f32 {
        // Square area (4) minus 4 corner gaps, each of area r² − πr²/4.
        4.0 - (4.0 - PI) * self.radius * self.radius
    }

    fn bounding_box_area(&self) -> f32 {
        4.0 // uniform over [-1,1]²
    }

    fn sample(&self, u: f32, v: f32) -> (f32, f32) {
        let mut u = u;
        let mut v = v;
        for _ in 0..32 {
            let a = u * 2.0 - 1.0;
            let b = v * 2.0 - 1.0;
            if self.contains(a, b) {
                return (a, b);
            }
            u = (u + 0.618_034).fract();
            v = (v + 0.618_034).fract();
        }
        (0.0, 0.0)
    }
}
