use crate::planar::Region2D;

/// Region type for a triangle (a ≥ 0, b ≥ 0, a+b ≤ 1).
#[derive(Clone)]
pub struct TriRegion;

impl Region2D for TriRegion {
    fn contains(&self, a: f32, b: f32) -> bool {
        a >= 0.0 && b >= 0.0 && (a + b) <= 1.0
    }

    fn area(&self) -> f32 {
        0.5
    }

    fn sample(&self, u: f32, v: f32) -> (f32, f32) {
        let sqrt_u = u.sqrt();
        let a = 1.0 - sqrt_u;
        let b = v * sqrt_u;
        (a, b)
    }
}
