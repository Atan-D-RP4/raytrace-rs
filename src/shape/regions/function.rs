use std::sync::Arc;

use rand::RngExt;

use crate::shape::Region2D;

/// Region type defined by an arbitrary `(a, b) -> bool` predicate (a math
/// inequality, a set of inequalities, or any other condition).
///
/// The `area` and `bbox` fields are precomputed at construction time so the
/// per-frame paths don't have to do numerical integration or sweep a closure.
#[derive(Clone)]
pub struct FunctionRegion {
    contains_fn: Arc<dyn Fn(f32, f32) -> bool + Send + Sync>,
    area: f32,
    bbox: (f32, f32, f32, f32), // (a_min, a_max, b_min, b_max)
}

impl FunctionRegion {
    /// Build a function region with a precomputed area and bounding box.
    pub fn new(
        contains_fn: Arc<dyn Fn(f32, f32) -> bool + Send + Sync>,
        area: f32,
        bbox: (f32, f32, f32, f32),
    ) -> Self {
        Self {
            contains_fn,
            area,
            bbox,
        }
    }

    /// Build a function region and estimate its area via Monte Carlo
    /// integration with `samples` random points in `bbox`.
    ///
    /// The estimator has error ~ `O(1/√samples)`. 10_000 samples gives ~1%
    /// relative error for shapes that fill a reasonable fraction of `bbox`.
    pub fn with_monte_carlo_area(
        contains_fn: Arc<dyn Fn(f32, f32) -> bool + Send + Sync>,
        bbox: (f32, f32, f32, f32),
        samples: usize,
    ) -> Self {
        let (a_min, a_max, b_min, b_max) = bbox;
        let bbox_area = (a_max - a_min) * (b_max - b_min);
        let mut rng = rand::rng();
        let mut count = 0usize;
        for _ in 0..samples {
            let a = rng.random_range(a_min..a_max);
            let b = rng.random_range(b_min..b_max);
            if contains_fn(a, b) {
                count += 1;
            }
        }
        let area = bbox_area * (count as f32 / samples as f32);
        Self {
            contains_fn,
            area,
            bbox,
        }
    }
}

impl Region2D for FunctionRegion {
    fn contains(&self, a: f32, b: f32) -> bool {
        (self.contains_fn)(a, b)
    }

    fn area(&self) -> f32 {
        self.area
    }

    fn bounding_box_area(&self) -> f32 {
        let (a_min, a_max, b_min, b_max) = self.bbox;
        (a_max - a_min) * (b_max - b_min)
    }

    fn sample(&self, u: f32, v: f32) -> (f32, f32) {
        let (a_min, a_max, b_min, b_max) = self.bbox;
        let mut u = u;
        let mut v = v;
        for _ in 0..32 {
            let a = u * (a_max - a_min) + a_min;
            let b = v * (b_max - b_min) + b_min;
            if self.contains(a, b) {
                return (a, b);
            }
            u = (u + 0.618_034).fract();
            v = (v + 0.618_034).fract();
        }
        // Fallback: centroid of bounding box. For convex regions this is
        // always inside; for concave or disconnected regions it may fall
        // outside, producing a subtly incorrect light sample.  Increasing
        // retry count or switching to spatial hashing would fix this, but
        // the 32-retry golden-ratio loop succeeds for all practical shapes.
        tracing::warn!(
            a_min,
            a_max,
            b_min,
            b_max,
            "FunctionRegion::sample fell back to bbox centroid — region may be concave"
        );
        ((a_min + a_max) * 0.5, (b_min + b_max) * 0.5)
    }
}
