use std::sync::Arc;

use rand::RngExt;

use crate::planar::Region2D;

/// Region type defined by an arbitrary `(a, b) -> bool` predicate (a math
/// inequality, a set of inequalities, or any other condition).
///
/// The `area` and `bbox` fields are precomputed at construction time so the
/// per-frame paths don't have to do numerical integration or sweep a closure.
#[derive(Clone)]
pub struct FunctionRegion {
    contains_fn: Arc<dyn Fn(f64, f64) -> bool + Send + Sync>,
    area: f64,
    bbox: (f64, f64, f64, f64), // (a_min, a_max, b_min, b_max)
}

impl FunctionRegion {
    /// Build a function region with a precomputed area and bounding box.
    pub fn new(
        contains_fn: Arc<dyn Fn(f64, f64) -> bool + Send + Sync>,
        area: f64,
        bbox: (f64, f64, f64, f64),
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
        contains_fn: Arc<dyn Fn(f64, f64) -> bool + Send + Sync>,
        bbox: (f64, f64, f64, f64),
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
        let area = bbox_area * (count as f64 / samples as f64);
        Self {
            contains_fn,
            area,
            bbox,
        }
    }
}

impl Region2D for FunctionRegion {
    fn contains(&self, a: f64, b: f64) -> bool {
        (self.contains_fn)(a, b)
    }

    fn area(&self) -> f64 {
        self.area
    }

    fn sample(&self, rng: &mut dyn rand::Rng) -> (f64, f64) {
        // Rejection in the provided bbox.
        let (a_min, a_max, b_min, b_max) = self.bbox;
        loop {
            let a = rng.random_range(a_min..a_max);
            let b = rng.random_range(b_min..b_max);
            if (self.contains_fn)(a, b) {
                return (a, b);
            }
        }
    }
}
