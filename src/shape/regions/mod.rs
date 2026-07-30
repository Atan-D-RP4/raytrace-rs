use crate::shape::Region2D;

mod annulus;
mod ellipse;
mod function;
mod polygon;
mod quad;
mod rounded_rect;
mod superellipse;
mod triangle;

pub use annulus::AnnulusRegion;
pub use ellipse::EllipseRegion;
pub use function::FunctionRegion;
pub use polygon::PolygonRegion;
pub use quad::QuadRegion;
pub use rounded_rect::RoundedRectRegion;
pub use superellipse::SuperellipseRegion;
pub use triangle::TriRegion;

/// Rejection sampling for arbitrary regions.
///
/// This is a simple method that repeatedly samples points in the unit square until one is found
/// that lies within the specified region.
///
/// The input `u` and `v` are used as starting points, and are updated with a golden-ratio-based
/// sequence to ensure that the sampling is not biased. The function returns a point `(a, b)` in the
/// region, or `(0.0, 0.0)` if no point is found after 32 attempts.
fn rejection_sample(u: f32, v: f32, region: &impl Region2D) -> (f32, f32) {
    (0..32)
        .scan((u, v), |state, _| {
            let (u, v) = state;
            let a = *u * 2.0 - 1.0;
            let b = *v * 2.0 - 1.0;
            if region.contains(a, b) {
                Some(Some((a, b)))
            } else {
                *u = (*u + 0.618_034).fract();
                *v = (*v + 0.618_034).fract();
                Some(None)
            }
        })
        .find_map(|x| x)
        .unwrap_or((0.0, 0.0))
}
