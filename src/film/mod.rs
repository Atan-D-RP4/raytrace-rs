pub mod rgb;
pub mod tile;

pub use rgb::RgbFilm;
pub use tile::FilmTile;

use std::path::Path;
use std::sync::{Arc, RwLock};

use glam::Vec3;
use image::{ImageResult, RgbImage};

use crate::vec3::Color3;

/// Post-process a linear RGB color to sRGB8 with exposure and optional tone mapping.
/// - `color`: linear RGB color in [0,1] range.
/// - `exposure`: exposure multiplier.
/// - `tone_map`: whether to apply tone mapping (Reinhard) before gamma correction
#[inline(always)]
fn post_process(color: Color3, exposure: f32, tone_map: bool) -> [u8; 3] {
    // Scale by sample count, exposure, and apply gamma correction.
    // Apply tone mapping operator before gamma if enabled, otherwise clamp to [0,1].
    let scaled = if tone_map {
        reinhard_tone_map(exposure, color)
    } else {
        color * exposure
    };

    // Gamma 2: sqrt() converts linear -> sRGB, then scale to [0,255].
    [
        (256.0 * linear_to_gamma(scaled.x()).clamp(0.0, 0.999)) as u8,
        (256.0 * linear_to_gamma(scaled.y()).clamp(0.0, 0.999)) as u8,
        (256.0 * linear_to_gamma(scaled.z()).clamp(0.0, 0.999)) as u8,
    ]
}

/// Reinhard tone mapping operator for HDR to LDR conversion.
/// - `exposure`: exposure multiplier.
/// - `color`: linear RGB color in [0,1] range.
#[inline(always)]
const fn reinhard_tone_map(exposure: f32, color: Color3) -> Color3 {
    let mapped = Vec3::new(
        color.0.x * exposure,
        color.0.y * exposure,
        color.0.z * exposure,
    );
    Color3::new(
        mapped.x / (1.0 + mapped.x),
        mapped.y / (1.0 + mapped.y),
        mapped.z / (1.0 + mapped.z),
    )
}

///
#[inline(always)]
fn linear_to_gamma(linear_component: f32) -> f32 {
    if linear_component > 0. {
        linear_component.sqrt()
    } else {
        0.
    }
}

/// A trait representing a film that accumulates samples and produces an image.
pub trait Film: Send + Sync {
    /// Add a sample to the film at pixel (x, y). Samples are equal-weight.
    fn add_sample(&mut self, x: u32, y: u32, color: Color3);

    /// Read the current image data as a packed RGB8 vector.
    fn read_image(&self) -> Vec<u8>;

    /// Write the current image data to a file at the given path.
    fn write_image(&self, path: impl AsRef<Path>) -> ImageResult<()>;

    /// Get the resolution of the film as (width, height).
    fn resolution(&self) -> (u32, u32);

    /// Reset the film to its initial state, clearing all accumulated samples.
    fn reset(&mut self);

    /// Merge a film tile into the film, accumulating its samples.
    fn merge_tile(&mut self, tile: &FilmTile);

    /// Produce a progressive RGB8 preview of the current film state.
    /// Uses per-pixel sample counts so adaptive sampling previews correctly.
    fn progressive(&self) -> impl Iterator<Item = u8> + '_;

    /// Returns the estimated per-pixel variance (max over RGB channels).
    /// Returns `f32::INFINITY` if fewer than 2 samples for this pixel.
    fn pixel_variance(&self, idx: usize) -> f32;

    /// Returns a fresh convergence mask: `true` = pixel variance is below threshold
    /// with at least `min_samples` accumulated. Allocates a new `Vec<bool>`.
    fn convergence_mask(
        &self,
        threshold_rel: f32,
        threshold_abs: f32,
        min_samples: u32,
    ) -> Vec<bool>;

    /// Refills an existing convergence mask `out` in place, avoiding allocation. The slice must
    /// have length == pixel count.
    ///
    /// Returns `true` if every pixel converged (allows early exit without a separate `all()` scan
    /// over the mask).
    fn reset_convergence_mask(
        &self,
        threshold_rel: f32,
        threshold_abs: f32,
        min_samples: u32,
        out: &mut [bool],
    ) -> bool;
}

/// Thread-safe framebuffer shared between UI thread and render thread.
///
/// - Render thread takes write lock, publishes progressive updates.
/// - UI thread takes read lock, blits current snapshot to window surface.
pub type SharedFramebuffer = Arc<RwLock<Framebuffer>>;

/// Shared RGB framebuffer used by live preview path.
///
/// Wraps an `RgbImage` (`ImageBuffer<Rgb<u8>, Vec<u8>>`) for the pixel data.
/// The image provides tightly packed RGB8 triples in row-major, top-left origin order,
/// with width and height metadata bundled in.
#[derive(Default)]
pub struct Framebuffer {
    /// The image buffer containing RGB8 pixel data.
    pub image: RgbImage,
    /// Signals render completion to UI redraw loop.
    pub finished: bool,
}

impl Framebuffer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_with(width: u32, height: u32) -> Self {
        Self {
            image: RgbImage::new(width, height),
            finished: false,
        }
    }

    pub fn set_dimensions(&mut self, width: u32, height: u32) {
        self.image = RgbImage::new(width, height);
    }
}
