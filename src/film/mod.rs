pub mod rgb;
pub mod tile;

pub use rgb::RgbFilm;
pub use tile::FilmTile;

use std::path::Path;
use std::sync::{Arc, RwLock};

use image::ImageResult;

use crate::vec3::{Color3, Vec3};

#[inline(always)]
fn post_process(color: Color3, exposure: f64, tone_map: bool) -> [u8; 3] {
    // Scale by sample count, exposure, and apply gamma correction.
    // Apply tone mapping operator before gamma if enabled, otherwise clamp to [0,1].
    let scaled = if tone_map {
        reinhard_tone_map(exposure, color)
    } else {
        color * exposure
    };

    // Gamma 2: sqrt() converts linear -> sRGB, then scale to [0,255].
    [
        (256.0 * linear_to_gamma(scaled.x).clamp(0.0, 0.999)) as u8,
        (256.0 * linear_to_gamma(scaled.y).clamp(0.0, 0.999)) as u8,
        (256.0 * linear_to_gamma(scaled.z).clamp(0.0, 0.999)) as u8,
    ]
}

#[inline(always)]
const fn reinhard_tone_map(exposure: f64, color: Color3) -> Color3 {
    let mapped = Vec3::from(color.x * exposure, color.y * exposure, color.z * exposure);
    Color3::from(
        mapped.x / (1.0 + mapped.x),
        mapped.y / (1.0 + mapped.y),
        mapped.z / (1.0 + mapped.z),
    )
}

#[inline(always)]
/// Converts a linear color channel to gamma-corrected (gamma=2) space.
fn linear_to_gamma(linear_component: f64) -> f64 {
    if linear_component > 0. {
        linear_component.sqrt()
    } else {
        0.
    }
}

pub trait Film: Send + Sync {
    /// Add a sample to the film at pixel (x, y) with the given color and weight.
    fn add_sample(&mut self, x: u32, y: u32, color: Color3, weight: f64);

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

    fn progressive(&self, samples_so_far: usize) -> impl Iterator<Item = u8> + '_;
}

/// Thread-safe framebuffer shared between UI thread and render thread.
///
/// - Render thread takes write lock, publishes progressive updates.
/// - UI thread takes read lock, blits current snapshot to window surface.
pub type SharedFramebuffer = Arc<RwLock<Framebuffer>>;

/// Shared RGB framebuffer used by live preview path.
///
/// `rgb` layout is tightly packed RGB8 triples:
/// `[R, G, B, R, G, B, ...]`, row-major, top-left origin.
pub struct Framebuffer {
    /// Pixel width of framebuffer.
    pub width: u32,
    /// Pixel height of framebuffer.
    pub height: u32,
    /// Packed RGB8 data, `width * height * 3` bytes.
    pub rgb: Vec<u8>,
    /// Signals render completion to UI redraw loop.
    pub finished: bool,
}

impl Framebuffer {
    /// Creates zero-initialized framebuffer for given dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            rgb: vec![0; (width * height * 3) as usize],
            finished: false,
        }
    }
}
