use std::path::Path;

use image::ImageResult;

use crate::film::{Film, post_process};
use crate::vec3::Color3;

use crate::film::FilmTile;

#[derive(Default, Clone)]
pub struct RgbFilm {
    width: u32,
    height: u32,
    /// Accumulated color for each pixel (linear space, not gamma-corrected).
    pixels: Vec<Color3>,
    sample_counts: Vec<u32>,
    // Exposure value for tone mapping.
    exposure: f64,
    // Whether to apply tone mapping to final colors.
    tone_map: bool,
}

impl RgbFilm {
    /// Create a new RGB film with the given resolution, exposure, and tone mapping settings.
    pub fn new(dimensions: (u32, u32), exposure: f64, tone_map: bool) -> Self {
        let (width, height) = dimensions;
        Self {
            width,
            height,
            pixels: vec![Color3::ZERO; (width * height) as usize],
            sample_counts: vec![0; (width * height) as usize],
            exposure,
            tone_map,
        }
    }

    /// Convert accumulated radiance to RGB8 output (moved from Camera's final conversion).
    pub fn to_rgb8(&self) -> Vec<u8> {
        self.pixels
            .iter()
            .zip(self.sample_counts.iter())
            .flat_map(|(color, &count)| {
                let avg_color = if count > 0 {
                    *color / (count as f64)
                } else {
                    *color
                };
                post_process(avg_color, self.exposure, self.tone_map)
            })
            .collect()
    }

    /// Progressive tonemap for live preview (moved from Camera's progressive block).
    pub fn progressive_rgb8(&self, samples_so_far: usize) -> impl Iterator<Item = u8> + '_ {
        self.pixels.iter().flat_map(move |color| {
            let avg_color = *color / (samples_so_far as f64);
            post_process(avg_color, self.exposure, self.tone_map)
        })
    }
}

impl Film for RgbFilm {
    fn add_sample(&mut self, x: u32, y: u32, color: Color3, weight: f64) {
        let index = (y * self.width + x) as usize;
        self.pixels[index] += color * weight;
        self.sample_counts[index] += 1;
    }

    fn read_image(&self) -> Vec<u8> {
        self.to_rgb8()
    }

    fn write_image(&self, path: impl AsRef<Path>) -> ImageResult<()> {
        let rgb_data = self.to_rgb8();
        image::save_buffer(
            path,
            &rgb_data,
            self.width,
            self.height,
            image::ColorType::Rgb8,
        )
    }

    fn resolution(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn reset(&mut self) {
        self.pixels.fill(Color3::ZERO);
        self.sample_counts.fill(0);
    }

    fn merge_tile(&mut self, tile: &FilmTile) {
        let (x_min, x_max, y_min, _y_max) = (
            tile.bounds[0],
            tile.bounds[1],
            tile.bounds[2],
            tile.bounds[3],
        );

        let tile_width = x_max - x_min;
        for (tile_idx, &color) in tile.pixels.iter().enumerate() {
            let tx = x_min + (tile_idx as u32 % tile_width);
            let ty = y_min + (tile_idx as u32 / tile_width);
            let index = (ty * self.width + tx) as usize;
            self.pixels[index] += color;
            self.sample_counts[index] += 1; // Assuming each tile contributes one sample per pixel
        }
    }

    fn progressive(&self, samples_so_far: usize) -> impl Iterator<Item = u8> + '_ {
        self.progressive_rgb8(samples_so_far)
    }
}
