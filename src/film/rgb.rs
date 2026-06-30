use std::path::Path;

use image::ImageResult;

use crate::film::{Film, post_process};
use crate::vec3::Color3;

use crate::film::FilmTile;

pub const LUMINANCE: Color3 = Color3 {
    x: 0.2126,
    y: 0.7152,
    z: 0.0722,
};

#[derive(Default, Clone)]
pub struct RgbFilm {
    width: u32,
    height: u32,
    /// Accumulated color for each pixel (linear space, not gamma-corrected).
    pixels: Vec<Color3>,
    /// Parallel vector to `pixels` that tracks the number of samples accumulated for each pixel.
    sample_counts: Vec<u32>,
    /// The running sum of squared differences from the current mean
    m_2: Vec<Color3>,
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
            m_2: vec![Color3::ZERO; (width * height) as usize],
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

    /// Progressive tonemap for live preview.
    /// Uses per-pixel sample counts so adaptive sampling works correctly
    /// (each pixel may have a different number of accumulated samples).
    pub fn progressive_rgb8(&self) -> impl Iterator<Item = u8> + '_ {
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
    }
}

impl Film for RgbFilm {
    fn add_sample(&mut self, x: u32, y: u32, color: Color3) {
        let index = (y * self.width + x) as usize;
        let n_prev = self.sample_counts[index];

        if n_prev == 0 {
            self.pixels[index] = color;
            self.sample_counts[index] = 1;
            // m_2 stays 0 — variance undefined for a single sample
        } else {
            let n_prev_f = n_prev as f64;
            let mean_prev = self.pixels[index] / n_prev_f;
            let delta = color - mean_prev;
            let n_new_f = n_prev_f + 1.0;

            self.pixels[index] += color;
            // Welford's online M2 update:
            //   M2 += delta² * n_prev / (n_prev + 1)
            self.m_2[index] += delta * delta * (n_prev_f / n_new_f);
            self.sample_counts[index] = n_prev + 1;
        }
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
        self.m_2.fill(Color3::ZERO);
    }

    fn merge_tile(&mut self, tile: &FilmTile) {
        let (x_min, x_max, y_min, _y_max) = (
            tile.bounds[0],
            tile.bounds[1],
            tile.bounds[2],
            tile.bounds[3],
        );

        let tile_width = x_max - x_min;
        for (tile_idx, (&color, &sampled)) in
            tile.pixels.iter().zip(tile.sampled.iter()).enumerate()
        {
            if !sampled {
                continue;
            }
            let tx = x_min + (tile_idx as u32 % tile_width);
            let ty = y_min + (tile_idx as u32 / tile_width);
            self.add_sample(tx, ty, color);
        }
    }

    fn progressive(&self) -> impl Iterator<Item = u8> + '_ {
        self.progressive_rgb8()
    }

    fn pixel_variance(&self, idx: usize) -> f64 {
        let m_2 = self.m_2[idx];
        let n = self.sample_counts[idx] as f64;

        if n < 2.0 {
            f64::INFINITY // Variance is undefined for n < 2
        } else {
            let variance = m_2 / (n - 1.0);
            // Use max across RGB channels: a single noisy channel should prevent
            // convergence — averaging could hide it and produce visible artifacts.
            variance.x.max(variance.y).max(variance.z)
        }
    }

    fn convergence_mask(
        &self,
        threshold_rel: f64,
        threshold_abs: f64,
        min_samples: u32,
    ) -> Vec<bool> {
        (0..self.pixels.len())
            .map(|idx| {
                let sample_count = self.sample_counts[idx];
                let variance = self.pixel_variance(idx);
                let var_mean = if sample_count > 0 {
                    variance / (sample_count as f64)
                } else {
                    f64::INFINITY
                };
                let mean = self.pixels[idx] / (self.sample_counts[idx] as f64);
                let luminance = LUMINANCE * mean;
                let luminance = luminance.x + luminance.y + luminance.z;

                self.sample_counts[idx] >= min_samples
                    && (var_mean < threshold_abs || var_mean / luminance.max(1e-6) < threshold_rel)
            })
            .collect()
    }

    fn reset_convergence_mask(
        &self,
        threshold_rel: f64,
        threshold_abs: f64,
        min_samples: u32,
        out: &mut [bool],
    ) -> bool {
        let mut all_converged = true;
        for (idx, entry) in out.iter_mut().enumerate() {
            let sample_count = self.sample_counts[idx];
            let variance = self.pixel_variance(idx);
            let var_mean = if sample_count > 0 {
                variance / (sample_count as f64)
            } else {
                f64::INFINITY
            };
            let mean = self.pixels[idx] / (self.sample_counts[idx] as f64);
            let luminance = LUMINANCE * mean;
            let luminance = luminance.x + luminance.y + luminance.z;
            let converged = self.sample_counts[idx] >= min_samples
                && (var_mean < threshold_abs || var_mean / luminance.max(1e-6) < threshold_rel);
            *entry = converged;
            all_converged = all_converged && converged;
        }
        all_converged
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Welford's online variance should converge to zero for constant samples.
    #[test]
    fn variance_converges_for_constant_samples() {
        let mut film = RgbFilm::new((4, 4), 1.0, false);
        let constant = Color3::from(0.5, 0.3, 0.2);

        // Add many identical samples — variance should shrink to zero.
        for _ in 0..1000 {
            film.add_sample(0, 0, constant);
        }

        let variance = film.pixel_variance(0);
        assert!(
            variance < 1e-10,
            "variance for constant samples should be ~0, got {variance}"
        );
    }

    /// Variance should be infinity for a single sample (undefined).
    #[test]
    fn variance_infinity_for_single_sample() {
        let mut film = RgbFilm::new((2, 2), 1.0, false);
        film.add_sample(0, 0, Color3::from(1.0, 0.5, 0.2));

        let variance = film.pixel_variance(0);
        assert!(
            variance.is_infinite(),
            "single-sample variance should be infinity"
        );
    }

    /// Variance should be > 0 for varying samples.
    #[test]
    fn variance_positive_for_varying_samples() {
        let mut film = RgbFilm::new((2, 2), 1.0, false);
        film.add_sample(0, 0, Color3::from(0.0, 0.0, 0.0));
        film.add_sample(0, 0, Color3::from(1.0, 1.0, 1.0));
        film.add_sample(0, 0, Color3::from(0.0, 0.0, 0.0));
        film.add_sample(0, 0, Color3::from(1.0, 1.0, 1.0));

        let variance = film.pixel_variance(0);
        assert!(
            variance > 0.1,
            "variance for alternating 0/1 samples should be significant, got {variance}"
        );
    }

    /// Convergence mask should mark unconverged pixels correctly.
    #[test]
    fn convergence_mask_basic() {
        let mut film = RgbFilm::new((2, 1), 1.0, false);

        // Pixel 0: many identical samples (low variance → converged)
        for _ in 0..200 {
            film.add_sample(0, 0, Color3::from(0.5, 0.5, 0.5));
        }
        // Pixel 1: no samples → should not converge
        // (pixel stays at ZERO with count=0)

        let mask = film.convergence_mask(1e-3, 1e-6, 100);
        assert!(mask[0], "constant-sampled pixel should be converged");
        assert!(!mask[1], "unsampled pixel should not be converged");
    }

    /// to_rgb8 should produce correct dimensions.
    #[test]
    fn to_rgb8_dimensions() {
        let mut film = RgbFilm::new((3, 2), 1.0, false);
        // Fill all pixels with at least one sample.
        for y in 0..2 {
            for x in 0..3 {
                film.add_sample(x, y, Color3::from(0.5, 0.5, 0.5));
            }
        }
        let rgb = film.to_rgb8();
        assert_eq!(rgb.len(), 3 * 2 * 3); // width * height * 3 channels
    }
}
