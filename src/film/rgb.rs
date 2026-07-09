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
    /// Accumulated weighted color for each pixel (linear space, not gamma-corrected).
    /// For tent-filtered samples: stores sum(color * weight).
    /// For unweighted samples: stores sum(color).
    pixels: Vec<Color3>,
    /// Running sum of sample weights for each pixel (weighted average denominator).
    /// For tent-filtered samples: sum(weight). For unweighted samples: sample count.
    sample_counts: Vec<f64>,
    /// Actual number of samples per pixel (independent of filter weights).
    /// Used by the convergence system to check min_samples thresholds.
    sample_num: Vec<u32>,
    /// Sum of raw (unweighted) sample colors. Used by Welford's algorithm
    /// to compute variance for convergence checking.
    raw_sum: Vec<Color3>,
    /// The running sum of squared differences from the unweighted mean.
    /// Updated by both add_sample and add_sample_weighted.
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
            sample_counts: vec![0.0; (width * height) as usize],
            sample_num: vec![0; (width * height) as usize],
            raw_sum: vec![Color3::ZERO; (width * height) as usize],
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
                let avg_color = if count > 0.0 { *color / count } else { *color };
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
                let avg_color = if count > 0.0 { *color / count } else { *color };
                post_process(avg_color, self.exposure, self.tone_map)
            })
    }

    /// Add a weighted sample to the film. Used when merging tiles that have
    /// reconstruction filter weights applied (e.g., tent filter).
    ///
    /// `color` is the raw sample color (NOT pre-multiplied by weight), and `weight`
    /// is the reconstruction filter weight. The weighted average is maintained as
    /// sum(color * weight) / sum(weight).
    ///
    /// Also runs Welford's m_2 update on the unweighted color sum so
    /// pixel_variance() produces a usable variance estimate for convergence.
    pub fn add_sample_weighted(&mut self, x: u32, y: u32, color: Color3, weight: f64) {
        let index = (y * self.width + x) as usize;
        self.pixels[index] += color * weight;
        self.sample_counts[index] += weight;

        // Welford's online variance update using the unweighted color sum.
        let n_prev = self.sample_num[index];
        let mean_prev = if n_prev == 0 {
            Color3::ZERO
        } else {
            self.raw_sum[index] / n_prev as f64
        };
        self.raw_sum[index] += color;

        if n_prev >= 1 {
            let delta = color - mean_prev;
            let n_new = (n_prev + 1) as f64;
            self.m_2[index] += delta * delta * (n_prev as f64 / n_new);
        }
        self.sample_num[index] = n_prev + 1;
    }
}

impl Film for RgbFilm {
    fn add_sample(&mut self, x: u32, y: u32, color: Color3) {
        let index = (y * self.width + x) as usize;
        let n_prev = self.sample_num[index];

        if n_prev == 0 {
            self.pixels[index] = color;
            self.sample_counts[index] = 1.0;
            self.raw_sum[index] = color;
            // m_2 stays 0 — variance undefined for a single sample
        } else {
            let mean_prev = self.raw_sum[index] / n_prev as f64;
            let delta = color - mean_prev;

            self.pixels[index] += color;
            self.sample_counts[index] += 1.0;
            self.raw_sum[index] += color;

            // Welford's online M2 update:
            //   M2 += delta² * n_prev / (n_prev + 1)
            let n_new = (n_prev + 1) as f64;
            self.m_2[index] += delta * delta * (n_prev as f64 / n_new);
        }
        self.sample_num[index] = n_prev + 1;
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
        self.sample_counts.fill(0.0);
        self.sample_num.fill(0);
        self.raw_sum.fill(Color3::ZERO);
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
        // Iterate over tile pixels, destructuring the 5 parallel vectors:
        // (color, raw, sampled, weight_sum, sample_count)
        for (tile_idx, (((&color, &raw), &weight_sum), &tile_count)) in tile
            .pixels
            .iter()
            .zip(tile.raw_sum.iter())
            .zip(tile.weight_sum.iter())
            .zip(tile.sample_count.iter())
            .enumerate()
        {
            if weight_sum == 0.0 {
                continue;
            }
            let tx = x_min + (tile_idx as u32 % tile_width);
            let ty = y_min + (tile_idx as u32 / tile_width);
            let idx = (ty * self.width + tx) as usize;

            // Accumulate weighted color and weight sum (for the final image).
            self.pixels[idx] += color;
            self.sample_counts[idx] += weight_sum;

            // Accumulate raw color sum and run Welford update (for variance).
            let n_prev = self.sample_num[idx];
            if n_prev >= 1 {
                let mean_prev = self.raw_sum[idx] / n_prev as f64;
                let delta = raw - mean_prev;
                let n_new = (n_prev + tile_count) as f64;
                // Approximate: apply the full tile's contribution as a batch.
                // For a single sample per pixel per pass, this is exact.
                self.m_2[idx] += delta * delta * (n_prev as f64 / n_new) * tile_count as f64;
            }
            self.raw_sum[idx] += raw;
            self.sample_num[idx] += tile_count;
        }
    }

    fn progressive(&self) -> impl Iterator<Item = u8> + '_ {
        self.progressive_rgb8()
    }

    fn pixel_variance(&self, idx: usize) -> f64 {
        let n = self.sample_num[idx];

        if n < 2 {
            f64::INFINITY // Variance is undefined for n < 2
        } else {
            let variance = self.m_2[idx] / (n as f64 - 1.0);
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
                let sample_count = self.sample_num[idx];
                let weight_sum = self.sample_counts[idx];
                let variance = self.pixel_variance(idx);
                let var_mean = if weight_sum > 0.0 {
                    variance / weight_sum
                } else {
                    f64::INFINITY
                };
                let mean = if weight_sum > 0.0 {
                    self.pixels[idx] / weight_sum
                } else {
                    self.pixels[idx]
                };
                let luminance = LUMINANCE * mean;
                let luminance = luminance.x + luminance.y + luminance.z;

                sample_count >= min_samples
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
            let sample_count = self.sample_num[idx];
            let weight_sum = self.sample_counts[idx];
            let variance = self.pixel_variance(idx);
            let var_mean = if weight_sum > 0.0 {
                variance / weight_sum
            } else {
                f64::INFINITY
            };
            let mean = if weight_sum > 0.0 {
                self.pixels[idx] / weight_sum
            } else {
                self.pixels[idx]
            };
            let luminance = LUMINANCE * mean;
            let luminance = luminance.x + luminance.y + luminance.z;
            let converged = sample_count >= min_samples
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
        let constant = Color3::new(0.5, 0.3, 0.2);

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
        film.add_sample(0, 0, Color3::new(1.0, 0.5, 0.2));

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
        film.add_sample(0, 0, Color3::new(0.0, 0.0, 0.0));
        film.add_sample(0, 0, Color3::new(1.0, 1.0, 1.0));
        film.add_sample(0, 0, Color3::new(0.0, 0.0, 0.0));
        film.add_sample(0, 0, Color3::new(1.0, 1.0, 1.0));

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
            film.add_sample(0, 0, Color3::new(0.5, 0.5, 0.5));
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
                film.add_sample(x, y, Color3::new(0.5, 0.5, 0.5));
            }
        }
        let rgb = film.to_rgb8();
        assert_eq!(rgb.len(), 3 * 2 * 3); // width * height * 3 channels
    }
}
