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

/// Reconstruction filter radius in pixels. Each sample is spread across
/// (2*R+1)² pixels via a tent (triangle) kernel.
pub const FILTER_RADIUS: u32 = 2;

#[derive(Default, Clone)]
pub struct RgbFilm {
    width: u32,
    height: u32,
    /// Accumulated raw sample colors: sum(color) per pixel.
    pixels: Vec<Color3>,
    /// Actual number of samples per pixel.
    sample_num: Vec<u32>,
    /// Sum of raw (unweighted) sample colors. Used by Welford's algorithm
    /// to compute variance for convergence checking.
    raw_sum: Vec<Color3>,
    /// The running sum of squared differences from the unweighted mean.
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
            sample_num: vec![0; (width * height) as usize],
            raw_sum: vec![Color3::ZERO; (width * height) as usize],
            m_2: vec![Color3::ZERO; (width * height) as usize],
            exposure,
            tone_map,
        }
    }

    /// Apply a tent (triangle) reconstruction filter to the accumulated image.
    /// Each pixel's contribution is spread to neighbors within FILTER_RADIUS,
    /// weighted by distance. This smooths silhouette aliasing.
    fn apply_tent_filter(&self) -> Vec<Color3> {
        let r = FILTER_RADIUS as i32;
        let mut filtered = vec![Color3::ZERO; (self.width * self.height) as usize];

        for y in 0..self.height {
            for x in 0..self.width {
                let count = self.sample_num[(y * self.width + x) as usize];
                if count == 0 {
                    continue;
                }
                let color = self.pixels[(y * self.width + x) as usize];

                // Spread this pixel's average to neighbors within the filter radius.
                for dy in -r..=r {
                    for dx in -r..=r {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        if nx < 0 || nx >= self.width as i32 || ny < 0 || ny >= self.height as i32 {
                            continue;
                        }

                        // Tent function: weight = max(0, 1 - |d| / R)
                        let wx = (1.0 - (dx as f64).abs() / r as f64).max(0.0);
                        let wy = (1.0 - (dy as f64).abs() / r as f64).max(0.0);
                        let weight = wx * wy;

                        let idx = (ny as u32 * self.width + nx as u32) as usize;
                        filtered[idx] += color * (weight / count as f64);
                    }
                }
            }
        }

        // Normalize by accumulated filter weights per pixel.
        for y in 0..self.height {
            for x in 0..self.width {
                let idx = (y * self.width + x) as usize;
                let mut total_weight = 0.0f64;
                for dy in -r..=r {
                    for dx in -r..=r {
                        let sx = x as i32 - dx;
                        let sy = y as i32 - dy;
                        if sx < 0 || sx >= self.width as i32 || sy < 0 || sy >= self.height as i32 {
                            continue;
                        }
                        if self.sample_num[(sy as u32 * self.width + sx as u32) as usize] == 0 {
                            continue;
                        }
                        let wx = (1.0 - (dx as f64).abs() / r as f64).max(0.0);
                        let wy = (1.0 - (dy as f64).abs() / r as f64).max(0.0);
                        total_weight += wx * wy;
                    }
                }
                if total_weight > 0.0 {
                    filtered[idx] /= total_weight;
                }
            }
        }

        filtered
    }

    /// Convert accumulated radiance to RGB8 output with tent filter applied.
    pub fn to_rgb8(&self) -> Vec<u8> {
        let filtered = self.apply_tent_filter();
        filtered
            .iter()
            .flat_map(|color| post_process(*color, self.exposure, self.tone_map))
            .collect()
    }

    /// Progressive tonemap for live preview (unfiltered, for speed).
    pub fn progressive_rgb8(&self) -> impl Iterator<Item = u8> + '_ {
        self.pixels
            .iter()
            .zip(self.sample_num.iter())
            .flat_map(|(color, &count)| {
                let avg_color = if count > 0 {
                    *color / count as f64
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
        let n_prev = self.sample_num[index];

        if n_prev == 0 {
            self.pixels[index] = color;
            self.raw_sum[index] = color;
            // m_2 stays 0 — variance undefined for a single sample
        } else {
            let mean_prev = self.raw_sum[index] / n_prev as f64;
            let delta = color - mean_prev;

            self.pixels[index] += color;
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
        self.sample_num.fill(0);
        self.raw_sum.fill(Color3::ZERO);
        self.m_2.fill(Color3::ZERO);
    }

    fn merge_tile(&mut self, tile: &FilmTile) {
        let x_min = tile.bounds[0];
        let x_max = tile.bounds[1].min(self.width);
        let y_min = tile.bounds[2];
        let _y_max = tile.bounds[3].min(self.height);

        let tile_width = x_max - x_min;
        for (tile_idx, &tile_count) in tile.sample_count.iter().enumerate() {
            if tile_count == 0 {
                continue;
            }
            let tx = x_min + (tile_idx as u32 % tile_width);
            let ty = y_min + (tile_idx as u32 / tile_width);
            if tx >= self.width || ty >= self.height {
                continue;
            }
            let idx = (ty * self.width + tx) as usize;
            let tile_idx = tile_idx;

            // Accumulate raw color sum.
            let tile_color = tile.pixels[tile_idx];
            self.pixels[idx] += tile_color;

            // Welford variance update: process each sample in the tile individually.
            let n_prev = self.sample_num[idx];
            for k in 0..tile_count {
                let sample_color = tile_color / tile_count as f64;
                let n = n_prev + k;
                if n >= 1 {
                    let mean_prev = self.raw_sum[idx] / n as f64;
                    let delta = sample_color - mean_prev;
                    let n_new = (n + 1) as f64;
                    self.m_2[idx] += delta * delta * (n as f64 / n_new);
                }
                self.raw_sum[idx] += sample_color;
            }
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
                let variance = self.pixel_variance(idx);
                let var_mean = if sample_count > 0 {
                    variance / sample_count as f64
                } else {
                    f64::INFINITY
                };
                let mean = if sample_count > 0 {
                    self.pixels[idx] / sample_count as f64
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
            let variance = self.pixel_variance(idx);
            let var_mean = if sample_count > 0 {
                variance / sample_count as f64
            } else {
                f64::INFINITY
            };
            let mean = if sample_count > 0 {
                self.pixels[idx] / sample_count as f64
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
