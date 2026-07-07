use crate::film::rgb::LUMINANCE;
use crate::vec3::{Color3, Vec3};
use image::Rgba32FImage;
use std::f64::consts::PI;

/// 1D piecewise-constant distribution with CDF-based sampling.
/// Used internally by Dist2D for the marginal and conditional distributions.
pub struct Dist1D {
    /// Cumulative distribution function (CDF) values, length n+1.
    cdfs: Vec<f64>,
    /// Normalized function values (weights ≥ 0).
    funcs: Vec<f64>,
    /// Sum of all function values. Zero if all weights are zero (uniform fallback).
    total: f64,
}

impl Dist1D {
    /// Build a 1D distribution from raw weight values.
    /// Non-positive values are clamped to zero; a zero-total distribution samples uniformly.
    pub fn new(values: &[f64]) -> Self {
        let n = values.len();
        let mut funcs = values.to_vec();

        let total = funcs.iter_mut().fold(0., |mut acc, value| {
            let weight = value.max(0.0);
            *value = weight;
            acc += weight;
            acc
        });

        let mut cdfs = vec![0.; n + 1];
        if total == 0. {
            (0..=n).for_each(|i| {
                cdfs[i] = i as f64 / n as f64;
            })
        } else {
            for i in 1..=n {
                cdfs[i] = cdfs[i - 1] + funcs[i - 1] / total;
            }
            cdfs[n] = 1.0;
        }

        Self { cdfs, funcs, total }
    }

    /// Sample the distribution with a unit-random value `u` ∈ [0, 1).
    /// Returns (index, PDF_value) where PDF_value uses the [0, 1] sample-space measure.
    pub fn sample(&self, u: f64) -> (usize, f64) {
        let u_clamp = &u.clamp(0., 1.0 - 1e-10);
        let offset = self.cdfs.binary_search_by(|&val| {
            if val <= *u_clamp {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        });
        let index = offset.unwrap_or_else(|idx| idx - 1);
        (index, self.pdf(index))
    }

    /// Evaluate the PDF at a given index. Returns 1.0 for the uniform fallback (zero total).
    pub fn pdf(&self, index: usize) -> f64 {
        if self.total == 0. {
            return 1.0;
        }

        (self.funcs[index] * self.count() as f64) / self.total
    }

    /// Number of bins in the distribution.
    pub fn count(&self) -> usize {
        self.funcs.len()
    }
}

/// 2D piecewise-constant distribution using a product of marginal + conditional 1D distributions.
/// Samples from the 2D CDF are drawn by first sampling the marginal (rows), then the conditional
/// (columns within the chosen row).
pub struct Dist2D {
    marginal: Dist1D,
    conditional: Vec<Dist1D>,
}

impl Dist2D {
    /// Build a 2D distribution from a flat array of shape (nv, nu) in row-major order.
    /// `nu` = columns (u-axis), `nv` = rows (v-axis).
    pub fn new(values: &[f64], nu: usize, nv: usize) -> Self {
        let mut row_sums = vec![0.; nv];
        for j in 0..nv {
            (0..nu).for_each(|i| {
                row_sums[j] += values[j * nu + i];
            });
        }
        let marginal = Dist1D::new(&row_sums);
        let conditional = (0..nv)
            .map(|j| {
                let row_start = j * nu;
                let row_end = row_start + nu;
                Dist1D::new(&values[row_start..row_end])
            })
            .collect();

        Self {
            marginal,
            conditional,
        }
    }

    /// Sample the 2D distribution with two unit-random values (u, v).
    /// Returns (column, row, PDF_value). `u` selects the column within the row,
    /// `v` selects the row from the marginal distribution.
    pub fn sample(&self, u: f64, v: f64) -> (usize, usize, f64) {
        let (row, marginal_pdf) = self.marginal.sample(v);

        let (col, conditional_pdf) = self.conditional[row].sample(u);

        let pdf = marginal_pdf * conditional_pdf;

        (col, row, pdf)
    }

    /// Evaluate the PDF at pixel (i, j) in the [0, 1]² sample-space measure.
    pub fn pdf(&self, i: usize, j: usize) -> f64 {
        self.marginal.pdf(j) * self.conditional[j].pdf(i)
    }
}

/// Equirectangular HDR environment map with sin(θ)-weighted luminance importance sampling.
/// The distribution is built once at construction and reused for all sample/pdf queries.
/// Radiance values are stored as-is (no tonemapping) — use `le()` for light evaluation.
pub struct EnvironmentMap {
    /// HDR pixel data (RGBA, linear space).
    image: Rgba32FImage,
    /// 2D pixel distribution weighted by luminance × sin(θ) (solid-angle correction).
    distribution: Dist2D,
    /// Total raw (unweighted) scene luminance. Useful for light-selection probability.
    #[allow(dead_code)]
    total_luminance: f64,
}

impl EnvironmentMap {
    /// Build an environment map from an equirectangular HDR image.
    /// The importance distribution weights each pixel by `luminance × sin(θ)` to account
    /// for sphere-area distortion — pixels near the poles cover less solid angle.
    pub fn new(image: Rgba32FImage) -> Self {
        let (width, height) = image.dimensions();
        let mut values = vec![0.0; (width * height) as usize];
        let mut total_luminance = 0.0;

        for j in 0..height {
            for i in 0..width {
                let pixel = image.get_pixel(i, j);

                let luminance = LUMINANCE.x * pixel[0] as f64
                    + LUMINANCE.y * pixel[1] as f64
                    + LUMINANCE.z * pixel[2] as f64;

                total_luminance += luminance;

                let theta = (j as f64 + 0.5) / height as f64 * PI;
                let weight = luminance * theta.sin();
                values[(j * width + i) as usize] = weight
            }
        }

        let distribution = Dist2D::new(&values, width as usize, height as usize);

        Self {
            image,
            distribution,
            total_luminance,
        }
    }

    /// Importance-sample the environment map using two unit-random values (u, v).
    /// Returns (column, row, PDF_value_in_pixel_domain). Use `EnvironmentMap::pdf()`
    /// to convert to solid-angle measure.
    pub fn sample(&self, u: f64, v: f64) -> (usize, usize, f64) {
        self.distribution.sample(u, v)
    }

    /// Evaluate the pixel-domain PDF at (i, j). For solid-angle PDF, divide by
    /// sin(θ) · 2π² (see `PdfEnum::value()` for the conversion).
    pub fn pdf(&self, i: usize, j: usize) -> f64 {
        self.distribution.pdf(i, j)
    }

    /// Read a raw pixel value from the HDR image as [R, G, B, A] floats.
    pub fn get_pixel(&self, i: usize, j: usize) -> [f32; 4] {
        let pixel = self.image.get_pixel(i as u32, j as u32);
        [pixel[0], pixel[1], pixel[2], pixel[3]]
    }

    /// Image width in pixels.
    pub fn width(&self) -> usize {
        self.image.width() as usize
    }

    /// Image height in pixels.
    pub fn height(&self) -> usize {
        self.image.height() as usize
    }

    /// Evaluate environment radiance (Le) in world-space `direction`.
    /// Performs nearest-neighbor lookup on the equirectangular map.
    pub fn le(&self, direction: Vec3) -> Color3 {
        let (i, j) = self.pixel_uv_from_direction(direction);

        let pixel = self.image.get_pixel(i as u32, j as u32);
        Color3::new(pixel[0] as f64, pixel[1] as f64, pixel[2] as f64)
    }

    /// Convert a world-space direction to equirectangular pixel coordinates (i, j).
    /// y-up convention: θ = 0 at north pole, φ ∈ [-π, π].
    pub fn pixel_uv_from_direction(&self, direction: Vec3) -> (usize, usize) {
        let w = direction.unit_vector(); // ensure unit length
        let theta = w.y.acos(); // y-up: θ = 0 at north pole
        let phi = w.z.atan2(w.x); // φ in [-π, π]

        // Map to [0, 1) texture coordinates
        let u = phi / (2.0 * PI); // [−½, ½]
        let u = u - u.floor(); // wrap to [0, 1)
        let v = theta / PI; // [0, 1]

        let width = self.image.width() as usize;
        let height = self.image.height() as usize;

        let i = (u * width as f64).floor() as usize % width;
        let j = ((v * height as f64).floor() as usize).min(height - 1);

        (i, j)
    }
}
