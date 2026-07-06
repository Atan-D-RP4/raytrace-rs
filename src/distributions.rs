use crate::film::rgb::LUMINANCE;
use crate::vec3::{Color3, Vec3};
use image::Rgba32FImage;
use std::f64::consts::PI;

pub struct Dist1D {
    cdfs: Vec<f64>,
    funcs: Vec<f64>,
    total: f64,
}

impl Dist1D {
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

    pub fn pdf(&self, index: usize) -> f64 {
        if self.total == 0. {
            return 1.0;
        }

        (self.funcs[index] * self.count() as f64) / self.total
    }

    pub fn count(&self) -> usize {
        self.funcs.len()
    }
}

pub struct Dist2D {
    marginal: Dist1D,
    conditional: Vec<Dist1D>,
}

impl Dist2D {
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

    pub fn sample(&self, u: f64, v: f64) -> (usize, usize, f64) {
        let (row, marginal_pdf) = self.marginal.sample(v);

        let (col, conditional_pdf) = self.conditional[row].sample(u);

        let pdf = marginal_pdf * conditional_pdf;

        (col, row, pdf)
    }

    pub fn pdf(&self, i: usize, j: usize) -> f64 {
        self.marginal.pdf(j) * self.conditional[j].pdf(i)
    }
}

pub struct EnvironmentMap {
    image: Rgba32FImage,
    distribution: Dist2D,
    /// Total raw (unweighted) scene luminance. Useful for light-selection probability.
    #[allow(dead_code)]
    total_luminance: f64,
}

impl EnvironmentMap {
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

    pub fn sample(&self, u: f64, v: f64) -> (usize, usize, f64) {
        self.distribution.sample(u, v)
    }

    pub fn pdf(&self, i: usize, j: usize) -> f64 {
        self.distribution.pdf(i, j)
    }

    pub fn get_pixel(&self, i: usize, j: usize) -> [f32; 4] {
        let pixel = self.image.get_pixel(i as u32, j as u32);
        [pixel[0], pixel[1], pixel[2], pixel[3]]
    }

    pub fn width(&self) -> usize {
        self.image.width() as usize
    }

    pub fn height(&self) -> usize {
        self.image.height() as usize
    }

    pub fn le(&self, direction: Vec3) -> Color3 {
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
        let j = (v * height as f64).floor() as usize % height;

        let pixel = self.image.get_pixel(i as u32, j as u32);
        Color3::new(pixel[0] as f64, pixel[1] as f64, pixel[2] as f64)
    }
}
