use std::f64::consts::PI;
use std::sync::Arc;

use crate::aabb::Aabb;
use crate::distributions::Dist2D;
use crate::film::rgb::LUMINANCE;
use crate::hittable::{Bounded, Intersectable, LightSample, MaterialHit, Sampleable};
use crate::interval::Interval;
use crate::ray::Ray;
use crate::vec3::{Color3, Vec3};

use image::Rgba32FImage;

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
    /// sin(θ) · 2π² (see `EnvironmentMap::to_solid_angle_pdf()`).
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

    pub fn to_solid_angle_pdf(&self, direction: Vec3) -> f64 {
        let (i, j) = self.pixel_uv_from_direction(direction);
        let pdf_pixel = self.pdf(i, j);

        let theta = direction.y.acos(); // y-up: θ = 0 at north pole
        let sin_theta = theta.sin().max(1e-10);

        pdf_pixel / (sin_theta * 2.0 * PI * PI)
    }
}

pub struct EnvironmentLight {
    env_map: Arc<EnvironmentMap>,
}

impl EnvironmentLight {
    pub fn new(env_map: Arc<EnvironmentMap>) -> Self {
        Self { env_map }
    }
}

impl Bounded for EnvironmentLight {
    fn bounding_box(&self) -> Aabb {
        // Environment light is at infinity, so it doesn't have a finite bounding box.
        Aabb::new()
    }
}

impl Intersectable for EnvironmentLight {
    fn intersect<'a>(&'a self, _ray: &Ray, _ray_t: Interval) -> Option<MaterialHit<'a>> {
        // Environment light is at infinity, so it doesn't have a finite intersection.
        // We can return None to indicate no intersection.
        None
    }
}

impl Sampleable for EnvironmentLight {
    fn pdf_value(&self, _origin: Vec3, direction: Vec3, _time: f64) -> f64 {
        self.env_map.to_solid_angle_pdf(direction)
    }

    fn random_direction(&self, _origin: Vec3, u: f64, v: f64, _time: f64) -> Vec3 {
        let (i, j, _pdf_pixel) = self.env_map.sample(u, v);
        let width = self.env_map.width();
        let height = self.env_map.height();

        // Convert pixel coordinates back to spherical coordinates
        let theta = (j as f64 + 0.5) / height as f64 * PI;
        let phi = (i as f64 + 0.5) / width as f64 * 2.0 * PI;

        // Convert spherical coordinates to Cartesian direction
        let sin_theta = theta.sin();
        let x = sin_theta * phi.cos();
        let y = theta.cos();
        let z = sin_theta * phi.sin();

        Vec3::new(x, y, z)
    }

    fn sample_light(&self, origin: Vec3, u: f64, v: f64, time: f64) -> LightSample {
        let direction = self.random_direction(origin, u, v, time);
        let pdf = self.pdf_value(origin, direction, time);
        let radiance = self.env_map.le(direction);

        LightSample {
            direction,
            normal: Vec3::ZERO,      // Environment light has no surface normal
            distance: f64::INFINITY, // Environment light is at infinity
            pdf,
            emission: radiance,
        }
    }
}
