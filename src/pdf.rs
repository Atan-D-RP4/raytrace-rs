use std::f64::consts::PI;
use std::sync::Arc;

use crate::environment::EnvironmentMap;
use crate::hittable::Sampleable;
use crate::material::PdfKind;
use crate::vec3::{Point3, Vec3, concentric_disk};

/// Power-heuristic MIS weight exponent.
///
/// β = 2 is the standard choice — provably optimal for piecewise-smooth
/// integrands and gives the best variance reduction in practice.
pub const BETA: f64 = 2.0;

/// Power-heuristic MIS weight: `p_i² / Σ(p_j²)`.
///
/// `pdf_sum_sq` must be the sum of squared PDF values (Σ p_j²).
/// Returns 0 when the denominator is near-zero (degenerate PDF).
#[inline(always)]
pub fn power_heuristic(pdf_i: f64, pdf_sum_sq: f64) -> f64 {
    if pdf_sum_sq <= 1e-20 {
        return 0.0;
    }
    (pdf_i * pdf_i / pdf_sum_sq).max(0.0)
}

/// Cosine-weighted hemisphere direction via concentric disk mapping.
///
/// Takes two uniform random values `(u, v)` in `[0, 1)` and returns a direction
/// on the unit hemisphere with PDF `cos(θ) / π`. The concentric disk mapping
/// avoids the rejection sampling of `sampler_cosine_direction`.
///
/// Reference: Shirley & Chiu, "A Low Distortion Map Between Disk and Square", 1997.
#[inline(always)]
pub fn cosine_hemisphere_direction(u: f64, v: f64) -> Vec3 {
    // Concentric disk mapping: map (u,v) in [0,1)^2 to (x,y) on the unit disk.
    let (x, y) = concentric_disk(u, v);
    Vec3::new(x, y, (1.0 - x * x - y * y).max(0.0).sqrt())
}

/// Uniform hemisphere direction via spherical coordinates.
///
/// Takes two uniform random values `(u, v)` in `[0, 1)` and
/// returns a direction on the unit hemisphere with PDF `1 / (2π)`.
#[inline(always)]
pub fn uniform_hemisphere_direction(u: f64, v: f64) -> Vec3 {
    let phi = 2.0 * PI * u;
    let (sin_phi, cos_phi) = phi.sin_cos();
    let z = v;
    let r = (1.0 - z * z).max(0.0).sqrt();
    Vec3::new(r * cos_phi, r * sin_phi, z)
}

/// Probability density function for sampling directions.
///
/// Implementations are pure functions of `(u, v)` — no internal cursor state.
/// The caller provides the random numbers; the PDF just maps them to directions.
pub trait PDF {
    /// Evaluates the PDF value for a given direction.
    fn value(&self, direction: Vec3) -> f64;

    /// Generates a random direction according to the PDF from `(u, v)` in [0, 1)².
    fn generate(&self, u: f64, v: f64) -> Vec3;
}

impl PDF for PdfKind {
    fn value(&self, direction: Vec3) -> f64 {
        PdfKind::value(self, direction)
    }

    fn generate(&self, u: f64, v: f64) -> Vec3 {
        PdfKind::generate(self, u, v)
    }
}

/// PDF for sampling directions from a set of light emitters
pub struct EmitterPDF<'a> {
    /// The set of light emitters to sample from.
    objects: &'a [Arc<dyn Sampleable>],
    /// The origin point from which to sample direction.
    origin: Point3,
    /// The time at which to sample the emitter.
    time: f64,
}

impl<'a> EmitterPDF<'a> {
    pub fn new(objects: &'a [Arc<dyn Sampleable>], origin: Point3, time: f64) -> Self {
        EmitterPDF {
            objects,
            origin,
            time,
        }
    }
}

impl<'a> PDF for EmitterPDF<'a> {
    fn value(&self, direction: Vec3) -> f64 {
        if self.objects.is_empty() {
            return 0.0;
        }
        let inv_len = 1.0 / self.objects.len() as f64;
        self.objects
            .iter()
            .map(|o| o.pdf_value(self.origin, direction, self.time) * inv_len)
            .sum()
    }

    /// Samples a direction from the emitter set.
    ///
    /// `u` selects the light (uniformly across emitters) and is also passed
    /// through to the selected light's [`random_direction()`](crate::hittable::Sampleable::random_direction).
    /// `v` is passed through to the selected light's `random_direction()` method
    /// which uses it for the secondary random dimension (e.g., surface position
    /// within the light or directional PDF sampling).
    fn generate(&self, u: f64, v: f64) -> Vec3 {
        if self.objects.is_empty() {
            return Vec3::ZERO;
        }
        let index = (u * self.objects.len() as f64).min(self.objects.len() as f64 - 1e-15) as usize;
        self.objects[index].random_direction(self.origin, u, v, self.time)
    }
}

/// PDF for a single sampleable light source.
///
/// Light selection is handled by the integrator — this PDF only generates
/// directions from the selected light. Wraps a reference to avoid cloning.
pub struct LightPDF<'a> {
    object: &'a Arc<dyn Sampleable>,
    origin: Point3,
    time: f64,
}

impl<'a> LightPDF<'a> {
    pub fn new(object: &'a Arc<dyn Sampleable>, origin: Point3, time: f64) -> Self {
        Self {
            object,
            origin,
            time,
        }
    }
}

impl<'a> PDF for LightPDF<'a> {
    fn value(&self, direction: Vec3) -> f64 {
        self.object.pdf_value(self.origin, direction, self.time)
    }

    fn generate(&self, u: f64, v: f64) -> Vec3 {
        self.object.random_direction(self.origin, u, v, self.time)
    }
}

/// Thin wrapper around an environment map that implements the [`PDF`] trait.
///
/// This allows the environment map to be used as a PDF for importance sampling directions from the
/// environment light.
pub struct EnvPdf<'a>(&'a Arc<EnvironmentMap>);

impl EnvPdf<'_> {
    pub fn new(env_map: &Arc<EnvironmentMap>) -> EnvPdf {
        EnvPdf(env_map)
    }
}

impl<'a> PDF for EnvPdf<'a> {
    fn value(&self, direction: Vec3) -> f64 {
        self.0.to_solid_angle_pdf(direction)
    }
    fn generate(&self, u: f64, v: f64) -> Vec3 {
        let (col, row, _) = self.0.sample(u, v);
        let theta = (row as f64 + 0.5) / self.0.height() as f64 * PI;
        let phi = (col as f64 + 0.5) / self.0.width() as f64 * 2.0 * PI;
        let sin_theta = theta.sin();
        Vec3::new(sin_theta * phi.cos(), theta.cos(), sin_theta * phi.sin())
    }
}
