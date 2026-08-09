use std::f32::consts::{FRAC_1_PI, PI};

use crate::light::Sampleable;
use crate::light::environment::EnvironmentMap;
use crate::math::onb::Onb;
use crate::math::vec3::{Direction3, Point3, concentric_disk};
use crate::primitives::LightPrimitive;

// ================================================================
// § PDF domain newtypes
//
// SolidAnglePdf and AreaPdf make the PDF domain explicit at the type
// level so callers cannot silently mix up solid-angle (sr⁻¹) and
// area (m⁻²) probability densities.  Conversions use the standard
// geometry relation:  PdfAtoW / PdfWtoA.
//
// Reference: luxrays/utils/mc.h lines 83-89
// ================================================================

/// Probability density w.r.t. solid angle (sr⁻¹).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolidAnglePdf(pub f32);

/// Probability density w.r.t. surface area (m⁻²).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AreaPdf(pub f32);

/// Geometry context required for solid-angle ↔ area PDF conversion.
#[derive(Clone, Copy, Debug)]
pub struct PdfConvCtx {
    /// Distance between the two points.
    pub dist: f32,
    /// Absolute cosine of the angle between the surface normal at the
    /// receiving point and the connecting direction.
    pub cos_there: f32,
}

impl From<(SolidAnglePdf, PdfConvCtx)> for AreaPdf {
    /// `pdfA = pdfW * |cosθ| / dist²`
    #[inline]
    fn from((pdf, ctx): (SolidAnglePdf, PdfConvCtx)) -> Self {
        AreaPdf(pdf.0 * ctx.cos_there.abs() / (ctx.dist * ctx.dist))
    }
}

impl From<(AreaPdf, PdfConvCtx)> for SolidAnglePdf {
    /// `pdfW = pdfA * dist² / |cosθ|`
    #[inline]
    fn from((pdf, ctx): (AreaPdf, PdfConvCtx)) -> Self {
        SolidAnglePdf(pdf.0 * ctx.dist * ctx.dist / ctx.cos_there.abs())
    }
}

// ================================================================
// § MIS heuristics
//
// Balance and Power (β=2) heuristics as first-class enum values,
// swappable at the call site.
//
// Reference: luxrays/utils/mc.h lines 69-81, pbrt-v4 sampling.h
// ================================================================

/// Multiple Importance Sampling heuristic strategy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MisHeuristic {
    /// Balance heuristic: w_f = f / (f + g)
    Balance,
    /// Power heuristic (β=2): w_f = f² / (f² + g²)
    Power,
}

impl MisHeuristic {
    /// N-strategy MIS weight for the selected strategy.
    ///
    /// `sel_idx` is the index of the strategy that generated the sample.
    /// `pdfs` is the array of all strategy PDFs.
    /// Returns 0 when the PDF sum is degenerate.
    #[inline]
    pub fn weight<const N: usize>(&self, sel_idx: usize, pdfs: &[f32; N]) -> f32 {
        let sel = pdfs[sel_idx];
        match self {
            MisHeuristic::Balance => {
                let sum: f32 = pdfs.iter().sum();
                if sum <= 0.0 { 0.0 } else { sel / sum }
            }
            MisHeuristic::Power => {
                let sum_sq: f32 = pdfs.iter().map(|p| p * p).sum();
                if sum_sq <= 0.0 {
                    0.0
                } else {
                    sel * sel / sum_sq
                }
            }
        }
    }
}

// ================================================================
// § PdfKind — concrete sampling strategies
//
// Lightweight enum returned by materials instead of heap-allocated
// `Box<dyn PDF>`. The integrator owns concrete PDF objects on the
// stack and updates them from the kind + parameters here.
// ================================================================

/// Describes which surface sampling PDF the integrator should use.
#[derive(Clone, Copy, Debug)]
pub enum PdfKind {
    /// Cosine-weighted hemisphere. `normal` defines the hemisphere orientation.
    Cosine {
        /// Surface normal.
        normal: Direction3,
    },
    /// Geometric G-buffer X (GGX) microfacet importance sampling. Samples half-vector from NDF, reflects.
    Ggx {
        /// Outgoing direction (surface → camera), world space.
        wo: Direction3,
        /// Surface normal.
        normal: Direction3,
        /// GGX alpha (roughness² clamped to [0.001, 1]).
        alpha: f32,
    },
    /// Uniform over the full sphere (isotropic volumes).
    UniformSphere,
    /// Uniform over the hemisphere oriented by `normal`.
    UniformHemisphere { normal: Direction3 },
}

impl PdfKind {
    /// Generate a direction from this PDF distribution.
    pub fn generate(&self, u: f32, v: f32) -> Direction3 {
        match self {
            PdfKind::Cosine { normal } => {
                let uvw = Onb::build_from_normal(*normal);
                uvw.local_to_world(cosine_hemisphere_direction(u, v))
            }
            PdfKind::Ggx { wo, normal, alpha } => {
                let onb = Onb::build_from_normal(*normal);
                let wo_unit = wo.normalize();
                let h_local = ggx_sample_h(*alpha, u, v);
                let h_world = onb.local_to_world(h_local);
                -wo_unit.reflect(h_world.into_inner())
            }
            PdfKind::UniformSphere => {
                let phi = 2.0 * PI * u;
                let (sin_phi, cos_phi) = phi.sin_cos();
                let z = 1.0 - 2.0 * v;
                let r = (1.0 - z * z).max(0.0).sqrt();
                Direction3::new(r * cos_phi, r * sin_phi, z)
            }
            PdfKind::UniformHemisphere { normal } => {
                let uvw = Onb::build_from_normal(*normal);
                let phi = 2.0 * PI * u;
                let (sin_phi, cos_phi) = phi.sin_cos();
                let z = v;
                let r = (1.0 - z * z).max(0.0).sqrt();
                let local_dir = Direction3::new(r * cos_phi, r * sin_phi, z);
                uvw.local_to_world(local_dir)
            }
        }
    }

    /// Evaluate the PDF value for a given direction.
    pub fn value(&self, direction: Direction3) -> f32 {
        match self {
            PdfKind::Cosine { normal } => {
                let uvw = Onb::build_from_normal(*normal);
                let cos_theta = direction.dot(uvw.w.into_inner());
                (cos_theta / PI).max(0.0)
            }
            PdfKind::Ggx { wo, normal, alpha } => {
                let onb = Onb::build_from_normal(*normal);
                let wo_unit = wo.normalize();
                let h = (wo_unit + direction).normalize();
                let cos_h = wo_unit.dot(h.into_inner()).abs();
                if cos_h <= 0.0 {
                    return 0.0;
                }
                let h_local = onb.world_to_local(h);
                let cos_h_n = h_local.z().max(0.0);
                let d = ggx_d(cos_h_n, *alpha);
                d * cos_h_n / (4.0 * cos_h)
            }
            PdfKind::UniformSphere => 1.0 / (4.0 * PI),
            PdfKind::UniformHemisphere { normal } => {
                let uvw = Onb::build_from_normal(*normal);
                let cos_theta = direction.dot(uvw.w.into_inner());
                if cos_theta > 0.0 {
                    1.0 / (2.0 * PI)
                } else {
                    0.0
                }
            }
        }
    }
}

// ================================================================
// § Concrete PDFs
// ================================================================

/// PDF for sampling directions from a set of light emitters.
pub struct EmitterPDF<'a> {
    /// The set of light emitters to sample from.
    objects: &'a [LightPrimitive],
    /// The origin point from which to sample direction.
    origin: Point3,
    /// The time at which to sample the emitter.
    time: f32,
}

impl<'a> EmitterPDF<'a> {
    pub fn new(objects: &'a [LightPrimitive], origin: Point3, time: f32) -> Self {
        EmitterPDF {
            objects,
            origin,
            time,
        }
    }

    /// Evaluates the PDF value for a given direction.
    pub fn value(&self, direction: Direction3) -> f32 {
        if self.objects.is_empty() {
            return 0.0;
        }
        let inv_len = 1.0 / self.objects.len() as f32;
        self.objects
            .iter()
            .map(|o| o.pdf_value(self.origin, direction, self.time) * inv_len)
            .sum()
    }
}

/// A concrete MIS sampling strategy — enum dispatch of `&dyn PDF` in the
/// integrator's strategy array. `Env` wraps the environment map; `Kind`
/// wraps a materialized [`PdfKind`].
#[derive(Clone, Copy)]
pub enum PdfStrategy<'a> {
    /// Environment map importance sampling.
    Env(&'a EnvironmentMap),
    /// Material PDF kind (cosine, GGX, uniform sphere/hemisphere).
    Kind(PdfKind),
}

impl PdfStrategy<'_> {
    /// Evaluates the PDF value for a given direction.
    pub fn value(&self, direction: Direction3) -> f32 {
        match self {
            PdfStrategy::Env(env) => env.to_solid_angle_pdf(direction).0,
            PdfStrategy::Kind(kind) => kind.value(direction),
        }
    }

    /// Generates a random direction according to the PDF from `(u, v)` in [0, 1)².
    pub fn generate(&self, u: f32, v: f32) -> Direction3 {
        match self {
            PdfStrategy::Env(env) => {
                let (col, row, _) = env.sample(u, v);
                let theta = (row as f32 + 0.5) / env.height() as f32 * PI;
                let phi = (col as f32 + 0.5) / env.width() as f32 * 2.0 * PI;
                let sin_theta = theta.sin();
                Direction3::new(sin_theta * phi.cos(), theta.cos(), sin_theta * phi.sin())
            }
            PdfStrategy::Kind(kind) => kind.generate(u, v),
        }
    }
}

// ================================================================
// § Sampling helpers
// ================================================================

/// Uniform hemisphere PDF: 1 / (2π)
#[inline(always)]
pub fn uniform_hemisphere_pdf() -> f32 {
    0.5 * FRAC_1_PI
}

/// Uniform sphere PDF: 1 / (4π)
#[inline(always)]
pub fn uniform_sphere_pdf() -> f32 {
    0.25 * FRAC_1_PI
}

/// Uniform cone PDF: 1 / (2π(1 − cosθmax))
#[inline]
pub fn uniform_cone_pdf(cos_theta_max: f32) -> f32 {
    if cos_theta_max >= 1.0 {
        return 0.0;
    }
    1.0 / (2.0 * PI * (1.0 - cos_theta_max))
}

/// Cosine-weighted hemisphere PDF: cosθ / π
#[inline(always)]
pub fn cosine_hemisphere_pdf(cos_theta: f32) -> f32 {
    cos_theta * FRAC_1_PI
}

/// Sample a discrete distribution by weight.
///
/// Returns `(index, pmf, u_remapped)` where `pmf` is the discrete PMF
/// of the selected bin and `u_remapped` ∈ [0, 1) is the variate
/// re-mapped within the selected bin.  Returns `None` when the weight
/// array is empty.
pub fn sample_discrete(weights: &[f32], u: f32) -> Option<(usize, f32, f32)> {
    if weights.is_empty() {
        return None;
    }
    let sum: f32 = weights.iter().sum();
    if sum <= 0.0 {
        // Uniform fallback: treat all weights as equal.
        let n = weights.len() as f32;
        let idx = (u * n).min(n - 1.0) as usize;
        let u_remapped = (u * n - idx as f32).min(1.0 - 1e-15);
        return Some((idx, 1.0 / n, u_remapped));
    }
    let mut up = u * sum;
    if up >= sum {
        up = sum.next_down();
    }

    let mut offset = 0usize;
    let mut running = 0.0f32;
    while running + weights[offset] <= up {
        running += weights[offset];
        offset += 1;
    }
    let pmf = weights[offset] / sum;
    let u_remapped = if weights[offset] > 0.0 {
        ((up - running) / weights[offset]).min(1.0 - 1e-15)
    } else {
        0.0
    };
    Some((offset, pmf, u_remapped))
}

/// Cosine-weighted hemisphere direction via concentric disk mapping.
///
/// Takes two uniform random values `(u, v)` in `[0, 1)` and returns a direction
/// on the unit hemisphere with PDF `cos(θ) / π`. The concentric disk mapping
/// avoids the rejection sampling of `sampler_cosine_direction`.
///
/// Reference: Shirley & Chiu, "A Low Distortion Map Between Disk and Square", 1997.
#[inline(always)]
pub fn cosine_hemisphere_direction(u: f32, v: f32) -> Direction3 {
    // Concentric disk mapping: map (u,v) in [0,1)^2 to (x,y) on the unit disk.
    let (x, y) = concentric_disk(u, v);
    Direction3::new(x, y, (1.0 - x * x - y * y).max(0.0).sqrt())
}

/// Uniform hemisphere direction via spherical coordinates.
///
/// Takes two uniform random values `(u, v)` in `[0, 1)` and
/// returns a direction on the unit hemisphere with PDF `1 / (2π)`.
#[inline(always)]
pub fn uniform_hemisphere_direction(u: f32, v: f32) -> Direction3 {
    let phi = 2.0 * PI * u;
    let (sin_phi, cos_phi) = phi.sin_cos();
    let z = v;
    let r = (1.0 - z * z).max(0.0).sqrt();
    Direction3::new(r * cos_phi, r * sin_phi, z)
}

/// GGX/Trowbridge-Reitz microfacet importance sampling.
///
/// Samples a half-vector H from the GGX NDF given roughness² `alpha` and uniform
/// random variables `u`, `v` in [0, 1). Returns H in tangent space (Z = normal).
pub fn ggx_sample_h(alpha: f32, u: f32, v: f32) -> Direction3 {
    let cos_theta = ((1.0 - v) / (1.0 + (alpha * alpha - 1.0) * v))
        .clamp(0.0, 1.0)
        .sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi = 2.0 * PI * u;
    let (sin_phi, cos_phi) = phi.sin_cos();
    Direction3::new(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta)
}

/// GGX/Trowbridge-Reitz normal distribution function (NDF).
///
/// Returns the probability density that a microfacet has half-vector H aligned
/// with the surface normal. `alpha` is roughness²; controls specular lobe width.
pub fn ggx_d(cos_theta_h: f32, alpha: f32) -> f32 {
    if cos_theta_h <= 0.0 {
        return 0.0;
    }
    let a2 = alpha * alpha;
    let denom = cos_theta_h * cos_theta_h * (a2 - 1.0) + 1.0;
    a2 / (PI * denom * denom)
}
