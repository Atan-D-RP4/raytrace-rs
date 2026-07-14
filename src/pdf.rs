use std::f32::consts::{FRAC_1_PI, PI};
use std::sync::Arc;

use glam::Vec3;

use crate::distributions::Sample1D;
use crate::environment::EnvironmentMap;
use crate::hittable::Sampleable;
use crate::onb::Onb;
use crate::vec3::{Point3, concentric_disk};

// ================================================================
// § PDF domain newtypes
//
// SolidAnglePdf and AreaPdf make the PDF domain explicit at the type
// level so callers cannot silently mix up solid-angle (sr⁻¹) and
// area (m⁻²) probability densities.  Conversions use the standard
// geometry term:  PdfAtoW / PdfWtoA.
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
    /// Two-technique MIS weight w(f_pdf, g_pdf).
    ///
    /// `f_pdf` is the PDF of the strategy that generated the sample;
    /// `g_pdf` is the PDF of the other strategy.
    #[inline]
    pub fn weight(self, f_pdf: f32, g_pdf: f32) -> f32 {
        match self {
            MisHeuristic::Balance => {
                let denom = f_pdf + g_pdf;
                if denom <= 0.0 { 0.0 } else { f_pdf / denom }
            }
            MisHeuristic::Power => {
                let denom = f_pdf * f_pdf + g_pdf * g_pdf;
                if denom <= 0.0 {
                    0.0
                } else {
                    f_pdf * f_pdf / denom
                }
            }
        }
    }

    /// Weighted variant with strategy counts: `w(nf·f, ng·g)`.
    #[inline]
    pub fn weight_n(self, nf: u32, f_pdf: f32, ng: u32, g_pdf: f32) -> f32 {
        self.weight(nf as f32 * f_pdf, ng as f32 * g_pdf)
    }

    /// Scalar squaring helper used in VCM-style MIS accumulators.
    #[inline(always)]
    pub fn vcm_term(a: f32) -> f32 {
        a * a
    }
}

/// Power-heuristic MIS weight: `p_i² / Σ(p_j²)`.
///
/// `pdf_sum_sq` must be the sum of squared PDF values (Σ p_j²).
/// Power-heuristic MIS weight (β=2) for N strategies.
///
/// Returns `p_i² / Σ(p_j²)`, the N-strategy form of the power heuristic.
/// `pdf_sum_sq` must be the sum of squared PDF values (Σ p_j²).
/// Returns 0 when the denominator is near-zero (degenerate PDF).
#[inline(always)]
pub fn power_heuristic(pdf_i: f32, pdf_sum_sq: f32) -> f32 {
    if pdf_sum_sq <= 1e-20 {
        return 0.0;
    }
    (pdf_i * pdf_i / pdf_sum_sq).max(0.0)
}

/// Balance-heuristic MIS weight: `p_i / Σ(p_j)`.
///
/// `pdf_sum` must be the sum of PDF values (Σ p_j).
/// Returns 0 when the denominator is near-zero (degenerate PDF).
///
/// This is the N-strategy form: `w_i = p_i / Σ(p_j)`.
#[inline(always)]
pub fn balance_heuristic(pdf_i: f32, pdf_sum: f32) -> f32 {
    if pdf_sum <= 1e-20 {
        return 0.0;
    }
    (pdf_i / pdf_sum).max(0.0)
}

/// Two-strategy power heuristic (pbrt-v4 style).
#[inline]
pub fn power_heuristic_2(nf: u32, f_pdf: f32, ng: u32, g_pdf: f32) -> f32 {
    MisHeuristic::Power.weight_n(nf, f_pdf, ng, g_pdf)
}

/// Two-strategy balance heuristic (pbrt-v4 style).
#[inline]
pub fn balance_heuristic_2(nf: u32, f_pdf: f32, ng: u32, g_pdf: f32) -> f32 {
    MisHeuristic::Balance.weight_n(nf, f_pdf, ng, g_pdf)
}

// ================================================================
// § Convenience PDF constants and functions
//
// Reference: luxrays/utils/mc.cpp, pbrt-v4 sampling.h
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

// ================================================================
// § sample_discrete — weighted discrete selection
//
// Reference: pbrt-v4 util/sampling.h lines 79-113
// ================================================================

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

// ================================================================
// § Distribution1DFixed — stack-allocated fixed-size 1D distribution
//
// Useful for small compile-time-known tables such as per-material
// lobe weights.
//
// Reference: luxrays Distribution1DFixed, pbrt-v4
// ================================================================

/// Stack-allocated 1D piecewise-constant distribution with `N` bins
/// known at compile time.
#[derive(Clone, Debug)]
pub struct Distribution1DFixed<const N: usize> {
    /// Normalized function values: `func[i] = w_i / total`.
    func: [f32; N],
    /// Cumulative distribution: `cdf[i]` = sum of func[0..=i].
    cdf: [f32; N],
    /// Total sum of raw input weights.
    func_int: f32,
    /// 1 / N — precomputed for speed.
    inv_count: f32,
}

impl<const N: usize> Distribution1DFixed<N> {
    /// Build from raw weight values. Non-positive weights are clamped to zero.
    /// A zero-total distribution falls back to uniform sampling.
    pub fn new(f: &[f32; N]) -> Self {
        let inv_count = 1.0 / N as f32;
        let mut func = [0.0f32; N];
        let mut total = 0.0f32;
        for (i, &v) in f.iter().enumerate() {
            let w = v.max(0.0);
            func[i] = w;
            total += w;
        }

        let mut cdf = [0.0f32; N];
        if total > 0.0 {
            let inv_total = 1.0 / total;
            let mut running = 0.0f32;
            for i in 0..N {
                running += func[i] * inv_total;
                cdf[i] = running;
            }
        } else {
            // Uniform fallback
            for i in 0..N {
                func[i] = 1.0;
                cdf[i] = (i + 1) as f32 * inv_count;
            }
        }

        Distribution1DFixed {
            func,
            cdf,
            func_int: total,
            inv_count,
        }
    }

    /// Integral of the original function over [0, 1).
    pub fn integral(&self) -> f32 {
        self.func_int
    }

    /// Bucket index for `u ∈ [0, 1)`.
    #[inline]
    pub fn offset(&self, u: f32) -> usize {
        ((u * N as f32) as usize).min(N - 1)
    }

    /// Continuous PDF at `u`.
    #[inline]
    pub fn pdf_continuous(&self, u: f32) -> f32 {
        self.func[self.offset(u)]
    }

    /// Discrete PDF for bucket at `offset`.
    #[inline]
    pub fn pdf_discrete(&self, offset: usize) -> f32 {
        self.func[offset] * self.inv_count
    }

    /// Sample a continuous position.
    pub fn sample_continuous(&self, u: f32) -> Sample1D {
        let n = N;
        if u <= 0.0 {
            return Sample1D::Continuous {
                x: 0.0,
                pdf: self.func[0],
                offset: 0,
            };
        }
        if u >= 1.0 {
            return Sample1D::Continuous {
                x: 1.0 - 1e-15,
                pdf: self.func[n - 1],
                offset: n - 1,
            };
        }
        let pos = self
            .cdf
            .partition_point(|&c| c <= u)
            .saturating_sub(1)
            .min(n - 1);
        let cdf_low = if pos == 0 { 0.0 } else { self.cdf[pos - 1] };
        let cdf_high = self.cdf[pos];
        let du = if cdf_high > cdf_low {
            ((u - cdf_low) / (cdf_high - cdf_low)).min(1.0 - 1e-15)
        } else {
            0.0
        };
        let x = ((pos as f32 + du) * self.inv_count).min(1.0 - 1e-15);
        Sample1D::Continuous {
            x,
            pdf: self.func[pos],
            offset: pos,
        }
    }

    /// Sample a discrete bucket.
    pub fn sample_discrete(&self, u: f32) -> Sample1D {
        let n = N;
        if u <= 0.0 {
            return Sample1D::Discrete {
                index: 0,
                pdf: self.func[0] * self.inv_count,
                du: 0.0,
            };
        }
        if u >= 1.0 {
            return Sample1D::Discrete {
                index: n - 1,
                pdf: self.func[n - 1] * self.inv_count,
                du: 1.0,
            };
        }
        let pos = self
            .cdf
            .partition_point(|&c| c <= u)
            .saturating_sub(1)
            .min(n - 1);
        let cdf_low = if pos == 0 { 0.0 } else { self.cdf[pos - 1] };
        let cdf_high = self.cdf[pos];
        let du = if cdf_high > cdf_low {
            ((u - cdf_low) / (cdf_high - cdf_low)).min(1.0)
        } else {
            0.0
        };
        Sample1D::Discrete {
            index: pos,
            pdf: self.func[pos] * self.inv_count,
            du,
        }
    }
}

/// Cosine-weighted hemisphere direction via concentric disk mapping.
///
/// Takes two uniform random values `(u, v)` in `[0, 1)` and returns a direction
/// on the unit hemisphere with PDF `cos(θ) / π`. The concentric disk mapping
/// avoids the rejection sampling of `sampler_cosine_direction`.
///
/// Reference: Shirley & Chiu, "A Low Distortion Map Between Disk and Square", 1997.
#[inline(always)]
pub fn cosine_hemisphere_direction(u: f32, v: f32) -> Vec3 {
    // Concentric disk mapping: map (u,v) in [0,1)^2 to (x,y) on the unit disk.
    let (x, y) = concentric_disk(u, v);
    Vec3::new(x, y, (1.0 - x * x - y * y).max(0.0).sqrt())
}

/// Uniform hemisphere direction via spherical coordinates.
///
/// Takes two uniform random values `(u, v)` in `[0, 1)` and
/// returns a direction on the unit hemisphere with PDF `1 / (2π)`.
#[inline(always)]
pub fn uniform_hemisphere_direction(u: f32, v: f32) -> Vec3 {
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
    fn value(&self, direction: Vec3) -> f32;

    /// Generates a random direction according to the PDF from `(u, v)` in [0, 1)².
    fn generate(&self, u: f32, v: f32) -> Vec3;
}

/// PDF for sampling directions from a set of light emitters
pub struct EmitterPDF<'a> {
    /// The set of light emitters to sample from.
    objects: &'a [Arc<dyn Sampleable>],
    /// The origin point from which to sample direction.
    origin: Point3,
    /// The time at which to sample the emitter.
    time: f32,
}

impl<'a> EmitterPDF<'a> {
    pub fn new(objects: &'a [Arc<dyn Sampleable>], origin: Point3, time: f32) -> Self {
        EmitterPDF {
            objects,
            origin,
            time,
        }
    }
}

impl<'a> PDF for EmitterPDF<'a> {
    fn value(&self, direction: Vec3) -> f32 {
        if self.objects.is_empty() {
            return 0.0;
        }
        let inv_len = 1.0 / self.objects.len() as f32;
        self.objects
            .iter()
            .map(|o| o.pdf_value(*self.origin, direction, self.time) * inv_len)
            .sum()
    }

    /// Samples a direction from the emitter set.
    ///
    /// `u` selects the light (uniformly across emitters) and is also passed
    /// through to the selected light's [`random_direction()`](crate::hittable::Sampleable::random_direction).
    /// `v` is passed through to the selected light's `random_direction()` method
    /// which uses it for the secondary random dimension (e.g., surface position
    /// within the light or directional PDF sampling).
    fn generate(&self, u: f32, v: f32) -> Vec3 {
        if self.objects.is_empty() {
            return Vec3::ZERO;
        }
        let index = (u * self.objects.len() as f32).min(self.objects.len() as f32 - 1e-15) as usize;
        self.objects[index].random_direction(*self.origin, u, v, self.time)
    }
}

/// PDF for a single sampleable light source.
///
/// Light selection is handled by the integrator — this PDF only generates
/// directions from the selected light. Wraps a reference to avoid cloning.
pub struct LightPDF<'a> {
    object: &'a Arc<dyn Sampleable>,
    origin: Point3,
    time: f32,
}

impl<'a> LightPDF<'a> {
    pub fn new(object: &'a Arc<dyn Sampleable>, origin: Point3, time: f32) -> Self {
        Self {
            object,
            origin,
            time,
        }
    }
}

impl<'a> PDF for LightPDF<'a> {
    fn value(&self, direction: Vec3) -> f32 {
        self.object.pdf_value(*self.origin, direction, self.time)
    }

    fn generate(&self, u: f32, v: f32) -> Vec3 {
        self.object.random_direction(*self.origin, u, v, self.time)
    }
}

/// Thin wrapper around an environment map that implements the [`PDF`] trait.
///
/// This allows the environment map to be used as a PDF for importance sampling directions from the
/// environment light.
pub struct EnvPdf<'a>(&'a Arc<EnvironmentMap>);

impl EnvPdf<'_> {
    pub fn new(env_map: &Arc<EnvironmentMap>) -> EnvPdf<'_> {
        EnvPdf(env_map)
    }
}

impl<'a> PDF for EnvPdf<'a> {
    fn value(&self, direction: Vec3) -> f32 {
        self.0.to_solid_angle_pdf(direction)
    }
    fn generate(&self, u: f32, v: f32) -> Vec3 {
        let (col, row, _) = self.0.sample(u, v);
        let theta = (row as f32 + 0.5) / self.0.height() as f32 * PI;
        let phi = (col as f32 + 0.5) / self.0.width() as f32 * 2.0 * PI;
        let sin_theta = theta.sin();
        Vec3::new(sin_theta * phi.cos(), theta.cos(), sin_theta * phi.sin())
    }
}

/// GGX/Trowbridge-Reitz microfacet importance sampling.
///
/// Samples a half-vector H from the GGX NDF given roughness² `alpha` and uniform
/// random variables `u`, `v` in [0, 1). Returns H in tangent space (Z = normal).
pub fn ggx_sample_h(alpha: f32, u: f32, v: f32) -> Vec3 {
    let cos_theta = ((1.0 - v) / (1.0 + (alpha * alpha - 1.0) * v))
        .clamp(0.0, 1.0)
        .sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi = 2.0 * PI * u;
    let (sin_phi, cos_phi) = phi.sin_cos();
    Vec3::new(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta)
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

/// Describes which surface sampling PDF the integrator should use.
///
/// Lightweight enum returned by materials instead of heap-allocated `Box<dyn PDF>`.
/// The integrator owns concrete PDF objects on the stack and updates them from
/// the kind + parameters here.
#[derive(Clone, Copy, Debug)]
pub enum PdfKind {
    /// Cosine-weighted hemisphere. `normal` defines the hemisphere orientation.
    Cosine { normal: Vec3 },
    /// GGX microfacet importance sampling. Samples half-vector from NDF, reflects.
    Ggx {
        /// Outgoing direction (surface → camera), world space.
        wo: Vec3,
        /// Surface normal.
        normal: Vec3,
        /// GGX alpha (roughness² clamped to [0.001, 1]).
        alpha: f32,
    },
    /// Uniform over the full sphere (isotropic volumes).
    UniformSphere,
    /// Uniform over the hemisphere oriented by `normal`.
    UniformHemisphere { normal: Vec3 },
}

impl PdfKind {
    /// Generate a direction from this PDF distribution.
    pub fn generate(&self, u: f32, v: f32) -> Vec3 {
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
                -wo_unit.reflect(h_world)
            }
            PdfKind::UniformSphere => {
                let phi = 2.0 * PI * u;
                let (sin_phi, cos_phi) = phi.sin_cos();
                let z = 1.0 - 2.0 * v;
                let r = (1.0 - z * z).max(0.0).sqrt();
                Vec3::new(r * cos_phi, r * sin_phi, z)
            }
            PdfKind::UniformHemisphere { normal } => {
                let uvw = Onb::build_from_normal(*normal);
                let phi = 2.0 * PI * u;
                let (sin_phi, cos_phi) = phi.sin_cos();
                let z = v;
                let r = (1.0 - z * z).max(0.0).sqrt();
                let local_dir = Vec3::new(r * cos_phi, r * sin_phi, z);
                uvw.local_to_world(local_dir)
            }
        }
    }

    /// Evaluate the PDF value for a given direction.
    pub fn value(&self, direction: Vec3) -> f32 {
        match self {
            PdfKind::Cosine { normal } => {
                let uvw = Onb::build_from_normal(*normal);
                let cos_theta = direction.dot(uvw.w);
                (cos_theta / PI).max(0.0)
            }
            PdfKind::Ggx { wo, normal, alpha } => {
                let onb = Onb::build_from_normal(*normal);
                let wo_unit = wo.normalize();
                let h = (wo_unit + direction).normalize();
                let cos_h = wo_unit.dot(h).abs();
                if cos_h <= 0.0 {
                    return 0.0;
                }
                let h_local = onb.world_to_local(h);
                let cos_h_n = h_local.z.max(0.0);
                let d = ggx_d(cos_h_n, *alpha);
                d * cos_h_n / (4.0 * cos_h)
            }
            PdfKind::UniformSphere => 1.0 / (4.0 * PI),
            PdfKind::UniformHemisphere { normal } => {
                let uvw = Onb::build_from_normal(*normal);
                let cos_theta = direction.dot(uvw.w);
                if cos_theta > 0.0 {
                    1.0 / (2.0 * PI)
                } else {
                    0.0
                }
            }
        }
    }
}

impl PDF for PdfKind {
    fn value(&self, direction: Vec3) -> f32 {
        PdfKind::value(self, direction)
    }

    fn generate(&self, u: f32, v: f32) -> Vec3 {
        PdfKind::generate(self, u, v)
    }
}
