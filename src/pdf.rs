use std::f64::consts::PI;
use std::sync::Arc;

use crate::hittable::Sampleable;
use crate::material::PdfKind;
use crate::material::ggx_d;
use crate::onb::Onb;
use crate::sampler::{DimCursor, Sampler};
use crate::vec3::{Point3, Vec3, concentric_disk, reflect};

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
    Vec3::from(x, y, (1.0 - x * x - y * y).max(0.0).sqrt())
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
    Vec3::from(r * cos_phi, r * sin_phi, z)
}

/// Type-erased wrapper that delegates to a concrete PDF type.
///
/// Avoids `dyn PDF` dispatch in the MIS hot path while allowing the integrator
/// to build PDFs from runtime `PdfKind` descriptors.
pub struct PdfEnum<S: Sampler> {
    inner: PdfEnumInner,
    _s: std::marker::PhantomData<S>,
}

enum PdfEnumInner {
    Cosine(CosinePDF),
    Ggx(GgxSamplePDF),
    UniformSphere(UniformSpherePDF),
    UniformHemisphere(UniformHemispherePDF),
}

impl<S: Sampler> PDF<S> for PdfEnum<S> {
    fn value(&self, direction: Vec3) -> f64 {
        match &self.inner {
            PdfEnumInner::Cosine(p) => <CosinePDF as PDF<S>>::value(p, direction),
            PdfEnumInner::Ggx(p) => <GgxSamplePDF as PDF<S>>::value(p, direction),
            PdfEnumInner::UniformSphere(p) => <UniformSpherePDF as PDF<S>>::value(p, direction),
            PdfEnumInner::UniformHemisphere(p) => {
                <UniformHemispherePDF as PDF<S>>::value(p, direction)
            }
        }
    }
    fn generate(&self, dim_offset: &mut DimCursor<S>) -> Vec3 {
        match &self.inner {
            PdfEnumInner::Cosine(p) => <CosinePDF as PDF<S>>::generate(p, dim_offset),
            PdfEnumInner::Ggx(p) => <GgxSamplePDF as PDF<S>>::generate(p, dim_offset),
            PdfEnumInner::UniformSphere(p) => <UniformSpherePDF as PDF<S>>::generate(p, dim_offset),
            PdfEnumInner::UniformHemisphere(p) => {
                <UniformHemispherePDF as PDF<S>>::generate(p, dim_offset)
            }
        }
    }
}

impl<S: Sampler> PdfEnum<S> {
    /// Construct a concrete PDF from a `PdfKind` descriptor.
    pub fn new(pk: &crate::material::PdfKind) -> Self {
        let inner = match pk {
            PdfKind::Cosine { normal } => PdfEnumInner::Cosine(CosinePDF::new(*normal)),
            PdfKind::Ggx { wo, normal, alpha } => {
                PdfEnumInner::Ggx(GgxSamplePDF::new(*wo, *normal, *alpha))
            }
            PdfKind::UniformSphere => PdfEnumInner::UniformSphere(UniformSpherePDF::new()),
            PdfKind::UniformHemisphere { normal } => {
                PdfEnumInner::UniformHemisphere(UniformHemispherePDF::new(*normal))
            }
            PdfKind::Delta => unreachable!(),
        };
        PdfEnum {
            inner,
            _s: std::marker::PhantomData,
        }
    }
}

pub trait PDF<S: Sampler> {
    /// Evaluates the PDF value for a given direction.
    fn value(&self, direction: Vec3) -> f64;

    /// Generates a random direction according to the PDF.
    fn generate(&self, dim_offset: &mut DimCursor<S>) -> Vec3;
}

pub struct UniformSpherePDF;

impl UniformSpherePDF {
    #[allow(clippy::new_without_default)]
    pub const fn new() -> Self {
        Self
    }
}

impl<S: Sampler> PDF<S> for UniformSpherePDF {
    fn value(&self, _direction: Vec3) -> f64 {
        1.0 / (4.0 * std::f64::consts::PI)
    }

    fn generate(&self, dim_offset: &mut DimCursor<S>) -> Vec3 {
        let u = dim_offset.next_sample();
        let v = dim_offset.next_sample();

        let phi = 2.0 * std::f64::consts::PI * u;
        let (sin_phi, cos_phi) = phi.sin_cos();
        let z = 1.0 - 2.0 * v;
        let r = (1.0 - z * z).max(0.0).sqrt();
        Vec3::from(r * cos_phi, r * sin_phi, z)
    }
}

/// Uniformly distributed directions over the hemisphere.
///
/// `value(d) = 1 / 2π` for directions in the hemisphere above `normal`, zero otherwise.
/// `generate()` produces uniform directions via ONB around the surface normal.
///
/// Geometric sampling primitive — no weighting, no texture, just
/// the hemisphere.  Used in mixture PDFs to give scattered rays a
/// controlled probability of escaping to background illumination.
pub struct UniformHemispherePDF {
    /// The normal vector defining the hemisphere for uniform sampling.
    pub uvw: Onb,
}

impl UniformHemispherePDF {
    pub fn new(normal: Vec3) -> Self {
        Self {
            uvw: Onb::build_from_normal(normal),
        }
    }
}

impl<S: Sampler> PDF<S> for UniformHemispherePDF {
    fn value(&self, direction: Vec3) -> f64 {
        let cos_theta = direction.dot(&self.uvw.w);
        if cos_theta > 0.0 {
            1.0 / (2.0 * PI)
        } else {
            0.0
        }
    }

    fn generate(&self, dim_offset: &mut DimCursor<S>) -> Vec3 {
        let u = dim_offset.next_sample();
        let v = dim_offset.next_sample();

        let phi = 2.0 * PI * u;
        let (sin_phi, cos_phi) = phi.sin_cos();
        let z = v;
        let r = (1.0 - z * z).max(0.0).sqrt();
        let local_dir = Vec3::from(r * cos_phi, r * sin_phi, z);

        self.uvw.local_to_world(local_dir)
    }
}

pub struct CosinePDF {
    /// The normal vector defining the hemisphere for cosine-weighted sampling.
    pub uvw: Onb,
}

impl CosinePDF {
    pub fn new(normal: Vec3) -> Self {
        Self {
            uvw: Onb::build_from_normal(normal),
        }
    }
}

impl<S: Sampler> PDF<S> for CosinePDF {
    fn value(&self, direction: Vec3) -> f64 {
        let cos_theta = direction.dot(&self.uvw.w);
        (cos_theta / PI).max(0.)
    }

    fn generate(&self, dim_offset: &mut DimCursor<S>) -> Vec3 {
        let u = dim_offset.next_sample();
        let v = dim_offset.next_sample();

        self.uvw.local_to_world(cosine_hemisphere_direction(u, v))
    }
}

pub struct EmitterPDF<'a> {
    objects: &'a [Arc<dyn Sampleable>],
    origin: Point3,
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

impl<'a, S: Sampler> PDF<S> for EmitterPDF<'a> {
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

    fn generate(&self, dim_offset: &mut DimCursor<S>) -> Vec3 {
        if self.objects.is_empty() {
            return Vec3::ZERO;
        }
        // Selection: 1 QMC sample to pick a light
        let u_select = dim_offset.next_sample();
        // Clamp to [0, N) to avoid out-of-bounds index when u_select == 1.0
        let index =
            (u_select * self.objects.len() as f64).min(self.objects.len() as f64 - 1e-15) as usize;
        // Direction: 2 QMC samples for (u, v)
        let u = dim_offset.next_sample();
        let v = dim_offset.next_sample();
        self.objects[index].random_direction(self.origin, u, v, self.time)
    }
}

// The GGX distribution is centered on the **surface normal**, not the camera
// direction. The PDF is `D(H) * (n·H) / (4 * |wo·H|)` — the standard form
// for microfacet importance sampling with the half-vector transformation.

/// PDF for sampling directions from the GGX importance-sampling distribution.
///
/// `wo` is the outgoing direction (from surface to camera). Importance-sampling
/// picks a microfacet normal `H` from the GGX NDF, then reflects `wo` about
/// `H` to get `wi`. The PDF of the resulting `wi` is:
///
///   p(wi) = D(H) * (n · H) / (4 * (wo · H))
///
/// The GGX distribution is centered on the **surface normal** `n`, not `wo`.
/// Both `value` and `generate` use an orthonormal basis aligned with `n`
/// so the GGX lobe is correctly oriented in the hemisphere.
pub struct GgxSamplePDF {
    alpha: f64,
    /// Pre-normalized outgoing direction (toward camera) — avoids redundant
    /// `unit_vector()` calls in value() and generate().
    wo_unit: Vec3,
    // /// The surface normal — the GGX distribution is centered on this.
    // normal: Vec3,
    onb: Onb,
}

impl GgxSamplePDF {
    pub fn new(wo: Vec3, normal: Vec3, alpha: f64) -> Self {
        // Evaluate D(H) in the local frame where the normal is the up-axis (z in the ONB
        // convention). Previously this used h.y (world-space Y), which is only correct when the
        // surface normal is world-up.
        // Build the ONB from the **surface normal**, not wo, so the GGX lobe is correctly centered
        // on the normal.
        let onb = Onb::build_from_normal(normal);
        let wo_unit = wo.unit_vector();

        Self {
            alpha,
            wo_unit,
            onb,
        }
    }
}

impl<S: Sampler> PDF<S> for GgxSamplePDF {
    fn value(&self, direction: Vec3) -> f64 {
        let wi = direction;
        let h = (self.wo_unit + wi).unit_vector();
        let cos_h = self.wo_unit.dot(&h).abs();
        if cos_h <= 0.0 {
            return 0.0;
        }

        let h_local = self.onb.world_to_local(h);
        let cos_h_n = h_local.z.max(0.0);

        let d = ggx_d(cos_h_n, self.alpha);
        d * cos_h_n / (4.0 * cos_h)
    }

    fn generate(&self, dim_offset: &mut DimCursor<S>) -> Vec3 {
        let u = dim_offset.next_sample();
        let v = dim_offset.next_sample();

        let a = self.alpha;

        let cos_theta = ((1.0 - v) / (1.0 + (a * a - 1.0) * v))
            .clamp(0.0, 1.0)
            .sqrt();
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();

        let phi = 2.0 * PI * u;
        let (sin_phi, cos_phi) = phi.sin_cos();

        // Local frame: x=bitangent, y=tangent, z=normal (matches ONB convention). Put cos_theta on
        // the normal axis (z).
        let h_local = Vec3::from(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);

        let h_world = self.onb.local_to_world(h_local);

        // Reflect wo about H to get wi. `reflect` expects the incident direction (toward surface),
        // so negate wo which points away from the surface.
        reflect(&-self.wo_unit, &h_world)
    }
}
