use std::f64::consts::PI;

use crate::hittable::Sampleable;
use crate::material::ggx_d;
use crate::onb::Onb;
use crate::sampler::{DimCursor, Sampler};
use crate::vec3::cosine_hemisphere_direction;
use crate::vec3::reflect;
use crate::vec3::{Point3, Vec3};

pub trait PDF<S: Sampler> {
    /// Evaluates the PDF value for a given direction.
    fn value(&self, direction: Vec3) -> f64;

    /// Generates a random direction according to the PDF.
    fn generate(&self, dim_offset: &mut DimCursor<S>) -> Vec3;
}

pub struct UniformSpherePDF;

#[allow(clippy::new_without_default)]
impl UniformSpherePDF {
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
        let z = 1.0 - 2.0 * v;
        let r = (1.0 - z * z).max(0.0).sqrt();
        Vec3::from(r * phi.cos(), r * phi.sin(), z)
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
        let cos_theta = direction.unit_vector().dot(&self.uvw.w);
        (cos_theta / PI).max(0.)
    }

    fn generate(&self, dim_offset: &mut DimCursor<S>) -> Vec3 {
        let u = dim_offset.next_sample();
        let v = dim_offset.next_sample();

        self.uvw.local_to_world(cosine_hemisphere_direction(u, v))
    }
}

pub struct HittablePDF<'a, S: Sampler> {
    objects: &'a dyn Sampleable<S>,
    origin: Point3,
}

impl<'a, S: Sampler> HittablePDF<'a, S> {
    pub fn new(objects: &'a dyn Sampleable<S>, origin: Point3) -> Self {
        HittablePDF { objects, origin }
    }
}

impl<'a, S: Sampler> PDF<S> for HittablePDF<'a, S> {
    fn value(&self, direction: Vec3) -> f64 {
        self.objects.pdf_value(self.origin, direction)
    }

    fn generate(&self, dim_offset: &mut DimCursor<S>) -> Vec3 {
        self.objects.random(self.origin, dim_offset)
    }
}

pub struct MixturePDF<'a, S: Sampler, const N: usize> {
    pdfs: &'a [&'a dyn PDF<S>; N],
    /// Uniform weights — fully stack-allocated, no heap, no runtime len field.
    weights: [f64; N],
}

impl<'a, S: Sampler, const N: usize> MixturePDF<'a, S, N> {
    /// Creates a mixture PDF with uniform weights per entry.
    ///
    /// Callers control the mixture by repeating entries: `[light, surface, surface]`
    /// gives surface double weight — each entry gets `1/n`, so `value()` returns
    /// `light/3 + 2*surface/3`.
    pub fn new(pdfs: &'a [&'a dyn PDF<S>; N]) -> Self {
        let inv_n = 1.0 / N as f64;
        let weights = [inv_n; N];
        MixturePDF { pdfs, weights }
    }
}

impl<'a, S: Sampler, const N: usize> PDF<S> for MixturePDF<'a, S, N> {
    fn value(&self, direction: Vec3) -> f64 {
        self.pdfs
            .iter()
            .zip(&self.weights)
            .map(|(pdf, &w)| w * pdf.value(direction))
            .sum()
    }

    fn generate(&self, dim_offset: &mut DimCursor<S>) -> Vec3 {
        let u = dim_offset.next_sample();
        let mut cumulative = 0.0;
        for (pdf, &weight) in self.pdfs.iter().zip(&self.weights) {
            cumulative += weight;
            if u < cumulative {
                return pdf.generate(dim_offset);
            }
        }

        self.pdfs[N - 1].generate(dim_offset)
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
        let wi = direction.unit_vector();
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

        // Local frame: x=bitangent, y=tangent, z=normal (matches ONB convention). Put cos_theta on
        // the normal axis (z).
        let h_local = Vec3::from(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta);

        let h_world = self.onb.local_to_world(h_local);

        // Reflect wo about H to get wi. `reflect` expects the incident direction (toward surface),
        // so negate wo which points away from the surface.
        reflect(&-self.wo_unit, &h_world)
    }
}
