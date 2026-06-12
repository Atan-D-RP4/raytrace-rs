use std::f64::consts::PI;

use crate::hittable::Hittable;
use crate::material::ggx_d;
use crate::onb::Onb;
use crate::sampler::{DimCursor, Sampler};
use crate::vec3::cosine_hemisphere_direction;
use crate::vec3::reflect;
use crate::vec3::{Point3, Vec3};

pub trait PDF {
    /// Evaluates the PDF value for a given direction.
    fn value(&self, direction: Vec3) -> f64;

    /// Generates a random direction according to the PDF.
    fn generate(
        &self,
        sampler: &dyn Sampler,
        sample_index: u32,
        dim_offset: &mut DimCursor,
    ) -> Vec3;
}

pub struct UniformSpherePDF;

#[allow(clippy::new_without_default)]
impl UniformSpherePDF {
    pub const fn new() -> Self {
        Self
    }
}

impl PDF for UniformSpherePDF {
    fn value(&self, _direction: Vec3) -> f64 {
        1.0 / (4.0 * std::f64::consts::PI)
    }

    fn generate(
        &self,
        sampler: &dyn Sampler,
        sample_index: u32,
        dim_offset: &mut DimCursor,
    ) -> Vec3 {
        let u = sampler.sample(sample_index, dim_offset.next_dim());
        let v = sampler.sample(sample_index, dim_offset.next_dim());
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

impl PDF for CosinePDF {
    fn value(&self, direction: Vec3) -> f64 {
        let cos_theta = direction.unit_vector().dot(&self.uvw.w);
        (cos_theta / PI).max(0.)
    }

    fn generate(
        &self,
        sampler: &dyn Sampler,
        sample_index: u32,
        dim_offset: &mut DimCursor,
    ) -> Vec3 {
        let u = sampler.sample(sample_index, dim_offset.next_dim());
        let v = sampler.sample(sample_index, dim_offset.next_dim());
        self.uvw.local_to_world(cosine_hemisphere_direction(u, v))
    }
}

pub struct HittablePDF<'a> {
    objects: &'a dyn Hittable,
    origin: Point3,
}

impl<'a> HittablePDF<'a> {
    pub fn new(objects: &'a dyn Hittable, origin: Point3) -> Self {
        HittablePDF { objects, origin }
    }
}

impl<'a> PDF for HittablePDF<'a> {
    fn value(&self, direction: Vec3) -> f64 {
        self.objects.pdf_value(self.origin, direction)
    }

    fn generate(
        &self,
        sampler: &dyn Sampler,
        sample_index: u32,
        dim_offset: &mut DimCursor,
    ) -> Vec3 {
        self.objects
            .random(self.origin, sampler, sample_index, dim_offset)
    }
}

pub struct DiracPDF {
    direction: Vec3,
}

impl DiracPDF {
    pub fn new(direction: Vec3) -> Self {
        Self { direction }
    }
}

impl PDF for DiracPDF {
    fn value(&self, _direction: Vec3) -> f64 {
        // Dirac delta: the sampling PDF matches the material's delta distribution.
        // In the MC estimator, both deltas cancel, leaving just the reflectance.
        1.0
    }

    fn generate(
        &self,
        _sampler: &dyn Sampler,
        _sample_index: u32,
        _dim_offset: &mut DimCursor,
    ) -> Vec3 {
        self.direction
    }
}

pub struct MixturePDF<'a> {
    pdfs: &'a [&'a dyn PDF],
    weights: Vec<f64>,
}

impl<'a> MixturePDF<'a> {
    /// Creates a mixture PDF with weights inferred from pointer identity.
    ///
    /// If a PDF appears multiple times in the slice, its weight is proportional
    /// to its count. For example, `[light_pdf, surface_pdf, surface_pdf]`
    /// yields weights `[1/3, 2/3]` — the surface PDF gets double weight.
    pub fn new(pdfs: &'a [&'a dyn PDF]) -> Self {
        let n = pdfs.len() as f64;
        let weights: Vec<f64> = pdfs
            .iter()
            .map(|pdf| {
                let ptr: *const (dyn PDF + 'a) = *pdf;
                let count = pdfs
                    .iter()
                    .filter(|p| {
                        let other: *const (dyn PDF + 'a) = **p;
                        std::ptr::eq(other, ptr)
                    })
                    .count();
                count as f64 / n
            })
            .collect();
        MixturePDF { pdfs, weights }
    }
}

impl<'a> PDF for MixturePDF<'a> {
    fn value(&self, direction: Vec3) -> f64 {
        self.pdfs
            .iter()
            .zip(&self.weights)
            .map(|(pdf, &w)| w * pdf.value(direction))
            .sum()
    }

    fn generate(
        &self,
        sampler: &dyn Sampler,
        sample_index: u32,
        dim_offset: &mut DimCursor,
    ) -> Vec3 {
        let u = sampler.sample(sample_index, dim_offset.next_dim());
        let mut cumulative = 0.0;
        for (pdf, &weight) in self.pdfs.iter().zip(&self.weights) {
            cumulative += weight;
            if u < cumulative {
                return pdf.generate(sampler, sample_index, dim_offset);
            }
        }
        self.pdfs
            .last()
            .unwrap()
            .generate(sampler, sample_index, dim_offset)
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
    /// The outgoing direction (toward camera), kept for value().
    wo: Vec3,
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

        Self { alpha, wo, onb }
    }
}

impl PDF for GgxSamplePDF {
    fn value(&self, direction: Vec3) -> f64 {
        let wi = direction.unit_vector();
        let h = (self.wo + wi).unit_vector();
        // let cos_h = h.dot(&wi); // |wo·H| == |wi·H| when H is the half-vector
        let cos_h = self.wo.unit_vector().dot(&h).abs();
        if cos_h <= 0.0 {
            return 0.0;
        }

        let h_local = self.onb.world_to_local(h);
        let cos_h_n = h_local.z.max(0.0);

        let d = ggx_d(cos_h_n, self.alpha);
        d * cos_h_n / (4.0 * cos_h)
    }

    fn generate(
        &self,
        sampler: &dyn Sampler,
        sample_index: u32,
        dim_offset: &mut DimCursor,
    ) -> Vec3 {
        let u1 = sampler.sample(sample_index, dim_offset.next_dim());
        let u2 = sampler.sample(sample_index, dim_offset.next_dim());

        let a = self.alpha;
        let cos_theta = ((1.0 - u2) / (1.0 + (a * a - 1.0) * u2))
            .clamp(0.0, 1.0)
            .sqrt();
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();

        let phi = 2.0 * PI * u1;

        // Local frame: x=bitangent, y=tangent, z=normal (matches ONB convention). Put cos_theta on
        // the normal axis (z).
        let h_local = Vec3::from(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta);

        let h_world = self.onb.local_to_world(h_local);

        // Reflect wo about H to get wi. `reflect` expects the incident direction (toward surface),
        // so negate wo which points away from the surface.
        let wo_unit = self.wo.unit_vector();
        reflect(&-wo_unit, &h_world)
    }
}
