use std::f64::consts::PI;

use crate::hittable::Hittable;
use crate::material::ggx_d;
use crate::onb::Onb;
use crate::vec3::reflect;
use crate::vec3::{Point3, Vec3, random_cosine_direction};

use rand::RngExt;

pub trait PDF {
    /// Evaluates the PDF value for a given direction.
    fn value(&self, direction: Vec3) -> f64;

    /// Generates a random direction according to the PDF.
    fn generate(&self, rng: &mut dyn rand::Rng) -> Vec3;
}

pub struct UniformSpherePDF;

impl PDF for UniformSpherePDF {
    fn value(&self, _direction: Vec3) -> f64 {
        1.0 / (4.0 * std::f64::consts::PI)
    }

    fn generate(&self, rng: &mut dyn rand::Rng) -> Vec3 {
        // Rejection sampling to generate a random unit vector uniformly distributed on the surface of the unit sphere.
        loop {
            let point = Vec3::from(
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
            );
            let len_squared = point.length_squared();
            if 1e-160 < len_squared && len_squared <= 1.0 {
                return point / len_squared.sqrt();
            }
        }
    }
}

#[allow(clippy::new_without_default)]
impl UniformSpherePDF {
    pub fn new() -> Self {
        Self
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

    fn generate(&self, rng: &mut dyn rand::Rng) -> Vec3 {
        self.uvw.local_to_world(random_cosine_direction(rng))
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

    fn generate(&self, rng: &mut dyn rand::Rng) -> Vec3 {
        self.objects.random(self.origin, rng)
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

    fn generate(&self, _rng: &mut dyn rand::Rng) -> Vec3 {
        self.direction
    }
}

pub struct MixturePDF<'a> {
    pdfs: &'a [&'a dyn PDF],
}

impl<'a> MixturePDF<'a> {
    pub fn new(pdfs: &'a [&'a dyn PDF]) -> Self {
        MixturePDF { pdfs }
    }
}

impl<'a> PDF for MixturePDF<'a> {
    fn value(&self, direction: Vec3) -> f64 {
        self.pdfs
            .iter()
            .map(|pdf| pdf.value(direction) / self.pdfs.len() as f64)
            .sum()
    }

    fn generate(&self, rng: &mut dyn rand::Rng) -> Vec3 {
        let index = rng.random_range(0..self.pdfs.len());
        self.pdfs[index].generate(rng)
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
    /// The surface normal — the GGX distribution is centered on this.
    normal: Vec3,
}

impl GgxSamplePDF {
    pub fn new(wo: Vec3, normal: Vec3, alpha: f64) -> Self {
        Self { alpha, wo, normal }
    }
}

impl PDF for GgxSamplePDF {
    fn value(&self, direction: Vec3) -> f64 {
        let wi = direction.unit_vector();
        let h = (self.wo + wi).unit_vector();
        let cos_h = h.dot(&wi); // |wo·H| == |wi·H| when H is the half-vector
        if cos_h <= 0.0 {
            return 0.0;
        }

        // Evaluate D(H) in the local frame where the normal is the up-axis (z in the ONB
        // convention). Previously this used h.y (world-space Y), which is only correct when the
        // surface normal is world-up.
        let onb = Onb::build_from_normal(self.normal);

        let h_local = onb.world_to_local(h);
        let cos_h_n = h_local.z.max(0.0);

        let d = ggx_d(cos_h_n, self.alpha);
        d * cos_h_n / (4.0 * cos_h)
    }

    fn generate(&self, rng: &mut dyn rand::Rng) -> Vec3 {
        // Sample H from the GGX distribution. In the local frame where the surface normal is +z,
        // this gives a half-vector distributed as D(H).
        let u1: f64 = rng.random();
        let u2: f64 = rng.random();

        let a = self.alpha;
        let cos_theta = ((1.0 - u2) / (1.0 + (a * a - 1.0) * u2))
            .clamp(0.0, 1.0)
            .sqrt();
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();

        let phi = 2.0 * PI * u1;

        // Local frame: x=bitangent, y=tangent, z=normal (matches ONB convention). Put cos_theta on
        // the normal axis (z).
        let h_local = Vec3::from(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta);

        // Build an ONB from the **surface normal**, not wo, so the GGX lobe is correctly centered
        // on the normal.
        let onb = Onb::build_from_normal(self.normal);

        let h_world = onb.local_to_world(h_local);

        // Reflect wo about H to get wi. `reflect` expects the incident direction (toward surface),
        // so negate wo which points away from the surface.
        let wo_unit = self.wo.unit_vector();
        reflect(&-wo_unit, &h_world)
    }
}
