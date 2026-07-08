//! Dielectric (glass/water) material with refraction and reflection.
//!
//! Transparent materials like glass or water. Light hitting the surface
//! either reflects or refracts, with the ratio governed by Fresnel's equations
//! and the refractive indices of the two media.
//!
//! **Snell's Law**: `η₁ sin(θ₁) = η₂ sin(θ₂)` determines the refraction angle.
//!
//! **Total Internal Reflection**: when light in a denser medium hits the
//! boundary at a steep angle, no refraction is possible — all light reflects.
//!
//! **Fresnel**: at normal incidence, `((η₁ - η₂) / (η₁ + η₂))²` reflects.
//! At grazing angles, nearly all light reflects regardless of IOR.
//!
//! This is a **delta material** — it scatters in a single determined direction,
//! not over a distribution. The integrator must skip MIS weighting and use the
//! sampled direction directly.

use crate::hittable::SurfaceInteraction;
use crate::vec3::{Color3, Vec3, reflect, refract};

use crate::material::fresnel_schlick;
use crate::material::{Bsdf, BsdfScatter, GpuMaterialBuffer, GpuMaterialNode, GpuMaterialType};
use crate::material::{GPU_NONE, PdfKind};
use crate::sampler::SampleDims;

/// Dielectric transmission/reflection controlled by refractive index.
#[derive(Clone)]
pub struct DielectricMaterial {
    /// Index of refraction (1.0 = air, 1.33 = water, 1.5 = glass, 2.42 = diamond).
    pub ior: f64,
    /// Optional tint color for colored glass. Pure white means no tint.
    pub tint: Color3,
    /// Precomputed Fresnel reflectance at normal incidence.
    pub r0: f64,
}

impl Bsdf for DielectricMaterial {
    /// Compute refraction ratio from the two media using Snell's Law.
    /// Then use Fresnel to decide between reflection and refraction.
    fn scatter(&self, wo: Vec3, si: &SurfaceInteraction, dims: SampleDims) -> Option<BsdfScatter> {
        // Determine the ratio of indices of refraction based on whether the ray is entering or exiting the material.
        let ri = if si.front_face() {
            1.0 / self.ior
        } else {
            self.ior
        };
        // Compute the cosine of the angle between the outgoing direction and the surface normal.
        let cos_theta = wo.dot(&si.shading_normal()).min(1.0);
        // Compute the sine of the angle using the identity sin²(θ) + cos²(θ) = 1.
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();

        // Use Fresnel's equations to determine the probability of reflection vs refraction.
        let direction = if ri * sin_theta > 1.0 || fresnel_schlick(cos_theta, self.r0) > dims.u {
            reflect(&-wo, &si.shading_normal())
        } else {
            refract(&-wo, &si.shading_normal(), ri)
        };
        // Return the chosen direction with unit attenuation (delta material — all energy goes one way).
        Some(BsdfScatter::Delta {
            wi: direction,
            f_cos: self.tint,
        })
    }

    /// Delta material — cannot evaluate at arbitrary directions. Returns zero.
    fn eval(&self, _wo: Vec3, _wi: Vec3, _si: &SurfaceInteraction) -> Color3 {
        Color3::ZERO
    }

    /// Delta material — probability of any specific direction is zero.
    fn pdf(&self, _wo: Vec3, _wi: Vec3, _si: &SurfaceInteraction) -> f64 {
        0.0
    }

    /// Delta material — no PDF kind for arbitrary directions.
    fn pdf_kind(&self, _wo: Vec3, _si: &SurfaceInteraction) -> Option<PdfKind> {
        None
    }

    /// Estimate the reflectance fraction for the coating layer. This is used in
    /// the integrator to determine how much light is reflected vs transmitted.
    fn reflectance_estimate(&self, wo: Vec3, si: &SurfaceInteraction) -> f64 {
        let cos_theta = wo.dot(&si.shading_normal()).abs();
        // Only the reflective fraction of the dielectric is returned to the
        // coating; transmitted light goes into the substrate and doesn't
        // contribute to the inter-reflection series.
        fresnel_schlick(cos_theta, self.r0)
    }

    fn is_delta(&self) -> bool {
        true
    }

    fn serialize_gpu(&self, buf: &mut GpuMaterialBuffer) -> u32 {
        let params = vec![self.ior];
        let param_offset = buf.params.len() as u32;
        buf.push_params(&params);
        buf.nodes.push(GpuMaterialNode {
            material_type: GpuMaterialType::Dielectric as u32,
            param_offset,
            child_a: GPU_NONE,
            child_b: GPU_NONE,
            texture_index: GPU_NONE,
        });
        buf.nodes.len() as u32 - 1
    }
}
