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

use std::sync::Arc;

use crate::hittable::SurfaceInteraction;
use crate::material::fresnel_schlick;
use crate::material::gpu::{GPU_NONE, GpuSerializable};
use crate::material::{Bsdf, BsdfScatter, GpuMaterialBuffer, GpuMaterialNode, GpuMaterialType};
use crate::pdf::PdfKind;
use crate::texture::{SolidColor, Texture};
use crate::vec3::{Color3, Direction3};

use crate::material::Material;

/// Dielectric transmission/reflection controlled by refractive index.
#[derive(Clone)]
pub struct DielectricMaterial {
    /// Index of refraction (1.0 = air, 1.33 = water, 1.5 = glass, 2.42 = diamond).
    pub ior: f32,
    /// Tint color for colored glass, sampled as a texture. Pure white means no tint.
    pub tint: Arc<dyn Texture>,
}

impl Bsdf for DielectricMaterial {
    /// Compute refraction ratio from the two media using Snell's Law.
    /// Then use Fresnel to decide between reflection and refraction.
    fn scatter(
        &self,
        wo: Direction3,
        si: &SurfaceInteraction,
        next_dim: &mut dyn FnMut() -> f32,
    ) -> Option<BsdfScatter> {
        // Determine the ratio of indices of refraction based on whether the ray is entering or exiting the material.
        let ri = if si.front_face() {
            1.0 / self.ior
        } else {
            self.ior
        };

        // Compute the cosine of the angle between the outgoing direction and the surface normal.
        let cos_theta = wo.dot(si.shading_normal().into_inner()).min(1.0);
        // Compute the sine of the angle using the identity sin²(θ) + cos²(θ) = 1.
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();

        // Use Fresnel's equations to determine the probability of reflection vs refraction.
        let u = next_dim();
        let direction = if ri * sin_theta > 1.0
            || fresnel_schlick(cos_theta, super::fresnel_r0(self.ior)) > u
        {
            (-wo).reflect(si.shading_normal().into_inner())
        } else {
            (-wo).refract(si.shading_normal().into_inner(), ri)
        };

        // Return the chosen direction with unit attenuation (delta material — all energy goes one way).
        let tint = self.tint.value(&si.texture_coords());
        Some(BsdfScatter::Delta {
            wi: direction,
            f_cos: tint,
            eta: Some(ri),
        })
    }

    /// Delta material — cannot evaluate at arbitrary directions. Returns zero.
    fn eval(&self, _wo: Direction3, _wi: Direction3, _si: &SurfaceInteraction) -> Color3 {
        Color3::ZERO
    }

    /// Delta material — probability of any specific direction is zero.
    fn pdf(&self, _wo: Direction3, _wi: Direction3, _si: &SurfaceInteraction) -> f32 {
        0.0
    }

    /// Delta material — no PDF kind for arbitrary directions.
    fn pdf_kind(&self, _wo: Direction3, _si: &SurfaceInteraction) -> Option<PdfKind> {
        None
    }

    /// Estimate the reflectance fraction for the coating layer. This is used in
    /// the integrator to determine how much light is reflected vs transmitted.
    fn reflectance_estimate(&self, wo: Direction3, si: &SurfaceInteraction) -> f32 {
        let cos_theta = wo.dot(si.shading_normal().into_inner()).abs();
        // Only the reflective fraction of the dielectric is returned to the
        // coating; transmitted light goes into the substrate and doesn't
        // contribute to the inter-reflection series.
        fresnel_schlick(cos_theta, super::fresnel_r0(self.ior))
    }

    /// Delta material
    fn is_delta(&self) -> bool {
        true
    }
}

impl DielectricMaterial {
    /// Create a dielectric (clear) with the given IOR. Tint defaults to white.
    pub fn new(ior: f32) -> Self {
        Self {
            ior,
            tint: Arc::new(SolidColor::new(Color3::ONE)),
        }
    }

    /// Create a tinted dielectric (colored glass).
    pub fn tinted(ior: f32, tint: Color3) -> Self {
        Self {
            ior,
            tint: Arc::new(SolidColor::new(tint)),
        }
    }

    /// Create a dielectric with a textured tint (spatially varying colored glass).
    pub fn textured(ior: f32, tint: Arc<dyn Texture>) -> Self {
        Self { ior, tint }
    }
}

impl From<DielectricMaterial> for Material {
    fn from(m: DielectricMaterial) -> Self {
        Material::Dielectric(m)
    }
}

impl GpuSerializable for DielectricMaterial {
    fn serialize_gpu(&self, buf: &mut GpuMaterialBuffer) -> u32 {
        let (r, g, b, texture_index) = buf.gpu_color(&self.tint);
        let params = vec![r, g, b, self.ior];
        let param_offset = buf.params.len() as u32;
        buf.push_params(&params);
        buf.nodes.push(GpuMaterialNode {
            material_type: GpuMaterialType::Dielectric as u32,
            param_offset,
            child_a: GPU_NONE,
            child_b: GPU_NONE,
            texture_index,
        });
        buf.nodes.len() as u32 - 1
    }
}
