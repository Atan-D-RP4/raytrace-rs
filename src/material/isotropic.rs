//! Isotropic volumetric scattering.
//!
//! Models participating media like smoke, fog, or clouds. Light entering the
//! medium scatters in a uniformly random direction — no preferred orientation
//! (hence "isotropic"). The scattering PDF is uniform over the full sphere:
//! `1 / (4π)`.
//!
//! The integrator handles volumes by sampling a distance along the ray (via
//! the medium's density function), then calling `sample()` to get the
//! attenuation color. The integrator itself generates the new scattered
//! direction — `sample()` returns `Vec3::ZERO` as a placeholder.

use std::f32::consts::PI;
use std::sync::Arc;

use crate::hittable::SurfaceInteraction;
use crate::material::gpu::{GPU_NONE, GpuSerializable};
use crate::material::{
    Bsdf, BsdfScatter, GpuMaterialBuffer, GpuMaterialNode, GpuMaterialType, MAX_BSDF_STRATS,
    PdfKind,
};
use crate::texture::Texture;
use crate::vec3::{Color3, Direction3};

use crate::material::Material;

/// Isotropic scattering medium (volumes).
#[derive(Clone)]
pub struct IsotropicMaterial {
    /// Base albedo color. Used as fallback when `tex` is set, and for GPU serialization.
    pub albedo: Color3,
    /// Optional texture for spatial albedo variation. CPU-only.
    pub tex: Option<Arc<dyn Texture>>,
}

impl Bsdf for IsotropicMaterial {
    /// Isotropic scattering is non-directional, so the BSDF does not depend on the outgoing
    /// direction `wo`. The integrator generates the actual scattered direction.
    fn scatter(
        &self,
        _wo: Direction3,
        _si: &SurfaceInteraction,
        _next_dim: &mut dyn FnMut() -> f32,
    ) -> Option<BsdfScatter> {
        let mut pk = [None; MAX_BSDF_STRATS];
        pk[0] = Some(PdfKind::UniformSphere);
        Some(BsdfScatter::NonDelta { pdf_kinds: pk })
    }

    /// Isotropic phase function: `albedo / (4π)`. Returns the attenuation
    /// regardless of direction — every scattering direction is equally likely.
    fn eval(&self, _wo: Direction3, _wi: Direction3, si: &SurfaceInteraction) -> Color3 {
        let attenuation = self
            .tex
            .as_ref()
            .map(|t| t.value(&si.texture_coords()))
            .unwrap_or(self.albedo);
        attenuation / (4.0 * PI)
    }

    /// Uniform sphere PDF: `1 / (4π)`.
    fn pdf(&self, _wo: Direction3, _wi: Direction3, _si: &SurfaceInteraction) -> f32 {
        1.0 / (4.0 * PI)
    }

    /// Isotropic scattering is non-directional, so the reflectance estimate is simply `1.0`.
    fn reflectance_estimate(&self, _wo: Direction3, _si: &SurfaceInteraction) -> f32 {
        1.0
    }
}

impl IsotropicMaterial {
    /// Create an isotropic material from a solid color.
    pub fn new(albedo: Color3) -> Self {
        Self { albedo, tex: None }
    }

    /// Create an isotropic material from a texture.
    pub fn textured(tex: Arc<dyn Texture>) -> Self {
        Self {
            albedo: Color3::ZERO,
            tex: Some(tex),
        }
    }
}

impl From<IsotropicMaterial> for Material {
    fn from(m: IsotropicMaterial) -> Self {
        Material::Isotropic(m)
    }
}

impl GpuSerializable for IsotropicMaterial {
    fn serialize_gpu(&self, buf: &mut GpuMaterialBuffer) -> u32 {
        let params = vec![self.albedo.x(), self.albedo.y(), self.albedo.z()];
        let param_offset = buf.params.len() as u32;
        buf.push_params(&params);
        buf.nodes.push(GpuMaterialNode {
            material_type: GpuMaterialType::Isotropic as u32,
            param_offset,
            child_a: GPU_NONE,
            child_b: GPU_NONE,
            texture_index: GPU_NONE,
        });
        buf.nodes.len() as u32 - 1
    }
}
