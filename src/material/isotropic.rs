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

use std::f64::consts::PI;
use std::sync::Arc;

use crate::hittable::HitRecord;
use crate::sampler::Sampler;
use crate::texture::Texture;
use crate::vec3::{Color3, Vec3};

use super::GPU_NONE;
use super::{Bsdf, BsdfSample, GpuMaterialBuffer, GpuMaterialNode, GpuMaterialType, PdfKind};

/// Isotropic scattering medium (volumes).
#[derive(Clone)]
pub struct IsotropicMaterial {
    /// Base albedo color. Used as fallback when `tex` is set, and for GPU serialization.
    pub albedo: Color3,
    /// Optional texture for spatial albedo variation. CPU-only.
    pub tex: Option<Arc<dyn Texture>>,
}

impl Bsdf for IsotropicMaterial {
    /// Returns the attenuation color (texture or solid albedo) with `Vec3::ZERO`
    /// as a placeholder direction. The integrator generates the actual scattered
    /// direction.
    fn sample(&self, _wo: Vec3, hit: &HitRecord, _sampler: &mut dyn Sampler) -> Option<BsdfSample> {
        let attenuation = self
            .tex
            .as_ref()
            .map(|t| t.value(&hit.texture_coords()))
            .unwrap_or(self.albedo);
        Some(BsdfSample {
            wi: Vec3::ZERO,
            f_cos: attenuation,
            pdf: 1.0,
            pdf_kind: PdfKind::UniformSphere,
        })
    }

    /// Isotropic phase function: `albedo / (4π)`. Returns the attenuation
    /// regardless of direction — every scattering direction is equally likely.
    fn eval(&self, _wo: Vec3, _wi: Vec3, hit: &HitRecord) -> Color3 {
        let attenuation = self
            .tex
            .as_ref()
            .map(|t| t.value(&hit.texture_coords()))
            .unwrap_or(self.albedo);
        attenuation / (4.0 * PI)
    }

    /// Uniform sphere PDF: `1 / (4π)`.
    fn pdf(&self, _wo: Vec3, _wi: Vec3, _hit: &HitRecord) -> f64 {
        1.0 / (4.0 * PI)
    }

    fn gpu_node(&self, buf: &mut GpuMaterialBuffer) -> Option<u32> {
        let params = vec![self.albedo.x, self.albedo.y, self.albedo.z];
        let param_offset = buf.params.len() as u32;
        buf.push_params(&params);
        buf.nodes.push(GpuMaterialNode {
            material_type: GpuMaterialType::Isotropic as u32,
            param_offset,
            child_a: GPU_NONE,
            child_b: GPU_NONE,
            texture_index: GPU_NONE,
        });
        Some(buf.nodes.len() as u32 - 1)
    }

    fn clone_box(&self) -> Box<dyn Bsdf> {
        Box::new(self.clone())
    }

    fn serialize_gpu(&self, buf: &mut GpuMaterialBuffer) -> u32 {
        let params = vec![self.albedo.x, self.albedo.y, self.albedo.z];
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
