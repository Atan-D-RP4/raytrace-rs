//! Emissive (area light) material.
//!
//! Emits radiance from its surface, acting as a physically-based area light.
//! Only emits from the front face — the back face is dark (the emitter has
//! a backing, like a real light panel).
//!
//! This material does not scatter incoming light. The integrator calls
//! `emitted()` when a ray hits this surface and adds the result to the
//! accumulated radiance. For importance sampling, the scene builder collects
//! all emissive materials to build a light PDF.

use std::sync::Arc;

use crate::hittable::HitRecord;
use crate::sampler::Sampler;
use crate::texture::Texture;
use crate::vec3::{Color3, Vec3};

use super::GPU_NONE;
use super::{Bsdf, BsdfSample, GpuMaterialBuffer, GpuMaterialNode, GpuMaterialType};

/// Light emitting surface.
#[derive(Clone)]
pub struct DiffuseLightMaterial {
    /// Emission color (radiance). Multiplied by texture if one is set.
    pub emit: Color3,
    /// Optional texture for spatial emission variation. CPU-only.
    pub tex: Option<Arc<dyn Texture>>,
}

impl Bsdf for DiffuseLightMaterial {
    /// Pure emitter — no scattering, always returns `None`.
    fn sample(
        &self,
        _wo: Vec3,
        _hit: &HitRecord,
        _sampler: &mut dyn Sampler,
    ) -> Option<BsdfSample> {
        None
    }

    /// No reflection — always zero.
    fn eval(&self, _wo: Vec3, _wi: Vec3, _hit: &HitRecord) -> Color3 {
        Color3::from(0., 0., 0.)
    }

    /// No scattering PDF — always zero.
    fn pdf(&self, _wo: Vec3, _wi: Vec3, _hit: &HitRecord) -> f64 {
        0.0
    }

    /// Returns the emission color if the hit is on the front face, zero otherwise.
    fn emitted(&self, hit: &HitRecord) -> Color3 {
        if hit.front_face {
            self.tex
                .as_ref()
                .map(|t| t.value(&hit.texture_coords()))
                .unwrap_or(self.emit)
        } else {
            Color3::from(0., 0., 0.)
        }
    }

    fn gpu_node(&self, buf: &mut GpuMaterialBuffer) -> Option<u32> {
        let params = vec![self.emit.x, self.emit.y, self.emit.z];
        let param_offset = buf.params.len() as u32;
        buf.push_params(&params);
        buf.nodes.push(GpuMaterialNode {
            material_type: GpuMaterialType::DiffuseLight as u32,
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
        let params = vec![self.emit.x, self.emit.y, self.emit.z];
        let param_offset = buf.params.len() as u32;
        buf.push_params(&params);
        buf.nodes.push(GpuMaterialNode {
            material_type: GpuMaterialType::DiffuseLight as u32,
            param_offset,
            child_a: GPU_NONE,
            child_b: GPU_NONE,
            texture_index: GPU_NONE,
        });
        buf.nodes.len() as u32 - 1
    }
}
