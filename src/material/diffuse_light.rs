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

use crate::hittable::SurfaceInteraction;
use crate::texture::Texture;
use crate::vec3::{Color3, Vec3};

use crate::material::{Bsdf, BsdfScatter, GpuMaterialBuffer, GpuMaterialNode, GpuMaterialType};
use crate::material::{PdfKind, GPU_NONE};

use super::gpu::GpuSerializable;

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
    fn scatter(
        &self,
        _wo: Vec3,
        _si: &SurfaceInteraction,
        _next_dim: &mut dyn FnMut() -> f64,
    ) -> Option<BsdfScatter> {
        None
    }

    /// No reflection — always zero.
    fn eval(&self, _wo: Vec3, _wi: Vec3, _si: &SurfaceInteraction) -> Color3 {
        Color3::new(0., 0., 0.)
    }

    /// No scattering PDF — always zero.
    fn pdf(&self, _wo: Vec3, _wi: Vec3, _si: &SurfaceInteraction) -> f64 {
        0.0
    }

    /// No scattering PDF kind — always `None`.
    fn pdf_kind(&self, _wo: Vec3, _si: &SurfaceInteraction) -> Option<PdfKind> {
        None
    }

    /// Returns the emission color if the hit is on the front face, zero otherwise.
    fn emitted(&self, _wo: Vec3, si: &SurfaceInteraction) -> Color3 {
        if si.front_face() {
            self.tex
                .as_ref()
                .map(|t| t.value(&si.texture_coords()))
                .unwrap_or(self.emit)
        } else {
            Color3::new(0., 0., 0.)
        }
    }

    /// No reflection — always zero.
    fn reflectance_estimate(&self, _wo: Vec3, _si: &SurfaceInteraction) -> f64 {
        0.0
    }
}

impl GpuSerializable for DiffuseLightMaterial {
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
