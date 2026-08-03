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

use crate::intersect::interaction::SurfaceInteraction;
use crate::material::gpu::{GPU_NONE, GpuSerializable};
use crate::material::{
    Bsdf, BsdfScatter, GpuMaterialBuffer, GpuMaterialNode, GpuMaterialType, Material,
};
use crate::math::vec3::{Color3, Direction3};
use crate::sampling::pdf::PdfKind;
use crate::texture::Texture;

/// Light emitting surface.
#[derive(Clone)]
pub struct DiffuseEmitterMaterial {
    /// Emission color (radiance) as a texture for spatial variation.
    pub emission: Arc<dyn Texture>,
}

impl DiffuseEmitterMaterial {
    /// Create an emissive material from a solid color.
    pub fn new(emission: Color3) -> Self {
        Self {
            emission: Arc::new(crate::texture::SolidColor::new(emission)),
        }
    }

    /// Create an emissive material from a texture.
    pub fn textured(tex: Arc<dyn Texture>) -> Self {
        Self { emission: tex }
    }
}

impl From<DiffuseEmitterMaterial> for Material {
    fn from(m: DiffuseEmitterMaterial) -> Self {
        Material::DiffuseEmitter(m)
    }
}

impl Bsdf for DiffuseEmitterMaterial {
    /// Pure emitter — no scattering, always returns `None`.
    fn scatter(
        &self,
        _wo: Direction3,
        _si: &SurfaceInteraction,
        _next_dim: &mut dyn FnMut() -> f32,
    ) -> Option<BsdfScatter> {
        None
    }

    /// No reflection — always zero.
    fn eval(&self, _wo: Direction3, _wi: Direction3, _si: &SurfaceInteraction) -> Color3 {
        Color3::ZERO
    }

    /// No scattering PDF — always zero.
    fn pdf(&self, _wo: Direction3, _wi: Direction3, _si: &SurfaceInteraction) -> f32 {
        0.0
    }

    /// No scattering PDF kind — always `None`.
    fn pdf_kind(&self, _wo: Direction3, _si: &SurfaceInteraction) -> Option<PdfKind> {
        None
    }

    /// Returns the emission color if the hit is on the front face, zero otherwise.
    fn emitted(&self, _wo: Direction3, si: &SurfaceInteraction) -> Color3 {
        if si.front_face() {
            self.emission.value(&si.texture_coords())
        } else {
            Color3::ZERO
        }
    }

    /// No reflection — always zero.
    fn reflectance_estimate(&self, _wo: Direction3, _si: &SurfaceInteraction) -> f32 {
        0.0
    }
}

impl GpuSerializable for DiffuseEmitterMaterial {
    fn serialize_gpu(&self, buf: &mut GpuMaterialBuffer) -> u32 {
        let (r, g, b, texture_index) = buf.gpu_color(&self.emission);
        let params = vec![r, g, b];
        let param_offset = buf.params.len() as u32;
        buf.push_params(&params);
        buf.nodes.push(GpuMaterialNode {
            material_type: GpuMaterialType::DiffuseEmitter as u32,
            param_offset,
            child_a: GPU_NONE,
            child_b: GPU_NONE,
            texture_index,
        });
        buf.nodes.len() as u32 - 1
    }
}
