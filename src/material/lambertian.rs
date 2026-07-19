//! Lambertian (ideal diffuse) reflectance.
//!
//! Models matte surfaces like paper, chalk, or unpolished wood. A Lambertian
//! surface reflects incoming light uniformly across the hemisphere — it looks
//! equally bright from every viewing angle.
//!
//! BRDF: `albedo / π` — the `1/π` normalizer ensures energy conservation
//! (total reflected energy equals incoming × albedo).
//!
//! Sampling: the integrator draws directions from a cosine-weighted hemisphere
//! PDF, so `sample()` returns `Vec3::ZERO` as a placeholder — the actual
//! direction comes from the PDF, not the material.

use std::f32::consts::PI;
use std::sync::Arc;

use crate::hittable::SurfaceInteraction;
use crate::material::GPU_NONE;
use crate::material::gpu::GpuSerializable;
use crate::material::{
    Bsdf, BsdfScatter, GpuMaterialBuffer, GpuMaterialNode, GpuMaterialType, MAX_BSDF_STRATS,
    PdfKind,
};
use crate::texture::Texture;
use crate::vec3::{Color3, Direction3};

use crate::material::Material;

/// Diffuse (Lambertian) surface.
#[derive(Clone)]
pub struct LambertianMaterial {
    /// Fraction of light reflected at each wavelength. Used as fallback when `tex` is set,
    /// and as the GPU serialization color (textures are CPU-only).
    pub albedo: Color3,
    /// Optional texture for spatial albedo variation. When set, `value()` is
    /// evaluated at the hit point instead of using `albedo`.
    pub tex: Option<Arc<dyn Texture>>,
}

impl Bsdf for LambertianMaterial {
    /// Returns `Vec3::ZERO` as a direction placeholder. The integrator samples
    /// the actual direction from the cosine-weighted hemisphere PDF indicated by
    /// `pdf_kind`. The `f_cos` field carries the albedo (texture or solid color).
    fn scatter(
        &self,
        _wo: Direction3,
        si: &SurfaceInteraction,
        _next_dim: &mut dyn FnMut() -> f32,
    ) -> Option<BsdfScatter> {
        let mut pk = [None; MAX_BSDF_STRATS];
        pk[0] = Some(PdfKind::Cosine {
            normal: si.shading_normal(),
        });
        Some(BsdfScatter::NonDelta { pdf_kinds: pk })
    }

    /// Lambertian BRDF: `albedo · cos(θ) / π`. Returns zero if `wi` is below the surface.
    fn eval(&self, _wo: Direction3, wi: Direction3, si: &SurfaceInteraction) -> Color3 {
        let attenuation = self
            .tex
            .as_ref()
            .map(|t| t.value(&si.texture_coords()))
            .unwrap_or(self.albedo);
        let cos_theta = si.shading_normal().into_inner().dot(wi.into_inner());
        if cos_theta < 0.0 {
            Color3::new(0., 0., 0.)
        } else {
            attenuation * cos_theta / PI
        }
    }

    /// Cosine-weighted hemisphere PDF: `cos(θ) / π`. Returns zero if `wi` is below the surface.
    fn pdf(&self, _wo: Direction3, wi: Direction3, si: &SurfaceInteraction) -> f32 {
        let cos_theta = si.shading_normal().into_inner().dot(wi.into_inner());
        if cos_theta < 0.0 { 0.0 } else { cos_theta / PI }
    }

    /// Returns `PdfKind::Cosine` for the cosine-weighted hemisphere PDF.
    fn pdf_kind(&self, _wo: Direction3, si: &SurfaceInteraction) -> Option<PdfKind> {
        Some(PdfKind::Cosine {
            normal: si.shading_normal(),
        })
    }

    fn reflectance_estimate(&self, _wo: Direction3, si: &SurfaceInteraction) -> f32 {
        let albedo = self
            .tex
            .as_ref()
            .map(|t| t.value(&si.texture_coords()))
            .unwrap_or(self.albedo);
        // Lambertian directional-hemispherical reflectance = albedo (exact:
        // ∫ (albedo/π) * cos θ dω = albedo). Average across RGB channels.
        (albedo.x() + albedo.y() + albedo.z()) / 3.0
    }
}

impl LambertianMaterial {
    /// Create a lambertian material from a solid color.
    pub fn new(albedo: Color3) -> Self {
        Self { albedo, tex: None }
    }

    /// Create a lambertian material with a texture. `albedo` is used as
    /// fallback when tex is None (e.g., GPU serialization).
    pub fn with_texture(albedo: Color3, tex: Arc<dyn Texture>) -> Self {
        Self {
            albedo,
            tex: Some(tex),
        }
    }
}

impl From<LambertianMaterial> for Material {
    fn from(m: LambertianMaterial) -> Self {
        Material::Lambertian(m)
    }
}

impl GpuSerializable for LambertianMaterial {
    fn serialize_gpu(&self, buf: &mut GpuMaterialBuffer) -> u32 {
        let params = vec![self.albedo.x(), self.albedo.y(), self.albedo.z()];
        let param_offset = buf.params.len() as u32;
        buf.push_params(&params);
        buf.nodes.push(GpuMaterialNode {
            material_type: GpuMaterialType::Lambertian as u32,
            param_offset,
            child_a: GPU_NONE,
            child_b: GPU_NONE,
            texture_index: GPU_NONE,
        });
        buf.nodes.len() as u32 - 1
    }
}
