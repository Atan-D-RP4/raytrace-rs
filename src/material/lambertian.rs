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
use crate::texture::{SolidColor, Texture};
use crate::vec3::{Color3, Direction3};

use crate::material::Material;

/// Diffuse (Lambertian) surface.
#[derive(Clone)]
pub struct LambertianMaterial {
    /// Fraction of light reflected at each wavelength as a texture for spatial variation.
    pub albedo: Arc<dyn Texture>,
}

impl LambertianMaterial {
    /// Create a lambertian material from a solid color.
    pub fn new(albedo: Color3) -> Self {
        Self {
            albedo: Arc::new(SolidColor::new(albedo)),
        }
    }

    /// Create a lambertian material from a texture.
    pub fn textured(albedo: Arc<dyn Texture>) -> Self {
        Self { albedo }
    }
}

impl From<LambertianMaterial> for Material {
    fn from(m: LambertianMaterial) -> Self {
        Material::Lambertian(m)
    }
}

impl Bsdf for LambertianMaterial {
    /// The integrator samples the actual direction from the cosine-weighted hemisphere PDF
    /// indicated by `pdf_kind`. The `f_cos` field carries the albedo (texture or solid color).
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
        let attenuation = self.albedo.value(&si.texture_coords());
        let cos_theta = si.shading_normal().dot(wi.into_inner());
        if cos_theta < 0.0 {
            Color3::ZERO
        } else {
            attenuation * cos_theta / PI
        }
    }

    /// Cosine-weighted hemisphere PDF: `cos(θ) / π`. Returns zero if `wi` is below the surface.
    fn pdf(&self, _wo: Direction3, wi: Direction3, si: &SurfaceInteraction) -> f32 {
        let cos_theta = si.shading_normal().dot(wi.into_inner());
        if cos_theta < 0.0 { 0.0 } else { cos_theta / PI }
    }

    /// Returns `PdfKind::Cosine` for the cosine-weighted hemisphere PDF.
    fn pdf_kind(&self, _wo: Direction3, si: &SurfaceInteraction) -> Option<PdfKind> {
        Some(PdfKind::Cosine {
            normal: si.shading_normal(),
        })
    }

    fn reflectance_estimate(&self, _wo: Direction3, si: &SurfaceInteraction) -> f32 {
        let albedo = self.albedo.value(&si.texture_coords());
        // Lambertian directional-hemispherical reflectance = albedo (exact:
        // ∫ (albedo/π) * cos θ dω = albedo). Average across RGB channels.
        albedo.into_inner().element_sum() / 3.0
    }
}

impl GpuSerializable for LambertianMaterial {
    fn serialize_gpu(&self, buf: &mut GpuMaterialBuffer) -> u32 {
        let (r, g, b, texture_index) = buf.gpu_color(&self.albedo);
        let params = vec![r, g, b];
        let param_offset = buf.params.len() as u32;
        buf.push_params(&params);
        buf.nodes.push(GpuMaterialNode {
            material_type: GpuMaterialType::Lambertian as u32,
            param_offset,
            child_a: GPU_NONE,
            child_b: GPU_NONE,
            texture_index,
        });
        buf.nodes.len() as u32 - 1
    }
}
