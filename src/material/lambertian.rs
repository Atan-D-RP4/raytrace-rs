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

use std::f64::consts::PI;
use std::sync::Arc;

use crate::hittable::SurfaceInteraction;
use crate::texture::Texture;
use crate::vec3::{Color3, Vec3};

use super::GPU_NONE;
use super::{Bsdf, BsdfSample, GpuMaterialBuffer, GpuMaterialNode, GpuMaterialType, PdfKind};

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
    fn sample(
        &self,
        _wo: Vec3,
        si: &SurfaceInteraction,
        _u: f64,
        _v: f64,
        _w: f64,
        _x: f64,
    ) -> Option<BsdfSample> {
        let attenuation = self
            .tex
            .as_ref()
            .map(|t| t.value(&si.texture_coords()))
            .unwrap_or(self.albedo);
        Some(BsdfSample {
            wi: Vec3::ZERO,
            f_cos: attenuation,
            pdf: 1.0,
            pdf_kind: PdfKind::Cosine {
                normal: si.shading_normal(),
            },
        })
    }

    /// Lambertian BRDF: `albedo · cos(θ) / π`. Returns zero if `wi` is below the surface.
    fn eval(&self, _wo: Vec3, wi: Vec3, si: &SurfaceInteraction) -> Color3 {
        let attenuation = self
            .tex
            .as_ref()
            .map(|t| t.value(&si.texture_coords()))
            .unwrap_or(self.albedo);
        let cos_theta = si.shading_normal().dot(&wi);
        if cos_theta < 0.0 {
            Color3::from(0., 0., 0.)
        } else {
            attenuation * cos_theta / PI
        }
    }

    /// Cosine-weighted hemisphere PDF: `cos(θ) / π`. Returns zero if `wi` is below the surface.
    fn pdf(&self, _wo: Vec3, wi: Vec3, si: &SurfaceInteraction) -> f64 {
        let cos_theta = si.shading_normal().dot(&wi);
        if cos_theta < 0.0 { 0.0 } else { cos_theta / PI }
    }

    fn gpu_node(&self, buf: &mut GpuMaterialBuffer) -> Option<u32> {
        let params = vec![self.albedo.x, self.albedo.y, self.albedo.z];
        let param_offset = buf.params.len() as u32;
        buf.push_params(&params);
        buf.nodes.push(GpuMaterialNode {
            material_type: GpuMaterialType::Lambertian as u32,
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
            material_type: GpuMaterialType::Lambertian as u32,
            param_offset,
            child_a: GPU_NONE,
            child_b: GPU_NONE,
            texture_index: GPU_NONE,
        });
        buf.nodes.len() as u32 - 1
    }
}
