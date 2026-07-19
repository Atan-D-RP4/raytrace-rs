//! GGX microfacet dielectric BSDF (glossy reflection).
//!
//! Similar to the metal BRDF but models dielectric surfaces (plastic, coated
//! wood, ceramic) rather than conductors. The key difference is that the
//! Fresnel term uses the material's IOR to determine the reflection/transmission
//! ratio, and the result is multiplied by the surface albedo.
//!
//! BSDF (same Cook-Torrance model as metal):
//!
//! ```text
//! f(ωo, ωi) = albedo · F · D · G / (4 · cos_o · cos_i)
//! ```
//!
//! `roughness` controls the GGX distribution width (0 = mirror, 1 = fully rough).
//! `ior` sets the index of refraction for the Fresnel term — higher IOR means
//! more reflection at normal incidence.

use std::sync::Arc;

use crate::hittable::SurfaceInteraction;
use crate::material::gpu::GpuSerializable;
use crate::material::{
    Bsdf, BsdfScatter, GPU_NONE, GpuMaterialBuffer, GpuMaterialNode, GpuMaterialType,
    MAX_BSDF_STRATS, PdfKind, fresnel_schlick, geometry_schlick_ggx, ggx_d, ggx_sample_h,
};
use crate::onb::Onb;
use crate::texture::Texture;
use crate::vec3::{Color3, Direction3};

use super::MIRROR_THRESHOLD;

use crate::material::Material;

/// Glossy microfacet BSDF (GGX).
#[derive(Clone)]
pub struct GlossyMaterial {
    /// Base reflectance color. Multiplied by the Cook-Torrance BRDF value.
    pub albedo: Color3,
    /// Optional texture for spatial albedo variation. CPU-only; GPU serialization
    /// falls back to `albedo`.
    pub tex: Option<Arc<dyn Texture>>,
    /// Surface smoothness: 0 = mirror, 1 = fully rough. GGX alpha = `roughness²`.
    pub roughness: f32,
    /// Index of refraction for the Fresnel term (1.5 = glass, 1.45 = typical plastic).
    pub ior: f32,
    /// Precomputed Fresnel reflectance at normal incidence.
    pub r0: f32,
}

impl Bsdf for GlossyMaterial {
    /// Importance-sample the GGX NDF: draw half-vector H, reflect `wo` about it.
    /// Returns `None` if the reflected direction is below the surface.
    ///
    /// When `roughness` is effectively zero (below 0.01), the surface is a
    /// near-mirror — returns `BsdfScatter::Delta` so the integrator skips
    /// the mixture PDF and uses the reflected direction directly.
    fn scatter(
        &self,
        wo: Direction3,
        si: &SurfaceInteraction,
        next_dim: &mut dyn FnMut() -> f32,
    ) -> Option<BsdfScatter> {
        // Near-mirror: delta path bypasses the mixture PDF entirely.
        if self.is_delta() {
            let wi = -wo.reflect(si.shading_normal().into_inner());
            if wi.dot(si.shading_normal().into_inner()) <= 0.0 {
                return None;
            }
            let cos_o = wo.dot(si.shading_normal().into_inner()).max(0.0);
            let f = fresnel_schlick(cos_o, self.r0);
            let albedo_ = self
                .tex
                .as_ref()
                .map(|t| t.value(&si.texture_coords()))
                .unwrap_or(self.albedo);
            return Some(BsdfScatter::Delta {
                wi,
                f_cos: albedo_ * f,
                eta: None,
            });
        }

        let alpha = self.ggx_alpha()?;
        // Sample H from GGX NDF.
        let u = next_dim();
        let v = next_dim();
        let h_local = ggx_sample_h(alpha, u, v);
        let onb = Onb::build_from_normal(si.shading_normal());
        let h_world = onb.local_to_world(h_local);

        let wi = -wo.reflect(h_world.into_inner());

        if wi.dot(si.shading_normal().into_inner()) <= 0.0 {
            return None;
        }

        let mut pk = [None; MAX_BSDF_STRATS];
        pk[0] = Some(PdfKind::Ggx {
            wo,
            normal: si.shading_normal(),
            alpha,
        });
        Some(BsdfScatter::NonDelta { pdf_kinds: pk })
    }

    /// Cook-Torrance BRDF: `albedo · F · D · G / (4 · cos_o · cos_i)`.
    fn eval(&self, wo: Direction3, wi: Direction3, si: &SurfaceInteraction) -> Color3 {
        if self.is_delta() {
            return Color3::ZERO;
        }
        let albedo = self
            .tex
            .as_ref()
            .map(|t| t.value(&si.texture_coords()))
            .unwrap_or(self.albedo);
        let alpha = self.ggx_alpha().unwrap_or(0.001);

        let h = (wo + wi).normalize();
        let cos_h_n = h.dot(si.shading_normal().into_inner()).max(0.0);
        let cos_h_o = wo.dot(h.into_inner()).max(0.0);

        let cos_o = wo.dot(si.shading_normal().into_inner()).max(0.0);
        let cos_i = wi.dot(si.shading_normal().into_inner()).max(0.0);

        if cos_h_o <= 0.0 || cos_o <= 0.0 || cos_i <= 0.0 {
            return Color3::new(0., 0., 0.);
        }

        let d = ggx_d(cos_h_n, alpha);
        let f = fresnel_schlick(cos_h_o, self.r0);
        let g = geometry_schlick_ggx(cos_o, self.roughness)
            * geometry_schlick_ggx(cos_i, self.roughness);

        albedo * f * d * g / (4.0 * cos_o)
    }

    /// GGX NDF sampling PDF: `D(H) · cos(H·N) / (4 · cos(H·O))`.
    fn pdf(&self, wo: Direction3, wi: Direction3, si: &SurfaceInteraction) -> f32 {
        if self.is_delta() {
            return 0.0;
        }

        let alpha = self.ggx_alpha().unwrap_or(0.001);

        let h = (wo + wi).normalize();
        let cos_h_n = h.dot(si.shading_normal().into_inner()).max(0.0);
        let cos_h_o = wo.dot(h.into_inner()).max(0.0);

        if cos_h_o <= 0.0 {
            return 0.0;
        }

        ggx_d(cos_h_n, alpha) * cos_h_n / (4.0 * cos_h_o)
    }

    /// Returns the PDF kind for the GGX distribution, which is used in mixture sampling.
    /// Returns `None` for near-mirror materials so the integrator skips GGX PDF strategy.
    fn pdf_kind(&self, wo: Direction3, si: &SurfaceInteraction) -> Option<PdfKind> {
        if self.is_delta() {
            None
        } else {
            let alpha = self.ggx_alpha().unwrap_or(0.001);
            Some(PdfKind::Ggx {
                wo,
                normal: si.shading_normal(),
                alpha,
            })
        }
    }

    fn reflectance_estimate(&self, wo: Direction3, si: &SurfaceInteraction) -> f32 {
        let albedo = self
            .tex
            .as_ref()
            .map(|t| t.value(&si.texture_coords()))
            .unwrap_or(self.albedo);
        let albedo_avg = (albedo.x() + albedo.y() + albedo.z()) / 3.0;
        let cos_theta = wo.dot(si.shading_normal().into_inner()).abs();
        let f = fresnel_schlick(cos_theta, self.r0);
        // Base color × Fresnel gives the specular reflectance; roughness adds
        // a small boost from multiple scattering making the surface appear brighter.
        let roughness_boost = self.roughness * 0.25;
        (albedo_avg * f + roughness_boost).min(1.0)
    }

    fn is_delta(&self) -> bool {
        self.roughness < MIRROR_THRESHOLD
    }

    fn ggx_alpha(&self) -> Option<f32> {
        if self.is_delta() {
            None
        } else {
            Some((self.roughness * self.roughness).clamp(0.001, 1.0))
        }
    }
}

impl GlossyMaterial {
    /// Create a glossy material.
    pub fn new(albedo: Color3, roughness: f32, ior: f32) -> Self {
        Self {
            albedo,
            tex: None,
            roughness,
            ior,
            r0: super::fresnel_r0(ior),
        }
    }

    /// Create a glossy material with a texture.
    pub fn textured(tex: Arc<dyn Texture>, roughness: f32, ior: f32) -> Self {
        Self {
            albedo: Color3::ZERO,
            tex: Some(tex),
            roughness,
            ior,
            r0: super::fresnel_r0(ior),
        }
    }
}

impl From<GlossyMaterial> for Material {
    fn from(m: GlossyMaterial) -> Self {
        Material::Glossy(m)
    }
}

impl GpuSerializable for GlossyMaterial {
    fn serialize_gpu(&self, buf: &mut GpuMaterialBuffer) -> u32 {
        let params = vec![
            self.albedo.x(),
            self.albedo.y(),
            self.albedo.z(),
            self.roughness,
            self.ior,
        ];
        let param_offset = buf.params.len() as u32;
        buf.push_params(&params);
        buf.nodes.push(GpuMaterialNode {
            material_type: GpuMaterialType::Glossy as u32,
            param_offset,
            child_a: GPU_NONE,
            child_b: GPU_NONE,
            texture_index: GPU_NONE,
        });
        buf.nodes.len() as u32 - 1
    }
}
