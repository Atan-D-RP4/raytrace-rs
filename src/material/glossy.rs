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
    MAX_BSDF_STRATS, PdfKind, fresnel_r0, fresnel_schlick, geometry_schlick_ggx, ggx_d,
    ggx_sample_h,
};
use crate::onb::Onb;
use crate::texture::{SolidColor, Texture};
use crate::vec3::{Color3, Direction3};

use super::MIRROR_THRESHOLD;

use crate::material::Material;

/// Glossy microfacet BSDF (GGX).
#[derive(Clone)]
pub struct GlossyMaterial {
    /// Base reflectance color (dielectric tint). Multiplied by the Cook-Torrance BRDF value.
    pub albedo: Arc<dyn Texture>,
    /// GGX roughness; the alpha is `roughness²` (0 = mirror, 1 = fully rough).
    /// Sampled as channel 0 of the texture (scalar convention).
    pub roughness: Arc<dyn Texture>,
    /// Index of refraction for the Fresnel term (1.5 = glass, 1.45 = typical plastic).
    pub ior: f32,
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
            let f = fresnel_schlick(cos_o, fresnel_r0(self.ior));
            let albedo_ = self.albedo.value(&si.texture_coords());
            // Delta (mirror) throughput: f_cos = albedo · F — the delta maps
            // solid angle 1:1, so the Cook-Torrance 1/(4·cos_o·cos_i)
            // denominator cancels against the reflection Jacobian. Albedo is
            // kept because Schlick F is colorless; Metal uses the conductor
            // Fresnel (already colored) and drops its albedo.
            return Some(BsdfScatter::Delta {
                wi,
                f_cos: albedo_ * f,
                eta: None,
            });
        }

        let alpha = self.ggx_alpha(si)?;
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
        let albedo = self.albedo.value(&si.texture_coords());
        let alpha = self.ggx_alpha(si).unwrap_or(0.001);

        let h = (wo + wi).normalize();
        let cos_h_n = h.dot(si.shading_normal().into_inner()).max(0.0);
        let cos_h_o = wo.dot(h.into_inner()).max(0.0);

        let cos_o = wo.dot(si.shading_normal().into_inner()).max(0.0);
        let cos_i = wi.dot(si.shading_normal().into_inner()).max(0.0);

        if cos_h_o <= 0.0 || cos_o <= 0.0 || cos_i <= 0.0 {
            return Color3::ZERO;
        }

        let d = ggx_d(cos_h_n, alpha);
        let f = fresnel_schlick(cos_h_o, fresnel_r0(self.ior));
        let roughness = self.roughness.value(&si.texture_coords()).x();
        let g = geometry_schlick_ggx(cos_o, roughness) * geometry_schlick_ggx(cos_i, roughness);

        albedo * f * d * g / (4.0 * cos_o)
    }

    /// GGX NDF sampling PDF: `D(H) · cos(H·N) / (4 · cos(H·O))`.
    fn pdf(&self, wo: Direction3, wi: Direction3, si: &SurfaceInteraction) -> f32 {
        if self.is_delta() {
            return 0.0;
        }

        let alpha = self.ggx_alpha(si).unwrap_or(0.001);

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
            let alpha = self.ggx_alpha(si).unwrap_or(0.001);
            Some(PdfKind::Ggx {
                wo,
                normal: si.shading_normal(),
                alpha,
            })
        }
    }

    fn reflectance_estimate(&self, wo: Direction3, si: &SurfaceInteraction) -> f32 {
        let albedo = self.albedo.value(&si.texture_coords());
        let albedo_avg = albedo.into_inner().element_sum() / 3.0;
        let cos_theta = wo.dot(si.shading_normal().into_inner()).abs();
        let f = fresnel_schlick(cos_theta, fresnel_r0(self.ior));
        // Base color × Fresnel gives the specular reflectance; roughness adds
        // a small boost from multiple scattering making the surface appear brighter.
        let roughness_boost = self.roughness.value(&si.texture_coords()).x() * 0.25;
        (albedo_avg * f + roughness_boost).min(1.0)
    }

    fn is_delta(&self) -> bool {
        if let Some(roughness) = self.roughness.as_constant() {
            roughness.x() < MIRROR_THRESHOLD
        } else {
            false
        }
    }

    fn ggx_alpha(&self, si: &SurfaceInteraction) -> Option<f32> {
        if self.is_delta() {
            None
        } else {
            let roughness = self.roughness.value(&si.texture_coords()).x();
            Some((roughness * roughness).clamp(0.001, 1.0))
        }
    }
}

impl GlossyMaterial {
    /// Create a glossy material.
    pub fn new(albedo: Color3, roughness: f32, ior: f32) -> Self {
        Self {
            albedo: Arc::new(SolidColor::new(albedo)),
            roughness: Arc::new(SolidColor::new(Color3::splat(roughness))),
            ior,
        }
    }

    /// Create a glossy material with textures for `albedo` and `roughness`.
    pub fn textured(albedo: Arc<dyn Texture>, roughness: Arc<dyn Texture>, ior: f32) -> Self {
        Self {
            albedo,
            roughness,
            ior,
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
        let index = buf.nodes.len() as u32;
        let param_offset = buf.params.len() as u32;

        let (ar, ag, ab, albedo_tex) = buf.gpu_color(&self.albedo);
        let (rr, _rg, _rb, rough_tex) = buf.gpu_color(&self.roughness); // channel-0 scalar
        buf.push_params(&[ar, ag, ab, rr, self.ior]);

        // Texture references ride after the baked colors. GPU_NONE (u32::MAX)
        // cannot round-trip through f32, so use −1.0 as the "no texture" sentinel.
        let tex = |i: u32| if i == GPU_NONE { -1.0 } else { i as f32 };
        buf.push_params(&[tex(albedo_tex), tex(rough_tex)]);

        buf.nodes.push(GpuMaterialNode {
            material_type: GpuMaterialType::Glossy as u32,
            param_offset,
            child_a: GPU_NONE,
            child_b: GPU_NONE,
            texture_index: GPU_NONE, // refs ride in params for multi-texture materials
        });
        index
    }
}
