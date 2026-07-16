//! GGX microfacet conductor BRDF.
//!
//! Models rough metals (gold, copper, aluminium) as a surface covered in
//! tiny perfect mirrors (microfacets). The macro-level shininess comes from
//! the statistical distribution of their orientations.
//!
//! The BRDF is the Cook-Torrance specular model:
//!
//! ```text
//! f(ωo, ωi) = F · D · G / (4 · cos_o · cos_i)
//! ```
//!
//! - **F** (Fresnel): fraction reflected at the microfacet. Uses Schlick's
//!   approximation with the material's IOR.
//! - **D** (NDF): GGX/Trowbridge-Reitz — probability that a microfacet has
//!   half-vector H. Controls the specular lobe width.
//! - **G** (Geometry): Smith's shadowing/masking via Schlick-GGX — microfacets
//!   blocking each other at grazing angles.
//!
//! Sampling importance-samples the GGX NDF to generate H, then reflects `wo`
//! about H to get `wi`. This concentrates samples where the BRDF has the most
//! energy, reducing noise.

use std::sync::Arc;

use crate::hittable::SurfaceInteraction;
use crate::material::gpu::{GPU_NONE, GpuSerializable};
use crate::material::{
    Bsdf, BsdfScatter, GpuMaterialBuffer, GpuMaterialNode, GpuMaterialType, MAX_BSDF_STRATS,
    PdfKind, fresnel_schlick, geometry_schlick_ggx, ggx_d, ggx_sample_h,
};
use crate::onb::Onb;
use crate::texture::Texture;
use crate::vec3::{Color3, Direction3};

use super::MIRROR_THRESHOLD;

/// Microfacet conductor BRDF (GGX).
#[derive(Clone)]
pub struct MetalMaterial {
    /// Base reflectance color. Multiplied by the Cook-Torrance BRDF value.
    pub albedo: Color3,
    /// Optional texture for spatial albedo variation. CPU-only; GPU serialization
    /// falls back to `albedo`.
    pub tex: Option<Arc<dyn Texture>>,
    /// Controls roughness: the GGX alpha is `fuzz²`. 0 = mirror, 1 = fully rough.
    pub roughness: f32,
    /// Index of refraction for the Fresnel term (typical metals: 2.5–3.0).
    pub ior: f32,
    /// Precomputed Fresnel reflectance at normal incidence.
    pub r0: f32,
}

impl Bsdf for MetalMaterial {
    /// Importance-sample the GGX NDF: draw a half-vector H from the distribution,
    /// then reflect `wo` about H to get `wi`. Returns `None` if the reflected
    /// direction ends up below the surface.
    ///
    /// When `roughness` is effectively zero (below 0.01), the microsurface is a
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

        // Reflect wo about H to get wi.
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
        // If the material is effectively a mirror, return zero for arbitrary directions.
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
        // If the material is effectively a mirror, return zero for arbitrary directions.
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

    /// Returns the PDF kind for the GGX NDF if `fuzz` is non-zero, otherwise `None`.
    fn pdf_kind(&self, wo: Direction3, si: &SurfaceInteraction) -> Option<PdfKind> {
        if self.is_delta() {
            None
        } else {
            Some(PdfKind::Ggx {
                wo,
                normal: si.shading_normal(),
                alpha: (self.roughness * self.roughness).clamp(0.001, 1.0),
            })
        }
    }

    /// Returns an estimate of the material's reflectance for a given outgoing
    /// direction. For a smooth conductor, this is approximately the Fresnel term with a roughness
    /// boost. For rough conductors, the effective reflectance is higher due to multiple scattering.
    fn reflectance_estimate(&self, wo: Direction3, si: &SurfaceInteraction) -> f32 {
        let cos_theta = wo.dot(si.shading_normal().into_inner()).abs();
        // For a smooth conductor, reflectance ≈ Fresnel(θ) — the GGX lobe
        // is narrow, so most energy is at the mirror direction. Roughness
        // increases the effective reflectance due to multiple scattering.
        let f = fresnel_schlick(cos_theta, self.r0);
        let roughness_boost = self.roughness * 0.25;
        (f + roughness_boost).min(1.0)
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

impl GpuSerializable for MetalMaterial {
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
            material_type: GpuMaterialType::Metal as u32,
            param_offset,
            child_a: GPU_NONE,
            child_b: GPU_NONE,
            texture_index: GPU_NONE,
        });
        buf.nodes.len() as u32 - 1
    }
}
