//! GGX microfacet dielectric BRDF (glossy reflection).
//!
//! Similar to the metal BRDF but models dielectric surfaces (plastic, coated
//! wood, ceramic) rather than conductors. The key difference is that the
//! Fresnel term uses the material's IOR to determine the reflection/transmission
//! ratio, and the result is multiplied by the surface albedo.
//!
//! BRDF (same Cook-Torrance model as metal):
//!
//! ```text
//! f(ωo, ωi) = albedo · F · D · G / (4 · cos_o · cos_i)
//! ```
//!
//! `roughness` controls the GGX distribution width (0 = mirror, 1 = fully rough).
//! `ior` sets the index of refraction for the Fresnel term — higher IOR means
//! more reflection at normal incidence.

use std::f64::consts::PI;
use std::sync::Arc;

use crate::hittable::SurfaceInteraction;
use crate::material::{
    Bsdf, BsdfSample, GPU_NONE, GpuMaterialBuffer, GpuMaterialNode, GpuMaterialType, PdfKind,
    fresnel_schlick, geometry_schlick_ggx, ggx_d,
};
use crate::onb::Onb;
use crate::texture::Texture;
use crate::vec3::{Color3, Vec3, reflect};

/// Glossy microfacet BRDF (GGX).
#[derive(Clone)]
pub struct GlossyMaterial {
    /// Base reflectance color. Multiplied by the Cook-Torrance BRDF value.
    pub albedo: Color3,
    /// Optional texture for spatial albedo variation. CPU-only; GPU serialization
    /// falls back to `albedo`.
    pub tex: Option<Arc<dyn Texture>>,
    /// Surface smoothness: 0 = mirror, 1 = fully rough. GGX alpha = `roughness²`.
    pub roughness: f64,
    /// Index of refraction for the Fresnel term (1.5 = glass, 1.45 = typical plastic).
    pub ior: f64,
}

impl Bsdf for GlossyMaterial {
    /// Importance-sample the GGX NDF: draw half-vector H, reflect `wo` about it.
    /// Returns `None` if the reflected direction is below the surface.
    ///
    /// When `roughness` is effectively zero (below 1e-4), the surface is a
    /// perfect mirror — returns `BsdfSample::Delta` so the integrator skips
    /// the mixture PDF and uses the reflected direction directly.
    fn sample(
        &self,
        wo: Vec3,
        si: &SurfaceInteraction,
        u: f64,
        v: f64,
        _w: f64,
        _x: f64,
        _y: f64,
        _z: f64,
    ) -> Option<BsdfSample> {
        // Near-mirror: delta path bypasses the mixture PDF entirely.
        if self.roughness < 1e-4 {
            let wi = reflect(&-wo, &si.shading_normal());
            if wi.dot(&si.shading_normal()) <= 0.0 {
                return None;
            }
            let cos_o = wo.dot(&si.shading_normal()).max(0.0);
            let f = fresnel_schlick(cos_o, self.ior);
            let albedo_ = self
                .tex
                .as_ref()
                .map(|t| t.value(&si.texture_coords()))
                .unwrap_or(self.albedo);
            return Some(BsdfSample::Delta {
                wi,
                f_cos: albedo_ * f,
            });
        }

        let alpha = (self.roughness * self.roughness).clamp(0.001, 1.0);
        // Sample H from GGX NDF.
        let u1 = u;
        let u2 = v;
        let cos_theta = ((1.0 - u2) / (1.0 + (alpha * alpha - 1.0) * u2))
            .clamp(0.0, 1.0)
            .sqrt();
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        let phi = 2.0 * PI * u1;
        let h_local = Vec3::from(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta);

        let onb = Onb::build_from_normal(si.shading_normal());
        let h_world = onb.local_to_world(h_local);

        let wi = reflect(&-wo, &h_world);

        if wi.dot(&si.shading_normal()) <= 0.0 {
            return None;
        }

        Some(BsdfSample::NonDelta {
            pdf_kind: PdfKind::Ggx {
                wo,
                normal: si.shading_normal(),
                alpha,
            },
        })
    }

    /// Cook-Torrance BRDF: `albedo · F · D · G / (4 · cos_o · cos_i)`.
    fn eval(&self, wo: Vec3, wi: Vec3, si: &SurfaceInteraction) -> Color3 {
        let albedo = self
            .tex
            .as_ref()
            .map(|t| t.value(&si.texture_coords()))
            .unwrap_or(self.albedo);
        let alpha = (self.roughness * self.roughness).clamp(0.001, 1.0);

        let h = (wo + wi).unit_vector();
        let cos_h_n = h.dot(&si.shading_normal()).max(0.0);
        let cos_h_o = wo.dot(&h).max(0.0);

        let cos_o = wo.dot(&si.shading_normal()).max(0.0);
        let cos_i = wi.dot(&si.shading_normal()).max(0.0);

        if cos_h_o <= 0.0 || cos_o <= 0.0 || cos_i <= 0.0 {
            return Color3::from(0., 0., 0.);
        }

        let d = ggx_d(cos_h_n, alpha);
        let f = fresnel_schlick(cos_h_o, self.ior);
        let g = geometry_schlick_ggx(cos_o, alpha) * geometry_schlick_ggx(cos_i, alpha);

        albedo * f * d * g / (4.0 * cos_o)
    }

    /// GGX NDF sampling PDF: `D(H) · cos(H·N) / (4 · cos(H·O))`.
    fn pdf(&self, wo: Vec3, wi: Vec3, si: &SurfaceInteraction) -> f64 {
        let alpha = (self.roughness * self.roughness).clamp(0.001, 1.0);

        let h = (wo + wi).unit_vector();
        let cos_h_n = h.dot(&si.shading_normal()).max(0.0);
        let cos_h_o = wo.dot(&h).max(0.0);

        if cos_h_o <= 0.0 {
            return 0.0;
        }

        ggx_d(cos_h_n, alpha) * cos_h_n / (4.0 * cos_h_o)
    }

    fn clone_box(&self) -> Box<dyn Bsdf> {
        Box::new(self.clone())
    }

    fn serialize_gpu(&self, buf: &mut GpuMaterialBuffer) -> u32 {
        let params = vec![
            self.albedo.x,
            self.albedo.y,
            self.albedo.z,
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
