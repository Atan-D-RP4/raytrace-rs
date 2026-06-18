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

use std::f64::consts::PI;
use std::sync::Arc;

use crate::hittable::SurfaceInteraction;
use crate::onb::Onb;
use crate::texture::Texture;
use crate::vec3::{Color3, Vec3, reflect};

use super::GPU_NONE;
use super::{Bsdf, BsdfSample, GpuMaterialBuffer, GpuMaterialNode, GpuMaterialType, PdfKind};

use super::{fresnel_schlick, geometry_schlick_ggx, ggx_d};

/// Microfacet conductor BRDF (GGX).
#[derive(Clone)]
pub struct MetalMaterial {
    /// Base reflectance color. Multiplied by the Cook-Torrance BRDF value.
    pub albedo: Color3,
    /// Optional texture for spatial albedo variation. CPU-only; GPU serialization
    /// falls back to `albedo`.
    pub tex: Option<Arc<dyn Texture>>,
    /// Controls roughness: the GGX alpha is `fuzz²`. 0 = mirror, 1 = fully rough.
    pub fuzz: f64,
    /// Index of refraction for the Fresnel term (typical metals: 2.5–3.0).
    pub ior: f64,
}

impl Bsdf for MetalMaterial {
    /// Importance-sample the GGX NDF: draw a half-vector H from the distribution,
    /// then reflect `wo` about H to get `wi`. Returns `None` if the reflected
    /// direction ends up below the surface.
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
        let albedo = self
            .tex
            .as_ref()
            .map(|t| t.value(&si.texture_coords()))
            .unwrap_or(self.albedo);
        let alpha = (self.fuzz * self.fuzz).clamp(0.001, 1.0);
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

        // Reflect wo about H to get wi.
        let wi = reflect(&-wo, &h_world);

        if wi.dot(&si.shading_normal()) <= 0.0 {
            return None;
        }

        // Note: D/F/G evaluation is deferred to eval() — the integrator
        // ignores f_cos and pdf for non-delta materials and recomputes
        // the BRDF via eval() with a direction from the mixture PDF.
        Some(BsdfSample {
            wi: Vec3::ZERO,
            f_cos: albedo,
            pdf: 1.0,
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
        let alpha = (self.fuzz * self.fuzz).clamp(0.001, 1.0);
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
        let alpha = (self.fuzz * self.fuzz).clamp(0.001, 1.0);
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
            self.fuzz,
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
