//! Rough dielectric (microfacet GGX) BSDF with both reflection and transmission.
//!
//! Models a smooth-to-rough dielectric surface (glass, water, diamond) using the
//! microfacet GGX distribution for both reflection and refraction lobes (Walter et
//! al. 2007). At `roughness < MIRROR_THRESHOLD` it degrades to a delta BSDF (Snell
//! + Fresnel with no microfacet spread).
//!
//! Reflection: standard Cook-Torrance microfacet BRDF `F·D·G / (4·cosθ_o)`.
//! Transmission: microfacet BTDF `D·G·(1-F)·η_i² / (η_o·cosθ_o + η_i·cosθ_i)²`
//!   times `|wi·H|·|wo·H|`, following Walter et al. Eq. 17.
//! PDF: reflection `D(ω_h)·|ω_h·n| / (4·|wo·ω_h|)`, transmission Eq. 33.

use crate::hittable::SurfaceInteraction;
use crate::material::gpu::{GPU_NONE, GpuSerializable};
use crate::material::{
    Bsdf, BsdfScatter, GpuMaterialBuffer, GpuMaterialNode, GpuMaterialType, MAX_BSDF_STRATS,
    fresnel_schlick, geometry_schlick_ggx, ggx_d, ggx_sample_h,
};
use crate::onb::Onb;
use crate::pdf::PdfKind;
use crate::vec3::{Color3, Direction3};

use super::MIRROR_THRESHOLD;

/// Rough dielectric material with microfacet-based reflection and refraction.
#[derive(Clone)]
pub struct RoughDielectricMaterial {
    /// Index of refraction (1.0 = air, 1.33 = water, 1.5 = glass, 2.42 = diamond).
    pub ior: f32,
    /// Roughness of the surface. Higher values produce more diffuse reflection and refraction.
    pub roughness: f32,
    /// Optional tint color for colored glass. Pure white means no tint.
    pub tint: Color3,
    /// Precomputed Fresnel reflectance at normal incidence.
    pub r0: f32,
}

impl Bsdf for RoughDielectricMaterial {
    /// Sample the BSDF: either a deterministic delta (mirror) or a GGX-microfacet
    /// direction split stochastically by Fresnel between reflection and transmission.
    fn scatter(
        &self,
        wo: Direction3,
        si: &SurfaceInteraction,
        next_dim: &mut dyn FnMut() -> f32,
    ) -> Option<BsdfScatter> {
        let ri = if si.front_face() {
            1.0 / self.ior
        } else {
            self.ior
        };

        if self.is_delta() {
            let cos_theta = wo.dot(si.shading_normal().into_inner()).min(1.0);
            let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();

            let u = next_dim();
            // TIR (ri·sinθ > 1) or stochastic Fresnel reflection → reflect,
            // otherwise refract through the surface.
            let direction = if ri * sin_theta > 1.0 || fresnel_schlick(cos_theta, self.r0) > u {
                (-wo).reflect(si.shading_normal().into_inner())
            } else {
                (-wo).refract(si.shading_normal().into_inner(), ri)
            };

            Some(BsdfScatter::Delta {
                wi: direction,
                f_cos: self.tint,
                eta: Some(ri),
            })
        } else {
            // ── Non-delta (rough) path: GGX-microfacet importance sampling ──
            let alpha = self.ggx_alpha().unwrap_or(0.001);

            // Sample a microfacet normal (half-vector) from the GGX distribution.
            let h_local = ggx_sample_h(alpha, next_dim(), next_dim());
            let onb = Onb::build_from_normal(si.shading_normal());
            let h_world = onb.local_to_world(h_local);
            let n = si.shading_normal().into_inner();

            let cos_h_o = wo.dot(h_world.into_inner()).max(0.0);
            let fresnel_val = fresnel_schlick(cos_h_o, self.r0);
            // Split stochastically: reflection (Fresnel) vs transmission (1-Fresnel).
            let is_reflection = next_dim() < fresnel_val;

            let wi = if is_reflection {
                -wo.reflect(h_world.into_inner())
            } else {
                (-wo).refract(h_world.into_inner(), ri)
            };

            // Validate hemisphere: reflection must stay on the same side as the
            // normal, transmission must cross to the opposite side. If the
            // generated direction violates this (e.g. TIR attempted a refraction
            // that actually reflected), reject the sample.
            let on_same_side = wi.dot(n) > 0.0;
            if is_reflection != on_same_side {
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
    }

    fn eval(&self, wo: Direction3, wi: Direction3, si: &SurfaceInteraction) -> Color3 {
        if self.is_delta() {
            return Color3::ZERO;
        }
        let alpha = self.ggx_alpha().unwrap_or(0.001);
        let n = si.shading_normal().into_inner();
        let cos_o = wo.dot(n).abs();
        let cos_i = wi.dot(n).abs();

        if cos_o <= 0.0 || cos_i <= 0.0 {
            return Color3::ZERO;
        }

        let g = geometry_schlick_ggx(cos_o, self.roughness)
            * geometry_schlick_ggx(cos_i, self.roughness);

        // Check if (wo, wi) are on the same side of the surface → reflection, else transmission.
        if wo.dot(n) * wi.dot(n) > 0.0 {
            // ── Reflection: standard Cook-Torrance BRDF × cos_i ──
            let h = (wo + wi).normalize();
            let cos_h_n = h.dot(n).max(0.0);
            let cos_h_o = wo.dot(h.into_inner()).max(0.0);

            if cos_h_o <= 0.0 {
                return Color3::ZERO;
            }

            let d = ggx_d(cos_h_n, alpha);
            let f = fresnel_schlick(cos_h_o, self.r0);

            // f_r × |cosθ_i| = F·D·G / (4·cosθ_o)
            self.tint * f * d * g / (4.0 * cos_o).max(1e-6)
        } else {
            // ── Transmission: microfacet BTDF × cos_i (Walter et al. 2007, Eq. 17) ──
            let ri = if si.front_face() {
                1.0 / self.ior
            } else {
                self.ior
            };
            let h = (wo + ri * wi).normalize();
            let cos_h_n = h.dot(n).max(0.0);
            let cos_h_o = wo.dot(h.into_inner()).abs();
            let cos_h_i = wi.dot(h.into_inner()).abs();

            if cos_h_n <= 0.0 || cos_h_o <= 0.0 || cos_h_i <= 0.0 {
                return Color3::ZERO;
            }

            let d = ggx_d(cos_h_n, alpha);
            let f = fresnel_schlick(cos_h_o, self.r0);

            // η_o = IOR of the medium containing wo, η_i = IOR of the medium containing wi
            let (eta_o, eta_i) = if si.front_face() {
                (1.0, self.ior) // wo in air, wi in material
            } else {
                (self.ior, 1.0) // wo in material, wi in air
            };

            let denom = eta_o * cos_h_o + eta_i * cos_h_i;

            // f_t × |cosθ_i| =
            //   |wi·H|·|wo·H| · D·G·(1-F)·η_i² / [cosθ_o · (η_o·|wo·H| + η_i·|wi·H|)²]
            self.tint * (1.0 - f) * d * g * cos_h_i * cos_h_o * eta_i * eta_i
                / (cos_o * denom * denom).max(1e-6)
        }
    }

    fn pdf(&self, wo: Direction3, wi: Direction3, si: &SurfaceInteraction) -> f32 {
        if self.is_delta() {
            return 0.0;
        }
        let alpha = self.ggx_alpha().unwrap_or(0.001);
        let n = si.shading_normal().into_inner();

        // Same-side → reflection PDF, opposite-side → transmission PDF
        if wo.dot(n) * wi.dot(n) > 0.0 {
            // ── Reflection PDF: D(H·N)·|H·N| / (4·|wo·H|) ──
            let h = (wo + wi).normalize();
            let cos_h_n = h.dot(n).max(0.0);
            let cos_h_o = wo.dot(h.into_inner()).max(0.0);

            if cos_h_o <= 0.0 {
                return 0.0;
            }

            ggx_d(cos_h_n, alpha) * cos_h_n / (4.0 * cos_h_o)
        } else {
            // ── Transmission PDF (Walter et al. 2007, Eq. 33) ──
            let ri = if si.front_face() {
                1.0 / self.ior
            } else {
                self.ior
            };
            let h = (wo + ri * wi).normalize();
            let cos_h_n = h.dot(n).max(0.0);
            let cos_h_o = wo.dot(h.into_inner()).abs();
            let cos_h_i = wi.dot(h.into_inner()).abs();

            if cos_h_n <= 0.0 || cos_h_o <= 0.0 || cos_h_i <= 0.0 {
                return 0.0;
            }

            let (eta_o, eta_i) = if si.front_face() {
                (1.0, self.ior)
            } else {
                (self.ior, 1.0)
            };

            let denom = eta_o * cos_h_o + eta_i * cos_h_i;

            // p_t(ω_i) = D(H·N)·|H·N| · η_i²·|wi·H| / (η_o·|wo·H| + η_i·|wi·H|)²
            ggx_d(cos_h_n, alpha) * cos_h_n * eta_i * eta_i * cos_h_i / (denom * denom)
        }
    }

    fn pdf_kind(&self, wo: Direction3, si: &SurfaceInteraction) -> Option<PdfKind> {
        // Delta: no PDF distribution to sample from (mirror direction is deterministic).
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

    /// Directional-hemispherical reflectance estimate for MIS weighting.
    /// Rough approximation: Fresnel term dominates at normal incidence (f) and
    /// transmission dominates at grazing (1-f), so `max(f, 1-f)` is a cheap upper bound.
    fn reflectance_estimate(&self, wo: Direction3, si: &SurfaceInteraction) -> f32 {
        let cos_theta = wo.dot(si.shading_normal().into_inner()).abs();
        let f = fresnel_schlick(cos_theta, self.r0);
        f.max(1. - f)
    }

    /// Surfaces below `MIRROR_THRESHOLD` roughness are visually indistinguishable
    /// from perfect mirrors — use a single delta direction (Snell + Fresnel) with
    /// no microfacet spread.
    fn is_delta(&self) -> bool {
        self.roughness < MIRROR_THRESHOLD
    }

    /// Convert perceptual roughness to GGX alpha = roughness². Returns `None`
    /// for delta surfaces (no GGX lobe to sample).
    fn ggx_alpha(&self) -> Option<f32> {
        if self.is_delta() {
            None
        } else {
            Some((self.roughness * self.roughness).clamp(0.001, 1.0))
        }
    }
}

impl GpuSerializable for RoughDielectricMaterial {
    fn serialize_gpu(&self, buf: &mut GpuMaterialBuffer) -> u32 {
        let params = vec![
            self.ior,
            self.roughness,
            self.tint.x(),
            self.tint.y(),
            self.tint.z(),
        ];
        let param_offset = buf.params.len() as u32;
        buf.push_params(&params);
        buf.nodes.push(GpuMaterialNode {
            material_type: GpuMaterialType::RoughDielectric as u32,
            param_offset,
            child_a: GPU_NONE,
            child_b: GPU_NONE,
            texture_index: GPU_NONE,
        });
        buf.nodes.len() as u32 - 1
    }
}
