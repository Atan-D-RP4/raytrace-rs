//! Dielectric (glass/water) material with refraction and reflection.
//!
//! Transparent materials like glass or water. Light hitting the surface
//! either reflects or refracts, with the ratio governed by Fresnel's equations
//! and the refractive indices of the two media.
//!
//! **Snell's Law**: `η₁ sin(θ₁) = η₂ sin(θ₂)` determines the refraction angle.
//!
//! **Total Internal Reflection**: when light in a denser medium hits the
//! boundary at a steep angle, no refraction is possible — all light reflects.
//!
//! **Fresnel**: at normal incidence, `((η₁ - η₂) / (η₁ + η₂))²` reflects.
//! At grazing angles, nearly all light reflects regardless of IOR.
//!
//! A smooth dielectric (`roughness: None`) is a **delta material** — it
//! scatters in a single determined direction (Snell + Fresnel split), not
//! over a distribution. The integrator must skip MIS weighting and use the
//! sampled direction directly.
//!
//! With `roughness` set, this is the rough dielectric (microfacet GGX) BSDF
//! with both reflection and transmission. Models a smooth-to-rough dielectric
//! surface (glass, water, diamond) using the microfacet GGX distribution for
//! both reflection and refraction lobes (Walter et al. 2007). At
//! `roughness < MIRROR_THRESHOLD` it degrades back to a delta BSDF (Snell +
//! Fresnel with no microfacet spread).
//!
//! Reflection: standard Cook-Torrance microfacet BRDF `F·D·G / (4·cosθ_o)`.
//! Transmission: microfacet BTDF `D·G·(1-F)·η_i² / (η_o·cosθ_o + η_i·cosθ_i)²`
//!   times `|wi·H|·|wo·H|`, following Walter et al. Eq. 17.
//! PDF: reflection `D(ω_h)·|ω_h·n| / (4·|wo·ω_h|)`, transmission Eq. 33.

use std::sync::Arc;

use crate::intersect::interaction::SurfaceInteraction;
use crate::material::gpu::{GPU_NONE, GpuSerializable};
use crate::material::{
    Bsdf, BsdfScatter, GpuMaterialBuffer, GpuMaterialNode, GpuMaterialType, MAX_BSDF_STRATS,
    MIRROR_THRESHOLD, PdfKind, fresnel_r0, fresnel_schlick, geometry_schlick_ggx, ggx_d,
    ggx_sample_h,
};
use crate::math::onb::Onb;
use crate::math::vec3::{Color3, Direction3};
use crate::texture::{SolidColor, Texture};

use crate::material::Material;

/// Dielectric transmission/reflection controlled by refractive index.
///
/// `roughness: None` is the smooth (classic) dielectric; `Some` enables the
/// GGX-microfacet rough path with a delta fallback below `MIRROR_THRESHOLD`.
#[derive(Clone)]
pub struct DielectricMaterial {
    /// Index of refraction (1.0 = air, 1.33 = water, 1.5 = glass, 2.42 = diamond).
    pub ior: f32,
    /// None = smooth (always delta, the classic dielectric). Some = rough
    /// GGX-microfacet path with delta fallback below MIRROR_THRESHOLD.
    pub roughness: Option<Arc<dyn Texture>>,
    /// Tint color for colored glass, sampled as a texture. Pure white means no tint.
    pub tint: Arc<dyn Texture>,
}

impl Bsdf for DielectricMaterial {
    /// Compute refraction ratio from the two media using Snell's Law.
    /// Then use Fresnel to decide between reflection and refraction.
    ///
    /// Smooth path (1 RNG draw): TIR (`ri·sinθ > 1`) or a stochastic Fresnel
    /// split chooses reflect vs refract.
    ///
    /// Rough path (3 RNG draws): sample one half-vector H from the GGX NDF,
    /// then Fresnel-split between reflection (same side as `wo`) and
    /// transmission (opposite side) — the coupled sampler for rough
    /// dielectrics, not two independent lobes.
    fn scatter(
        &self,
        wo: Direction3,
        si: &SurfaceInteraction,
        next_dim: &mut dyn FnMut() -> f32,
    ) -> Option<BsdfScatter> {
        // Determine the ratio of indices of refraction based on whether the ray is entering or exiting the material.
        let ri = if si.front_face() {
            1.0 / self.ior
        } else {
            self.ior
        };

        if self.is_delta() {
            // Compute the cosine of the angle between the outgoing direction and the surface normal.
            let cos_theta = wo.dot(si.shading_normal().into_inner()).min(1.0);
            // Compute the sine of the angle using the identity sin²(θ) + cos²(θ) = 1.
            let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();

            // Use Fresnel's equations to determine the probability of reflection vs refraction.
            let u = next_dim();
            let direction =
                if ri * sin_theta > 1.0 || fresnel_schlick(cos_theta, fresnel_r0(self.ior)) > u {
                    (-wo).reflect(si.shading_normal().into_inner())
                } else {
                    (-wo).refract(si.shading_normal().into_inner(), ri)
                };

            // Return the chosen direction with unit attenuation (delta material — all energy goes one way).
            let tint = self.tint.value(&si.texture_coords());
            Some(BsdfScatter::Delta {
                wi: direction,
                f_cos: tint,
                eta: Some(ri),
            })
        } else {
            // ── Non-delta (rough) path: GGX-microfacet importance sampling ──
            let alpha = self.ggx_alpha(si).unwrap_or(0.001);

            // Sample a microfacet normal (half-vector) from the GGX distribution.
            let h_local = ggx_sample_h(alpha, next_dim(), next_dim());
            let onb = Onb::build_from_normal(si.shading_normal());
            let h_world = onb.local_to_world(h_local);
            let n = si.shading_normal().into_inner();

            let cos_h_o = wo.dot(h_world.into_inner()).max(0.0);
            let fresnel_val = fresnel_schlick(cos_h_o, fresnel_r0(self.ior));
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

    /// Delta material — cannot evaluate at arbitrary directions. Returns zero.
    fn eval(&self, wo: Direction3, wi: Direction3, si: &SurfaceInteraction) -> Color3 {
        if self.is_delta() {
            return Color3::ZERO;
        }
        // Invariant: is_delta() returned false above, so ggx_alpha is Some.
        let alpha = self
            .ggx_alpha(si)
            .expect("non-delta roughness => Some alpha");
        let n = si.shading_normal().into_inner();
        let cos_o = wo.dot(n).abs();
        let cos_i = wi.dot(n).abs();

        if cos_o <= 0.0 || cos_i <= 0.0 {
            return Color3::ZERO;
        }

        let tint = self.tint.value(&si.texture_coords());
        let rough = self
            .roughness
            .as_ref()
            .unwrap()
            .value(&si.texture_coords())
            .x();

        let g = geometry_schlick_ggx(cos_o, rough) * geometry_schlick_ggx(cos_i, rough);

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
            let f = fresnel_schlick(cos_h_o, fresnel_r0(self.ior));

            // f_r × |cosθ_i| = F·D·G / (4·cosθ_o)
            // The 1e-6 denominator guard keeps the grazing-angle boundary between
            // the reflection and transmission lobes numerically stable in this
            // coupled dielectric path. MicrofacetReflector is reflection-only and
            // has no such boundary, so it carries no epsilon.
            tint * f * d * g / (4.0 * cos_o).max(1e-6)
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
            let f = fresnel_schlick(cos_h_o, fresnel_r0(self.ior));

            // η_o = IOR of the medium containing wo, η_i = IOR of the medium containing wi
            let (eta_o, eta_i) = if si.front_face() {
                (1.0, self.ior) // wo in air, wi in material
            } else {
                (self.ior, 1.0) // wo in material, wi in air
            };

            let denom = eta_o * cos_h_o + eta_i * cos_h_i;

            // f_t × |cosθ_i| =
            //   |wi·H|·|wo·H| · D·G·(1-F)·η_i² / [cosθ_o · (η_o·|wo·H| + η_i·|wi·H|)²]
            // Same epsilon rationale as the reflection branch above: the guard
            // stabilizes the reflection/transmission boundary at grazing angles;
            // reflection-only MicrofacetReflector needs no such guard.
            tint * (1.0 - f) * d * g * cos_h_i * cos_h_o * eta_i * eta_i
                / (cos_o * denom * denom).max(1e-6)
        }
    }

    /// Delta material — probability of any specific direction is zero.
    fn pdf(&self, wo: Direction3, wi: Direction3, si: &SurfaceInteraction) -> f32 {
        if self.is_delta() {
            return 0.0;
        }
        // Invariant: is_delta() returned false above, so ggx_alpha is Some.
        let alpha = self
            .ggx_alpha(si)
            .expect("non-delta roughness => Some alpha");
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

    /// Delta material — no PDF kind for arbitrary directions.
    fn pdf_kind(&self, wo: Direction3, si: &SurfaceInteraction) -> Option<PdfKind> {
        // Delta: no PDF distribution to sample from (mirror direction is deterministic).
        if self.is_delta() {
            None
        } else {
            // Invariant: is_delta() returned None above, so ggx_alpha is
            // guaranteed to be Some here — no fallback needed.
            let alpha = self
                .ggx_alpha(si)
                .expect("non-delta roughness => Some alpha");
            Some(PdfKind::Ggx {
                wo,
                normal: si.shading_normal(),
                alpha,
            })
        }
    }

    /// Estimate the reflectance fraction for the coating layer. This is used in
    /// the integrator to determine how much light is reflected vs transmitted.
    fn reflectance_estimate(&self, wo: Direction3, si: &SurfaceInteraction) -> f32 {
        let cos_theta = wo.dot(si.shading_normal().into_inner()).abs();
        match &self.roughness {
            None => {
                // Smooth dielectric: only the reflective fraction of the dielectric
                // is returned to the coating; transmitted light goes into the
                // substrate and doesn't contribute to the inter-reflection series.
                fresnel_schlick(cos_theta, fresnel_r0(self.ior))
            }
            Some(_) => {
                // Directional-hemispherical reflectance estimate for MIS weighting.
                // Rough approximation: Fresnel term dominates at normal incidence (f)
                // and transmission dominates at grazing (1-f), so `max(f, 1-f)` is a
                // cheap upper bound.
                let f = fresnel_schlick(cos_theta, fresnel_r0(self.ior));
                f.max(1.0 - f)
            }
        }
    }

    /// Delta material (smooth) — or, for the rough path, surfaces below
    /// `MIRROR_THRESHOLD` roughness are visually indistinguishable from perfect
    /// mirrors — use a single delta direction (Snell + Fresnel) with no
    /// microfacet spread.
    fn is_delta(&self) -> bool {
        match &self.roughness {
            None => true,
            Some(roughness) => {
                if let Some(roughness) = roughness.as_constant() {
                    roughness.x() < MIRROR_THRESHOLD
                } else {
                    false
                }
            }
        }
    }

    /// Convert perceptual roughness to GGX alpha = roughness². Returns `None`
    /// for delta surfaces (no GGX lobe to sample).
    fn ggx_alpha(&self, si: &SurfaceInteraction) -> Option<f32> {
        if self.is_delta() {
            None
        } else {
            let roughness = self.roughness.as_ref()?.value(&si.texture_coords()).x();
            Some((roughness * roughness).clamp(0.001, 1.0))
        }
    }
}

impl DielectricMaterial {
    /// Create a dielectric (clear) with the given IOR. Tint defaults to white.
    pub fn new(ior: f32) -> Self {
        Self {
            ior,
            roughness: None,
            tint: Arc::new(SolidColor::new(Color3::ONE)),
        }
    }

    /// Create a tinted dielectric (colored glass).
    pub fn tinted(ior: f32, tint: Color3) -> Self {
        Self {
            ior,
            roughness: None,
            tint: Arc::new(SolidColor::new(tint)),
        }
    }

    /// Create a dielectric with a textured tint (spatially varying colored glass).
    pub fn textured(ior: f32, tint: Arc<dyn Texture>) -> Self {
        Self {
            ior,
            roughness: None,
            tint,
        }
    }

    /// Create a rough dielectric. Tint defaults to white.
    pub fn rough(ior: f32, roughness: f32) -> Self {
        Self {
            ior,
            roughness: Some(Arc::new(SolidColor::new(Color3::splat(roughness)))),
            tint: Arc::new(SolidColor::new(Color3::ONE)),
        }
    }

    /// Create a tinted rough dielectric.
    pub fn rough_tinted(ior: f32, roughness: f32, tint: Color3) -> Self {
        Self {
            ior,
            roughness: Some(Arc::new(SolidColor::new(Color3::splat(roughness)))),
            tint: Arc::new(SolidColor::new(tint)),
        }
    }

    /// Create a rough dielectric with textures for `roughness` and `tint`.
    pub fn rough_textured(ior: f32, roughness: Arc<dyn Texture>, tint: Arc<dyn Texture>) -> Self {
        Self {
            ior,
            roughness: Some(roughness),
            tint,
        }
    }
}

impl From<DielectricMaterial> for Material {
    fn from(m: DielectricMaterial) -> Self {
        Material::Dielectric(m)
    }
}

impl GpuSerializable for DielectricMaterial {
    /// Fixed-width layout: 8 f32 params.
    ///
    /// `[tint.rgb, ior, roughness, is_rough, tex(tint), tex(roughness)]`.
    /// `is_rough` = 0 for the smooth delta path (roughness stays 0.0) and
    /// 1 for the rough path. `ior` always bakes.
    fn serialize_gpu(&self, buf: &mut GpuMaterialBuffer) -> u32 {
        let index = buf.nodes.len() as u32;
        let param_offset = buf.params.len() as u32;

        let (tr, tg, tb, tint_tex) = buf.gpu_color(&self.tint);
        // Texture references ride after the baked values. GPU_NONE (u32::MAX)
        // cannot round-trip through f32, so use −1.0 as the "no texture" sentinel.
        let tex = |i: u32| if i == GPU_NONE { -1.0 } else { i as f32 };

        match &self.roughness {
            None => {
                buf.push_params(&[tr, tg, tb, self.ior, 0.0, 0.0]);
                buf.push_params(&[tex(tint_tex), -1.0]);
            }
            Some(roughness) => {
                let (rr, _rg, _rb, rough_tex) = buf.gpu_color(roughness); // channel-0 scalar
                buf.push_params(&[tr, tg, tb, self.ior, rr, 1.0]);
                buf.push_params(&[tex(tint_tex), tex(rough_tex)]);
            }
        }

        buf.nodes.push(GpuMaterialNode {
            material_type: GpuMaterialType::Dielectric as u32,
            param_offset,
            child_a: GPU_NONE,
            child_b: GPU_NONE,
            texture_index: GPU_NONE, // refs ride in params for multi-texture materials
        });
        index
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smooth dielectric (roughness None) is delta: exactly ONE Fresnel-split
    /// draw, then either mirror-reflect (u < F) or transmit (u >= F), with
    /// eta = ior ratio (front face → 1/ior) and f_cos = white tint.
    #[test]
    fn smooth_dielectric_delta_scatter_uses_one_fresnel_draw() {
        let mat = DielectricMaterial::new(1.5);
        let sn = Direction3::new(0.0, 0.0, 1.0);
        let si = SurfaceInteraction::test_surface(&Material::Void, sn);
        let wo = Direction3::new(0.0, 0.0, 1.0); // normal incidence
        let r0 = ((1.0f32 - 1.5) / (1.0 + 1.5)).powi(2); // ≈ 0.04
        let ri = 1.0 / 1.5;

        // u < fresnel(1.0) = r0 → reflect: wi == wo, tint stays, eta = ri.
        let mut draws = 0;
        let mut next_dim = || {
            draws += 1;
            r0 - 0.01 // below r0
        };
        match mat
            .scatter(wo, &si, &mut next_dim)
            .expect("smooth dielectric scatters")
        {
            BsdfScatter::Delta { wi, f_cos, eta } => {
                assert_eq!(draws, 1, "smooth dielectric draws exactly once");
                assert!(
                    (wi.x().abs() < 1e-6) && (wi.y().abs() < 1e-6) && ((wi.z() - 1.0).abs() < 1e-6),
                    "mirror at normal incidence: got {wi:?}"
                );
                assert!((f_cos.x() - 1.0).abs() < 1e-6, "white tint");
                let eta = eta.expect("delta carries the IOR ratio");
                assert!((eta - ri).abs() < 1e-6, "eta = {eta}, expected {ri}");
            }
            other => panic!("expected Delta, got {other:?}"),
        }

        // u >= fresnel(1.0) → refract: wi = (0,0,-1), eta still ri.
        let mut draws = 0;
        let mut next_dim = || {
            draws += 1;
            r0 + 0.01 // above r0
        };
        match mat
            .scatter(wo, &si, &mut next_dim)
            .expect("smooth dielectric scatters")
        {
            BsdfScatter::Delta { wi, f_cos, eta } => {
                assert_eq!(draws, 1, "smooth dielectric draws exactly once");
                assert!(
                    (wi.x().abs() < 1e-6) && (wi.y().abs() < 1e-6) && ((wi.z() + 1.0).abs() < 1e-6),
                    "transmit straight through at normal incidence: got {wi:?}"
                );
                assert!((f_cos.y() - 1.0).abs() < 1e-6, "white tint");
                let eta = eta.expect("delta carries the IOR ratio");
                assert!((eta - ri).abs() < 1e-6, "eta = {eta}, expected {ri}");
            }
            other => panic!("expected Delta, got {other:?}"),
        }
    }

    /// Rough dielectric scatter: exactly 3 draws (2 for the GGX half-vector +
    /// 1 for the Fresnel split), one coupled H shared by both lobes. With
    /// draws (0, 0) the H = (0,0,1) so the split chooses reflect vs transmit
    /// cleanly; an extreme-H reflection below the surface is rejected by the
    /// side check.
    #[test]
    fn rough_dielectric_scatter_draws_three_coupled() {
        let mat = DielectricMaterial::rough(1.5, 0.3);
        let sn = Direction3::new(0.0, 0.0, 1.0);
        let si = SurfaceInteraction::test_surface(&Material::Void, sn);
        let wo = Direction3::new(0.0, 0.0, 1.0);
        let alpha = 0.3f32 * 0.3; // roughness² = 0.09

        // (u=0, v=0) → cosθ_H = 1 → H = (0,0,1). Split 0.01 < fresnel(1) = r0 → reflect.
        let mut draws = 0;
        let mut draws_seq = vec![0.0f32, 0.0, 0.01];
        let mut next_dim = || {
            draws += 1;
            draws_seq.remove(0)
        };
        match mat
            .scatter(wo, &si, &mut next_dim)
            .expect("reflection accepted")
        {
            BsdfScatter::NonDelta { pdf_kinds } => {
                assert_eq!(draws, 3, "exactly 2 H draws + 1 Fresnel split");
                match pdf_kinds[0] {
                    Some(PdfKind::Ggx { alpha: a, .. }) => assert!((a - alpha).abs() < 1e-6),
                    other => panic!("expected Ggx pdf kind, got {other:?}"),
                }
            }
            other => panic!("expected NonDelta, got {other:?}"),
        }

        // Same H, split 0.5 >= fresnel(1) → transmit. Refracted wi = (0,0,-1)
        // is below the surface, matching the transmission decision → accepted.
        let mut draws = 0;
        let mut draws_seq = vec![0.0f32, 0.0, 0.5];
        let mut next_dim = || {
            draws += 1;
            draws_seq.remove(0)
        };
        assert!(
            mat.scatter(wo, &si, &mut next_dim).is_some(),
            "transmission half of the coupled split must be accepted"
        );
        assert_eq!(draws, 3, "exactly 2 H draws + 1 Fresnel split");

        // Extreme tilt (v = 0.995 → cosθ_H ≈ 0.62) with a reflection split:
        // the reflected wi goes below the surface → rejected by the side check.
        let mut draws = 0;
        let mut draws_seq = vec![0.5f32, 0.995, 0.01];
        let mut next_dim = || {
            draws += 1;
            draws_seq.remove(0)
        };
        assert!(
            mat.scatter(wo, &si, &mut next_dim).is_none(),
            "below-surface reflection must be rejected"
        );
        assert_eq!(draws, 3, "draws happen before the side check");
    }

    /// Rough dielectric eval, reflection branch: tint·F·D·G/(4·cosθ_o).
    /// Hand-checkable at normal incidence: D = 1/(π·α²) = 39.2975 (α = 0.09),
    /// G = 1 at cosθ = 1, F = F0 = 0.04 → 0.04·39.2975/4 = 0.39298.
    #[test]
    fn rough_dielectric_eval_reflection_branch() {
        let mat = DielectricMaterial::rough(1.5, 0.3);
        let sn = Direction3::new(0.0, 0.0, 1.0);
        let si = SurfaceInteraction::test_surface(&Material::Void, sn);
        let wo = Direction3::new(0.0, 0.0, 1.0);
        let wi = Direction3::new(0.0, 0.0, 1.0);

        let alpha = 0.3f32 * 0.3;
        let d = ggx_d(1.0, alpha);
        let g = geometry_schlick_ggx(1.0, 0.3) * geometry_schlick_ggx(1.0, 0.3);
        let f = fresnel_schlick(1.0, fresnel_r0(1.5)); // = F0 = 0.04
        let expected = f * d * g / 4.0; // tint = white

        let got = mat.eval(wo, wi, &si);
        assert!(
            (got.x() - expected).abs() < 1e-3,
            "reflection eval: got {}, expected {}",
            got.x(),
            expected
        );
    }

    /// Rough dielectric eval, transmission branch: Walter et al. 2007 Eq. 17
    /// (× |cosθ_i|). At normal incidence h = (0,0,1), denom = η_o + η_i = 2.5,
    /// so expected = (1−F)·D·G·η_i²/(cosθ_o·denom²) = 0.96·39.2975·2.25/6.25.
    #[test]
    fn rough_dielectric_eval_transmission_branch() {
        let mat = DielectricMaterial::rough(1.5, 0.3);
        let sn = Direction3::new(0.0, 0.0, 1.0);
        let si = SurfaceInteraction::test_surface(&Material::Void, sn);
        let wo = Direction3::new(0.0, 0.0, 1.0);
        let wi = Direction3::new(0.0, 0.0, -1.0);

        let alpha = 0.3f32 * 0.3;
        let d = ggx_d(1.0, alpha);
        let g = geometry_schlick_ggx(1.0, 0.3) * geometry_schlick_ggx(1.0, 0.3);
        let f = fresnel_schlick(1.0, fresnel_r0(1.5)); // = F0 = 0.04
        // front_face → eta_o = 1, eta_i = 1.5; denom = 1·1 + 1.5·1 = 2.5.
        let expected = (1.0 - f) * d * g * (1.5f32 * 1.5) / (1.0 * 2.5 * 2.5);

        let got = mat.eval(wo, wi, &si);
        assert!(
            (got.x() - expected).abs() < 1e-3,
            "transmission eval: got {}, expected {}",
            got.x(),
            expected
        );
    }
}
