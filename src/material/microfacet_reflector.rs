//! GGX microfacet reflector — conductor (was Metal) or dielectric (was Glossy).
//!
//! Unifies the former Metal and Glossy materials: both are the Cook-Torrance
//! specular BRDF over a GGX microsurface, differing only in the [`Fresnel`]
//! term and whether the surface carries a color-bearing tint.
//!
//! Models rough metals (gold, copper, aluminium) and dielectrics (plastic,
//! coated wood, ceramic) as a surface covered in tiny perfect mirrors
//! (microfacets). The macro-level shininess comes from the statistical
//! distribution of their orientations.
//!
//! The BRDF is the Cook-Torrance specular model:
//!
//! ```text
//! f(ωo, ωi) = F · D · G / (4 · cos_o · cos_i)
//! ```
//!
//! - **F** (Fresnel): fraction reflected at the microfacet. Full complex-IOR
//!   conductor Fresnel (η + iκ per channel) — this is what gives metals
//!   their color — or Schlick's dielectric approximation scaled by the
//!   surface albedo.
//! - **D** (NDF): GGX/Trowbridge-Reitz — probability that a microfacet has
//!   half-vector H. Controls the specular lobe width.
//! - **G** (Geometry): Smith's shadowing/masking via Schlick-GGX — microfacets
//!   blocking each other at grazing angles.
//!
//! Sampling importance-samples the GGX NDF to generate H, then reflects `wo`
//! about H to get `wi`. This concentrates samples where the BRDF has the most
//! energy, reducing noise.

use std::sync::Arc;

use glam::Vec3;

use crate::hittable::SurfaceInteraction;
use crate::material::gpu::GpuSerializable;
use crate::material::{
    Bsdf, BsdfScatter, Fresnel, GPU_NONE, GpuMaterialBuffer, GpuMaterialNode, GpuMaterialType,
    MAX_BSDF_STRATS, MIRROR_THRESHOLD, PdfKind, fresnel_conductor, fresnel_r0, fresnel_schlick,
    geometry_schlick_ggx, ggx_d, ggx_sample_h,
};
use crate::onb::Onb;
use crate::texture::{SolidColor, Texture};
use crate::vec3::{Color3, Direction3};

use crate::material::Material;

/// Microfacet GGX reflector. The [`Fresnel`] term selects conductor (color
/// from the complex IOR, no albedo) or dielectric (albedo × Schlick) behavior.
#[derive(Clone)]
pub struct MicrofacetReflector {
    /// Fresnel reflectance model — also decides where the surface color comes from.
    pub fresnel: Fresnel,
    /// Dielectric tint × Cook-Torrance BRDF. `None` for conductor (color from Fresnel).
    pub albedo: Option<Arc<dyn Texture>>,
    /// GGX roughness; the alpha is `fuzz²` / `roughness²` (0 = mirror, 1 = fully rough).
    /// Sampled as channel 0 of the texture (scalar convention).
    pub roughness: Arc<dyn Texture>,
}

impl MicrofacetReflector {
    /// Creates a conductor reflector from a complex index of refraction.
    /// `eta` and `k` are the real and imaginary parts of the complex index of
    /// refraction, respectively. Default roughness is 0.5 (moderately rough).
    /// Use `with_roughness` to set a different roughness.
    pub fn conductor(eta: Color3, k: Color3) -> Self {
        Self {
            fresnel: Fresnel::Conductor {
                eta: Arc::new(SolidColor::new(eta)),
                k: Arc::new(SolidColor::new(k)),
            },
            albedo: None,
            roughness: Arc::new(SolidColor::new(Color3::splat(0.5))),
        }
    }

    /// Creates a conductor reflector with textures for `eta`, `k`, and `roughness`.
    pub fn conductor_textured(
        eta: Arc<dyn Texture>,
        k: Arc<dyn Texture>,
        roughness: Arc<dyn Texture>,
    ) -> Self {
        Self {
            fresnel: Fresnel::Conductor { eta, k },
            albedo: None,
            roughness,
        }
    }

    /// Creates a conductor reflector from a dielectric-IOR approximation.
    /// For metals, the extinction coefficient k is often approximated as equal
    /// to the real part of the IOR.
    pub fn conductor_from_ior(ior: Color3, roughness: f32) -> Self {
        Self {
            fresnel: Fresnel::Conductor {
                eta: Arc::new(SolidColor::new(ior)),
                k: Arc::new(SolidColor::new(ior)),
            },
            albedo: None,
            roughness: Arc::new(SolidColor::new(Color3::splat(roughness))),
        }
    }

    /// Fit per-channel IORs so the normal-incidence reflectance equals `base`.
    ///
    /// Inverts F0 = ((η−1)/(η+1))² (the κ = 0 dielectric limit) per channel:
    /// η = (1 + √F0)/(1 − √F0). Exact at normal incidence; the angular
    /// profile is the exact dielectric Fresnel, which Schlick's approximation
    /// tracks to within ~0.5%. This reproduces the legacy tinted-Schlick
    /// metal look (albedo × Fresnel(ior)) and is the artist-friendly way to
    /// target a head-on metal color. True conductors (κ > 0) need
    /// [`Self::conductor`] with explicit `eta`/`k`.
    pub fn conductor_from_reflectance(base: Color3, roughness: f32) -> Self {
        // F0 → 1 would need η → ∞; clamp keeps the fit finite.
        let base = base.into_inner().min(Vec3::splat(0.999));
        let eta = Color3((1.0 + base.sqrt()) / (1.0 - base.sqrt())); // η = (1 + √F0)/(1 − √F0)
        Self {
            fresnel: Fresnel::Conductor {
                eta: Arc::new(SolidColor::new(eta)),
                k: Arc::new(SolidColor::new(Color3::ZERO)),
            },
            albedo: None,
            roughness: Arc::new(SolidColor::new(Color3::splat(roughness))),
        }
    }

    /// Create a dielectric reflector (glossy surface).
    pub fn dielectric(albedo: Color3, roughness: f32, ior: f32) -> Self {
        Self {
            fresnel: Fresnel::Dielectric { ior },
            albedo: Some(Arc::new(SolidColor::new(albedo))),
            roughness: Arc::new(SolidColor::new(Color3::splat(roughness))),
        }
    }

    /// Create a dielectric reflector with textures for `albedo` and `roughness`.
    pub fn dielectric_textured(
        albedo: Arc<dyn Texture>,
        roughness: Arc<dyn Texture>,
        ior: f32,
    ) -> Self {
        Self {
            fresnel: Fresnel::Dielectric { ior },
            albedo: Some(albedo),
            roughness,
        }
    }

    /// Sets the roughness of the material. Roughness should be in the range [0, 1], where 0 is a
    /// perfect mirror and 1 is fully rough.
    pub fn with_roughness(mut self, roughness: f32) -> Self {
        self.roughness = Arc::new(SolidColor::new(Color3::splat(roughness)));
        self
    }
}

impl From<MicrofacetReflector> for Material {
    fn from(m: MicrofacetReflector) -> Self {
        Material::MicrofacetReflector(m)
    }
}

impl Bsdf for MicrofacetReflector {
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
            return match &self.fresnel {
                Fresnel::Conductor { eta, k } => {
                    let cos_i = wi.dot(si.shading_normal().into_inner()).max(0.0);
                    let eta = eta.value(&si.texture_coords());
                    let k = k.value(&si.texture_coords());
                    let f = fresnel_conductor(cos_i, eta, k);
                    // Mirror throughput: f_cos = f · cos_i = F. The delta maps solid
                    // angle 1:1, so the Cook-Torrance 1/(4·cos_o·cos_i) denominator
                    // cancels against the reflection Jacobian — only the Fresnel
                    // reflectance remains (D = G = 1 for a perfect mirror).
                    // The conductor Fresnel is already colored, so there is no
                    // albedo multiply.
                    Some(BsdfScatter::Delta {
                        wi,
                        f_cos: f,
                        eta: None, // No change in medium for a perfect mirror.
                    })
                }
                Fresnel::Dielectric { ior } => {
                    let cos_o = wo.dot(si.shading_normal().into_inner()).max(0.0);
                    let f = fresnel_schlick(cos_o, fresnel_r0(*ior));
                    let albedo = self.albedo.as_ref()?.value(&si.texture_coords());
                    // Delta (mirror) throughput: f_cos = albedo · F — the delta maps
                    // solid angle 1:1, so the Cook-Torrance 1/(4·cos_o·cos_i)
                    // denominator cancels against the reflection Jacobian. Albedo is
                    // kept because Schlick F is colorless; the conductor uses the
                    // complex Fresnel (already colored) and drops its albedo.
                    Some(BsdfScatter::Delta {
                        wi,
                        f_cos: albedo * f,
                        eta: None,
                    })
                }
            };
        }

        let alpha = self.ggx_alpha(si)?;
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

    /// Cook-Torrance BRDF as `f_cos` (× cos_i folded into the estimator):
    /// `F · D · G / (4 · cos_o)`. For a dielectric, `F` is `albedo × Schlick`.
    fn eval(&self, wo: Direction3, wi: Direction3, si: &SurfaceInteraction) -> Color3 {
        // If the material is effectively a mirror, return zero for arbitrary directions.
        if self.is_delta() {
            return Color3::ZERO;
        }
        // Invariant: is_delta() returned false above, so ggx_alpha is Some.
        let alpha = self
            .ggx_alpha(si)
            .expect("non-delta roughness => Some alpha");
        let h = (wo + wi).normalize();
        let cos_h_n = h.dot(si.shading_normal().into_inner()).max(0.0);
        let cos_h_o = wo.dot(h.into_inner()).max(0.0);
        let cos_o = wo.dot(si.shading_normal().into_inner()).max(0.0);
        let cos_i = wi.dot(si.shading_normal().into_inner()).max(0.0);
        if cos_h_o <= 0.0 || cos_o <= 0.0 || cos_i <= 0.0 {
            return Color3::ZERO;
        }
        let d = ggx_d(cos_h_n, alpha);
        let roughness = self.roughness.value(&si.texture_coords()).x();
        let g = geometry_schlick_ggx(cos_o, roughness) * geometry_schlick_ggx(cos_i, roughness);

        let f = match &self.fresnel {
            Fresnel::Conductor { eta, k } => fresnel_conductor(
                cos_h_o,
                eta.value(&si.texture_coords()),
                k.value(&si.texture_coords()),
            ),
            Fresnel::Dielectric { ior } => {
                // Invariant: dielectric constructors always set Some(albedo);
                // conductors (albedo = None) take the Conductor arm above, so this
                // branch never sees a missing albedo. Debug builds assert that
                // explicitly; release builds fall back to white instead of panicking.
                let albedo = if let Some(albedo) = &self.albedo {
                    albedo.value(&si.texture_coords())
                } else {
                    debug_assert!(
                        self.albedo.is_some(),
                        "dielectric MicrofacetReflector has no albedo"
                    );
                    Color3::ONE
                };
                albedo * fresnel_schlick(cos_h_o, fresnel_r0(*ior))
            }
        };

        f * d * g / (4.0 * cos_o)
    }

    /// GGX NDF sampling PDF: `D(H) · cos(H·N) / (4 · cos(H·O))`.
    fn pdf(&self, wo: Direction3, wi: Direction3, si: &SurfaceInteraction) -> f32 {
        // If the material is effectively a mirror, return zero for arbitrary directions.
        if self.is_delta() {
            return 0.0;
        }
        // Invariant: is_delta() returned false above, so ggx_alpha is Some.
        let alpha = self
            .ggx_alpha(si)
            .expect("non-delta roughness => Some alpha");
        let h = (wo + wi).normalize();
        let cos_h_n = h.dot(si.shading_normal().into_inner()).max(0.0);
        let cos_h_o = wo.dot(h.into_inner()).max(0.0);
        if cos_h_o <= 0.0 {
            return 0.0;
        }
        ggx_d(cos_h_n, alpha) * cos_h_n / (4.0 * cos_h_o)
    }

    /// Returns the PDF kind for the GGX NDF if the surface is not a near-mirror,
    /// otherwise `None`.
    fn pdf_kind(&self, wo: Direction3, si: &SurfaceInteraction) -> Option<PdfKind> {
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

    /// Returns an estimate of the material's reflectance for a given outgoing
    /// direction. For a smooth surface, this is approximately the Fresnel term
    /// with a roughness boost. For rough surfaces, the effective reflectance is
    /// higher due to multiple scattering.
    fn reflectance_estimate(&self, wo: Direction3, si: &SurfaceInteraction) -> f32 {
        let cos_theta = wo.dot(si.shading_normal().into_inner()).abs();
        match &self.fresnel {
            Fresnel::Conductor { eta, k } => {
                // For a smooth conductor, reflectance ≈ Fresnel(θ) — the GGX lobe
                // is narrow, so most energy is at the mirror direction. Roughness
                // increases the effective reflectance due to multiple scattering.
                let f = fresnel_conductor(
                    cos_theta,
                    eta.value(&si.texture_coords()),
                    k.value(&si.texture_coords()),
                )
                .into_inner()
                .max_element();
                let roughness_boost = self.roughness.value(&si.texture_coords()).x() * 0.25;
                (f + roughness_boost).min(1.0)
            }
            Fresnel::Dielectric { ior } => {
                let albedo = self.albedo.as_ref().map(|a| a.value(&si.texture_coords()));
                let albedo_avg = albedo
                    .map(|a| a.into_inner().element_sum() / 3.0)
                    .unwrap_or(0.0);
                let f = fresnel_schlick(cos_theta, fresnel_r0(*ior));
                // Base color × Fresnel gives the specular reflectance; roughness adds
                // a small boost from multiple scattering making the surface appear brighter.
                let roughness_boost = self.roughness.value(&si.texture_coords()).x() * 0.25;
                (albedo_avg * f + roughness_boost).min(1.0)
            }
        }
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

impl GpuSerializable for MicrofacetReflector {
    /// Fixed-width layout: 15 f32 params.
    ///
    /// `[albedo.rgb, roughness, fresnel_kind, eta.rgb, k.rgb, tex(albedo),
    /// tex(roughness), tex(eta), tex(k)]`. For a conductor the albedo slots are
    /// zero (color comes from the Fresnel term); for a dielectric the eta slots
    /// carry the IOR splat and k is zero.
    fn serialize_gpu(&self, buf: &mut GpuMaterialBuffer) -> u32 {
        let index = buf.nodes.len() as u32;
        let param_offset = buf.params.len() as u32;

        let (ar, ag, ab, albedo_tex) = match &self.albedo {
            Some(albedo) => buf.gpu_color(albedo),
            None => (0.0, 0.0, 0.0, GPU_NONE),
        };
        let (rr, _rg, _rb, rough_tex) = buf.gpu_color(&self.roughness); // channel-0 scalar

        let (fresnel_kind, er, eg, eb, eta_tex, kr, kg, kb, k_tex) = match &self.fresnel {
            Fresnel::Conductor { eta, k } => {
                let (er, eg, eb, eta_tex) = buf.gpu_color(eta);
                let (kr, kg, kb, k_tex) = buf.gpu_color(k);
                (0.0, er, eg, eb, eta_tex, kr, kg, kb, k_tex)
            }
            Fresnel::Dielectric { ior } => {
                (1.0, *ior, *ior, *ior, GPU_NONE, 0.0, 0.0, 0.0, GPU_NONE)
            }
        };

        buf.push_params(&[ar, ag, ab, rr, fresnel_kind, er, eg, eb, kr, kg, kb]);
        // Texture references ride after the baked values. GPU_NONE (u32::MAX)
        // cannot round-trip through f32, so use −1.0 as the "no texture" sentinel.
        let tex = |i: u32| if i == GPU_NONE { -1.0 } else { i as f32 };
        buf.push_params(&[tex(albedo_tex), tex(rough_tex), tex(eta_tex), tex(k_tex)]);

        buf.nodes.push(GpuMaterialNode {
            material_type: GpuMaterialType::MicrofacetReflector as u32,
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
    use std::f32::consts::FRAC_1_SQRT_2;

    use super::*;

    use crate::material::fresnel_conductor;

    /// At normal incidence the conductor Fresnel must equal the closed form
    /// F0 = ((η−1)²+κ²)/((η+1)²+κ²) — catches cross-term algebra slips.
    #[test]
    fn fresnel_conductor_f0_matches_closed_form() {
        for (eta, k) in [
            (0.183, 3.424),
            (0.421, 2.346),
            (1.374, 1.771),
            (1.5, 0.0),
            (2.5, 2.5),
        ] {
            let got = fresnel_conductor(1.0, Color3::splat(eta), Color3::splat(k)).x();
            let expected = ((eta - 1.0).powi(2) + k * k) / ((eta + 1.0).powi(2) + k * k);
            assert!(
                (got - expected).abs() < 1e-5,
                "eta={eta}, k={k}: got {got}, expected {expected}"
            );
        }
    }

    /// Gold's tint must fall out of the physics: F0 per channel derived from
    /// the closed form ((η−1)²+κ²)/((η+1)²+κ²) for the standard measured
    /// RGB complex IOR. This is the test that catches the κ ≠ 0 bug (using
    /// √(u²+v²) instead of u in the cross term).
    #[test]
    fn fresnel_conductor_gold_tint() {
        let eta = Color3::new(0.183, 0.421, 1.374);
        let k = Color3::new(3.424, 2.346, 1.771);
        let f = fresnel_conductor(1.0, eta, k);
        // Closed-form values: R = 12.391/13.123 = 0.9442, G = 5.839/7.523 =
        // 0.7762, B = 3.276/8.772 = 0.3735.
        let expected = [0.9442, 0.7762, 0.3735];
        for (c, &exp) in expected.iter().enumerate() {
            assert!(
                (f.into_inner()[c] - exp).abs() < 1e-3,
                "channel {c}: got {}, expected {}",
                f.into_inner()[c],
                exp
            );
        }
    }

    /// With κ = 0 the conductor Fresnel must equal the analytic unpolarized
    /// dielectric Fresnel (Snell-based) at every angle — NOT Schlick.
    #[test]
    fn fresnel_conductor_matches_dielectric_at_k_zero() {
        let eta: f32 = 1.5;
        for cos_theta in [0.1_f32, 0.3, 0.5, FRAC_1_SQRT_2, 0.9, 0.99] {
            let sin2 = 1.0 - cos_theta * cos_theta;
            let cos_t = (1.0 - sin2 / (eta * eta)).sqrt(); // Snell: cos of transmitted angle
            let rs = ((cos_theta - eta * cos_t) / (cos_theta + eta * cos_t)).powi(2);
            let rp = ((cos_t - eta * cos_theta) / (cos_t + eta * cos_theta)).powi(2);
            let expected = 0.5 * (rs + rp);
            let got = fresnel_conductor(cos_theta, Color3::splat(eta), Color3::ZERO).x();
            assert!(
                (got - expected).abs() < 1e-5,
                "cos={cos_theta}: got {got}, expected {expected}"
            );
        }
    }

    /// Grazing incidence reflects essentially everything; reflectance rises
    /// monotonically as incidence approaches grazing.
    ///
    /// Note: F is NOT globally monotone in cosθ — the p-polarized component
    /// dips below F0 near the pseudo-Brewster angle, dragging the unpolarized
    /// average slightly below its normal-incidence value. Only the grazing
    /// limit (→ 1) and the rise toward grazing are guaranteed.
    #[test]
    fn fresnel_conductor_grazing_and_monotonicity() {
        let eta = Color3::splat(2.5);
        let k = Color3::splat(2.5);
        assert!(fresnel_conductor(1e-6, eta, k).x() > 0.999);
        // Rises toward grazing (θ=84° vs θ=60°), for both the synthetic
        // eta=k=2.5 case and real gold.
        let f_graz = fresnel_conductor(0.1, eta, k).x();
        let f_mid = fresnel_conductor(0.5, eta, k).x();
        assert!(
            f_graz > f_mid,
            "F(0.1)={f_graz} should exceed F(0.5)={f_mid}"
        );
        let gold = (
            Color3::new(0.183, 0.421, 1.374),
            Color3::new(3.424, 2.346, 1.771),
        );
        let g_graz = fresnel_conductor(0.1, gold.0, gold.1).x();
        let g_mid = fresnel_conductor(0.5, gold.0, gold.1).x();
        assert!(
            g_graz > g_mid,
            "gold F(0.1)={g_graz} should exceed F(0.5)={g_mid}"
        );
    }

    /// from_reflectance fits per-channel IORs so F0 == base exactly (κ = 0
    /// dielectric limit) — the equivalence guarantee for the scene migration.
    #[test]
    fn from_reflectance_reproduces_base_f0() {
        for base in [
            Color3::new(0.147, 0.147, 0.165),
            Color3::new(0.2, 0.4, 0.6),
            Color3::splat(0.999), // clamp boundary
        ] {
            let m = MicrofacetReflector::conductor_from_reflectance(base, 0.3);
            let eta = match &m.fresnel {
                Fresnel::Conductor { eta, .. } => eta.as_constant().expect("constant eta"),
                _ => panic!("expected conductor fresnel"),
            };
            let f = fresnel_conductor(1.0, eta, Color3::ZERO);
            for c in 0..3 {
                let expected = base.into_inner()[c];
                let got = f.into_inner()[c];
                let tol = if expected > 0.99 { 1e-3 } else { 1e-5 };
                assert!(
                    (got - expected).abs() < tol,
                    "channel {c}: got {got}, expected {expected}"
                );
            }
        }
    }

    /// Conductor delta path (roughness < MIRROR_THRESHOLD): mirror reflection
    /// with ZERO RNG draws. `f_cos` = conductor Fresnel only (no albedo
    /// multiply — the color comes from η/κ), `eta` = None (same medium).
    #[test]
    fn conductor_delta_scatter_is_zero_draw_specular() {
        // base 0.5 → per-channel F0 = 0.5 by construction of the reflectance fit.
        let mat = MicrofacetReflector::conductor_from_reflectance(Color3::splat(0.5), 0.0);
        let sn = Direction3::new(0.0, 0.0, 1.0);
        let si = SurfaceInteraction::test_surface(&Material::Void, sn);
        let wo = Direction3::new(0.0, 0.0, 1.0); // normal incidence

        let mut draws = 0;
        let mut next_dim = || {
            draws += 1;
            0.5 // must never be consumed
        };
        match mat.scatter(wo, &si, &mut next_dim).expect("delta scatters") {
            BsdfScatter::Delta { wi, f_cos, eta } => {
                assert_eq!(draws, 0, "conductor delta path must draw no RNG values");
                // Mirror at normal incidence maps wo onto itself.
                assert!(
                    (wi.x().abs() < 1e-6) && (wi.y().abs() < 1e-6) && ((wi.z() - 1.0).abs() < 1e-6),
                    "got {wi:?}"
                );
                for c in 0..3 {
                    assert!(
                        (f_cos.into_inner()[c] - 0.5).abs() < 1e-4,
                        "channel {c}: got {}, expected F0 = 0.5",
                        f_cos.into_inner()[c]
                    );
                }
                assert!(eta.is_none(), "conductor reflection keeps the same medium");
            }
            other => panic!("expected Delta scatter, got {other:?}"),
        }
    }

    /// Dielectric delta path: also ZERO RNG draws, but f_cos = albedo × Schlick
    /// (the tint multiplies because Schlick F is colorless — the asymmetry with
    /// the conductor path, where color comes from the complex Fresnel).
    #[test]
    fn dielectric_delta_scatter_is_zero_draw_albedo_times_schlick() {
        let albedo = Color3::new(0.8, 0.4, 0.2);
        let mat = MicrofacetReflector::dielectric(albedo, 0.0, 1.5);
        let sn = Direction3::new(0.0, 0.0, 1.0);
        let si = SurfaceInteraction::test_surface(&Material::Void, sn);
        let wo = Direction3::new(0.0, 0.0, 1.0); // normal incidence

        let mut draws = 0;
        let mut next_dim = || {
            draws += 1;
            0.5 // must never be consumed
        };
        match mat.scatter(wo, &si, &mut next_dim).expect("delta scatters") {
            BsdfScatter::Delta { wi, f_cos, eta } => {
                assert_eq!(draws, 0, "dielectric delta path must draw no RNG values");
                assert!(
                    (wi.x().abs() < 1e-6) && (wi.y().abs() < 1e-6) && ((wi.z() - 1.0).abs() < 1e-6),
                    "got {wi:?}"
                );
                // F0 = ((η−1)/(η+1))² at η = 1.5 → 0.04.
                let r0 = ((1.0f32 - 1.5) / (1.0 + 1.5)).powi(2);
                let expected = albedo * r0;
                for c in 0..3 {
                    assert!(
                        (f_cos.into_inner()[c] - expected.into_inner()[c]).abs() < 1e-6,
                        "channel {c}: got {}, expected {}",
                        f_cos.into_inner()[c],
                        expected.into_inner()[c]
                    );
                }
                assert!(eta.is_none(), "dielectric delta keeps the same medium");
            }
            other => panic!("expected Delta scatter, got {other:?}"),
        }
    }

    /// eval (conductor) = F·D·G/(4·cosθ_o) at known normal-incidence inputs.
    /// Hand-checkable: D = 1/(π·α²) = 5.0930 at α = 0.25, G = 1 at cosθ = 1,
    /// F = 0.5 from the reflectance fit → 0.5·5.0930/4 = 0.6366.
    #[test]
    fn eval_matches_cook_torrance_conductor() {
        let mat = MicrofacetReflector::conductor_from_reflectance(Color3::splat(0.5), 0.5);
        let sn = Direction3::new(0.0, 0.0, 1.0);
        let si = SurfaceInteraction::test_surface(&Material::Void, sn);
        let wo = Direction3::new(0.0, 0.0, 1.0);
        let wi = Direction3::new(0.0, 0.0, 1.0);

        let eta = match &mat.fresnel {
            Fresnel::Conductor { eta, .. } => eta.as_constant().expect("constant eta"),
            _ => panic!("expected conductor fresnel"),
        };
        let alpha = (0.5f32 * 0.5).clamp(0.001, 1.0);
        let d = ggx_d(1.0, alpha);
        let g = geometry_schlick_ggx(1.0, 0.5) * geometry_schlick_ggx(1.0, 0.5);
        let f = fresnel_conductor(1.0, eta, Color3::ZERO);
        let expected = f * d * g / 4.0;

        let got = mat.eval(wo, wi, &si);
        for c in 0..3 {
            assert!(
                (got.into_inner()[c] - expected.into_inner()[c]).abs() < 1e-4,
                "channel {c}: got {}, expected {}",
                got.into_inner()[c],
                expected.into_inner()[c]
            );
        }
    }

    /// eval (dielectric) = albedo × Schlick · D·G/(4·cosθ_o): the dielectric
    /// Fresnel is colorless, so the albedo tint rides on F.
    #[test]
    fn eval_matches_cook_torrance_dielectric() {
        let albedo = Color3::new(0.7, 0.3, 0.1);
        let mat = MicrofacetReflector::dielectric(albedo, 0.5, 1.5);
        let sn = Direction3::new(0.0, 0.0, 1.0);
        let si = SurfaceInteraction::test_surface(&Material::Void, sn);
        let wo = Direction3::new(0.0, 0.0, 1.0);
        let wi = Direction3::new(0.0, 0.0, 1.0);

        let alpha = (0.5f32 * 0.5).clamp(0.001, 1.0);
        let d = ggx_d(1.0, alpha);
        let g = geometry_schlick_ggx(1.0, 0.5) * geometry_schlick_ggx(1.0, 0.5);
        let f = fresnel_schlick(1.0, fresnel_r0(1.5)); // = F0 = 0.04
        let expected = albedo * f * d * g / 4.0;

        let got = mat.eval(wo, wi, &si);
        for c in 0..3 {
            assert!(
                (got.into_inner()[c] - expected.into_inner()[c]).abs() < 1e-4,
                "channel {c}: got {}, expected {}",
                got.into_inner()[c],
                expected.into_inner()[c]
            );
        }
    }
}
