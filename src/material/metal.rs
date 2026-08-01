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
//! - **F** (Fresnel): fraction reflected at the microfacet. Full complex-IOR
//!   conductor Fresnel (η + iκ per channel) — this is what gives metals
//!   their color.
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
use crate::material::Material;
use crate::material::gpu::{GPU_NONE, GpuSerializable};
use crate::material::{
    Bsdf, BsdfScatter, GpuMaterialBuffer, GpuMaterialNode, GpuMaterialType, MAX_BSDF_STRATS,
    MIRROR_THRESHOLD, PdfKind, geometry_schlick_ggx, ggx_d, ggx_sample_h,
};
use crate::onb::Onb;
use crate::texture::{SolidColor, Texture};
use crate::vec3::{Color3, Direction3};

/// Fresnel reflectance for a complex refractive index (conductor).
///
/// `eta` is the real part of the index and `k` the imaginary part (extinction coefficient): η̃ = η +
/// iκ. The per-channel extinction is what gives metals their color — at normal incidence this
/// reduces to F0 = ((η−1)² + κ²) / ((η+1)² + κ²), and with κ = 0 it degenerates to the exact
/// dielectric Fresnel equations. Grazing incidence → 1.
///
/// Returns the unpolarized reflectance, the average of the s and p polarization components.
fn fresnel_conductor(cos_theta: f32, eta: Color3, k: Color3) -> Color3 {
    let eta = eta.into_inner();
    let k = k.into_inner();

    let cos_theta = cos_theta.clamp(0.0, 1.0);
    let cos_theta2 = cos_theta * cos_theta;
    let sin_theta2 = 1.0 - cos_theta2;
    let eta2 = eta * eta;
    let k2 = k * k;

    // w = √(ε̃ − sin²θ) = u + iv; disc = u² + v² = |w|².
    let t0 = eta2 - k2 - Vec3::splat(sin_theta2);
    let disc = (t0 * t0 + 4.0 * eta2 * k2).sqrt();
    let u = ((disc + t0) * 0.5).sqrt(); // real part of w — NOT √disc

    let rs = (disc + Vec3::splat(cos_theta2) - 2.0 * cos_theta * u)
        / (disc + Vec3::splat(cos_theta2) + 2.0 * cos_theta * u);

    let t3 = disc * Vec3::splat(cos_theta2) + Vec3::splat(sin_theta2 * sin_theta2);
    let t4 = 2.0 * cos_theta * sin_theta2 * u;
    let rp = rs * (t3 - t4) / (t3 + t4);

    Color3((rs + rp) * 0.5)
}

/// Microfacet conductor BRDF (GGX).
#[derive(Clone)]
pub struct MetalMaterial {
    /// Real part of the complex index of refraction, per RGB channel.
    pub eta: Arc<dyn Texture>,
    /// Imaginary part (extinction coefficient), per RGB channel.
    pub k: Arc<dyn Texture>,
    /// GGX roughness; the alpha is `fuzz²` (0 = mirror, 1 = fully rough).
    /// Sampled as channel 0 of the texture (scalar convention).
    pub roughness: Arc<dyn Texture>,
}

impl MetalMaterial {
    /// Creates a new `MetalMaterial`.
    /// `eta` and `k` are the real and imaginary parts of the complex index of refraction, respectively.
    /// Default roughness is 0.5 (moderately rough). Use `with_roughness` to set a different roughness.
    pub fn new(eta: Color3, k: Color3) -> Self {
        Self {
            eta: Arc::new(SolidColor::new(eta)),
            k: Arc::new(SolidColor::new(k)),
            roughness: Arc::new(SolidColor::new(Color3::splat(0.5))),
        }
    }

    pub fn from_ior(ior: Color3, roughness: f32) -> Self {
        // For metals, the extinction coefficient k is often approximated as equal to the real part of the IOR.
        Self {
            eta: Arc::new(SolidColor::new(ior)),
            k: Arc::new(SolidColor::new(ior)),
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
    /// [`Self::new`] with explicit `eta`/`k`.
    pub fn from_reflectance(base: Color3, roughness: f32) -> Self {
        // F0 → 1 would need η → ∞; clamp keeps the fit finite.
        let base = base.into_inner().min(Vec3::splat(0.999));
        let eta = Color3((1.0 + base.sqrt()) / (1.0 - base.sqrt())); // η = (1 + √F0)/(1 − √F0)
        Self {
            eta: Arc::new(SolidColor::new(eta)),
            k: Arc::new(SolidColor::new(Color3::ZERO)),
            roughness: Arc::new(SolidColor::new(Color3::splat(roughness))),
        }
    }

    /// Creates a new `MetalMaterial` with textures for `eta`, `k`, and `roughness`.
    pub fn textured(
        eta: Arc<dyn Texture>,
        k: Arc<dyn Texture>,
        roughness: Arc<dyn Texture>,
    ) -> Self {
        Self { eta, k, roughness }
    }

    /// Sets the roughness of the material. Roughness should be in the range [0, 1], where 0 is a
    /// perfect mirror and 1 is fully rough.
    pub fn with_roughness(mut self, roughness: f32) -> Self {
        self.roughness = Arc::new(SolidColor::new(Color3::splat(roughness)));
        self
    }
}

impl From<MetalMaterial> for Material {
    fn from(m: MetalMaterial) -> Self {
        Material::Metal(m)
    }
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
            let cos_i = wi.dot(si.shading_normal().into_inner()).max(0.0);

            let eta = self.eta.value(&si.texture_coords());
            let k = self.k.value(&si.texture_coords());

            let f = fresnel_conductor(cos_i, eta, k);
            // Mirror throughput: f_cos = f · cos_i = F. The delta maps solid
            // angle 1:1, so the Cook-Torrance 1/(4·cos_o·cos_i) denominator
            // cancels against the reflection Jacobian — only the Fresnel
            // reflectance remains (D = G = 1 for a perfect mirror).
            let f_cos = f;

            return Some(BsdfScatter::Delta {
                wi,
                f_cos,
                eta: None, // No change in medium for a perfect mirror.
            });
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
    /// `F · D · G / (4 · cos_o)`.
    fn eval(&self, wo: Direction3, wi: Direction3, si: &SurfaceInteraction) -> Color3 {
        // If the material is effectively a mirror, return zero for arbitrary directions.
        if self.is_delta() {
            return Color3::ZERO;
        }
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
        let f = fresnel_conductor(
            cos_h_o,
            self.eta.value(&si.texture_coords()),
            self.k.value(&si.texture_coords()),
        );
        let roughness = self.roughness.value(&si.texture_coords()).x();
        let g = geometry_schlick_ggx(cos_o, roughness) * geometry_schlick_ggx(cos_i, roughness);

        f * d * g / (4.0 * cos_o)
    }

    /// GGX NDF sampling PDF: `D(H) · cos(H·N) / (4 · cos(H·O))`.
    fn pdf(&self, wo: Direction3, wi: Direction3, si: &SurfaceInteraction) -> f32 {
        // If the material is effectively a mirror, return zero for arbitrary directions.
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

    /// Returns the PDF kind for the GGX NDF if `fuzz` is non-zero, otherwise `None`.
    fn pdf_kind(&self, wo: Direction3, si: &SurfaceInteraction) -> Option<PdfKind> {
        if self.is_delta() {
            None
        } else {
            let roughness = self.roughness.value(&si.texture_coords()).x();

            Some(PdfKind::Ggx {
                wo,
                normal: si.shading_normal(),
                alpha: (roughness * roughness).clamp(0.001, 1.0),
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
        let f = fresnel_conductor(
            cos_theta,
            self.eta.value(&si.texture_coords()),
            self.k.value(&si.texture_coords()),
        )
        .into_inner()
        .max_element();
        let roughness_boost = self.roughness.value(&si.texture_coords()).x() * 0.25;
        (f + roughness_boost).min(1.0)
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

impl GpuSerializable for MetalMaterial {
    fn serialize_gpu(&self, buf: &mut GpuMaterialBuffer) -> u32 {
        let index = buf.nodes.len() as u32;
        let param_offset = buf.params.len() as u32;

        // Each parameter bakes to (r,g,b) when constant, or serializes into
        // buf.textures and returns its node index (GPU_NONE when the texture
        // has no GPU representation).
        let (er, eg, eb, eta_tex) = buf.gpu_color(&self.eta);
        let (kr, kg, kb, k_tex) = buf.gpu_color(&self.k);
        let (rr, rg, rb, r_tex) = buf.gpu_color(&self.roughness);

        buf.push_params(&[er, eg, eb, kr, kg, kb, rr, rg, rb]);
        // Texture references ride after the baked colors. GPU_NONE (u32::MAX)
        // cannot round-trip through f32, so use −1.0 as the "no texture" sentinel.
        let tex = |i: u32| if i == GPU_NONE { -1.0 } else { i as f32 };
        buf.push_params(&[tex(eta_tex), tex(k_tex), tex(r_tex)]);

        buf.nodes.push(GpuMaterialNode {
            material_type: GpuMaterialType::Metal as u32,
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
        for c in 0..3 {
            assert!(
                (f.into_inner()[c] - expected[c]).abs() < 1e-3,
                "channel {c}: got {}, expected {}",
                f.into_inner()[c],
                expected[c]
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
            let m = MetalMaterial::from_reflectance(base, 0.3);
            let eta = m.eta.as_constant().expect("constant eta");
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
}
