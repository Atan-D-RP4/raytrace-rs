//! Material models and BSDF sampling for path tracing.
//!
//! [`Material`] is a recursive enum — composition variants ([`Material::Mix`],
//! [`Material::Coated`]) wrap dedicated structs ([`MixMaterial`], [`CoatedMaterial`])
//! that contain `Arc<Material>` children. No cycles by construction.
//!
//! # Authoring
//!
//! ```ignore
//! use std::sync::Arc;
//! use raytrace_rs::material::{
//!     DielectricMaterial, DiffuseReflector, Material, MicrofacetReflector,
//! };
//! use raytrace_rs::vec3::Color3;
//!
//! let red = Material::from(DiffuseReflector::new(Color3::new(0.8, 0.2, 0.2)));
//! let paint = red.mix(
//!     Material::from(MicrofacetReflector::conductor(
//!         Color3::new(0.9, 0.9, 0.9),
//!         Color3::new(3.4, 2.3, 1.8),
//!     )),
//!     0.5,
//! );
//! let car_paint = red.coated(Material::from(DielectricMaterial::new(1.5)));
//! ```
//!
//! # Extensibility
//!
//! Implement [`Bsdf`] for custom materials, wrap in [`Material::Custom`]:
//!
//! ```ignore
//! struct MyBrdf { ... }
//! impl Bsdf for MyBrdf { ... }
//! let mat = Material::custom(MyBrdf { ... });
//! ```
//!
//! # GPU Serialization
//!
//! Flatten via [`Material::to_gpu_buffer`] into [`GpuMaterialNode`]s with
//! child indices. Custom materials serialize as `Passthrough` (not uploaded).

use std::sync::Arc;

use crate::intersect::interaction::SurfaceInteraction;
use crate::math::vec3::{Color3, Direction3};
use crate::sampling::pdf::{PdfKind, ggx_d, ggx_sample_h};
use crate::texture::Texture;

mod coated;
mod dielectric;
mod diffuse_emitter;
mod diffuse_reflector;
mod gpu;
mod isotropic;
mod microfacet_reflector;
mod mix;

use glam::Vec3;
use gpu::GPU_NONE;

pub use coated::CoatedMaterial;
pub use dielectric::DielectricMaterial;
pub use diffuse_emitter::DiffuseEmitterMaterial;
pub use diffuse_reflector::DiffuseReflector;
pub use gpu::{GpuMaterialBuffer, GpuMaterialNode, GpuMaterialType, GpuSerializable};
pub use isotropic::IsotropicMaterial;
pub use microfacet_reflector::MicrofacetReflector;
pub use mix::MixMaterial;

/// Maximum number of BSDF sampling strategies produced by any material.
/// Used as the fixed capacity for `BsdfScatter::NonDelta` / `Split`.
/// Current max ~2 (Mix + Coated); bumped to 4 as safety margin.
pub const MAX_BSDF_STRATS: usize = 4;

const MIRROR_THRESHOLD: f32 = 0.01;

/// Fresnel reflectance model for a [`MicrofacetReflector`].
///
/// Selects both the reflection probability at the microfacet level and where
/// the surface color comes from.
#[derive(Clone)]
pub enum Fresnel {
    /// Complex refractive index (conductor: η + iκ per RGB channel).
    /// Color comes from the Fresnel term itself — no albedo multiply.
    Conductor {
        eta: Arc<dyn Texture>,
        k: Arc<dyn Texture>,
    },
    /// Dielectric Fresnel (Schlick approximation with scalar IOR).
    /// Used by the dielectric path of MicrofacetReflector (albedo × Schlick)
    /// and by DielectricMaterial.
    Dielectric { ior: f32 },
}

/// Fresnel reflectance for a complex refractive index (conductor).
///
/// `eta` is the real part of the index and `k` the imaginary part (extinction coefficient): η̃ = η +
/// iκ. The per-channel extinction is what gives metals their color — at normal incidence this
/// reduces to F0 = ((η−1)² + κ²) / ((η+1)² + κ²), and with κ = 0 it degenerates to the exact
/// dielectric Fresnel equations. Grazing incidence → 1.
///
/// Returns the unpolarized reflectance, the average of the s and p polarization components.
pub(super) fn fresnel_conductor(cos_theta: f32, eta: Color3, k: Color3) -> Color3 {
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

/// Smith's geometry function (Schlick-GGX approximation).
///
/// Models microfacet self-shadowing at grazing angles. Returns a multiplier
/// in [0, 1]. `roughness` is RMS surface slope (not squared).
pub(super) fn geometry_schlick_ggx(cos_theta: f32, roughness: f32) -> f32 {
    if cos_theta <= 0.0 {
        return 0.0;
    }
    let k = (roughness + 1.0).powi(2) / 8.0;
    cos_theta / (cos_theta * (1.0 - k) + k)
}

/// Precomputed Fresnel reflectance at normal incidence for a given IOR.
#[inline(always)]
pub(super) fn fresnel_r0(ior: f32) -> f32 {
    ((1.0 - ior) / (1.0 + ior)).powi(2)
}

/// Schlick Fresnel reflectance for unpolarized light.
///
/// Approximates the fraction of light reflected at a dielectric interface.
/// Approaches 1 at grazing angles. `r0` is reflectance at normal incidence.
pub(super) fn fresnel_schlick(cos_theta: f32, r0: f32) -> f32 {
    r0 + (1.0 - r0) * (1.0 - cos_theta).powi(5)
}

/// Convert a blackbody temperature (Kelvin) to an RGB color approximation.
pub(super) fn blackbody(temp: f32) -> Color3 {
    // Planck's law: spectral radiance of a blackbody at temperature T.
    // This is a simplified approximation for RGB color. For more accurate
    // rendering, use spectral rendering or a proper color matching function.
    let t = temp.clamp(1000.0, 10000.0);
    let r = (t / 1000.0).powf(3.0) * 0.5;
    let g = (t / 1000.0).powf(2.0) * 0.7;
    let b = (t / 1000.0).powf(1.5) * 1.0;
    Color3::new(r, g, b).clamp(Vec3::ZERO, Vec3::ONE)
}

/// Material sample result for one bounce.
#[derive(Clone, Copy, Debug)]
pub enum BsdfScatter {
    /// Perfect specular — used directly without MIS weighting.
    Delta {
        /// Scattered direction (toward camera for reflection, through surface for refraction).
        wi: Direction3,
        /// BSDF × cosine. Tint for dielectrics, white for lossless coatings.
        f_cos: Color3,
        /// GGX alpha for microfacet materials, or `None` for non-GGX materials.
        eta: Option<f32>,
    },
    /// Non-specular — integrator evaluates the BSDF and uses MIS weighting.
    /// Each slot is `Some(PdfKind)` for valid entries, `None` for unused slots.
    NonDelta {
        /// Surface PDF descriptors (first N entries valid, rest is spare capacity).
        pdf_kinds: [Option<PdfKind>; MAX_BSDF_STRATS],
    },
    /// One deterministic delta branch and one non-delta branch — the integrator
    /// traces both paths, splitting the path in two at this vertex.
    Split {
        delta_wi: Direction3,
        delta_f_cos: Color3,
        delta_eta: Option<f32>,
        non_delta_pdf_kinds: [Option<PdfKind>; MAX_BSDF_STRATS],
    },
}

/// Bi-directional scattering distribution function (BSDF) sampling interface — reflection (BRDF),
/// transmission (BTDF), or volumetric scattering. The returned [`BsdfScatter`] encodes the scattered
/// direction and BSDF×cosine throughput.
///
/// Custom materials implement this and integrate via [`Material::Custom`].
pub trait Bsdf: Send + Sync + GpuSerializable {
    /// Sample an outgoing direction. Returns `None` for pure emitters.
    ///
    /// wo: Outgoing direction (surface → camera), world space.
    /// wi: Incoming direction (surface → light), world space.
    /// next_dim: RNG closure for sampling the next dimension (e.g., u1, u2).
    fn scatter(
        &self,
        wo: Direction3,
        si: &SurfaceInteraction,
        next_dim: &mut dyn FnMut() -> f32,
    ) -> Option<BsdfScatter>;

    /// Evaluate the BSDF for an externally-sampled direction pair.
    ///
    /// wo: Outgoing direction (surface → camera), world space.
    /// wi: Incoming direction (surface → light), world space.
    ///
    /// Should be zero for delta materials, which cannot be evaluated over a distribution.
    fn eval(&self, wo: Direction3, wi: Direction3, si: &SurfaceInteraction) -> Color3;

    /// Evaluate the material's sampling PDF for a given direction pair.
    ///
    /// wo: Outgoing direction (surface → camera), world space.
    /// wi: Incoming direction (surface → light), world space.
    ///
    /// Should be zero for delta materials, which cannot be evaluated over a distribution.
    fn pdf(&self, wo: Direction3, wi: Direction3, si: &SurfaceInteraction) -> f32;

    /// Returns the sampling PDF kind for MIS strategy selection.
    ///
    ///  wo: Outgoing direction (surface → camera), world space.
    fn pdf_kind(&self, _wo: Direction3, _si: &SurfaceInteraction) -> Option<PdfKind> {
        None
    }

    /// Returns emitted light at the hit point. Default: no emission. Emission is not a BSDF
    /// property, but this method is provided for convenience in integrators that treat it as such
    /// (e.g., DiffuseEmitter).
    ///
    /// `wo` is the outgoing direction (surface → camera), world space.
    /// Pass `Vec3::ZERO` as a sentinel when direction is unavailable (e.g., NEE).
    ///
    /// Should be zero for non-emissive materials.
    fn emitted(&self, _wo: Direction3, _si: &SurfaceInteraction) -> Color3 {
        Color3::ZERO
    }

    /// Returns `true` if this material emits light.
    fn is_emissive(&self) -> bool {
        false
    }

    /// Rough estimate of the directional-hemispherical reflectance at `wo`, averaged across color
    /// channels. Returns a value in [0, 1]. Used by layered/coated materials to approximate
    /// multi-bounce inter-reflection without a full Monte Carlo random walk.
    ///
    /// This is the integral over the hemisphere of `f(wo, wi) * |cos θ_i| dω_i`. Each material
    /// overrides with its best bounded estimate — the default of 1.0 is a safe upper bound. wo:
    /// Outgoing direction (surface → camera), world space.
    ///
    /// Should be zero for delta materials, which cannot be evaluated over a distribution.
    fn reflectance_estimate(&self, _wo: Direction3, _si: &SurfaceInteraction) -> f32 {
        1.0
    }

    /// Returns `true` if this BSDF is a delta distribution (perfect specular). The integrator skips
    /// MIS weighting for delta materials.
    fn is_delta(&self) -> bool {
        false
    }

    /// Returns the GGX alpha for microfacet materials, or `None` for non-GGX materials. Used by
    /// layered materials (Coated) to include the coating's distribution in MIS strategies,
    /// preventing eval/PDF mismatches that cause fireflies.
    fn ggx_alpha(&self, _si: &SurfaceInteraction) -> Option<f32> {
        None
    }
}

/// Supported material models.
///
/// Wraps concrete structs for built-in materials and delegates to their
/// [`Bsdf`] implementations. Library consumers use [`Material::Custom`].
#[derive(Clone, Default)]
pub enum Material {
    /// Absence of material — all BSDF methods return zero/None.
    /// Used for importance targets where only geometry matters for sampling.
    #[default]
    Void,
    /// Diffuse (Lambertian) surface.
    DiffuseReflector(DiffuseReflector),
    /// Microfacet GGX reflector — conductor (was Metal) or dielectric (was
    /// Glossy), dispatched by the [`Fresnel`] term.
    MicrofacetReflector(MicrofacetReflector),
    /// Dielectric transmission/reflection. Smooth (no roughness) is a delta
    /// BSDF; with `roughness` set it is the microfacet GGX BTDF (was
    /// RoughDielectric).
    Dielectric(DielectricMaterial),
    /// Light emitting surface.
    DiffuseEmitter(DiffuseEmitterMaterial),
    /// Isotropic scattering medium.
    Isotropic(IsotropicMaterial),
    /// Stochastic mix of two materials. `weight` is the probability of choosing `b`.
    Mix(MixMaterial),
    /// Vertical layer: light hits `coating` first; if it transmits, interacts with `substrate`.
    Coated(CoatedMaterial),
    /// Custom material provided by a library consumer.
    Custom(Arc<dyn Bsdf>),
}

impl Material {
    /// Sample this material. Returns `None` for emitters or invalid directions.
    pub fn scatter(
        &self,
        wo: Direction3,
        si: &SurfaceInteraction,
        next_dim: &mut dyn FnMut() -> f32,
    ) -> Option<BsdfScatter> {
        match self {
            Material::Void => None,
            Material::DiffuseReflector(inner) => inner.scatter(wo, si, next_dim),
            Material::MicrofacetReflector(inner) => inner.scatter(wo, si, next_dim),
            Material::Dielectric(inner) => inner.scatter(wo, si, next_dim),
            Material::DiffuseEmitter(inner) => inner.scatter(wo, si, next_dim),
            Material::Isotropic(inner) => inner.scatter(wo, si, next_dim),
            Material::Mix(inner) => inner.scatter(wo, si, next_dim),
            Material::Coated(inner) => inner.scatter(wo, si, next_dim),
            Material::Custom(inner) => inner.scatter(wo, si, next_dim),
        }
    }

    /// Evaluate the BSDF for a direction pair not sampled by this material.
    ///
    /// Called by the integrator when the direction was sampled externally
    /// (e.g., from a light source PDF) and we need the material's response.
    pub fn eval(&self, wo: Direction3, wi: Direction3, si: &SurfaceInteraction) -> Color3 {
        match self {
            Material::Void => Color3::ZERO,
            Material::DiffuseReflector(inner) => inner.eval(wo, wi, si),
            Material::MicrofacetReflector(inner) => inner.eval(wo, wi, si),
            Material::Dielectric(inner) => inner.eval(wo, wi, si),
            Material::DiffuseEmitter(inner) => inner.eval(wo, wi, si),
            Material::Isotropic(inner) => inner.eval(wo, wi, si),
            Material::Mix(inner) => inner.eval(wo, wi, si),
            Material::Coated(inner) => inner.eval(wo, wi, si),
            Material::Custom(inner) => inner.eval(wo, wi, si),
        }
    }

    /// Evaluate the material's sampling PDF for a given direction pair.
    pub fn pdf(&self, wo: Direction3, wi: Direction3, si: &SurfaceInteraction) -> f32 {
        match self {
            Material::Void => 0.0,
            Material::DiffuseReflector(inner) => inner.pdf(wo, wi, si),
            Material::MicrofacetReflector(inner) => inner.pdf(wo, wi, si),
            Material::Dielectric(inner) => inner.pdf(wo, wi, si),
            Material::DiffuseEmitter(inner) => inner.pdf(wo, wi, si),
            Material::Isotropic(inner) => inner.pdf(wo, wi, si),
            Material::Mix(inner) => inner.pdf(wo, wi, si),
            Material::Coated(inner) => inner.pdf(wo, wi, si),
            Material::Custom(inner) => inner.pdf(wo, wi, si),
        }
    }

    /// Returns the sampling PDF kind for MIS strategy selection.
    pub fn pdf_kind(&self, wo: Direction3, si: &SurfaceInteraction) -> Option<PdfKind> {
        match self {
            Material::Void => None,
            Material::DiffuseReflector(inner) => inner.pdf_kind(wo, si),
            Material::MicrofacetReflector(inner) => inner.pdf_kind(wo, si),
            Material::Dielectric(inner) => inner.pdf_kind(wo, si),
            Material::DiffuseEmitter(inner) => inner.pdf_kind(wo, si),
            Material::Isotropic(inner) => inner.pdf_kind(wo, si),
            Material::Mix(inner) => inner.pdf_kind(wo, si),
            Material::Coated(inner) => inner.pdf_kind(wo, si),
            Material::Custom(inner) => inner.pdf_kind(wo, si),
        }
    }

    /// Returns emitted light at the hit point. Default: no emission.
    /// `wo` is the outgoing direction (surface → camera), world space.
    /// For Coated materials, computes Beer's law attenuation through the coating layer.
    pub fn emitted(&self, wo: Direction3, si: &SurfaceInteraction) -> Color3 {
        match self {
            Material::Void => Color3::ZERO,
            Material::DiffuseReflector(inner) => inner.emitted(wo, si),
            Material::MicrofacetReflector(inner) => inner.emitted(wo, si),
            Material::Dielectric(inner) => inner.emitted(wo, si),
            Material::DiffuseEmitter(inner) => inner.emitted(wo, si),
            Material::Isotropic(inner) => inner.emitted(wo, si),
            Material::Mix(inner) => inner.emitted(wo, si),
            Material::Coated(inner) => inner.emitted(wo, si),
            Material::Custom(inner) => inner.emitted(wo, si),
        }
    }

    /// Returns `true` if this material emits light.
    /// Recursively checks composition variants.
    pub fn is_emissive(&self) -> bool {
        match self {
            Material::DiffuseEmitter(_) => true,
            Material::Mix(inner) => inner.is_emissive(),
            Material::Coated(inner) => inner.is_emissive(),
            _ => false,
        }
    }

    /// Rough estimate of the directional-hemispherical reflectance, averaged
    /// across color channels. Bounded in [0, 1]. Used by layered materials
    /// for the multi-bounce inter-reflection series approximation.
    pub fn reflectance_estimate(&self, wo: Direction3, si: &SurfaceInteraction) -> f32 {
        match self {
            Material::Void => 0.0,
            Material::DiffuseReflector(inner) => inner.reflectance_estimate(wo, si),
            Material::MicrofacetReflector(inner) => inner.reflectance_estimate(wo, si),
            Material::Dielectric(inner) => inner.reflectance_estimate(wo, si),
            Material::DiffuseEmitter(inner) => inner.reflectance_estimate(wo, si),
            Material::Isotropic(inner) => inner.reflectance_estimate(wo, si),
            Material::Mix(inner) => inner.reflectance_estimate(wo, si),
            Material::Coated(inner) => inner.reflectance_estimate(wo, si),
            Material::Custom(inner) => inner.reflectance_estimate(wo, si),
        }
    }

    /// Returns `true` if this material is a pure delta distribution.
    /// Recursively checks composition variants: `Mix` is delta iff both children are.
    pub fn is_delta(&self) -> bool {
        match self {
            Material::MicrofacetReflector(inner) => inner.is_delta(),
            Material::Dielectric(inner) => inner.is_delta(),
            Material::Mix(inner) => inner.is_delta(),
            Material::Coated(inner) => inner.is_delta(),
            Material::Custom(inner) => inner.is_delta(),
            _ => false,
        }
    }
}

impl Bsdf for Material {
    fn scatter(
        &self,
        wo: Direction3,
        si: &SurfaceInteraction,
        next_dim: &mut dyn FnMut() -> f32,
    ) -> Option<BsdfScatter> {
        self.scatter(wo, si, next_dim)
    }

    fn eval(&self, wo: Direction3, wi: Direction3, si: &SurfaceInteraction) -> Color3 {
        self.eval(wo, wi, si)
    }

    fn pdf(&self, wo: Direction3, wi: Direction3, si: &SurfaceInteraction) -> f32 {
        self.pdf(wo, wi, si)
    }

    fn pdf_kind(&self, wo: Direction3, si: &SurfaceInteraction) -> Option<PdfKind> {
        self.pdf_kind(wo, si)
    }

    fn emitted(&self, wo: Direction3, si: &SurfaceInteraction) -> Color3 {
        self.emitted(wo, si)
    }

    fn is_emissive(&self) -> bool {
        Material::is_emissive(self)
    }

    fn reflectance_estimate(&self, wo: Direction3, si: &SurfaceInteraction) -> f32 {
        Material::reflectance_estimate(self, wo, si)
    }

    fn is_delta(&self) -> bool {
        Material::is_delta(self)
    }
}

impl GpuSerializable for Material {
    /// Recursive GPU serialization for the material tree.
    ///
    /// Returns the index of the node just pushed.
    fn serialize_gpu(&self, buf: &mut GpuMaterialBuffer) -> u32 {
        match self {
            // Void material has no parameters or children, so just push a node.
            Material::Void => {
                let param_offset = buf.params.len() as u32;
                buf.nodes.push(GpuMaterialNode {
                    material_type: GpuMaterialType::Void as u32,
                    param_offset,
                    child_a: GPU_NONE,
                    child_b: GPU_NONE,
                    texture_index: GPU_NONE,
                });
                buf.nodes.len() as u32 - 1
            }
            // Composition variants.
            Material::Mix(material) => material.serialize_gpu(buf),
            Material::Coated(material) => material.serialize_gpu(buf),
            // Leaf variants delegate to their struct's serialize_gpu.
            Material::DiffuseReflector(inner) => inner.serialize_gpu(buf),
            Material::MicrofacetReflector(inner) => inner.serialize_gpu(buf),
            Material::Dielectric(inner) => inner.serialize_gpu(buf),
            Material::DiffuseEmitter(inner) => inner.serialize_gpu(buf),
            Material::Isotropic(inner) => inner.serialize_gpu(buf),

            // Custom materials have no GPU representation — push a passthrough.
            Material::Custom(inner) => inner.serialize_gpu(buf),
        }
    }
}

impl Material {
    /// Absence of material. All BSDF methods return zero/None.
    pub fn void() -> Self {
        Material::Void
    }

    /// Stochastic mix of two materials. `weight` ∈ [0, 1] is the probability
    /// of choosing `b`.
    pub fn mix(self, other: Material, weight: f32) -> Self {
        let weight = weight.clamp(0.0, 1.0);
        Material::Mix(MixMaterial {
            a: Arc::new(self),
            b: Arc::new(other),
            weight,
        })
    }

    /// Coat this material with a clear-coat layer (thin dielectric).
    pub fn coated(self, coat: Material) -> Self {
        // Extract IOR and tint from dielectric coat if possible. A textured
        // tint has no constant color to bake into the coating layer, so fall
        // back to white (no tint) in that case.
        let (coating_ior, coating_tint) = match &coat {
            Material::Dielectric(d) => (d.ior, d.tint.as_constant().unwrap_or(Color3::ONE)),
            _ => (1.5, Color3::ONE),
        };
        // CoatedMaterial::new clamps the tint to [0, 1] per component.
        // Values > 1 would amplify via powf (physically invalid Beer's law).
        Material::Coated(CoatedMaterial::new(
            Arc::new(self) as Arc<Material>,
            Arc::new(coat) as Arc<Material>,
            coating_ior,
            coating_tint,
            0.01,
        ))
    }

    /// Wrap a custom [`Bsdf`] implementation in a `Material`.
    pub fn custom(bsdf: impl Bsdf + 'static) -> Self {
        Material::Custom(Arc::new(bsdf))
    }
}

impl Material {
    /// Flatten this material tree into a GPU-friendly buffer.
    pub fn to_gpu_buffer(&self) -> GpuMaterialBuffer {
        let mut buf = GpuMaterialBuffer::new();
        self.serialize_gpu(&mut buf);
        buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intersect::interaction::Hit;
    use crate::math::vec3::Point3;
    use crate::texture::{CheckerTexture, SolidColor};

    /// Construct a minimal [`SurfaceInteraction`] for unit tests.
    ///
    /// The geometry is a trivial default (position at origin, zero curvature, no UV
    /// gradients). Only `material` and `shading_normal` are meaningful — set them
    /// to control what the BSDF code path sees.
    #[cfg(test)]
    impl<'a> SurfaceInteraction<'a> {
        pub fn test_surface(
            material: &'a Material,
            shading_normal: Direction3,
        ) -> SurfaceInteraction<'a> {
            SurfaceInteraction::new(
                Hit::new(0.0, Point3::ZERO, Point3::ZERO, shading_normal, None, None),
                shading_normal,
                true,
                material,
                None,
            )
        }
    }

    /// Smoke test: GPU buffer generation for a flat material.
    #[test]
    fn gpu_buffer_diffuse() {
        let mat = Material::from(DiffuseReflector::new(Color3::new(0.5, 0.3, 0.1)));
        let buf = mat.to_gpu_buffer();
        assert_eq!(buf.nodes.len(), 1);
        assert_eq!(
            buf.nodes[0].material_type,
            GpuMaterialType::DiffuseReflector as u32
        );
        assert_eq!(buf.nodes[0].child_a, GPU_NONE);
        assert_eq!(buf.nodes[0].child_b, GPU_NONE);
        // Constant color bakes into params; no texture reference.
        assert_eq!(buf.nodes[0].texture_index, GPU_NONE);
        // 3 f32 params = 12 bytes
        assert_eq!(buf.params.len(), 12);
    }

    /// GPU buffer for a Mix material should produce 3 nodes (mix + 2
    /// children) with the mix node pointing to both children.
    #[test]
    fn gpu_buffer_mix() {
        let mat = Material::from(DiffuseReflector::new(Color3::new(0.5, 0.3, 0.1))).mix(
            Material::from(MicrofacetReflector::conductor_from_reflectance(
                Color3::splat(0.9) * 0.1837,
                0.0,
            )),
            0.5,
        );
        let buf = mat.to_gpu_buffer();
        assert_eq!(buf.nodes.len(), 3);
        // Last node is the mix itself.
        let mix = &buf.nodes[2];
        assert_eq!(mix.material_type, GpuMaterialType::Mix as u32);
        assert_eq!(mix.child_a, 0);
        assert_eq!(mix.child_b, 1);
        // Children are diffuse (node 0) and microfacet conductor (node 1).
        assert_eq!(
            buf.nodes[0].material_type,
            GpuMaterialType::DiffuseReflector as u32
        );
        assert_eq!(
            buf.nodes[1].material_type,
            GpuMaterialType::MicrofacetReflector as u32
        );
        // Microfacet conductor node: 15 f32 params (60 bytes) starting after
        // the diffuse node's 3 floats.
        let off = buf.nodes[1].param_offset as usize / 4;
        assert_eq!(off, 3);
        let f = |i: usize| read_f32(&buf.params, i);
        // Conductor: albedo slots are zero (color comes from Fresnel).
        assert_eq!(f(off), 0.0);
        assert_eq!(f(off + 1), 0.0);
        assert_eq!(f(off + 2), 0.0);
        assert_eq!(f(off + 3), 0.0); // roughness 0.0
        // fresnel_kind: conductor eta = (1 + √base)/(1 − √base), baked per channel.
        assert_eq!(f(off + 4), 0.0);
        let base = Color3::splat(0.9) * 0.1837;
        let expected_eta = (1.0 + base.x().sqrt()) / (1.0 - base.x().sqrt());
        assert!((f(off + 5) - expected_eta).abs() < 1e-5);
        assert!((f(off + 6) - expected_eta).abs() < 1e-5);
        assert!((f(off + 7) - expected_eta).abs() < 1e-5);
        // k = 0 for a reflectance fit.
        assert_eq!(f(off + 8), 0.0);
        assert_eq!(f(off + 9), 0.0);
        assert_eq!(f(off + 10), 0.0);
        // All texture refs ride as -1.0 (baked/absent).
        assert_eq!(f(off + 11), -1.0);
        assert_eq!(f(off + 12), -1.0);
        assert_eq!(f(off + 13), -1.0);
        assert_eq!(f(off + 14), -1.0);
        // Mix node: 1 float. Total = 3 + 15 + 1 = 19 floats = 76 bytes.
        assert_eq!(buf.params.len(), 76);
    }

    /// GPU buffer for a Coated material: coating first, then substrate.
    #[test]
    fn gpu_buffer_coated() {
        let mat = Material::from(DiffuseReflector::new(Color3::new(0.7, 0.2, 0.2)))
            .coated(DielectricMaterial::new(1.5).into());
        let buf = mat.to_gpu_buffer();
        assert_eq!(buf.nodes.len(), 3);
        // Last node is the coat.
        let coat = &buf.nodes[2];
        assert_eq!(coat.material_type, GpuMaterialType::Coated as u32);
        assert_eq!(coat.child_a, 0); // coating (dielectric)
        assert_eq!(coat.child_b, 1); // substrate (diffuse)
        // Coated wire: [coating_ior, thickness, tint.rgb, tex(tint)] = 6 f32s.
        // Children share the flat buffer: dielectric (8) + diffuse (3) + coated (6) = 17 f32s total.
        assert_eq!(buf.params.len(), 68);
        assert_eq!(coat.param_offset, 44);
        let off = coat.param_offset as usize / 4;
        let f = |i: usize| read_f32(&buf.params, i);
        assert!((f(off) - 1.5).abs() < 1e-6); // coating ior from the dielectric
        assert!((f(off + 1) - 0.01).abs() < 1e-6); // default thickness
        assert!((f(off + 2) - 1.0).abs() < 1e-6); // white tint baked
        assert!((f(off + 3) - 1.0).abs() < 1e-6);
        assert!((f(off + 4) - 1.0).abs() < 1e-6);
        assert_eq!(f(off + 5), -1.0); // no tint texture
        assert!(buf.textures.nodes.is_empty());
    }

    /// Coated with a textured tint: the tint serializes as a Mapped node in the
    /// texture buffer (index 0, zero color in the params, ref in slot 5).
    #[test]
    fn gpu_buffer_coated_textured() {
        let mat = Material::from(CoatedMaterial::textured(
            Arc::new(DiffuseReflector::new(Color3::new(0.7, 0.2, 0.2)).into()),
            Arc::new(DielectricMaterial::new(1.5).into()),
            1.5,
            CheckerTexture::with_scale(0.5, Color3::new(0.9, 0.1, 0.1), Color3::new(0.2, 0.8, 0.9)),
            0.01,
        ));
        let buf = mat.to_gpu_buffer();
        assert_eq!(buf.nodes.len(), 3);
        let coat = &buf.nodes[2];
        assert_eq!(coat.material_type, GpuMaterialType::Coated as u32);
        assert_eq!(coat.texture_index, GPU_NONE); // texture refs ride in params
        // Children share the flat buffer: dielectric (8) + diffuse (3) +
        // coated (6) = 17 f32s total.
        assert_eq!(buf.params.len(), 68);
        assert_eq!(coat.param_offset, 44);
        let off = coat.param_offset as usize / 4;
        let f = |i: usize| read_f32(&buf.params, i);
        assert!((f(off) - 1.5).abs() < 1e-6);
        assert!((f(off + 1) - 0.01).abs() < 1e-6);
        // Tint is sampled → zeros stay in the params.
        assert_eq!(f(off + 2), 0.0);
        assert_eq!(f(off + 3), 0.0);
        assert_eq!(f(off + 4), 0.0);
        assert_eq!(f(off + 5), 0.0); // Mapped node index 0
        assert_eq!(buf.textures.nodes.len(), 1);
        assert_eq!(
            buf.textures.nodes[0].texture_type,
            crate::texture::GpuTextureType::Mapped as u32
        );
    }

    /// Nested composition: a mixed material that's also coated.
    #[test]
    fn gpu_buffer_nested() {
        let inner = Material::from(DiffuseReflector::new(Color3::splat(0.5))).mix(
            Material::from(MicrofacetReflector::conductor_from_reflectance(
                Color3::splat(0.9) * 0.1837,
                0.5,
            )),
            0.5,
        );
        let mat = inner.coated(DielectricMaterial::tinted(1.5, Color3::new(1.0, 0.8, 0.8)).into());
        let buf = mat.to_gpu_buffer();
        // Serialization order: coating (dielectric=1 node) → substrate (mix=3
        // nodes) → coated node = 5 total.
        assert_eq!(buf.nodes.len(), 5);
        // The last node is the coat.
        let coat = buf.nodes.last().unwrap();
        assert_eq!(coat.material_type, GpuMaterialType::Coated as u32);
        // child_a is coating (dielectric), child_b is substrate (mix).
        assert_eq!(
            buf.nodes[coat.child_a as usize].material_type,
            GpuMaterialType::Dielectric as u32
        );
        assert_eq!(
            buf.nodes[coat.child_b as usize].material_type,
            GpuMaterialType::Mix as u32
        );
    }

    /// All built-in material types should produce GPU buffers.
    #[test]
    fn gpu_buffer_all_types() {
        let materials: Vec<Material> = vec![
            DiffuseReflector::new(Color3::splat(0.5)).into(),
            MicrofacetReflector::conductor_from_reflectance(Color3::splat(0.9) * 0.1837, 0.0)
                .into(),
            DielectricMaterial::new(1.5).into(),
            DiffuseEmitterMaterial::new(Color3::splat(4.0)).into(),
            IsotropicMaterial::new(Color3::splat(0.5)).into(),
            MicrofacetReflector::dielectric(Color3::splat(0.9), 0.3, 1.5).into(),
            DielectricMaterial::rough(1.5, 0.3).into(),
        ];
        for mat in &materials {
            let buf = mat.to_gpu_buffer();
            assert!(!buf.nodes.is_empty(), "GPU buffer should not be empty");
        }
    }

    /// Reads the first 3 f32s of a params buffer as a [`Color3`].
    fn read_color(params: &[u8]) -> Color3 {
        let vals: [f32; 3] =
            unsafe { std::ptr::read_unaligned(params.as_ptr() as *const [f32; 3]) };
        Color3((vals[0], vals[1], vals[2]).into())
    }

    /// Reads a single f32 at float offset `i` of a params buffer.
    fn read_f32(params: &[u8], i: usize) -> f32 {
        unsafe { std::ptr::read_unaligned(params.as_ptr().add(i * 4) as *const f32) }
    }

    /// Diffuse with a texture still produces a valid GPU buffer.
    ///
    /// A constant texture (SolidColor) bakes its color into the flat params;
    /// a sampled texture serializes into the texture buffer and is referenced
    /// by `texture_index` (zero color stays in the material params).
    #[test]
    fn gpu_buffer_diffuse_textured() {
        // Constant texture: the color bakes into the params buffer.
        let constant = Material::from(DiffuseReflector::textured(Arc::new(SolidColor::new(
            Color3::new(0.7, 0.3, 0.1),
        ))));
        let buf = constant.to_gpu_buffer();
        assert_eq!(buf.nodes.len(), 1);
        assert_eq!(buf.params.len(), 12);
        assert_eq!(buf.nodes[0].texture_index, GPU_NONE);
        let baked = read_color(&buf.params);
        assert!((baked.x() - 0.7).abs() < 1e-6);
        assert!((baked.y() - 0.3).abs() < 1e-6);
        assert!((baked.z() - 0.1).abs() < 1e-6);

        // Sampled texture: `with_scale` wraps the CheckerTexture in a
        // MappedTexture, which serializes as a Mapped node in the texture
        // buffer (the CheckerTexture child itself has no GPU representation
        // yet — it is evaluated in the shader). The material references the
        // node by index; no constant is baked into the params.
        let sampled = Material::from(DiffuseReflector::textured(CheckerTexture::with_scale(
            1.0,
            Color3::new(0.7, 0.3, 0.1),
            Color3::splat(0.2),
        )));
        let buf = sampled.to_gpu_buffer();
        assert_eq!(buf.params.len(), 12);
        assert_eq!(read_color(&buf.params), Color3::ZERO);
        assert_eq!(buf.nodes[0].texture_index, 0); // Mapped node in the texture buffer
        assert_eq!(buf.textures.nodes.len(), 1);
        assert_eq!(
            buf.textures.nodes[0].texture_type,
            crate::texture::GpuTextureType::Mapped as u32
        );
        assert_eq!(buf.textures.nodes[0].child_a, GPU_NONE);
    }

    /// Microfacet dielectric (was Glossy) with baked constants serializes as
    /// 15 f32 params
    /// `[albedo.rgb, roughness, fresnel_kind, eta.rgb (ior splat), k.rgb (0),
    /// tex(albedo), tex(roughness), tex(eta), tex(k)]`
    /// and no texture references (all ride as `-1.0`).
    #[test]
    fn gpu_buffer_microfacet_dielectric() {
        let mat = Material::from(MicrofacetReflector::dielectric(
            Color3::new(0.7, 0.3, 0.1),
            0.2,
            1.5,
        ));
        let buf = mat.to_gpu_buffer();
        assert_eq!(buf.nodes.len(), 1);
        assert_eq!(
            buf.nodes[0].material_type,
            GpuMaterialType::MicrofacetReflector as u32
        );
        assert_eq!(buf.nodes[0].child_a, GPU_NONE);
        assert_eq!(buf.nodes[0].child_b, GPU_NONE);
        assert_eq!(buf.nodes[0].texture_index, GPU_NONE);
        // 15 f32 params = 60 bytes
        assert_eq!(buf.params.len(), 60);
        let f = |i: usize| read_f32(&buf.params, i);
        // Baked constants: albedo, roughness (channel-0), fresnel_kind.
        assert!((f(0) - 0.7).abs() < 1e-6);
        assert!((f(1) - 0.3).abs() < 1e-6);
        assert!((f(2) - 0.1).abs() < 1e-6);
        assert!((f(3) - 0.2).abs() < 1e-6);
        // fresnel_kind: dielectric
        assert_eq!(f(4), 1.0);
        // eta slots carry the IOR splat.
        assert!((f(5) - 1.5).abs() < 1e-6);
        assert!((f(6) - 1.5).abs() < 1e-6);
        assert!((f(7) - 1.5).abs() < 1e-6);
        // k = 0 for dielectrics.
        assert_eq!(f(8), 0.0);
        assert_eq!(f(9), 0.0);
        assert_eq!(f(10), 0.0);
        // All textures baked → no texture references.
        assert_eq!(f(11), -1.0);
        assert_eq!(f(12), -1.0);
        assert_eq!(f(13), -1.0);
        assert_eq!(f(14), -1.0);
        assert!(buf.textures.nodes.is_empty());
    }

    /// Microfacet dielectric with a sampled albedo texture and constant
    /// roughness: the albedo serializes as a Mapped node in the texture buffer
    /// (index 0, zero color in the params), the roughness bakes as `(0.3, 0.3,
    /// 0.3)` with no texture reference.
    #[test]
    fn gpu_buffer_microfacet_dielectric_textured() {
        let mat = Material::from(MicrofacetReflector::dielectric_textured(
            CheckerTexture::with_scale(0.5, Color3::new(0.9, 0.1, 0.1), Color3::new(0.2, 0.8, 0.9)),
            Arc::new(SolidColor::new(Color3::splat(0.3))),
            1.5,
        ));
        let buf = mat.to_gpu_buffer();
        assert_eq!(buf.nodes.len(), 1);
        assert_eq!(
            buf.nodes[0].material_type,
            GpuMaterialType::MicrofacetReflector as u32
        );
        assert_eq!(buf.nodes[0].texture_index, GPU_NONE);
        assert_eq!(buf.params.len(), 60);
        let f = |i: usize| read_f32(&buf.params, i);
        // Albedo is a sampled texture → zero color stays in the params.
        assert_eq!(read_color(&buf.params), Color3::ZERO);
        // Roughness is a constant → channel-0 scalar bakes.
        assert!((f(3) - 0.3).abs() < 1e-6);
        // fresnel_kind: dielectric.
        assert_eq!(f(4), 1.0);
        // IOR splat.
        assert!((f(5) - 1.5).abs() < 1e-6);
        assert!((f(6) - 1.5).abs() < 1e-6);
        assert!((f(7) - 1.5).abs() < 1e-6);
        assert_eq!(f(8), 0.0);
        assert_eq!(f(9), 0.0);
        assert_eq!(f(10), 0.0);
        // Texture refs: albedo → Mapped node index 0; roughness → -1.0 (baked);
        // eta/k → -1.0 (not textured).
        assert_eq!(f(11), 0.0);
        assert_eq!(f(12), -1.0);
        assert_eq!(f(13), -1.0);
        assert_eq!(f(14), -1.0);
        // Exactly one texture node: the Mapped wrapper for the checker.
        assert_eq!(buf.textures.nodes.len(), 1);
        assert_eq!(
            buf.textures.nodes[0].texture_type,
            crate::texture::GpuTextureType::Mapped as u32
        );
    }

    /// Rough dielectric (was RoughDielectric) with baked constants serializes
    /// as 8 f32 params `[tint.rgb, ior, roughness, is_rough, tex(tint),
    /// tex(roughness)]` and no texture references (both ride as `-1.0`).
    #[test]
    fn gpu_buffer_dielectric_rough() {
        let mat = Material::from(DielectricMaterial::rough(1.5, 0.3));
        let buf = mat.to_gpu_buffer();
        assert_eq!(buf.nodes.len(), 1);
        assert_eq!(
            buf.nodes[0].material_type,
            GpuMaterialType::Dielectric as u32
        );
        assert_eq!(buf.nodes[0].child_a, GPU_NONE);
        assert_eq!(buf.nodes[0].child_b, GPU_NONE);
        assert_eq!(buf.nodes[0].texture_index, GPU_NONE);
        // 8 f32 params = 32 bytes
        assert_eq!(buf.params.len(), 32);
        let f = |i: usize| read_f32(&buf.params, i);
        // Baked constants: tint (white), ior, roughness (channel-0).
        assert!((f(0) - 1.0).abs() < 1e-6);
        assert!((f(1) - 1.0).abs() < 1e-6);
        assert!((f(2) - 1.0).abs() < 1e-6);
        assert!((f(3) - 1.5).abs() < 1e-6);
        assert!((f(4) - 0.3).abs() < 1e-6);
        assert_eq!(f(5), 1.0); // is_rough
        // Both textures baked → no texture references.
        assert_eq!(f(6), -1.0);
        assert_eq!(f(7), -1.0);
        assert!(buf.textures.nodes.is_empty());
    }

    /// Rough dielectric with a sampled tint texture and constant roughness: the
    /// tint serializes as a Mapped node in the texture buffer (index 0, zero
    /// color in the params), the roughness bakes with no texture reference.
    #[test]
    fn gpu_buffer_dielectric_rough_textured() {
        let mat = Material::from(DielectricMaterial::rough_textured(
            1.5,
            Arc::new(SolidColor::new(Color3::splat(0.3))),
            CheckerTexture::with_scale(0.5, Color3::new(0.9, 0.1, 0.1), Color3::new(0.2, 0.8, 0.9)),
        ));
        let buf = mat.to_gpu_buffer();
        assert_eq!(buf.nodes.len(), 1);
        assert_eq!(
            buf.nodes[0].material_type,
            GpuMaterialType::Dielectric as u32
        );
        assert_eq!(buf.nodes[0].texture_index, GPU_NONE);
        assert_eq!(buf.params.len(), 32);
        let f = |i: usize| read_f32(&buf.params, i);
        // Tint is a sampled texture → zero color stays in the params.
        assert_eq!(read_color(&buf.params), Color3::ZERO);
        // IOR bakes.
        assert!((f(3) - 1.5).abs() < 1e-6);
        // Roughness is a constant → channel-0 scalar bakes.
        assert!((f(4) - 0.3).abs() < 1e-6);
        assert_eq!(f(5), 1.0); // is_rough
        // Texture refs: tint → Mapped node index 0; roughness → -1.0 (baked).
        assert_eq!(f(6), 0.0);
        assert_eq!(f(7), -1.0);
        // Exactly one texture node: the Mapped wrapper for the checker.
        assert_eq!(buf.textures.nodes.len(), 1);
        assert_eq!(
            buf.textures.nodes[0].texture_type,
            crate::texture::GpuTextureType::Mapped as u32
        );
    }

    /// Smooth dielectric with a baked tint serializes as 8 f32 params
    /// `[tint.rgb, ior, roughness, is_rough, tex(tint), tex(roughness)]`;
    /// `is_rough` is 0 and the roughness slots stay zero.
    #[test]
    fn gpu_buffer_dielectric() {
        let mat = Material::from(DielectricMaterial::tinted(1.5, Color3::new(0.8, 0.5, 0.2)));
        let buf = mat.to_gpu_buffer();
        assert_eq!(buf.nodes.len(), 1);
        assert_eq!(
            buf.nodes[0].material_type,
            GpuMaterialType::Dielectric as u32
        );
        assert_eq!(buf.nodes[0].texture_index, GPU_NONE);
        // 8 f32 params = 32 bytes
        assert_eq!(buf.params.len(), 32);
        let f = |i: usize| read_f32(&buf.params, i);
        assert!((f(0) - 0.8).abs() < 1e-6);
        assert!((f(1) - 0.5).abs() < 1e-6);
        assert!((f(2) - 0.2).abs() < 1e-6);
        assert!((f(3) - 1.5).abs() < 1e-6);
        assert_eq!(f(4), 0.0); // roughness 0.0 for smooth
        assert_eq!(f(5), 0.0); // is_rough 0.0
        assert_eq!(f(6), -1.0); // tint baked
        assert_eq!(f(7), -1.0); // roughness n/a
        assert!(buf.textures.nodes.is_empty());
    }

    /// Dielectric with a textured tint: the tint serializes as a Mapped node in
    /// the texture buffer (index 0, zero color in the params) referenced via
    /// the params slot (refs ride in params, not `texture_index`).
    #[test]
    fn gpu_buffer_dielectric_textured() {
        let mat = Material::from(DielectricMaterial::textured(
            1.5,
            CheckerTexture::with_scale(0.5, Color3::new(0.9, 0.1, 0.1), Color3::new(0.2, 0.8, 0.9)),
        ));
        let buf = mat.to_gpu_buffer();
        assert_eq!(buf.nodes.len(), 1);
        assert_eq!(
            buf.nodes[0].material_type,
            GpuMaterialType::Dielectric as u32
        );
        assert_eq!(buf.nodes[0].texture_index, GPU_NONE); // refs ride in params
        assert_eq!(buf.params.len(), 32);
        // Tint is sampled → zeros stay in the params; ior bakes.
        assert_eq!(read_f32(&buf.params, 0), 0.0);
        assert_eq!(read_f32(&buf.params, 1), 0.0);
        assert_eq!(read_f32(&buf.params, 2), 0.0);
        assert!((read_f32(&buf.params, 3) - 1.5).abs() < 1e-6);
        assert_eq!(read_f32(&buf.params, 4), 0.0); // roughness 0.0
        assert_eq!(read_f32(&buf.params, 5), 0.0); // is_rough 0.0
        assert_eq!(read_f32(&buf.params, 6), 0.0); // Mapped node index 0
        assert_eq!(read_f32(&buf.params, 7), -1.0); // no roughness texture
        assert_eq!(buf.textures.nodes.len(), 1);
        assert_eq!(
            buf.textures.nodes[0].texture_type,
            crate::texture::GpuTextureType::Mapped as u32
        );
    }

    /// Custom material returns empty GPU buffer (no GPU representation).
    #[test]
    fn gpu_buffer_custom() {
        // A simple custom material for testing.
        struct DummyBsdf;
        impl Bsdf for DummyBsdf {
            fn scatter(
                &self,
                _wo: Direction3,
                _si: &SurfaceInteraction,
                _next_dim: &mut dyn FnMut() -> f32,
            ) -> Option<BsdfScatter> {
                None
            }
            fn eval(&self, _wo: Direction3, _wi: Direction3, _si: &SurfaceInteraction) -> Color3 {
                Color3::ZERO
            }
            fn pdf(&self, _wo: Direction3, _wi: Direction3, _si: &SurfaceInteraction) -> f32 {
                0.0
            }
        }
        impl GpuSerializable for DummyBsdf {}
        impl Clone for DummyBsdf {
            fn clone(&self) -> Self {
                DummyBsdf
            }
        }
        let mat = Material::custom(DummyBsdf);
        let buf = mat.to_gpu_buffer();
        // Custom material serializes as a Passthrough node (unknown type).
        assert_eq!(buf.nodes.len(), 1);
        assert_eq!(
            buf.nodes[0].material_type,
            GpuMaterialType::Passthrough as u32
        );
    }
}
