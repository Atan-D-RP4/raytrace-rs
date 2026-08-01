//! Material models and BSDF sampling for path tracing.
//!
//! [`Material`] is a recursive enum — composition variants ([`Material::Mix`],
//! [`Material::Coated`]) wrap dedicated structs ([`MixMaterial`], [`CoatedMaterial`])
//! that contain `Arc<dyn Bsdf>` children. No cycles by construction.
//!
//! # Authoring
//!
//! ```ignore
//! use std::sync::Arc;
//! use raytrace_rs::material::Material;
//! use raytrace_rs::texture::SolidColor;
//! use raytrace_rs::vec3::Color3;
//!
//! let red = Material::lambertian(Arc::new(SolidColor::new(Color3::new(0.8, 0.2, 0.2))));
//! let paint = red.mix(Material::metal(Color3::new(0.9, 0.9, 0.9), 0.0), 0.5);
//! let car_paint = red.coated(Material::dielectric(1.5));
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

use crate::hittable::SurfaceInteraction;
use crate::pdf::{PdfKind, ggx_d, ggx_sample_h};
use crate::vec3::{Color3, Direction3};

mod coated;
mod dielectric;
mod diffuse_light;
mod glossy;
mod gpu;
mod isotropic;
mod lambertian;
mod metal;
mod mix;
mod rough_dielectric;

use glam::Vec3;
use gpu::GPU_NONE;

pub use coated::CoatedMaterial;
pub use dielectric::DielectricMaterial;
pub use diffuse_light::DiffuseLightMaterial;
pub use glossy::GlossyMaterial;
pub use gpu::{GpuMaterialBuffer, GpuMaterialNode, GpuMaterialType, GpuSerializable};
pub use isotropic::IsotropicMaterial;
pub use lambertian::LambertianMaterial;
pub use metal::MetalMaterial;
pub use mix::MixMaterial;
pub use rough_dielectric::RoughDielectricMaterial;

/// Maximum number of BSDF sampling strategies produced by any material.
/// Used as the fixed capacity for `BsdfScatter::NonDelta` / `Split`.
/// Current max ~2 (Mix + Coated); bumped to 4 as safety margin.
pub const MAX_BSDF_STRATS: usize = 4;

const MIRROR_THRESHOLD: f32 = 0.01;

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

    /// Returns emitted light at the hit point. Default: no emission.
    /// Emission is not a BSDF property, but this method is provided for
    /// convenience in integrators that treat it as such (e.g., DiffuseLight).
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
    /// overrides with its best bounded estimate — the default of 1.0 is a safe upper bound.
    /// wo: Outgoing direction (surface → camera), world space.
    ///
    /// Should be zero for delta materials, which cannot be evaluated over a distribution.
    fn reflectance_estimate(&self, _wo: Direction3, _si: &SurfaceInteraction) -> f32 {
        1.0
    }

    /// Returns `true` if this BSDF is a delta distribution (perfect specular).
    /// The integrator skips MIS weighting for delta materials.
    fn is_delta(&self) -> bool {
        false
    }

    /// Returns the GGX alpha for microfacet materials, or `None` for non-GGX
    /// materials. Used by layered materials (Coated) to include the coating's
    /// distribution in MIS strategies, preventing eval/PDF mismatches that cause
    /// fireflies.
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
    Lambertian(LambertianMaterial),
    /// Microfacet conductor BRDF (GGX).
    Metal(MetalMaterial),
    /// Dielectric transmission/reflection.
    Dielectric(DielectricMaterial),
    /// Rough dielectric microfacet BTDF (GGX) with Beer's law absorption.
    RoughDielectric(RoughDielectricMaterial),
    /// Light emitting surface.
    DiffuseLight(DiffuseLightMaterial),
    /// Isotropic scattering medium.
    Isotropic(IsotropicMaterial),
    /// Glossy microfacet BSDF (GGX).
    Glossy(GlossyMaterial),
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
            Material::Lambertian(inner) => inner.scatter(wo, si, next_dim),
            Material::Metal(inner) => inner.scatter(wo, si, next_dim),
            Material::Dielectric(inner) => inner.scatter(wo, si, next_dim),
            Material::RoughDielectric(inner) => inner.scatter(wo, si, next_dim),
            Material::DiffuseLight(inner) => inner.scatter(wo, si, next_dim),
            Material::Isotropic(inner) => inner.scatter(wo, si, next_dim),
            Material::Glossy(inner) => inner.scatter(wo, si, next_dim),
            Material::Custom(inner) => inner.scatter(wo, si, next_dim),
            Material::Mix(inner) => inner.scatter(wo, si, next_dim),
            Material::Coated(inner) => inner.scatter(wo, si, next_dim),
        }
    }

    /// Evaluate the BSDF for a direction pair not sampled by this material.
    ///
    /// Called by the integrator when the direction was sampled externally
    /// (e.g., from a light source PDF) and we need the material's response.
    pub fn eval(&self, wo: Direction3, wi: Direction3, si: &SurfaceInteraction) -> Color3 {
        match self {
            Material::Void => Color3::ZERO,
            Material::Lambertian(inner) => inner.eval(wo, wi, si),
            Material::Metal(inner) => inner.eval(wo, wi, si),
            Material::Dielectric(inner) => inner.eval(wo, wi, si),
            Material::RoughDielectric(inner) => inner.eval(wo, wi, si),
            Material::DiffuseLight(inner) => inner.eval(wo, wi, si),
            Material::Isotropic(inner) => inner.eval(wo, wi, si),
            Material::Glossy(inner) => inner.eval(wo, wi, si),
            Material::Custom(inner) => inner.eval(wo, wi, si),
            Material::Mix(inner) => inner.eval(wo, wi, si),
            Material::Coated(inner) => inner.eval(wo, wi, si),
        }
    }

    /// Evaluate the material's sampling PDF for a given direction pair.
    pub fn pdf(&self, wo: Direction3, wi: Direction3, si: &SurfaceInteraction) -> f32 {
        match self {
            Material::Void => 0.0,
            Material::Lambertian(inner) => inner.pdf(wo, wi, si),
            Material::Metal(inner) => inner.pdf(wo, wi, si),
            Material::Dielectric(inner) => inner.pdf(wo, wi, si),
            Material::RoughDielectric(inner) => inner.pdf(wo, wi, si),
            Material::DiffuseLight(inner) => inner.pdf(wo, wi, si),
            Material::Isotropic(inner) => inner.pdf(wo, wi, si),
            Material::Glossy(inner) => inner.pdf(wo, wi, si),
            Material::Custom(inner) => inner.pdf(wo, wi, si),
            Material::Mix(inner) => inner.pdf(wo, wi, si),
            Material::Coated(inner) => inner.pdf(wo, wi, si),
        }
    }

    /// Returns the sampling PDF kind for MIS strategy selection.
    pub fn pdf_kind(&self, wo: Direction3, si: &SurfaceInteraction) -> Option<PdfKind> {
        match self {
            Material::Void => None,
            Material::Lambertian(inner) => inner.pdf_kind(wo, si),
            Material::Metal(inner) => inner.pdf_kind(wo, si),
            Material::Dielectric(inner) => inner.pdf_kind(wo, si),
            Material::RoughDielectric(inner) => inner.pdf_kind(wo, si),
            Material::DiffuseLight(inner) => inner.pdf_kind(wo, si),
            Material::Isotropic(inner) => inner.pdf_kind(wo, si),
            Material::Glossy(inner) => inner.pdf_kind(wo, si),
            Material::Custom(inner) => inner.pdf_kind(wo, si),
            Material::Mix(inner) => inner.pdf_kind(wo, si),
            Material::Coated(inner) => inner.pdf_kind(wo, si),
        }
    }

    /// Returns emitted light at the hit point. Default: no emission.
    /// `wo` is the outgoing direction (surface → camera), world space.
    /// For Coated materials, computes Beer's law attenuation through the coating layer.
    pub fn emitted(&self, wo: Direction3, si: &SurfaceInteraction) -> Color3 {
        match self {
            Material::Void => Color3::ZERO,
            Material::Lambertian(inner) => inner.emitted(wo, si),
            Material::Metal(inner) => inner.emitted(wo, si),
            Material::Dielectric(inner) => inner.emitted(wo, si),
            Material::RoughDielectric(inner) => inner.emitted(wo, si),
            Material::DiffuseLight(inner) => inner.emitted(wo, si),
            Material::Isotropic(inner) => inner.emitted(wo, si),
            Material::Glossy(inner) => inner.emitted(wo, si),
            Material::Custom(inner) => inner.emitted(wo, si),
            Material::Mix(inner) => inner.emitted(wo, si),
            Material::Coated(inner) => inner.emitted(wo, si),
        }
    }

    /// Returns `true` if this material emits light.
    /// Recursively checks composition variants.
    pub fn is_emissive(&self) -> bool {
        match self {
            Material::DiffuseLight(_) => true,
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
            Material::Lambertian(inner) => inner.reflectance_estimate(wo, si),
            Material::Metal(inner) => inner.reflectance_estimate(wo, si),
            Material::Dielectric(inner) => inner.reflectance_estimate(wo, si),
            Material::RoughDielectric(inner) => inner.reflectance_estimate(wo, si),
            Material::DiffuseLight(inner) => inner.reflectance_estimate(wo, si),
            Material::Isotropic(inner) => inner.reflectance_estimate(wo, si),
            Material::Glossy(inner) => inner.reflectance_estimate(wo, si),
            Material::Custom(inner) => inner.reflectance_estimate(wo, si),
            Material::Mix(inner) => inner.reflectance_estimate(wo, si),
            Material::Coated(inner) => inner.reflectance_estimate(wo, si),
        }
    }

    /// Returns `true` if this material is a pure delta distribution.
    /// Recursively checks composition variants: `Mix` is delta iff both children are.
    pub fn is_delta(&self) -> bool {
        match self {
            Material::Dielectric(inner) => inner.is_delta(),
            Material::Metal(inner) => inner.is_delta(),
            Material::Glossy(inner) => inner.is_delta(),
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
            Material::Lambertian(inner) => inner.serialize_gpu(buf),
            Material::Metal(inner) => inner.serialize_gpu(buf),
            Material::Dielectric(inner) => inner.serialize_gpu(buf),
            Material::RoughDielectric(inner) => inner.serialize_gpu(buf),
            Material::DiffuseLight(inner) => inner.serialize_gpu(buf),
            Material::Isotropic(inner) => inner.serialize_gpu(buf),
            Material::Glossy(inner) => inner.serialize_gpu(buf),

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
            a: Arc::new(self) as Arc<dyn Bsdf>,
            b: Arc::new(other) as Arc<dyn Bsdf>,
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
            Arc::new(self) as Arc<dyn Bsdf>,
            Arc::new(coat) as Arc<dyn Bsdf>,
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
    use crate::texture::{CheckerTexture, SolidColor};

    /// Smoke test: GPU buffer generation for a flat material.
    #[test]
    fn gpu_buffer_lambertian() {
        let mat = Material::from(LambertianMaterial::new(Color3::new(0.5, 0.3, 0.1)));
        let buf = mat.to_gpu_buffer();
        assert_eq!(buf.nodes.len(), 1);
        assert_eq!(
            buf.nodes[0].material_type,
            GpuMaterialType::Lambertian as u32
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
        let mat = Material::from(LambertianMaterial::new(Color3::new(0.5, 0.3, 0.1))).mix(
            Material::from(MetalMaterial::from_reflectance(
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
        // Children are lambertian (node 0) and metal (node 1).
        assert_eq!(
            buf.nodes[0].material_type,
            GpuMaterialType::Lambertian as u32
        );
        assert_eq!(buf.nodes[1].material_type, GpuMaterialType::Metal as u32);
    }

    /// GPU buffer for a Coated material: coating first, then substrate.
    #[test]
    fn gpu_buffer_coated() {
        let mat = Material::from(LambertianMaterial::new(Color3::new(0.7, 0.2, 0.2)))
            .coated(DielectricMaterial::new(1.5).into());
        let buf = mat.to_gpu_buffer();
        assert_eq!(buf.nodes.len(), 3);
        // Last node is the coat.
        let coat = &buf.nodes[2];
        assert_eq!(coat.material_type, GpuMaterialType::Coated as u32);
        assert_eq!(coat.child_a, 0); // coating (dielectric)
        assert_eq!(coat.child_b, 1); // substrate (lambertian)
        // Coated wire: [coating_ior, thickness, tint.rgb, tex(tint)] = 6 f32s.
        // Children share the flat buffer: dielectric (4) + lambertian (3) +
        // coated (6) = 13 f32s total.
        assert_eq!(buf.params.len(), 52);
        assert_eq!(coat.param_offset, 28);
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
            Arc::new(LambertianMaterial::new(Color3::new(0.7, 0.2, 0.2))) as Arc<dyn Bsdf>,
            Arc::new(DielectricMaterial::new(1.5)) as Arc<dyn Bsdf>,
            1.5,
            CheckerTexture::with_scale(0.5, Color3::new(0.9, 0.1, 0.1), Color3::new(0.2, 0.8, 0.9)),
            0.01,
        ));
        let buf = mat.to_gpu_buffer();
        assert_eq!(buf.nodes.len(), 3);
        let coat = &buf.nodes[2];
        assert_eq!(coat.material_type, GpuMaterialType::Coated as u32);
        assert_eq!(coat.texture_index, GPU_NONE); // texture refs ride in params
        // Children share the flat buffer: dielectric (4) + lambertian (3) +
        // coated (6) = 13 f32s total.
        assert_eq!(buf.params.len(), 52);
        assert_eq!(coat.param_offset, 28);
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
        let inner = Material::from(LambertianMaterial::new(Color3::splat(0.5))).mix(
            Material::from(MetalMaterial::from_reflectance(
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
            LambertianMaterial::new(Color3::splat(0.5)).into(),
            MetalMaterial::from_reflectance(Color3::splat(0.9) * 0.1837, 0.0).into(),
            DielectricMaterial::new(1.5).into(),
            DiffuseLightMaterial::new(Color3::splat(4.0)).into(),
            IsotropicMaterial::new(Color3::splat(0.5)).into(),
            GlossyMaterial::new(Color3::splat(0.9), 0.3, 1.5).into(),
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

    /// Lambertian with a texture still produces a valid GPU buffer.
    ///
    /// A constant texture (SolidColor) bakes its color into the flat params;
    /// a sampled texture serializes into the texture buffer and is referenced
    /// by `texture_index` (zero color stays in the material params).
    #[test]
    fn gpu_buffer_lambertian_textured() {
        // Constant texture: the color bakes into the params buffer.
        let constant = Material::from(LambertianMaterial::textured(Arc::new(SolidColor::new(
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
        let sampled = Material::from(LambertianMaterial::textured(CheckerTexture::with_scale(
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

    /// Glossy with baked constants serializes as 7 f32 params
    /// `[albedo.r, albedo.g, albedo.b, roughness, ior, tex(albedo), tex(roughness)]`
    /// and no texture references (both ride as `-1.0`).
    #[test]
    fn gpu_buffer_glossy() {
        let mat = Material::from(GlossyMaterial::new(Color3::new(0.7, 0.3, 0.1), 0.2, 1.5));
        let buf = mat.to_gpu_buffer();
        assert_eq!(buf.nodes.len(), 1);
        assert_eq!(buf.nodes[0].material_type, GpuMaterialType::Glossy as u32);
        assert_eq!(buf.nodes[0].child_a, GPU_NONE);
        assert_eq!(buf.nodes[0].child_b, GPU_NONE);
        assert_eq!(buf.nodes[0].texture_index, GPU_NONE);
        // 7 f32 params = 28 bytes
        assert_eq!(buf.params.len(), 28);
        let f = |i: usize| read_f32(&buf.params, i);
        // Baked constants: albedo, roughness (channel-0), ior.
        assert!((f(0) - 0.7).abs() < 1e-6);
        assert!((f(1) - 0.3).abs() < 1e-6);
        assert!((f(2) - 0.1).abs() < 1e-6);
        assert!((f(3) - 0.2).abs() < 1e-6);
        assert!((f(4) - 1.5).abs() < 1e-6);
        // Both textures baked → no texture references.
        assert_eq!(f(5), -1.0);
        assert_eq!(f(6), -1.0);
        assert!(buf.textures.nodes.is_empty());
    }

    /// Glossy with a sampled albedo texture and constant roughness: the albedo
    /// serializes as a Mapped node in the texture buffer (index 0, zero color
    /// in the params), the roughness bakes as `(0.3, 0.3, 0.3)` with no texture
    /// reference.
    #[test]
    fn gpu_buffer_glossy_textured() {
        let mat = Material::from(GlossyMaterial::textured(
            CheckerTexture::with_scale(0.5, Color3::new(0.9, 0.1, 0.1), Color3::new(0.2, 0.8, 0.9)),
            Arc::new(SolidColor::new(Color3::splat(0.3))),
            1.5,
        ));
        let buf = mat.to_gpu_buffer();
        assert_eq!(buf.nodes.len(), 1);
        assert_eq!(buf.nodes[0].material_type, GpuMaterialType::Glossy as u32);
        assert_eq!(buf.nodes[0].texture_index, GPU_NONE);
        assert_eq!(buf.params.len(), 28);
        let f = |i: usize| read_f32(&buf.params, i);
        // Albedo is a sampled texture → zero color stays in the params.
        assert_eq!(read_color(&buf.params), Color3::ZERO);
        // Roughness is a constant → channel-0 scalar bakes.
        assert!((f(3) - 0.3).abs() < 1e-6);
        // IOR bakes.
        assert!((f(4) - 1.5).abs() < 1e-6);
        // Texture refs: albedo → Mapped node index 0; roughness → -1.0 (baked).
        assert_eq!(f(5), 0.0);
        assert_eq!(f(6), -1.0);
        // Exactly one texture node: the Mapped wrapper for the checker.
        assert_eq!(buf.textures.nodes.len(), 1);
        assert_eq!(
            buf.textures.nodes[0].texture_type,
            crate::texture::GpuTextureType::Mapped as u32
        );
    }

    /// Rough dielectric with baked constants serializes as 7 f32 params
    /// `[tint.r, tint.g, tint.b, ior, roughness, tex(tint), tex(roughness)]`
    /// and no texture references (both ride as `-1.0`).
    #[test]
    fn gpu_buffer_rough_dielectric() {
        let mat = Material::from(RoughDielectricMaterial::new(1.5, 0.3));
        let buf = mat.to_gpu_buffer();
        assert_eq!(buf.nodes.len(), 1);
        assert_eq!(
            buf.nodes[0].material_type,
            GpuMaterialType::RoughDielectric as u32
        );
        assert_eq!(buf.nodes[0].child_a, GPU_NONE);
        assert_eq!(buf.nodes[0].child_b, GPU_NONE);
        assert_eq!(buf.nodes[0].texture_index, GPU_NONE);
        // 7 f32 params = 28 bytes
        assert_eq!(buf.params.len(), 28);
        let f = |i: usize| read_f32(&buf.params, i);
        // Baked constants: tint (white), ior, roughness (channel-0).
        assert!((f(0) - 1.0).abs() < 1e-6);
        assert!((f(1) - 1.0).abs() < 1e-6);
        assert!((f(2) - 1.0).abs() < 1e-6);
        assert!((f(3) - 1.5).abs() < 1e-6);
        assert!((f(4) - 0.3).abs() < 1e-6);
        // Both textures baked → no texture references.
        assert_eq!(f(5), -1.0);
        assert_eq!(f(6), -1.0);
        assert!(buf.textures.nodes.is_empty());
    }

    /// Rough dielectric with a sampled tint texture and constant roughness: the
    /// tint serializes as a Mapped node in the texture buffer (index 0, zero
    /// color in the params), the roughness bakes with no texture reference.
    #[test]
    fn gpu_buffer_rough_dielectric_textured() {
        let mat = Material::from(RoughDielectricMaterial {
            ior: 1.5,
            roughness: Arc::new(SolidColor::new(Color3::splat(0.3))),
            tint: CheckerTexture::with_scale(
                0.5,
                Color3::new(0.9, 0.1, 0.1),
                Color3::new(0.2, 0.8, 0.9),
            ),
        });
        let buf = mat.to_gpu_buffer();
        assert_eq!(buf.nodes.len(), 1);
        assert_eq!(
            buf.nodes[0].material_type,
            GpuMaterialType::RoughDielectric as u32
        );
        assert_eq!(buf.nodes[0].texture_index, GPU_NONE);
        assert_eq!(buf.params.len(), 28);
        let f = |i: usize| read_f32(&buf.params, i);
        // Tint is a sampled texture → zero color stays in the params.
        assert_eq!(read_color(&buf.params), Color3::ZERO);
        // IOR bakes.
        assert!((f(3) - 1.5).abs() < 1e-6);
        // Roughness is a constant → channel-0 scalar bakes.
        assert!((f(4) - 0.3).abs() < 1e-6);
        // Texture refs: tint → Mapped node index 0; roughness → -1.0 (baked).
        assert_eq!(f(5), 0.0);
        assert_eq!(f(6), -1.0);
        // Exactly one texture node: the Mapped wrapper for the checker.
        assert_eq!(buf.textures.nodes.len(), 1);
        assert_eq!(
            buf.textures.nodes[0].texture_type,
            crate::texture::GpuTextureType::Mapped as u32
        );
    }

    /// Dielectric with a baked tint serializes as 4 f32 params
    /// `[tint.r, tint.g, tint.b, ior]`; the tint bakes (no texture ref).
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
        // 4 f32 params = 16 bytes
        assert_eq!(buf.params.len(), 16);
        let f = |i: usize| read_f32(&buf.params, i);
        assert!((f(0) - 0.8).abs() < 1e-6);
        assert!((f(1) - 0.5).abs() < 1e-6);
        assert!((f(2) - 0.2).abs() < 1e-6);
        assert!((f(3) - 1.5).abs() < 1e-6);
        assert!(buf.textures.nodes.is_empty());
    }

    /// Dielectric with a textured tint: the tint serializes as a Mapped node in
    /// the texture buffer (index 0, zero color in the params) referenced via
    /// `texture_index`.
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
        assert_eq!(buf.nodes[0].texture_index, 0); // Mapped node in the texture buffer
        assert_eq!(buf.params.len(), 16);
        // Tint is sampled → zeros stay in the params; ior bakes.
        assert_eq!(read_f32(&buf.params, 0), 0.0);
        assert_eq!(read_f32(&buf.params, 1), 0.0);
        assert_eq!(read_f32(&buf.params, 2), 0.0);
        assert!((read_f32(&buf.params, 3) - 1.5).abs() < 1e-6);
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
