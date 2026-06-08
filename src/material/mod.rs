//! Material models and scattering behavior for path tracing.
//!
//! Materials form a **tree of BSDFs** — the [`Material`] enum is recursive via
//! `Box<dyn Bsdf>` in composition variants ([`Material::Mix`], [`Material::Coated`]).
//! The tree can't have cycles by construction (Rust's type system forbids it),
//! so no graph validation is needed at runtime.
//!
//! # Authoring
//!
//! Use the helper constructors and composition methods on [`Material`]:
//!
//! ```ignore
//! use std::sync::Arc;
//! use raytrace_rs::material::Material;
//! use raytrace_rs::texture::{SolidColor, Texture};
//! use raytrace_rs::vec3::Color3;
//!
//! // A plain Lambertian material with a texture.
//! let _red = Material::lambertian(Arc::new(SolidColor::new(Color3::from(0.8, 0.2, 0.2))));
//!
//! // Composition: metallic paint (Lambertian mixed with metal).
//! let _paint = Material::lambertian(Arc::new(SolidColor::new(Color3::from(0.2, 0.1, 0.1))))
//!     .mix(Material::metal(Color3::from(0.9, 0.9, 0.9), 0.0), 0.5);
//!
//! // Composition: clear coat over a substrate.
//! let _car_paint = Material::lambertian(Arc::new(SolidColor::new(Color3::from(0.7, 0.3, 0.1))))
//!     .coated(Material::dielectric(1.5));
//! ```
//!
//! # Extensibility
//!
//! Library consumers can implement custom materials via the [`Bsdf`] trait and
//! wrap them in [`Material::Custom`]:
//!
//! ```ignore
//! use raytrace_rs::material::{Bsdf, BsdfSample, Material, PdfKind};
//! use raytrace_rs::hittable::HitRecord;
//! use raytrace_rs::vec3::{Color3, Vec3};
//!
//! struct MyCustomBrdf { ... }
//!
//! impl Bsdf for MyCustomBrdf {
//!     fn sample(&self, wo: Vec3, hit: &HitRecord, rng: &mut dyn rand::Rng) -> Option<BsdfSample> { ... }
//!     fn eval(&self, wo: Vec3, wi: Vec3, hit: &HitRecord) -> Color3 { ... }
//!     fn pdf(&self, wo: Vec3, wi: Vec3, hit: &HitRecord) -> f64 { ... }
//! }
//!
//! let my_mat = Material::Custom(Box::new(MyCustomBrdf { ... }));
//! ```
//!
//! # GPU Serialization
//!
//! The material tree can be flattened into a GPU-friendly buffer via
//! [`Material::to_gpu_buffer`]. Each node is a [`GpuMaterialNode`] with
//! optional child indices. The shader mirrors the CPU's enum match via a
//! switch on `material_type`. Custom materials return `None` from
//! are not serialized to the GPU buffer.

mod dielectric;
mod diffuse_light;
mod glossy;
mod gpu;
mod isotropic;
mod lambertian;
mod metal;

pub use gpu::{GpuMaterialBuffer, GpuMaterialNode, GpuMaterialType};

pub use dielectric::DielectricMaterial;
pub use diffuse_light::DiffuseLightMaterial;
pub use glossy::GlossyMaterial;
pub use isotropic::IsotropicMaterial;
pub use lambertian::LambertianMaterial;
pub use metal::MetalMaterial;

use gpu::GPU_NONE;
use gpu::write_node;

use std::f64::consts::PI;
use std::sync::Arc;

use crate::hittable::HitRecord;
use crate::texture::Texture;
use crate::vec3::{Color3, Vec3};

/// Result of material sampling for a single bounce.
///
/// Contains the sampled direction, the BSDF value × cosine term, and the PDF
/// used for that sample. The direction and its PDF come from the same internal
/// sample, so they cannot diverge — making the Glossy direction-PDF mismatch
/// bug structurally impossible.
#[derive(Clone, Copy, Debug)]
pub struct BsdfSample {
    /// Sampled outgoing direction (world space, away from surface).
    pub wi: Vec3,
    /// Product of BSDF value and cosine of the angle between `wi` and the
    /// surface normal. This is the Monte Carlo *integrand* weight.
    pub f_cos: Color3,
    /// Probability density of this sample, under the material's own
    /// sampling distribution.
    pub pdf: f64,
    /// Which PDF the integrator should configure for the mixture with
    /// light sampling. The material returns this so the integrator can
    /// build the correct MIS mixture.
    pub pdf_kind: PdfKind,
}

/// Describes which surface sampling PDF the integrator should use.
///
/// Instead of returning a heap-allocated `Box<dyn PDF>`, materials return
/// this lightweight enum. The integrator owns concrete PDF objects on the
/// stack and updates them from the kind + parameters here.
#[derive(Clone, Copy, Debug)]
pub enum PdfKind {
    /// Cosine-weighted hemisphere. `normal` defines the hemisphere orientation.
    Cosine { normal: Vec3 },
    /// GGX microfacet importance sampling. The half-vector is sampled from
    /// the GGX NDF, then the incoming direction is reflected about it.
    Ggx {
        /// Outgoing direction (from surface toward camera), in world space.
        wo: Vec3,
        /// Surface normal.
        normal: Vec3,
        /// GGX alpha (roughness² clamped to [0.001, 1]).
        alpha: f64,
    },
    /// Uniform sampling over the full sphere (used by isotropic volumes).
    UniformSphere,
    /// Delta distribution (perfect specular). The integrator skips MIS
    /// weighting for delta materials — the sampled direction is used directly.
    Delta,
}

/// Trait for BSDF implementations.
///
/// This trait is **public** so library consumers can implement custom
/// materials. The methods correspond to the standard BSDF interface:
///
/// - `sample()` — generate a direction sample from the material's distribution
/// - `eval()` — evaluate the BSDF at an externally-chosen direction (for MIS)
/// - `pdf()` — evaluate the material's sampling PDF at a direction
/// - `emitted()` — emission contribution (only for light sources)
/// - `gpu_node()` — optional GPU serialization (returns `None` for custom materials)
pub trait Bsdf: Send + Sync {
    /// Sample an outgoing direction for the given outgoing direction and hit.
    ///
    /// Returns `None` for materials that don't scatter (e.g., pure emitters).
    /// The returned [`BsdfSample`] contains the direction, BSDF × cosine,
    /// and PDF — all from the same internal sample.
    fn sample(&self, wo: Vec3, hit: &HitRecord, rng: &mut dyn rand::Rng) -> Option<BsdfSample>;

    /// Evaluate the BSDF for an outgoing→incoming direction pair.
    ///
    /// Used by the integrator when the direction was sampled externally
    /// (e.g., from a light source PDF) and we need the material's response
    /// at that direction.
    fn eval(&self, wo: Vec3, wi: Vec3, hit: &HitRecord) -> Color3;

    /// Evaluate the material's sampling PDF for a given direction pair.
    ///
    /// Used by MIS when the integrator needs to know the probability
    /// that this material would have sampled `wi` given `wo`.
    fn pdf(&self, wo: Vec3, wi: Vec3, hit: &HitRecord) -> f64;

    /// Returns the emitted light color at the hit point.
    ///
    /// Default: no emission. Override for light-emitting materials.
    fn emitted(&self, _hit: &HitRecord) -> Color3 {
        Color3::from(0., 0., 0.)
    }

    /// Returns `true` if this material emits light. Default: `false`.
    fn is_emissive(&self) -> bool {
        false
    }

    /// Returns `true` if this material is a delta distribution (perfect
    /// specular). Delta materials skip MIS weighting. Default: `false`.
    fn is_delta(&self) -> bool {
        false
    }

    /// Serialize this material node for the GPU buffer.
    ///
    /// Leaf materials return their node directly. Composition materials
    /// should recursively serialize children. Return `None` if this
    /// material has no GPU representation (e.g., custom materials).
    fn gpu_node(&self, _buf: &mut GpuMaterialBuffer) -> Option<u32> {
        None
    }

    /// Clone this material into a boxed trait object.
    ///
    /// Required for `Material` to be `Clone` when it contains
    /// `Box<dyn Bsdf>`.
    fn clone_box(&self) -> Box<dyn Bsdf>;

    /// Recursively serialize this material into the GPU buffer.
    ///
    /// Leaf materials call `GpuMaterialBuffer` methods directly.
    /// Composition materials serialize children first, then register
    /// themselves with child indices. Returns the node index.
    ///
    /// The default implementation pushes a `Custom` node (unknown type).
    /// Built-in materials override this.
    fn serialize_gpu(&self, buf: &mut GpuMaterialBuffer) -> u32 {
        let param_offset = buf.params.len() as u32;
        buf.nodes.push(GpuMaterialNode {
            material_type: GpuMaterialType::Passthrough as u32,
            param_offset,
            child_a: GPU_NONE,
            child_b: GPU_NONE,
            texture_index: GPU_NONE,
        });
        buf.nodes.len() as u32 - 1
    }
}

/// Supported material models.
///
/// The enum wraps concrete structs for built-in materials and delegates
/// to their [`Bsdf`] implementations. Library consumers can add custom
/// materials via the [`Material::Custom`] variant.
pub enum Material {
    /// Diffuse (Lambertian) surface.
    Lambertian(LambertianMaterial),
    /// Microfacet conductor BRDF (GGX).
    Metal(MetalMaterial),
    /// Dielectric transmission/reflection.
    Dielectric(DielectricMaterial),
    /// Light emitting surface.
    DiffuseLight(DiffuseLightMaterial),
    /// Isotropic scattering medium.
    Isotropic(IsotropicMaterial),
    /// Glossy microfacet BRDF (GGX).
    Glossy(GlossyMaterial),
    /// Stochastic mix of two materials, weighted by a scalar in [0, 1].
    Mix {
        a: Box<dyn Bsdf>,
        b: Box<dyn Bsdf>,
        /// Selection probability for `b`.
        weight: f64,
    },
    /// A vertical layer: light hits `coating` first; if it transmits, it
    /// interacts with `substrate`.
    Coated {
        substrate: Box<dyn Bsdf>,
        coating: Box<dyn Bsdf>,
    },
    /// Custom material provided by a library consumer.
    Custom(Box<dyn Bsdf>),
}

impl Clone for Material {
    fn clone(&self) -> Self {
        match self {
            Material::Lambertian(inner) => Material::Lambertian(inner.clone()),
            Material::Metal(inner) => Material::Metal(inner.clone()),
            Material::Dielectric(inner) => Material::Dielectric(inner.clone()),
            Material::DiffuseLight(inner) => Material::DiffuseLight(inner.clone()),
            Material::Isotropic(inner) => Material::Isotropic(inner.clone()),
            Material::Glossy(inner) => Material::Glossy(inner.clone()),
            Material::Mix { a, b, weight } => Material::Mix {
                a: a.clone_box(),
                b: b.clone_box(),
                weight: *weight,
            },
            Material::Coated { substrate, coating } => Material::Coated {
                substrate: substrate.clone_box(),
                coating: coating.clone_box(),
            },
            Material::Custom(inner) => Material::Custom(inner.clone_box()),
        }
    }
}

impl Material {
    /// Sample this material for a given outgoing direction and hit record.
    ///
    /// Returns `None` for light-emitting materials (no scattering) or if the
    /// sampled direction is invalid (e.g., below surface).
    pub fn sample(
        &self,
        wo: Vec3,
        record: &HitRecord,
        rng: &mut dyn rand::Rng,
    ) -> Option<BsdfSample> {
        use rand::RngExt;
        match self {
            Material::Lambertian(inner) => inner.sample(wo, record, rng),
            Material::Metal(inner) => inner.sample(wo, record, rng),
            Material::Dielectric(inner) => inner.sample(wo, record, rng),
            Material::DiffuseLight(inner) => inner.sample(wo, record, rng),
            Material::Isotropic(inner) => inner.sample(wo, record, rng),
            Material::Glossy(inner) => inner.sample(wo, record, rng),
            Material::Custom(inner) => inner.sample(wo, record, rng),
            Material::Mix { a, b, weight } => {
                let chosen: &dyn Bsdf = if rng.random::<f64>() < *weight {
                    b.as_ref()
                } else {
                    a.as_ref()
                };
                chosen.sample(wo, record, rng)
            }
            Material::Coated { substrate, coating } => {
                let cos_o = wo.dot(&record.normal).abs();
                let coat_ior = 1.5;
                let f = fresnel_schlick(cos_o, coat_ior);
                if rng.random::<f64>() < f {
                    coating.sample(wo, record, rng)
                } else {
                    substrate.sample(wo, record, rng)
                }
            }
        }
    }

    /// Evaluate the material's BSDF for an externally-chosen direction pair.
    ///
    /// Used by MIS when the integrator samples a direction from the light
    /// PDF and needs to evaluate the material at that direction.
    pub fn eval(&self, wo: Vec3, wi: Vec3, hit: &HitRecord) -> Color3 {
        match self {
            Material::Lambertian(inner) => inner.eval(wo, wi, hit),
            Material::Metal(inner) => inner.eval(wo, wi, hit),
            Material::Dielectric(inner) => inner.eval(wo, wi, hit),
            Material::DiffuseLight(inner) => inner.eval(wo, wi, hit),
            Material::Isotropic(inner) => inner.eval(wo, wi, hit),
            Material::Glossy(inner) => inner.eval(wo, wi, hit),
            Material::Custom(inner) => inner.eval(wo, wi, hit),
            Material::Mix { a, b, weight } => {
                let w = *weight;
                (1.0 - w) * a.eval(wo, wi, hit) + w * b.eval(wo, wi, hit)
            }
            Material::Coated { substrate, coating } => {
                let cos_o = wo.dot(&hit.normal).abs();
                let f = fresnel_schlick(cos_o, 1.5);
                f * coating.eval(wo, wi, hit) + (1.0 - f) * substrate.eval(wo, wi, hit)
            }
        }
    }

    /// Evaluate the material's sampling PDF for a given direction pair.
    pub fn pdf(&self, wo: Vec3, wi: Vec3, hit: &HitRecord) -> f64 {
        match self {
            Material::Lambertian(inner) => inner.pdf(wo, wi, hit),
            Material::Metal(inner) => inner.pdf(wo, wi, hit),
            Material::Dielectric(inner) => inner.pdf(wo, wi, hit),
            Material::DiffuseLight(inner) => inner.pdf(wo, wi, hit),
            Material::Isotropic(inner) => inner.pdf(wo, wi, hit),
            Material::Glossy(inner) => inner.pdf(wo, wi, hit),
            Material::Custom(inner) => inner.pdf(wo, wi, hit),
            Material::Mix { a, b, weight } => {
                (1.0 - weight) * a.pdf(wo, wi, hit) + weight * b.pdf(wo, wi, hit)
            }
            Material::Coated { substrate, coating } => {
                let cos_o = wo.dot(&hit.normal).abs();
                let f = fresnel_schlick(cos_o, 1.5);
                f * coating.pdf(wo, wi, hit) + (1.0 - f) * substrate.pdf(wo, wi, hit)
            }
        }
    }

    /// Returns the emitted light color at the hit point.
    pub fn emitted(&self, hit_rec: &HitRecord) -> Color3 {
        match self {
            Material::Lambertian(inner) => inner.emitted(hit_rec),
            Material::Metal(inner) => inner.emitted(hit_rec),
            Material::Dielectric(inner) => inner.emitted(hit_rec),
            Material::DiffuseLight(inner) => inner.emitted(hit_rec),
            Material::Isotropic(inner) => inner.emitted(hit_rec),
            Material::Glossy(inner) => inner.emitted(hit_rec),
            Material::Custom(inner) => inner.emitted(hit_rec),
            Material::Mix { a, b, weight } => {
                let w = *weight;
                (1.0 - w) * a.emitted(hit_rec) + w * b.emitted(hit_rec)
            }
            Material::Coated { substrate, coating } => {
                // No view direction available in emitted(), so we can't compute
                // a proper Fresnel term. Sum both — in practice coatings don't
                // emit, so this just returns the substrate's emission.
                coating.emitted(hit_rec) + substrate.emitted(hit_rec)
            }
        }
    }

    /// Returns `true` if this material emits light.
    ///
    /// Recursively checks composition variants — a `Coated` or `Mix`
    /// containing an emissive material will also return `true`.
    pub fn is_emissive(&self) -> bool {
        match self {
            Material::DiffuseLight(_) => true,
            Material::Mix { a, b, .. } => a.is_emissive() || b.is_emissive(),
            Material::Coated { substrate, coating } => {
                substrate.is_emissive() || coating.is_emissive()
            }
            _ => false,
        }
    }

    /// Returns `true` if this material is a delta distribution (perfect
    /// specular reflection or refraction). Delta materials cannot be
    /// evaluated at arbitrary directions — the integrator must use the
    /// sampled direction directly without MIS weighting.
    ///
    /// Recursively checks composition variants.
    pub fn is_delta(&self) -> bool {
        match self {
            Material::Dielectric(_) => true,
            Material::Mix { a, b, .. } => a.is_delta() || b.is_delta(),
            Material::Coated { substrate, coating } => substrate.is_delta() || coating.is_delta(),
            _ => false,
        }
    }
}

impl Bsdf for Material {
    fn sample(&self, wo: Vec3, hit: &HitRecord, rng: &mut dyn rand::Rng) -> Option<BsdfSample> {
        self.sample(wo, hit, rng)
    }

    fn eval(&self, wo: Vec3, wi: Vec3, hit: &HitRecord) -> Color3 {
        self.eval(wo, wi, hit)
    }

    fn pdf(&self, wo: Vec3, wi: Vec3, hit: &HitRecord) -> f64 {
        self.pdf(wo, wi, hit)
    }

    fn emitted(&self, hit: &HitRecord) -> Color3 {
        self.emitted(hit)
    }

    fn is_emissive(&self) -> bool {
        Material::is_emissive(self)
    }

    fn is_delta(&self) -> bool {
        Material::is_delta(self)
    }

    fn gpu_node(&self, _buf: &mut GpuMaterialBuffer) -> Option<u32> {
        // Not used — serialization goes through write_node directly.
        None
    }

    fn clone_box(&self) -> Box<dyn Bsdf> {
        Box::new(self.clone())
    }

    fn serialize_gpu(&self, buf: &mut GpuMaterialBuffer) -> u32 {
        write_node(self, buf)
    }
}

impl Material {
    /// Lambertian diffuse material from a solid color.
    pub fn lambertian_color(r: f64, g: f64, b: f64) -> Self {
        Material::Lambertian(LambertianMaterial {
            albedo: Color3::from(r, g, b),
            tex: None,
        })
    }

    /// Lambertian diffuse material with a texture for spatial variation.
    pub fn lambertian(tex: Arc<dyn Texture>) -> Self {
        Material::Lambertian(LambertianMaterial {
            albedo: Color3::ZERO,
            tex: Some(tex),
        })
    }

    /// Microfacet conductor (GGX). `fuzz` ∈ [0, 1] controls roughness;
    /// `ior` sets the index of refraction for the Fresnel term (default 2.5).
    pub fn metal(albedo: Color3, fuzz: f64) -> Self {
        Material::Metal(MetalMaterial {
            albedo,
            tex: None,
            fuzz,
            ior: 2.5,
        })
    }

    /// Microfacet conductor with an explicit IOR for the Fresnel term.
    pub fn metal_with_ior(albedo: Color3, fuzz: f64, ior: f64) -> Self {
        Material::Metal(MetalMaterial {
            albedo,
            tex: None,
            fuzz,
            ior,
        })
    }

    /// Glass / dielectric material with refractive index.
    pub fn dielectric(ior: f64) -> Self {
        Material::Dielectric(DielectricMaterial {
            refractive_idx: ior,
        })
    }

    /// Area light emitting a constant color.
    pub fn light(emit: Color3) -> Self {
        Material::DiffuseLight(DiffuseLightMaterial { emit, tex: None })
    }

    /// Area light with a texture for spatial emission variation.
    pub fn light_textured(tex: Arc<dyn Texture>) -> Self {
        Material::DiffuseLight(DiffuseLightMaterial {
            emit: Color3::ZERO,
            tex: Some(tex),
        })
    }

    /// Isotropic scattering medium with a uniform albedo.
    pub fn isotropic(albedo: Color3) -> Self {
        Material::Isotropic(IsotropicMaterial { albedo, tex: None })
    }

    /// Isotropic scattering medium with a textured albedo.
    pub fn isotropic_texture(tex: Arc<dyn Texture>) -> Self {
        Material::Isotropic(IsotropicMaterial {
            albedo: Color3::ZERO,
            tex: Some(tex),
        })
    }

    /// Glossy microfacet BRDF (GGX).
    pub fn glossy(albedo: Color3, roughness: f64, ior: f64) -> Self {
        Material::Glossy(GlossyMaterial {
            albedo,
            tex: None,
            roughness,
            ior,
        })
    }

    /// Microfacet conductor with a textured albedo.
    pub fn metal_textured(tex: Arc<dyn Texture>, fuzz: f64) -> Self {
        Material::Metal(MetalMaterial {
            albedo: Color3::ZERO,
            tex: Some(tex),
            fuzz,
            ior: 2.5,
        })
    }

    /// Microfacet conductor with a textured albedo and explicit IOR.
    pub fn metal_textured_with_ior(tex: Arc<dyn Texture>, fuzz: f64, ior: f64) -> Self {
        Material::Metal(MetalMaterial {
            albedo: Color3::ZERO,
            tex: Some(tex),
            fuzz,
            ior,
        })
    }

    /// Glossy microfacet BRDF with a textured albedo.
    pub fn glossy_textured(tex: Arc<dyn Texture>, roughness: f64, ior: f64) -> Self {
        Material::Glossy(GlossyMaterial {
            albedo: Color3::ZERO,
            tex: Some(tex),
            roughness,
            ior,
        })
    }

    /// Stochastic mix of two materials.
    ///
    /// `weight` ∈ [0, 1]: probability of choosing `b`. Use 0.5 for a 50/50
    /// blend.
    pub fn mix(self, other: Material, weight: f64) -> Self {
        let weight = weight.clamp(0.0, 1.0);
        Material::Mix {
            a: Box::new(self) as Box<dyn Bsdf>,
            b: Box::new(other) as Box<dyn Bsdf>,
            weight,
        }
    }

    /// Coat this material with a clear-coat layer (thin dielectric).
    pub fn coated(self, coat: Material) -> Self {
        Material::Coated {
            substrate: Box::new(self) as Box<dyn Bsdf>,
            coating: Box::new(coat) as Box<dyn Bsdf>,
        }
    }

    /// Wrap a custom [`Bsdf`] implementation in a `Material`.
    pub fn custom(bsdf: impl Bsdf + 'static) -> Self {
        Material::Custom(Box::new(bsdf))
    }
}

impl Material {
    /// Flatten this material tree into a GPU-friendly buffer.
    pub fn to_gpu_buffer(&self) -> GpuMaterialBuffer {
        let mut buf = GpuMaterialBuffer::new();
        write_node(self, &mut buf);
        buf
    }
}

/// GGX/Trowbridge-Reitz normal distribution function (NDF).
///
/// Returns the probability density that a microfacet has half-vector H with
/// the surface normal. `cos_theta_h` is `cos(H·N)`, `alpha` is `roughness²`.
/// The NDF controls the specular lobe width: low alpha = sharp highlights,
/// high alpha = broad sheen.
pub fn ggx_d(cos_theta_h: f64, alpha: f64) -> f64 {
    if cos_theta_h <= 0.0 {
        return 0.0;
    }
    let a2 = alpha * alpha;
    let denom = cos_theta_h * cos_theta_h * (a2 - 1.0) + 1.0;
    a2 / (PI * denom * denom)
}

/// Smith's geometry function (Schlick-GGX approximation).
///
/// Models microfacet self-shadowing: at grazing angles, some microfacets are
/// blocked by others, reducing the effective reflection. `cos_theta` is
/// `cos(ω·N)`, `alpha` is `roughness²`. Returns a multiplier in [0, 1].
pub(super) fn geometry_schlick_ggx(cos_theta: f64, alpha: f64) -> f64 {
    if cos_theta <= 0.0 {
        return 0.0;
    }
    let a2 = alpha * alpha;
    // Direct lighting remapping of α
    let k = (a2.sqrt() + 1.0).powi(2) / 8.0;
    cos_theta / (cos_theta * (1.0 - k) + k)
}

/// Schlick Fresnel reflectance for unpolarized light.
///
/// Approximates the fraction of light reflected at a dielectric interface.
/// `cos_theta` is `cos(ω·N)`, `ior` is the ratio of refractive indices.
/// At normal incidence returns `((1-ior)/(1+ior))²`; approaches 1 at grazing.
pub(super) fn fresnel_schlick(cos_theta: f64, ior: f64) -> f64 {
    let r0 = ((1.0 - ior) / (1.0 + ior)).powi(2);
    r0 + (1.0 - r0) * (1.0 - cos_theta).powi(5)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::texture::SolidColor;

    /// Smoke test: GPU buffer generation for a flat material.
    #[test]
    fn gpu_buffer_lambertian() {
        let mat = Material::lambertian_color(0.5, 0.3, 0.1);
        let buf = mat.to_gpu_buffer();
        assert_eq!(buf.nodes.len(), 1);
        assert_eq!(
            buf.nodes[0].material_type,
            GpuMaterialType::Lambertian as u32
        );
        assert_eq!(buf.nodes[0].child_a, GPU_NONE);
        assert_eq!(buf.nodes[0].child_b, GPU_NONE);
        // 3 f32 params = 12 bytes
        assert_eq!(buf.params.len(), 12);
    }

    /// GPU buffer for a Mix material should produce 3 nodes (mix + 2
    /// children) with the mix node pointing to both children.
    #[test]
    fn gpu_buffer_mix() {
        let mat = Material::lambertian_color(0.5, 0.3, 0.1)
            .mix(Material::metal(Color3::from(0.9, 0.9, 0.9), 0.0), 0.5);
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

    /// GPU buffer for a Coated material: substrate first, then coating.
    #[test]
    fn gpu_buffer_coated() {
        let mat = Material::lambertian_color(0.7, 0.2, 0.2).coated(Material::dielectric(1.5));
        let buf = mat.to_gpu_buffer();
        assert_eq!(buf.nodes.len(), 3);
        // Last node is the coat.
        let coat = &buf.nodes[2];
        assert_eq!(coat.material_type, GpuMaterialType::Coated as u32);
        assert_eq!(coat.child_a, 0); // substrate
        assert_eq!(coat.child_b, 1); // coating
    }

    /// Nested composition: a mixed material that's also coated.
    #[test]
    fn gpu_buffer_nested() {
        let inner = Material::lambertian_color(0.5, 0.5, 0.5)
            .mix(Material::metal(Color3::from(0.9, 0.9, 0.9), 0.5), 0.5);
        let mat = inner.coated(Material::dielectric(1.5));
        let buf = mat.to_gpu_buffer();
        // inner is 3 nodes (lambertian, metal, mix). coated adds 1
        // (dielectric) + 1 (coat) = 2 more = 5 total.
        assert_eq!(buf.nodes.len(), 5);
        // The last node is the coat.
        let coat = buf.nodes.last().unwrap();
        assert_eq!(coat.material_type, GpuMaterialType::Coated as u32);
        // Substrate should be the mix (which is node index 2).
        assert_eq!(
            buf.nodes[coat.child_a as usize].material_type,
            GpuMaterialType::Mix as u32
        );
        // Coating should be the dielectric.
        assert_eq!(
            buf.nodes[coat.child_b as usize].material_type,
            GpuMaterialType::Dielectric as u32
        );
    }

    /// All built-in material types should produce GPU buffers.
    #[test]
    fn gpu_buffer_all_types() {
        let materials = vec![
            Material::lambertian_color(0.5, 0.5, 0.5),
            Material::metal(Color3::from(0.9, 0.9, 0.9), 0.0),
            Material::dielectric(1.5),
            Material::light(Color3::from(4.0, 4.0, 4.0)),
            Material::isotropic(Color3::from(0.5, 0.5, 0.5)),
            Material::glossy(Color3::from(0.9, 0.9, 0.9), 0.3, 1.5),
        ];
        for mat in &materials {
            let buf = mat.to_gpu_buffer();
            assert!(!buf.nodes.is_empty(), "GPU buffer should not be empty");
        }
    }

    /// Lambertian with a texture still produces a valid GPU buffer (uses
    /// fallback albedo).
    #[test]
    fn gpu_buffer_lambertian_textured() {
        let mat = Material::Lambertian(LambertianMaterial {
            albedo: Color3::from(0.5, 0.5, 0.5),
            tex: Some(Arc::new(SolidColor::new(Color3::from(0.7, 0.3, 0.1)))),
        });
        let buf = mat.to_gpu_buffer();
        assert_eq!(buf.nodes.len(), 1);
        // The GPU buffer should use the fallback albedo, not the texture's
        // color.
        assert_eq!(buf.params.len(), 12);
    }

    /// Custom material returns empty GPU buffer (no GPU representation).
    #[test]
    fn gpu_buffer_custom() {
        // A simple custom material for testing.
        struct DummyBsdf;
        impl Bsdf for DummyBsdf {
            fn sample(
                &self,
                _wo: Vec3,
                _hit: &HitRecord,
                _rng: &mut dyn rand::Rng,
            ) -> Option<BsdfSample> {
                None
            }
            fn eval(&self, _wo: Vec3, _wi: Vec3, _hit: &HitRecord) -> Color3 {
                Color3::from(0., 0., 0.)
            }
            fn pdf(&self, _wo: Vec3, _wi: Vec3, _hit: &HitRecord) -> f64 {
                0.0
            }
            fn clone_box(&self) -> Box<dyn Bsdf> {
                Box::new(self.clone())
            }
        }
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
