//! Material models and BSDF sampling for path tracing.
//!
//! [`Material`] is a recursive enum — composition variants ([`Material::Mix`],
//! [`Material::Coated`]) contain `Box<dyn Bsdf>` children. No cycles by construction.
//!
//! # Authoring
//!
//! ```ignore
//! use std::sync::Arc;
//! use raytrace_rs::material::Material;
//! use raytrace_rs::texture::SolidColor;
//! use raytrace_rs::vec3::Color3;
//!
//! let red = Material::lambertian(Arc::new(SolidColor::new(Color3::from(0.8, 0.2, 0.2))));
//! let paint = red.mix(Material::metal(Color3::from(0.9, 0.9, 0.9), 0.0), 0.5);
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
//! let mat = Material::Custom(Box::new(MyBrdf { ... }));
//! ```
//!
//! # GPU Serialization
//!
//! Flatten via [`Material::to_gpu_buffer`] into [`GpuMaterialNode`]s with
//! child indices. Custom materials serialize as `Passthrough` (not uploaded).

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

use crate::hittable::SurfaceInteraction;
use crate::sampler::SampleDims;
use crate::texture::Texture;
use crate::vec3::{Color3, Vec3, reflect};

/// Material sample result for one bounce.
///
/// [`Delta`]: integrator uses direction and throughput directly.
/// [`NonDelta`]: integrator samples from mixture PDF; only [`PdfKind`] matters.
#[derive(Clone, Copy, Debug)]
pub enum BsdfSample {
    /// Perfect specular — used directly without MIS weighting.
    Delta {
        /// Scattered direction (toward camera for reflection, through surface for refraction).
        wi: Vec3,
        /// BSDF × cosine. Tint for dielectrics, white for lossless coatings.
        f_cos: Color3,
    },
    /// Non-specular — integrator evaluates the BRDF and uses MIS weighting.
    /// `count` indicates how many PDF descriptors are valid (1 for leaf, up to 2 for Mix).
    NonDelta {
        /// Up to 2 surface PDF descriptors (first from chosen child, second from other).
        pdf_kinds: [PdfKind; 2],
        /// Number of valid entries in `pdf_kinds` (1 or 2).
        count: u8,
    },
}

/// Describes which surface sampling PDF the integrator should use.
///
/// Lightweight enum returned by materials instead of heap-allocated `Box<dyn PDF>`.
/// The integrator owns concrete PDF objects on the stack and updates them from
/// the kind + parameters here.
#[derive(Clone, Copy, Debug)]
pub enum PdfKind {
    /// Cosine-weighted hemisphere. `normal` defines the hemisphere orientation.
    Cosine { normal: Vec3 },
    /// GGX microfacet importance sampling. Samples half-vector from NDF, reflects.
    Ggx {
        /// Outgoing direction (surface → camera), world space.
        wo: Vec3,
        /// Surface normal.
        normal: Vec3,
        /// GGX alpha (roughness² clamped to [0.001, 1]).
        alpha: f64,
    },
    /// Uniform over the full sphere (isotropic volumes).
    UniformSphere,
    /// Uniform over the hemisphere oriented by `normal`.
    UniformHemisphere { normal: Vec3 },
    /// Delta distribution (perfect specular). Integrator skips MIS weighting.
    Delta,
}

/// BSDF sampling interface. Public so library consumers can implement custom materials.
pub trait Bsdf: Send + Sync {
    /// Sample an outgoing direction. Returns `None` for pure emitters.
    fn sample(&self, wo: Vec3, si: &SurfaceInteraction, dims: SampleDims) -> Option<BsdfSample>;

    /// Evaluate the BSDF for an externally-sampled direction pair.
    fn eval(&self, wo: Vec3, wi: Vec3, si: &SurfaceInteraction) -> Color3;

    /// Evaluate the material's sampling PDF for a given direction pair.
    fn pdf(&self, wo: Vec3, wi: Vec3, si: &SurfaceInteraction) -> f64;

    /// Returns the sampling PDF kind for MIS strategy selection.
    fn pdf_kind(&self, _wo: Vec3, _si: &SurfaceInteraction) -> Option<PdfKind> {
        None
    }

    /// Returns emitted light at the hit point. Default: no emission.
    fn emitted(&self, _si: &SurfaceInteraction) -> Color3 {
        Color3::from(0., 0., 0.)
    }

    /// Returns `true` if this material emits light.
    fn is_emissive(&self) -> bool {
        false
    }

    /// Returns `true` if this BSDF is a delta distribution (perfect specular).
    /// The integrator skips MIS weighting for delta materials.
    fn is_delta(&self) -> bool {
        false
    }

    /// Clone into a boxed trait object. Required for `Material: Clone`.
    fn clone_box(&self) -> Box<dyn Bsdf>;

    /// Recursively serialize into the GPU buffer. Returns the node index.
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
/// Wraps concrete structs for built-in materials and delegates to their
/// [`Bsdf`] implementations. Library consumers use [`Material::Custom`].
pub enum Material {
    /// Absence of material — all BSDF methods return zero/None.
    /// Used for importance targets where only geometry matters for sampling.
    Void,
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
    /// Stochastic mix of two materials. `weight` is the probability of choosing `b`.
    Mix {
        /// Material chosen with probability `(1 - weight)`.
        a: Box<dyn Bsdf>,
        /// Material chosen with probability `weight`.
        b: Box<dyn Bsdf>,
        /// Selection probability for `b`. ∈ [0, 1].
        weight: f64,
    },
    /// Vertical layer: light hits `coating` first; if it transmits, interacts with `substrate`.
    Coated {
        /// Bottom layer (absorbs transmitted light).
        substrate: Box<dyn Bsdf>,
        /// Top layer (thin dielectric, reflects some light via Fresnel).
        coating: Box<dyn Bsdf>,
    },
    /// Custom material provided by a library consumer.
    Custom(Box<dyn Bsdf>),
}

impl Clone for Material {
    fn clone(&self) -> Self {
        match self {
            Material::Void => Material::Void,
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
    /// Sample this material. Returns `None` for emitters or invalid directions.
    pub fn sample(
        &self,
        wo: Vec3,
        si: &SurfaceInteraction,
        dims: SampleDims,
    ) -> Option<BsdfSample> {
        match self {
            Material::Void => None,
            Material::Lambertian(inner) => inner.sample(wo, si, dims),
            Material::Metal(inner) => inner.sample(wo, si, dims),
            Material::Dielectric(inner) => inner.sample(wo, si, dims),
            Material::DiffuseLight(inner) => inner.sample(wo, si, dims),
            Material::Isotropic(inner) => inner.sample(wo, si, dims),
            Material::Glossy(inner) => inner.sample(wo, si, dims),
            Material::Custom(inner) => inner.sample(wo, si, dims),
            Material::Mix { a, b, weight } => {
                let (chosen, selection_prob) = if dims.u < *weight {
                    (b.as_ref() as &dyn Bsdf, *weight)
                } else {
                    (a.as_ref() as &dyn Bsdf, 1.0 - *weight)
                };
                // Dims: `u` consumed for selection, pass v-w for child directional
                // sampling, x-y-z as padding. `z` is recycled for child's z — no
                // material reads z, so the dependency is semantically harmless.
                let mut result = chosen.sample(
                    wo,
                    si,
                    SampleDims {
                        u: dims.v,
                        v: dims.w,
                        w: dims.x,
                        x: dims.y,
                        y: dims.z,
                        z: dims.z,
                    },
                )?;
                // The child was selected with probability `selection_prob`. For Delta
                // paths the direction comes directly from the child (no MIS mixture),
                // so f_cos must be divided by the selection probability. NonDelta paths
                // are sampled from the integrator's mixture PDF, which doesn't depend
                // on the Mix's internal selection — eval() handles the blend.
                match &mut result {
                    BsdfSample::Delta { f_cos, .. } => {
                        *f_cos /= selection_prob;
                    }
                    BsdfSample::NonDelta { pdf_kinds, count } => {
                        let other = if dims.u < *weight {
                            a.as_ref()
                        } else {
                            b.as_ref()
                        };
                        if let Some(other_kind) = other.pdf_kind(wo, si)
                            && (*count as usize) < pdf_kinds.len()
                        {
                            pdf_kinds[*count as usize] = other_kind;
                            *count += 1;
                        }
                    }
                }
                Some(result)
            }
            Material::Coated {
                substrate,
                coating: _coating,
            } => {
                // NOTE: `_coating` is intentionally unused in `sample()` because
                // the outer Fresnel branch handles coating reflection directly
                // (delta BSDF). The coating IS used in `eval()` and `pdf()` for
                // MIS weighting — this is correct because the coating's eval/pdf
                // return zero for non-specular directions, so the mixture resolves
                // identically. If the coating BSDF ever gains a non-delta component,
                // this branch will need to be updated to keep sample/eval/pdf
                // consistent.
                let cos_o = wo.dot(&si.shading_normal()).abs();
                let f = fresnel_schlick(cos_o, COATED_R0);
                if dims.u < f {
                    // Compute coating reflection directly — delegating to the
                    // dielectric's sample() would apply a second Fresnel check
                    // (on dimension v) causing double-counting or selecting
                    // refraction when the outer branch already chose reflection.
                    //
                    // MC weight: Fresnel f is both branch probability and delta
                    // BSDF value (lossless dielectric), so f/f = 1.
                    let wi = reflect(&-wo, &si.shading_normal());
                    Some(BsdfSample::Delta {
                        wi,
                        f_cos: Color3::from(1., 1., 1.),
                    })
                } else {
                    // Transmit through coating; importance-sample substrate with
                    // transmission probability (1-f) weight.
                    let mut bsdf = substrate.sample(
                        wo,
                        si,
                        SampleDims {
                            u: dims.v,
                            v: dims.w,
                            w: dims.x,
                            x: dims.y,
                            y: dims.z,
                            z: dims.z,
                        },
                    )?;
                    match &mut bsdf {
                        BsdfSample::Delta { f_cos, .. } => {
                            *f_cos /= 1.0 - f;
                        }
                        BsdfSample::NonDelta { .. } => {}
                    }
                    Some(bsdf)
                }
            }
        }
    }

    /// Evaluate the BSDF for a direction pair not sampled by this material.
    ///
    /// Called by the integrator when the direction was sampled externally
    /// (e.g., from a light source PDF) and we need the material's response.
    pub fn eval(&self, wo: Vec3, wi: Vec3, si: &SurfaceInteraction) -> Color3 {
        match self {
            Material::Void => Color3::ZERO,
            Material::Lambertian(inner) => inner.eval(wo, wi, si),
            Material::Metal(inner) => inner.eval(wo, wi, si),
            Material::Dielectric(inner) => inner.eval(wo, wi, si),
            Material::DiffuseLight(inner) => inner.eval(wo, wi, si),
            Material::Isotropic(inner) => inner.eval(wo, wi, si),
            Material::Glossy(inner) => inner.eval(wo, wi, si),
            Material::Custom(inner) => inner.eval(wo, wi, si),
            Material::Mix { a, b, weight } => {
                let w = *weight;
                (1.0 - w) * a.eval(wo, wi, si) + w * b.eval(wo, wi, si)
            }
            Material::Coated { substrate, coating } => {
                let cos_o = wo.dot(&si.shading_normal()).abs();
                let f = fresnel_schlick(cos_o, COATED_R0);
                f * coating.eval(wo, wi, si) + (1.0 - f) * substrate.eval(wo, wi, si)
            }
        }
    }

    /// Evaluate the material's sampling PDF for a given direction pair.
    pub fn pdf(&self, wo: Vec3, wi: Vec3, si: &SurfaceInteraction) -> f64 {
        match self {
            Material::Void => 0.0,
            Material::Lambertian(inner) => inner.pdf(wo, wi, si),
            Material::Metal(inner) => inner.pdf(wo, wi, si),
            Material::Dielectric(inner) => inner.pdf(wo, wi, si),
            Material::DiffuseLight(inner) => inner.pdf(wo, wi, si),
            Material::Isotropic(inner) => inner.pdf(wo, wi, si),
            Material::Glossy(inner) => inner.pdf(wo, wi, si),
            Material::Custom(inner) => inner.pdf(wo, wi, si),
            Material::Mix { a, b, weight } => {
                (1.0 - weight) * a.pdf(wo, wi, si) + weight * b.pdf(wo, wi, si)
            }
            Material::Coated { substrate, coating } => {
                let cos_o = wo.dot(&si.shading_normal()).abs();
                let f = fresnel_schlick(cos_o, COATED_R0);
                f * coating.pdf(wo, wi, si) + (1.0 - f) * substrate.pdf(wo, wi, si)
            }
        }
    }

    /// Returns the sampling PDF kind for MIS strategy selection.
    pub fn pdf_kind(&self, wo: Vec3, si: &SurfaceInteraction) -> Option<PdfKind> {
        match self {
            Material::Void => None,
            Material::Lambertian(inner) => inner.pdf_kind(wo, si),
            Material::Metal(inner) => inner.pdf_kind(wo, si),
            Material::Dielectric(inner) => inner.pdf_kind(wo, si),
            Material::DiffuseLight(inner) => inner.pdf_kind(wo, si),
            Material::Isotropic(inner) => inner.pdf_kind(wo, si),
            Material::Glossy(inner) => inner.pdf_kind(wo, si),
            Material::Custom(inner) => inner.pdf_kind(wo, si),
            Material::Mix { a, b, weight } => {
                // Pick the PDF kind based on the stochastic weight (which child
                // would be selected), not a Fresnel term — Mix has no Fresnel
                // parameter (that's Coated).
                if *weight > 0.5 {
                    b.pdf_kind(wo, si)
                } else {
                    a.pdf_kind(wo, si)
                }
            }
            Material::Coated { substrate, coating } => {
                let cos_o = wo.dot(&si.shading_normal()).abs();
                let f = fresnel_schlick(cos_o, COATED_R0);
                if f > 0.5 {
                    coating.pdf_kind(wo, si)
                } else {
                    substrate.pdf_kind(wo, si)
                }
            }
        }
    }

    /// Returns emitted light at the hit point. Default: no emission.
    pub fn emitted(&self, si: &SurfaceInteraction) -> Color3 {
        match self {
            Material::Void => Color3::ZERO,
            Material::Lambertian(inner) => inner.emitted(si),
            Material::Metal(inner) => inner.emitted(si),
            Material::Dielectric(inner) => inner.emitted(si),
            Material::DiffuseLight(inner) => inner.emitted(si),
            Material::Isotropic(inner) => inner.emitted(si),
            Material::Glossy(inner) => inner.emitted(si),
            Material::Custom(inner) => inner.emitted(si),
            Material::Mix { a, b, weight } => {
                let w = *weight;
                (1.0 - w) * a.emitted(si) + w * b.emitted(si)
            }
            Material::Coated { substrate, coating } => {
                // No view direction available in emitted(), so we can't compute
                // a proper Fresnel term. Sum both — in practice coatings don't
                // emit, so this just returns the substrate's emission.
                coating.emitted(si) + substrate.emitted(si)
            }
        }
    }

    /// Returns `true` if this material emits light.
    /// Recursively checks composition variants.
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

    /// Returns `true` if this material is a pure delta distribution.
    /// Recursively checks composition variants: `Mix` is delta iff both children are.
    pub fn is_delta(&self) -> bool {
        match self {
            Material::Dielectric(_) => true,
            Material::Metal(inner) => inner.fuzz < 1e-4,
            Material::Glossy(inner) => inner.roughness < 1e-4,
            Material::Mix { a, b, .. } => a.is_delta() && b.is_delta(),
            Material::Coated { substrate, coating } => substrate.is_delta() && coating.is_delta(),
            Material::Custom(inner) => inner.is_delta(),
            _ => false,
        }
    }
}

impl Bsdf for Material {
    fn sample(&self, wo: Vec3, si: &SurfaceInteraction, dims: SampleDims) -> Option<BsdfSample> {
        self.sample(wo, si, dims)
    }

    fn eval(&self, wo: Vec3, wi: Vec3, si: &SurfaceInteraction) -> Color3 {
        self.eval(wo, wi, si)
    }

    fn pdf(&self, wo: Vec3, wi: Vec3, si: &SurfaceInteraction) -> f64 {
        self.pdf(wo, wi, si)
    }

    fn pdf_kind(&self, wo: Vec3, si: &SurfaceInteraction) -> Option<PdfKind> {
        self.pdf_kind(wo, si)
    }

    fn emitted(&self, si: &SurfaceInteraction) -> Color3 {
        self.emitted(si)
    }

    fn is_emissive(&self) -> bool {
        Material::is_emissive(self)
    }

    fn is_delta(&self) -> bool {
        Material::is_delta(self)
    }

    fn clone_box(&self) -> Box<dyn Bsdf> {
        Box::new(self.clone())
    }

    fn serialize_gpu(&self, buf: &mut GpuMaterialBuffer) -> u32 {
        write_node(self, buf)
    }
}

impl Material {
    /// Absence of material. All BSDF methods return zero/None.
    pub fn void() -> Self {
        Material::Void
    }

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

    /// Microfacet conductor (GGX). `fuzz` ∈ [0, 1] controls roughness.
    pub fn metal(albedo: Color3, fuzz: f64) -> Self {
        Material::Metal(MetalMaterial {
            albedo,
            tex: None,
            fuzz,
            ior: 2.5,
            r0: fresnel_r0(2.5),
        })
    }

    /// Microfacet conductor with an explicit IOR for the Fresnel term.
    pub fn metal_with_ior(albedo: Color3, fuzz: f64, ior: f64) -> Self {
        Material::Metal(MetalMaterial {
            albedo,
            tex: None,
            fuzz,
            ior,
            r0: fresnel_r0(ior),
        })
    }

    /// Glass / dielectric material with refractive index.
    pub fn dielectric(ior: f64) -> Self {
        Material::Dielectric(DielectricMaterial {
            refractive_idx: ior,
            tint: Color3::from(1., 1., 1.),
            r0: fresnel_r0(ior),
        })
    }

    /// Dielectric with a colored tint (absorption per channel).
    pub fn dielectric_tinted(ior: f64, tint: Color3) -> Self {
        Material::Dielectric(DielectricMaterial {
            refractive_idx: ior,
            tint,
            r0: fresnel_r0(ior),
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
            r0: fresnel_r0(ior),
        })
    }

    /// Microfacet conductor with a textured albedo.
    pub fn metal_textured(tex: Arc<dyn Texture>, fuzz: f64) -> Self {
        Material::Metal(MetalMaterial {
            albedo: Color3::ZERO,
            tex: Some(tex),
            fuzz,
            ior: 2.5,
            r0: fresnel_r0(2.5),
        })
    }

    /// Microfacet conductor with a textured albedo and explicit IOR.
    pub fn metal_textured_with_ior(tex: Arc<dyn Texture>, fuzz: f64, ior: f64) -> Self {
        Material::Metal(MetalMaterial {
            albedo: Color3::ZERO,
            tex: Some(tex),
            fuzz,
            ior,
            r0: fresnel_r0(ior),
        })
    }

    /// Glossy microfacet BRDF with a textured albedo.
    pub fn glossy_textured(tex: Arc<dyn Texture>, roughness: f64, ior: f64) -> Self {
        Material::Glossy(GlossyMaterial {
            albedo: Color3::ZERO,
            tex: Some(tex),
            roughness,
            ior,
            r0: fresnel_r0(ior),
        })
    }

    /// Stochastic mix of two materials. `weight` ∈ [0, 1] is the probability
    /// of choosing `b`.
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
/// Returns the probability density that a microfacet has half-vector H aligned
/// with the surface normal. `alpha` is roughness²; controls specular lobe width.
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
/// Models microfacet self-shadowing at grazing angles. Returns a multiplier
/// in [0, 1]. `roughness` is RMS surface slope (not squared).
pub(super) fn geometry_schlick_ggx(cos_theta: f64, roughness: f64) -> f64 {
    if cos_theta <= 0.0 {
        return 0.0;
    }
    let k = (roughness + 1.0).powi(2) / 8.0;
    cos_theta / (cos_theta * (1.0 - k) + k)
}

/// Precomputed Fresnel reflectance at normal incidence for a given IOR.
#[inline(always)]
pub(super) fn fresnel_r0(ior: f64) -> f64 {
    ((1.0 - ior) / (1.0 + ior)).powi(2)
}

/// Fresnel r0 for the Coated material's hardcoded IOR of 1.5.
pub(super) const COATED_R0: f64 = 0.04;

/// Schlick Fresnel reflectance for unpolarized light.
///
/// Approximates the fraction of light reflected at a dielectric interface.
/// Approaches 1 at grazing angles. `r0` is reflectance at normal incidence.
pub(super) fn fresnel_schlick(cos_theta: f64, r0: f64) -> f64 {
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
        let mat = inner.coated(Material::dielectric_tinted(
            1.5,
            Color3::from(1.0, 0.8, 0.8),
        ));
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
                _si: &SurfaceInteraction,
                _dims: SampleDims,
            ) -> Option<BsdfSample> {
                None
            }
            fn eval(&self, _wo: Vec3, _wi: Vec3, _si: &SurfaceInteraction) -> Color3 {
                Color3::from(0., 0., 0.)
            }
            fn pdf(&self, _wo: Vec3, _wi: Vec3, _si: &SurfaceInteraction) -> f64 {
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
