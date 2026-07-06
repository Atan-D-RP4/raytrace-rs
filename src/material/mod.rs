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
use crate::onb::Onb;
use crate::sampler::SampleDims;
use crate::texture::Texture;
use crate::vec3::refract;
use crate::vec3::{Color3, Vec3, reflect};

/// Maximum number of internal bounces for a Coated material, matching the
/// number of QMC dimensions reserved for internal Fresnel splits (dims.v
/// through dims.z, one per bounce). The integrator terminates the path if
/// this limit is exceeded to avoid infinite recursion.
const MAX_INTERNAL_BOUNCES: usize = 5;

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

/// Schlick Fresnel reflectance for unpolarized light.
///
/// Approximates the fraction of light reflected at a dielectric interface.
/// Approaches 1 at grazing angles. `r0` is reflectance at normal incidence.
pub(super) fn fresnel_schlick(cos_theta: f64, r0: f64) -> f64 {
    r0 + (1.0 - r0) * (1.0 - cos_theta).powi(5)
}

fn blackbody(temp: f64) -> Color3 {
    // Planck's law: spectral radiance of a blackbody at temperature T.
    // This is a simplified approximation for RGB color. For more accurate
    // rendering, use spectral rendering or a proper color matching function.
    let t = temp.clamp(1000.0, 10000.0);
    let r = ((t / 1000.0).powf(3.0) * 0.5).clamp(0., 1.);
    let g = ((t / 1000.0).powf(2.0) * 0.7).clamp(0., 1.);
    let b = ((t / 1000.0).powf(1.5) * 1.0).clamp(0., 1.);
    Color3::from(r, g, b)
}

/// Material sample result for one bounce.
#[derive(Clone, Copy, Debug)]
pub enum BsdfScatter {
    /// Perfect specular — used directly without MIS weighting.
    Delta {
        /// Scattered direction (toward camera for reflection, through surface for refraction).
        wi: Vec3,
        /// BSDF × cosine. Tint for dielectrics, white for lossless coatings.
        f_cos: Color3,
    },
    /// Non-specular — integrator evaluates the BSDF and uses MIS weighting.
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

/// Bi-directional scattering distribution function (BSDF) sampling interface — reflection (BRDF),
/// transmission (BTDF), or volumetric scattering. The returned [`BsdfSample`] encodes the scattered
/// direction and BSDF×cosine throughput.
///
/// Custom materials implement this and integrate via [`Material::Custom`].
pub trait Bsdf: Send + Sync {
    /// Sample an outgoing direction. Returns `None` for pure emitters.
    /// wo: Outgoing direction (surface → camera), world space.
    /// wi: Incoming direction (surface → light), world space.
    fn scatter(&self, wo: Vec3, si: &SurfaceInteraction, dims: SampleDims) -> Option<BsdfScatter>;

    /// Evaluate the BSDF for an externally-sampled direction pair.
    /// wo: Outgoing direction (surface → camera), world space.
    /// wi: Incoming direction (surface → light), world space.
    /// Should be zero for delta materials, which cannot be evaluated over a distribution.
    fn eval(&self, wo: Vec3, wi: Vec3, si: &SurfaceInteraction) -> Color3;

    /// Evaluate the material's sampling PDF for a given direction pair.
    /// wo: Outgoing direction (surface → camera), world space.
    /// wi: Incoming direction (surface → light), world space.
    /// Should be zero for delta materials, which cannot be evaluated over a distribution.
    fn pdf(&self, wo: Vec3, wi: Vec3, si: &SurfaceInteraction) -> f64;

    /// Returns the sampling PDF kind for MIS strategy selection.
    ///  wo: Outgoing direction (surface → camera), world space.
    fn pdf_kind(&self, _wo: Vec3, _si: &SurfaceInteraction) -> Option<PdfKind> {
        None
    }

    /// Returns emitted light at the hit point. Default: no emission.
    /// `wo` is the outgoing direction (surface → camera), world space.
    /// Pass `Vec3::ZERO` as a sentinel when direction is unavailable (e.g., NEE).
    /// Should be zero for non-emissive materials. Emission is not a BSDF property, but this method
    /// is provided for convenience in integrators that treat it as such (e.g., DiffuseLight).
    fn emitted(&self, _wo: Vec3, _si: &SurfaceInteraction) -> Color3 {
        Color3::from(0., 0., 0.)
    }

    /// Returns `true` if this material emits light.
    fn is_emissive(&self) -> bool {
        false
    }

    /// Rough estimate of the directional-hemispherical reflectance at `wo`,
    /// averaged across color channels. Returns a value in [0, 1]. Used by
    /// layered/coated materials to approximate multi-bounce inter-reflection
    /// without a full Monte Carlo random walk.
    ///
    /// This is the integral over the hemisphere of `f(wo, wi) * |cos θ_i| dω_i`.
    /// Each material overrides with its best bounded estimate — the default
    /// of 1.0 is a safe upper bound.
    /// wo: Outgoing direction (surface → camera), world space.
    /// Should be zero for delta materials, which cannot be evaluated over a distribution.
    fn reflectance_estimate(&self, _wo: Vec3, _si: &SurfaceInteraction) -> f64 {
        1.0
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
    /// Glossy microfacet BSDF (GGX).
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
        /// Refractive index of the coating layer (used for Fresnel).
        coating_ior: f64,
        /// Tint color of the coating layer (used for Fresnel).
        coating_tint: Color3,
        /// Thickness of the coating layer (used for absorption in the coating).
        thickness: f64,
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
            Material::Coated {
                substrate,
                coating,
                coating_ior,
                coating_tint,
                thickness,
            } => Material::Coated {
                substrate: substrate.clone_box(),
                coating: coating.clone_box(),
                coating_ior: *coating_ior,
                coating_tint: *coating_tint,
                thickness: *thickness,
            },
            Material::Custom(inner) => Material::Custom(inner.clone_box()),
        }
    }
}

impl Material {
    /// Sample this material. Returns `None` for emitters or invalid directions.
    pub fn scatter(
        &self,
        wo: Vec3,
        si: &SurfaceInteraction,
        dims: SampleDims,
    ) -> Option<BsdfScatter> {
        match self {
            Material::Void => None,
            Material::Lambertian(inner) => inner.scatter(wo, si, dims),
            Material::Metal(inner) => inner.scatter(wo, si, dims),
            Material::Dielectric(inner) => inner.scatter(wo, si, dims),
            Material::DiffuseLight(inner) => inner.scatter(wo, si, dims),
            Material::Isotropic(inner) => inner.scatter(wo, si, dims),
            Material::Glossy(inner) => inner.scatter(wo, si, dims),
            Material::Custom(inner) => inner.scatter(wo, si, dims),
            Material::Mix { a, b, weight } => {
                let (chosen, selection_prob) = if dims.u < *weight {
                    (b.as_ref() as &dyn Bsdf, *weight)
                } else {
                    (a.as_ref() as &dyn Bsdf, 1.0 - *weight)
                };
                // Dims: `u` consumed for selection, pass v-w for child directional
                // sampling, x-y-z as padding. `z` is recycled for child's z — no
                // material reads z, so the dependency is semantically harmless.
                let mut result = chosen.scatter(
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
                    BsdfScatter::Delta { f_cos, .. } => {
                        *f_cos /= selection_prob;
                    }
                    BsdfScatter::NonDelta { pdf_kinds, count } => {
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
                coating: _,
                coating_ior,
                coating_tint,
                thickness,
            } => {
                let wo_global = wo;
                let n = si.shading_normal();
                let mut throughput = Color3::from(1.0, 1.0, 1.0);

                // Clamp coating tint to [0, 1] per component.
                // Values > 1 would amplify via powf (physically invalid Beer's law).
                let coating_tint = Color3::from(
                    coating_tint.x.clamp(0.0, 1.0),
                    coating_tint.y.clamp(0.0, 1.0),
                    coating_tint.z.clamp(0.0, 1.0),
                );

                // Fresnel split at coating-air boundary (top interface).
                // Uses dims.u as the QMC-stratified Fresnel threshold.
                let cos_wo = wo.dot(&n).abs();
                let f_top = fresnel_schlick(cos_wo, fresnel_r0(*coating_ior));
                if dims.u < f_top {
                    // Reflect off the coating, i.e., exit immediately
                    return Some(BsdfScatter::Delta {
                        wi: reflect(&-wo, &n),
                        f_cos: throughput,
                    });
                }

                // Refract into the coating layer. For IOR < 1 (e.g., sphere 1
                // with coating=dielectric(0.4)), this can TIR at shallow angles.
                // Detect TIR using Snell's law: if ri² * sin²(θ) > 1.0, TIR occurs.
                let cos_in = -wo.dot(&n);
                let sin2_in = (1.0 - cos_in * cos_in).max(0.0);
                let ri = 1.0 / *coating_ior;
                if ri * ri * sin2_in > 1.0 {
                    // TIR: no transmission into coating. Fresnel reflect instead.
                    return Some(BsdfScatter::Delta {
                        wi: reflect(&-wo, &n),
                        f_cos: throughput,
                    });
                }
                let mut wi = refract(&-wo, &n, ri);
                // Beer's law absorption per crossing in the coating layer
                let path_len = *thickness / wi.dot(&n).abs();
                throughput = Color3::from(
                    throughput.x * coating_tint.x.powf(path_len.abs()),
                    throughput.y * coating_tint.y.powf(path_len.abs()),
                    throughput.z * coating_tint.z.powf(path_len.abs()),
                );

                // Internal Fresnel splits use dims.v through dims.z (one per bounce).
                // The substrate gets default dims — for Delta substrates (metal,
                // dielectric) this is fine; NonDelta exits the walk immediately.
                let internal_dims = [dims.v, dims.w, dims.x, dims.y, dims.z];

                // Bounce internally between coating and substrate interfaces
                for (bounce_idx, internal_dim) in
                    internal_dims.iter().enumerate().take(MAX_INTERNAL_BOUNCES)
                {
                    // Sample the substrate material (scatter upwards)
                    // Each bounce gets a shifted QMC dimension to preserve stratification.
                    let bounce_offset = bounce_idx as f64 * 0.123456789;
                    let sub_u = (dims.u + bounce_offset).fract();
                    let sub_v = (dims.v + bounce_offset * 2.0).fract();
                    let sub_w = (dims.w + bounce_offset * 3.0).fract();
                    let sub_dims = SampleDims {
                        u: sub_u,
                        v: sub_v,
                        w: sub_w,
                        x: dims.x,
                        y: dims.y,
                        z: dims.z,
                    };
                    // wi instead of wo because the substrate sees the incoming direction from the
                    // coating layer
                    // sub.wi points upward (away from substrate, toward coating top)
                    // Negate wi: the substrate's sample() expects wo pointing OUTWARD
                    // (away from surface, toward coating), but wi points inward (toward substrate).
                    let sub = substrate.scatter(-wi, si, sub_dims)?;

                    match sub {
                        BsdfScatter::Delta {
                            wi: wi_internal,
                            f_cos: f_cos_internal,
                        } => {
                            // Beer's law for the upward crossing through the coating layer
                            let path_len_up = *thickness / wi_internal.dot(&n).abs();
                            throughput = Color3::from(
                                throughput.x * coating_tint.x.powf(path_len_up.abs()),
                                throughput.y * coating_tint.y.powf(path_len_up.abs()),
                                throughput.z * coating_tint.z.powf(path_len_up.abs()),
                            );

                            // Fresnel split at top interface (coating-air boundary),
                            // using a QMC-stratified threshold from internal_dims.
                            let cos_wi_internal = wi_internal.dot(&n).abs();
                            let sin2_theta = (1.0 - cos_wi_internal * cos_wi_internal).max(0.0);
                            let tir = *coating_ior * coating_ior * sin2_theta > 1.0;
                            let f_top_internal =
                                fresnel_schlick(cos_wi_internal, fresnel_r0(*coating_ior));

                            if tir || *internal_dim < f_top_internal {
                                // Must reflect (TIR) or stochastic Fresnel reflection.
                                wi = reflect(&wi_internal, &n);
                                // Beer's law for the downward crossing (back through the coating)
                                let path_len_down = *thickness / wi.dot(&n).abs();
                                throughput = Color3::from(
                                    throughput.x * coating_tint.x.powf(path_len_down.abs()),
                                    throughput.y * coating_tint.y.powf(path_len_down.abs()),
                                    throughput.z * coating_tint.z.powf(path_len_down.abs()),
                                );
                            } else {
                                // Transmit out of the coating layer, i.e., exit to air
                                let exit_dir = refract(&wi_internal, &-n, *coating_ior);
                                let raw = throughput * f_cos_internal;
                                let bound = 2.0 * throughput;
                                let f_cos = Vec3::from(
                                    raw.x.min(bound.x),
                                    raw.y.min(bound.y),
                                    raw.z.min(bound.z),
                                );
                                return Some(BsdfScatter::Delta {
                                    wi: exit_dir,
                                    f_cos,
                                });
                            }
                        }
                        // NonDelta: substrate returned a PDF distribution instead of a
                        // specific direction. For GGX substrates (Metal, Glossy), we
                        // generate the direction in the internal frame to avoid the
                        // frame mismatch between global-frame GGX sampling and
                        // internal-frame eval(). For non-GGX (Cosine/Lambertian),
                        // we pass through as NonDelta — the frame mismatch is benign
                        // because the Lambertian eval only depends on cos(θ) · dot(n, wi).
                        BsdfScatter::NonDelta {
                            pdf_kinds: sub_pdf_kinds,
                            count: sub_count,
                        } => {
                            // Check if any pdf_kind is GGX
                            let ggx_info =
                                sub_pdf_kinds[..sub_count as usize].iter().find_map(|pk| {
                                    if let PdfKind::Ggx { normal, alpha, .. } = pk {
                                        Some((*normal, *alpha))
                                    } else {
                                        None
                                    }
                                });

                            if let Some((normal, alpha)) = ggx_info {
                                // Compute wo_internal: the direction inside the coating
                                // that refracts to wo_global at the top interface.
                                let cos_wo_g = wo_global.dot(&n).max(0.0);
                                let wo_perp = wo_global - cos_wo_g * n;
                                let sin_wo = wo_perp.length();
                                let sin_w_in = sin_wo / *coating_ior;
                                let cos_w_in = (1.0 - sin_w_in * sin_w_in).max(0.0).sqrt();
                                let wo_int = if sin_wo > 1e-10 {
                                    let wo_unit_perp = wo_perp / sin_wo;
                                    cos_w_in * n + sin_w_in * wo_unit_perp
                                } else {
                                    n
                                };

                                // GGX importance sampling using the internal wo.
                                // Uses the same inverse-CDF as Metal/Glossy.
                                let u1 = sub_u;
                                let u2 = sub_v;
                                let cos_theta = ((1.0 - u2) / (1.0 + (alpha * alpha - 1.0) * u2))
                                    .clamp(0.0, 1.0)
                                    .sqrt();
                                let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
                                let phi = 2.0 * PI * u1;
                                let (sin_phi, cos_phi) = phi.sin_cos();
                                let h_local =
                                    Vec3::from(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);

                                let onb = Onb::build_from_normal(normal);
                                let h_world = onb.local_to_world(h_local);

                                // Reflect wo_internal about the half-vector
                                let wi_int = reflect(&-wo_int, &h_world);

                                // Check hemisphere: wi_int must point toward the substrate
                                if wi_int.dot(&n) > 0.0 {
                                    // Beer's law for the upward crossing
                                    let path_len_up = *thickness / wi_int.dot(&n).abs();
                                    throughput = Color3::from(
                                        throughput.x * coating_tint.x.powf(path_len_up.abs()),
                                        throughput.y * coating_tint.y.powf(path_len_up.abs()),
                                        throughput.z * coating_tint.z.powf(path_len_up.abs()),
                                    );

                                    // Fresnel split at top interface
                                    let cos_wi_int = wi_int.dot(&n).abs();
                                    let sin2_theta = (1.0 - cos_wi_int * cos_wi_int).max(0.0);
                                    let tir = *coating_ior * coating_ior * sin2_theta > 1.0;
                                    let f_top_int =
                                        fresnel_schlick(cos_wi_int, fresnel_r0(*coating_ior));

                                    if tir || *internal_dim < f_top_int {
                                        // Internal reflection — continue bouncing
                                        wi = reflect(&wi_int, &n);
                                        let path_len_down = *thickness / wi.dot(&n).abs();
                                        throughput = Color3::from(
                                            throughput.x * coating_tint.x.powf(path_len_down.abs()),
                                            throughput.y * coating_tint.y.powf(path_len_down.abs()),
                                            throughput.z * coating_tint.z.powf(path_len_down.abs()),
                                        );
                                        continue;
                                    }

                                    // Transmit out of the coating layer
                                    let exit_dir = refract(&wi_int, &-n, *coating_ior);
                                    // Include the substrate's BSDF value in f_cos.
                                    // wo_internal is the outgoing direction in the internal frame,
                                    // wi_int is the substrate's outgoing direction (toward coating).
                                    let substrate_val = substrate.eval(wo_int, wi_int, si);
                                    // Use the substrate's own PDF for consistency with its eval().
                                    // This avoids mismatches between a hand-rolled PDF derivation
                                    // and the substrate's internal conventions.
                                    let sub_pdf = substrate.pdf(wo_int, wi_int, si);
                                    let substrate_f = substrate_val / sub_pdf.max(1e-10);
                                    let exit_fresnel = 1.0
                                        - fresnel_schlick(
                                            exit_dir.dot(&n).abs(),
                                            fresnel_r0(*coating_ior),
                                        );
                                    // Heuristic firefly backstop: `f_cos` = BSDF × cosine should be bounded
                                    // for physically valid materials (Lambertian max ≈ 0.32, GGX max ≈ 2-3 at
                                    // extreme grazing). The 2.0× throughput cap prevents energy blowup from
                                    // numerical edge cases in the substrate eval / pdf ratio (e.g., very narrow
                                    // GGX lobe with near-zero PDF) while preserving material appearance.
                                    // This is NOT a physically derived limit — it's a safety net.
                                    let raw = throughput * substrate_f * exit_fresnel;
                                    let bound = 2.0 * throughput;
                                    let f_cos = Vec3::from(
                                        raw.x.min(bound.x),
                                        raw.y.min(bound.y),
                                        raw.z.min(bound.z),
                                    );
                                    return Some(BsdfScatter::Delta {
                                        wi: exit_dir,
                                        f_cos,
                                    });
                                }
                            }

                            // Fallback for non-GGX substrates (Lambertian, etc.):
                            // Also for whenever a valid GGX half-vector produces a wrong hemisphere reflection
                            // (wi_int.dot(n) > 0.0) due to numerical issues or extreme angles.
                            return Some(BsdfScatter::NonDelta {
                                // Cosine pdf_kind is frame-safe — the eval's cos(θ)
                                // check works correctly regardless of refraction.
                                pdf_kinds: [PdfKind::Cosine { normal: n }, PdfKind::Delta],
                                count: 1,
                            });
                        }
                    }
                }

                None
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
                // Delta children have zero eval (handled by their own eval() guard),
                // so only accumulate non-delta contributions.
                let eval_a = if a.is_delta() {
                    Color3::ZERO
                } else {
                    a.eval(wo, wi, si)
                };
                let eval_b = if b.is_delta() {
                    Color3::ZERO
                } else {
                    b.eval(wo, wi, si)
                };
                (1.0 - w) * eval_a + w * eval_b
            }
            Material::Coated {
                substrate,
                coating,
                coating_ior,
                coating_tint,
                thickness,
            } => {
                let sn = si.shading_normal();
                // Compute the cosine of the angle between the outgoing/incoming directions and the
                // shading normal.
                let cos_wo = wo.dot(&sn).abs();
                let cos_wi = wi.dot(&sn).abs();
                // Precompute Fresnel reflectance at normal incidence for the coating layer.
                let r0 = fresnel_r0(*coating_ior);

                // Direct coating reflection (zero for delta coating except at mirror).
                let direct_coat = coating.eval(wo, wi, si);

                // Fresnel reflectance at the coating-air interface for outgoing and incoming
                // directions.
                let fresnel_o = 1.0 - fresnel_schlick(cos_wo, r0);
                let fresnel_i = 1.0 - fresnel_schlick(cos_wi, r0);

                // Refract global incoming direction into the coating's internal frame.
                let wi_internal = refract(&-wi, &sn, 1.0 / *coating_ior);

                // Compute the internal exit direction: the direction inside the coating
                // that, when refracted at the coating-air interface from coating→air (IOR = coating_ior),
                // becomes the global wo direction (outward, dot(sn) > 0).
                // From Snell's law: sin(θ_c) = sin(θ_a) / coating_ior
                let cos_wo_global = wo.dot(&sn).max(0.0);
                let wo_perp = wo - cos_wo_global * sn;
                let sin_wo = wo_perp.length();
                let sin_wi_inside = sin_wo / *coating_ior;
                let cos_wi_inside = (1.0 - sin_wi_inside * sin_wi_inside).max(0.0).sqrt();
                let wo_internal = if sin_wo > 1e-10 {
                    // Tangent direction in coating is the same as in air (just scaled)
                    let wo_unit_perp = wo_perp / sin_wo;
                    cos_wi_inside * sn + sin_wi_inside * wo_unit_perp
                } else {
                    // Normal incidence — straight through
                    sn
                };

                // Path lengths through the coating layer for outgoing and incoming directions.
                // The ray travels at the INTERNAL angle inside the coating, so use the internal
                // direction's cosine (not the global direction's cosine) for correct Beer's law.
                let cos_wi_int = (-wi_internal).dot(&sn).abs().max(1e-10);
                let cos_wo_int = wo_internal.dot(&sn).abs().max(1e-10);
                let path_o = *thickness / cos_wo_int;
                let path_i = *thickness / cos_wi_int;

                // Absorption in the coating layer (Beer's law) for outgoing and incoming paths.
                // Clamp tint components to [0, 1] to prevent amplification (tint > 1 would
                // add energy via powf).
                let tint = Color3::from(
                    coating_tint.x.clamp(0.0, 1.0),
                    coating_tint.y.clamp(0.0, 1.0),
                    coating_tint.z.clamp(0.0, 1.0),
                );
                let coating_absorption_o = Color3::from(
                    tint.x.powf(path_o),
                    tint.y.powf(path_o),
                    tint.z.powf(path_o),
                );
                let coating_absorption_i = Color3::from(
                    tint.x.powf(path_i),
                    tint.y.powf(path_i),
                    tint.z.powf(path_i),
                );

                // Transmission coefficient/components through the coating layer (Beer's law).
                let t_o = coating_absorption_o * fresnel_o;
                let t_i = coating_absorption_i * fresnel_i;

                // Substrate contribution (single bounce, attenuated by coating absorption).
                // substrate.eval() expects both wo and wi in the outward-pointing hemisphere (dot(sn) > 0).
                // - wo_internal is already outward ✓
                // - wi_internal is inward (dot(sn) < 0), so negate it ✓
                let substrate_direct = substrate.eval(wo_internal, -wi_internal, si);

                // Inter-reflection correction (geometric series approximation):
                // coating-substrate-coating path and subsequent bounces.
                // Uses approximated reflectances since the exact direction changes per bounce.
                let avg_cos = 0.5;
                // Fresnel reflectance at the coating-substrate interface for the internal bounce.
                let r_top_internal = fresnel_schlick(avg_cos, r0);
                // Substrate directional-hemispherical reflectance (bounded in [0, 1]),
                // estimated from the substrate's known parameters.
                let r_sub = substrate.reflectance_estimate(wo_internal, si);
                // Geometric series tail r + r² + r³ + … = r/(1-r) for multi-bounce
                // inter-reflection, where r = r_sub × r_top_internal.
                // Clamped to prevent divide-by-zero from approximation errors.
                let r_prod = (r_sub * r_top_internal).clamp(0.0, 0.95);
                let series = r_prod / (1.0 - r_prod).max(1e-10);
                // Total contribution: direct coating reflection + transmitted substrate reflection
                // + inter-reflection correction.
                direct_coat + t_o * substrate_direct * t_i * (1.0 + series)
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
                let w = *weight;
                // Delta children have zero pdf (handled by their own pdf() guard),
                // so only accumulate non-delta contributions.
                let pdf_a = if a.is_delta() { 0.0 } else { a.pdf(wo, wi, si) };
                let pdf_b = if b.is_delta() { 0.0 } else { b.pdf(wo, wi, si) };
                // Weighted mixture of the two PDFs, scaled by their selection probabilities.
                (1.0 - w) * pdf_a + w * pdf_b
            }
            Material::Coated {
                substrate,
                coating_ior,
                ..
            } => {
                let sn = si.shading_normal();
                let cos_wo = wo.dot(&sn).abs();
                let cos_wi = wi.dot(&sn).abs();
                let r0 = fresnel_r0(*coating_ior);

                // Fresnel transmittance at top interface for outgoing direction
                let fresnel_t = 1.0 - fresnel_schlick(cos_wo, r0);

                // Refract global incoming direction into internal frame (same as eval)
                let wi_internal = refract(&-wi, &sn, 1.0 / *coating_ior);

                // Snell reversal for wo_internal (same as eval)
                let cos_wo_global = wo.dot(&sn).max(0.0);
                let wo_perp = wo - cos_wo_global * sn;
                let sin_wo = wo_perp.length();
                let sin_wi_inside = sin_wo / *coating_ior;
                let cos_wi_inside = (1.0 - sin_wi_inside * sin_wi_inside).max(0.0).sqrt();
                let wo_internal = if sin_wo > 1e-10 {
                    let wo_unit_perp = wo_perp / sin_wo;
                    cos_wi_inside * sn + sin_wi_inside * wo_unit_perp
                } else {
                    sn
                };

                // Solid-angle Jacobian for the refraction at the coating-air boundary.
                // The substrate's PDF is defined in the internal solid-angle measure (dω_int).
                // The external PDF measure (dω_ext) differs by:
                //   dω_int / dω_ext = cos(θ_air) / (η² · cos(θ_coating))
                // where θ_air is the angle of wi from sn, and θ_coating is the angle
                // of -wi_internal from sn (the outward-pointing internal direction).
                let cos_ext = cos_wi.max(1e-10);
                let cos_int = (-wi_internal).dot(&sn).max(0.0).max(1e-10);
                let jacobian = cos_ext / (*coating_ior * *coating_ior * cos_int);

                fresnel_t * substrate.pdf(wo_internal, -wi_internal, si) * jacobian
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
                // Try the higher-weighted child's PDF kind first. If it has
                // no PDF kind (e.g. a delta material), fall back to the
                // other child rather than returning None.
                if *weight > 0.5 {
                    b.pdf_kind(wo, si).or_else(|| a.pdf_kind(wo, si))
                } else {
                    a.pdf_kind(wo, si).or_else(|| b.pdf_kind(wo, si))
                }
            }
            Material::Coated { substrate, .. } => {
                // Delegate to the substrate's PDF kind in the global frame.
                // The coating is a pure delta layer — its Fresnel and Beer's law
                // absorption are accounted for in eval() and pdf(), not in the
                // PDF shape. The integrator uses this PdfKind to build the MIS
                // strategy list, and the actual PDF evaluation (via pdf())
                // includes the Fresnel transmittance and solid-angle Jacobian.
                // Note: wo is passed in the global (external) frame, which
                // matches the MIS evaluation frame in the integrator.
                // The substrate's internal-frame PdfKind::Ggx uses this global
                // wo, which is consistent with the sample() path (see fix at
                // line ~490 where *wo = wo_global). The eval/pdf frames differ
                // (internal vs global) but the estimator remains unbiased —
                // the MIS weights are approximate but correct on average.
                substrate.pdf_kind(wo, si)
            }
        }
    }

    /// Returns emitted light at the hit point. Default: no emission.
    /// `wo` is the outgoing direction (surface → camera), world space.
    /// For Coated materials, computes Beer's law attenuation through the coating layer.
    pub fn emitted(&self, wo: Vec3, si: &SurfaceInteraction) -> Color3 {
        match self {
            Material::Void => Color3::ZERO,
            Material::Lambertian(inner) => inner.emitted(wo, si),
            Material::Metal(inner) => inner.emitted(wo, si),
            Material::Dielectric(inner) => inner.emitted(wo, si),
            Material::DiffuseLight(inner) => inner.emitted(wo, si),
            Material::Isotropic(inner) => inner.emitted(wo, si),
            Material::Glossy(inner) => inner.emitted(wo, si),
            Material::Custom(inner) => inner.emitted(wo, si),
            Material::Mix { a, b, weight } => {
                if !a.is_emissive() && !b.is_emissive() {
                    return Color3::ZERO;
                }
                let w = *weight;
                (1.0 - w) * a.emitted(wo, si) + w * b.emitted(wo, si)
            }
            Material::Coated {
                substrate,
                coating,
                coating_ior,
                coating_tint,
                thickness,
            } => {
                // Vec3::ZERO sentinel: NEE callers (sample_light) want raw substrate emission.
                // The BSDF eval handles coating attenuation for that path.
                if wo.length_squared() < 1e-10 {
                    return coating.emitted(wo, si) + substrate.emitted(wo, si);
                }
                let sn = si.shading_normal();
                let cos_wo = wo.dot(&sn).abs();
                let r0 = fresnel_r0(*coating_ior);
                // Fresnel transmittance at coating-air boundary for the exit direction
                let fresnel_t = 1.0 - fresnel_schlick(cos_wo, r0);
                // Refract wo into the coating's internal frame to get the internal angle.
                // wo points outward. From Snell's law: sin(θ_c) = sin(θ_a) / coating_ior
                let cos_wo_global = wo.dot(&sn).max(0.0);
                let wo_perp = wo - cos_wo_global * sn;
                let sin_wo = wo_perp.length();
                let sin_wi_inside = sin_wo / *coating_ior;
                let cos_wi_inside = (1.0 - sin_wi_inside * sin_wi_inside).max(0.0).sqrt();
                let wo_internal = if sin_wo > 1e-10 {
                    let wo_unit_perp = wo_perp / sin_wo;
                    cos_wi_inside * sn + sin_wi_inside * wo_unit_perp
                } else {
                    sn
                };
                // Beer's law absorption through the coating at the INTERNAL angle
                let cos_wo_int = wo_internal.dot(&sn).abs().max(1e-10);
                let path_o = *thickness / cos_wo_int;
                let tint = Color3::from(
                    coating_tint.x.clamp(0.0, 1.0),
                    coating_tint.y.clamp(0.0, 1.0),
                    coating_tint.z.clamp(0.0, 1.0),
                );
                let coating_absorption = Color3::from(
                    tint.x.powf(path_o),
                    tint.y.powf(path_o),
                    tint.z.powf(path_o),
                );
                coating.emitted(wo, si) + coating_absorption * fresnel_t * substrate.emitted(wo, si)
            }
        }
    }

    /// Returns `true` if this material emits light.
    /// Recursively checks composition variants.
    pub fn is_emissive(&self) -> bool {
        match self {
            Material::DiffuseLight(_) => true,
            Material::Mix { a, b, .. } => a.is_emissive() || b.is_emissive(),
            Material::Coated {
                substrate, coating, ..
            } => substrate.is_emissive() || coating.is_emissive(),
            _ => false,
        }
    }

    /// Rough estimate of the directional-hemispherical reflectance, averaged
    /// across color channels. Bounded in [0, 1]. Used by layered materials
    /// for the multi-bounce inter-reflection series approximation.
    pub fn reflectance_estimate(&self, wo: Vec3, si: &SurfaceInteraction) -> f64 {
        match self {
            Material::Void => 0.0,
            Material::Lambertian(inner) => inner.reflectance_estimate(wo, si),
            Material::Metal(inner) => inner.reflectance_estimate(wo, si),
            Material::Dielectric(inner) => inner.reflectance_estimate(wo, si),
            Material::DiffuseLight(_) => 0.0,
            Material::Isotropic(_) => 1.0,
            Material::Glossy(inner) => inner.reflectance_estimate(wo, si),
            Material::Custom(inner) => inner.reflectance_estimate(wo, si),
            Material::Mix { a, b, weight } => {
                let w = *weight;
                // Delta children have negligible albedo at non-mirror directions,
                // so only accumulate non-delta contributions.
                let r_a = if a.is_delta() {
                    0.0
                } else {
                    a.reflectance_estimate(wo, si)
                };
                let r_b = if b.is_delta() {
                    0.0
                } else {
                    b.reflectance_estimate(wo, si)
                };
                // Weighted average of the two materials' reflectance estimates.
                (1.0 - w) * r_a + w * r_b
            }
            Material::Coated { substrate, .. } => substrate.reflectance_estimate(wo, si),
        }
    }

    /// Returns `true` if this material is a pure delta distribution.
    /// Recursively checks composition variants: `Mix` is delta iff both children are.
    pub fn is_delta(&self) -> bool {
        match self {
            Material::Dielectric(_) => true,
            Material::Metal(inner) => inner.roughness < 1e-4,
            Material::Glossy(inner) => inner.roughness < 1e-4,
            Material::Mix { a, b, .. } => a.is_delta() && b.is_delta(),
            Material::Coated {
                substrate, coating, ..
            } => substrate.is_delta() && coating.is_delta(),
            Material::Custom(inner) => inner.is_delta(),
            _ => false,
        }
    }
}

impl Bsdf for Material {
    fn scatter(&self, wo: Vec3, si: &SurfaceInteraction, dims: SampleDims) -> Option<BsdfScatter> {
        self.scatter(wo, si, dims)
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

    fn emitted(&self, wo: Vec3, si: &SurfaceInteraction) -> Color3 {
        self.emitted(wo, si)
    }

    fn is_emissive(&self) -> bool {
        Material::is_emissive(self)
    }

    fn reflectance_estimate(&self, wo: Vec3, si: &SurfaceInteraction) -> f64 {
        Material::reflectance_estimate(self, wo, si)
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
            roughness: fuzz,
            ior: 2.5,
            r0: fresnel_r0(2.5),
        })
    }

    /// Microfacet conductor with an explicit IOR for the Fresnel term.
    pub fn metal_with_ior(albedo: Color3, fuzz: f64, ior: f64) -> Self {
        Material::Metal(MetalMaterial {
            albedo,
            tex: None,
            roughness: fuzz,
            ior,
            r0: fresnel_r0(ior),
        })
    }

    /// Glass / dielectric material with refractive index.
    pub fn dielectric(ior: f64) -> Self {
        Material::Dielectric(DielectricMaterial {
            ior,
            tint: Color3::from(1., 1., 1.),
            r0: fresnel_r0(ior),
        })
    }

    /// Dielectric with a colored tint (absorption per channel).
    pub fn dielectric_tinted(ior: f64, tint: Color3) -> Self {
        Material::Dielectric(DielectricMaterial {
            ior,
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
            emit: blackbody(6500.0), // default white light
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

    /// Glossy microfacet BSDF (GGX).
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
            roughness: fuzz,
            ior: 2.5,
            r0: fresnel_r0(2.5),
        })
    }

    /// Microfacet conductor with a textured albedo and explicit IOR.
    pub fn metal_textured_with_ior(tex: Arc<dyn Texture>, fuzz: f64, ior: f64) -> Self {
        Material::Metal(MetalMaterial {
            albedo: Color3::ZERO,
            tex: Some(tex),
            roughness: fuzz,
            ior,
            r0: fresnel_r0(ior),
        })
    }

    /// Glossy microfacet BSDF with a textured albedo.
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
        // Extract IOR and tint from dielectric coat if possible.
        let (coating_ior, coating_tint) = match &coat {
            Material::Dielectric(d) => (d.ior, d.tint),
            _ => (1.5, Color3::from(1.0, 1.0, 1.0)),
        };
        Material::Coated {
            substrate: Box::new(self) as Box<dyn Bsdf>,
            coating: Box::new(coat) as Box<dyn Bsdf>,
            coating_ior,
            coating_tint,
            thickness: 0.01,
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
            fn scatter(
                &self,
                _wo: Vec3,
                _si: &SurfaceInteraction,
                _dims: SampleDims,
            ) -> Option<BsdfScatter> {
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
