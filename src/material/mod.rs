//! Material models and scattering behavior for path tracing.
//!
//! Materials form a **tree of BSDFs** — the [`Material`] enum is recursive via
//! `Box<Material>` in composition variants ([`Material::Mix`], [`Material::Coated`]).
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
//! # GPU Serialization
//!
//! The material tree can be flattened into a GPU-friendly buffer via
//! [`Material::to_gpu_buffer`]. Each node is a [`GpuMaterialNode`] with
//! optional child indices. The shader mirrors the CPU's enum match via a
//! switch on `material_type`.

mod gpu;

#[cfg(test)]
use gpu::GPU_NONE;
use gpu::write_node;

pub use gpu::{GpuMaterialBuffer, GpuMaterialNode, GpuMaterialType};

use std::f64::consts::PI;
use std::sync::Arc;

use rand::RngExt;

use crate::hittable::HitRecord;
use crate::onb::Onb;
use crate::pdf::GgxSamplePDF;
use crate::pdf::{CosinePDF, PDF, UniformSpherePDF};
use crate::ray::Ray;
use crate::texture::Texture;
use crate::vec3::{Color3, Vec3, random_unit_vector_with_rng, reflect, refract};

/// Result of material sampling for a single bounce.
///
/// Diffuse materials return only the surface PDF — the integrator builds the
/// mixture with light sampling and generates the direction. Specular
/// materials return the fully-determined scattered ray.
pub enum Scatter<'a> {
    /// Diffuse path: integrator samples direction from a mixture of the
    /// surface PDF and light PDF.
    Diffuse {
        /// Multiplicative color throughput for this bounce.
        attenuation: Color3,
        /// Material's surface sampling PDF (e.g. cosine-weighted hemisphere).
        /// The integrator combines this with light sampling into a mixture.
        surface_pdf: Box<dyn PDF + 'a>,
    },
    /// Specular path: direction is fully determined by the material
    /// (mirror reflection or dielectric refraction).
    Specular {
        /// Multiplicative color throughput for this bounce.
        attenuation: Color3,
        /// Outgoing ray with the determined direction.
        scattered: Ray,
    },
}

/// Supported material models.
#[derive(Clone)]
pub enum Material {
    /// Diffuse (Lambertian) surface.
    Lambertian {
        albedo: Color3,
        /// Optional texture for spatial variation. When set, the texture's
        /// `value()` is used at hit time instead of `albedo`. The GPU buffer
        /// representation falls back to `albedo` (CPU-only feature).
        tex: Option<Arc<dyn Texture>>,
    },
    /// Mirror-like reflection with optional roughness (`fuzz`).
    Metal { albedo: Color3, fuzz: f64 },
    /// Dielectric transmission/reflection controlled by refractive index.
    Dielectric { refractive_idx: f64 },
    /// Light emitting surface.
    DiffuseLight {
        emit: Color3,
        /// Optional texture for emission. CPU-only.
        tex: Option<Arc<dyn Texture>>,
    },
    /// Isotropic scattering medium (volumes).
    Isotropic {
        albedo: Color3,
        /// Optional texture for spatial variation. CPU-only.
        tex: Option<Arc<dyn Texture>>,
    },
    /// Glossy microfacet BRDF (GGX). Use `roughness` ∈ [0,1] for surface
    /// smoothness; `ior` is the index of refraction used by the Fresnel
    /// term.
    Glossy {
        albedo: Color3,
        roughness: f64,
        ior: f64,
    },
    /// Stochastic mix of two materials, weighted by a scalar in [0, 1].
    ///
    /// At each scattering event, we pick either `a` or `b` with probability
    /// `weight` (where `weight = 0` means always `a`, `weight = 1` always
    /// `b`). This is PBRT-v4's `MixMaterial`.
    Mix {
        a: Box<Material>,
        b: Box<Material>,
        /// Selection probability for `b`.
        weight: f64,
    },
    /// A vertical layer: light hits `coating` first; if it transmits, it
    /// interacts with `substrate`. Used for clear coats, varnishes, etc.
    ///
    /// This is a simplified single-bounce approximation: the coating is
    /// treated as a thin dielectric that reflects via Schlick Fresnel and
    /// transmits (1 - F) of the energy to the substrate. PBRT-v4's full
    /// `LayeredBxDF` does multi-bounce transport; we defer that.
    Coated {
        substrate: Box<Material>,
        coating: Box<Material>,
    },
}

impl Material {
    /// Samples this material for a given incoming ray and hit record.
    ///
    /// For diffuse materials, returns a [`Scatter::Diffuse`] containing only
    /// the surface PDF. The integrator is responsible for building the
    /// mixture PDF with light sampling and generating the scattered
    /// direction.
    ///
    /// For specular materials, returns a [`Scatter::Specular`] with the
    /// fully-determined scattered ray.
    ///
    /// Returns `None` for light-emitting materials (no scattering).
    /// Ray time is preserved across bounces for motion blur.
    pub fn scatter<R: rand::Rng + ?Sized>(
        &self,
        ray: &Ray,
        record: &HitRecord,
        rng: &mut R,
    ) -> Option<Scatter<'_>> {
        match self {
            Material::Lambertian { albedo, tex } => {
                let attenuation = tex
                    .as_ref()
                    .map(|t| t.value(&record.texture_coords()))
                    .unwrap_or(*albedo);
                let surface_pdf = Box::new(CosinePDF::new(record.normal));
                Some(Scatter::Diffuse {
                    attenuation,
                    surface_pdf,
                })
            }
            Material::Metal { albedo, fuzz } => {
                let reflected = reflect(&ray.direction.unit_vector(), &record.normal);
                let direction = reflected + (*fuzz * random_unit_vector_with_rng(rng));
                let length_sq = direction.length_squared();
                if length_sq < 1e-12 {
                    return None;
                }
                let scattered =
                    Ray::new_with_time(record.point, direction / length_sq.sqrt(), ray.time);
                if scattered.direction.dot(&record.normal) > 0.0 {
                    if *fuzz > 0.0 {
                        let surface_pdf = Box::new(CosinePDF::new(reflected));
                        Some(Scatter::Diffuse {
                            attenuation: *albedo,
                            surface_pdf,
                        })
                    } else {
                        Some(Scatter::Specular {
                            attenuation: *albedo,
                            scattered,
                        })
                    }
                } else {
                    None
                }
            }
            Material::Dielectric { refractive_idx } => {
                let attenuation = Color3::from(1., 1., 1.);
                let ri = if record.front_face {
                    1.0 / refractive_idx
                } else {
                    *refractive_idx
                };
                let unit_dir = ray.direction.unit_vector();
                let cos_theta = (-unit_dir).dot(&record.normal).min(1.0);
                let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
                let direction = if ri * sin_theta > 1.0
                    || fresnel_schlick(cos_theta, *refractive_idx) > rng.random::<f64>()
                {
                    reflect(&unit_dir, &record.normal)
                } else {
                    refract(&unit_dir, &record.normal, ri)
                };
                let scattered = Ray::new_with_time(record.point, direction, ray.time);
                Some(Scatter::Specular {
                    attenuation,
                    scattered,
                })
            }
            Material::Isotropic { albedo, tex } => {
                let attenuation = tex
                    .as_ref()
                    .map(|t| t.value(&record.texture_coords()))
                    .unwrap_or(*albedo);
                let surface_pdf = Box::new(UniformSpherePDF::new());
                Some(Scatter::Diffuse {
                    attenuation,
                    surface_pdf,
                })
            }
            Material::Glossy {
                albedo,
                roughness,
                ior,
            } => {
                let alpha = (roughness * roughness).clamp(0.001, 1.0);
                let wo = -ray.direction.unit_vector();
                // Sample H from GGX (in the local frame where the normal is +y).
                let u1: f64 = rng.random();
                let u2: f64 = rng.random();
                let cos_theta = ((1.0 - u2) / (1.0 + (alpha * alpha - 1.0) * u2))
                    .clamp(0.0, 1.0)
                    .sqrt();
                let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
                let phi = 2.0 * PI * u1;
                let h_local = Vec3::from(sin_theta * phi.cos(), cos_theta, sin_theta * phi.sin());

                // Build ONB aligned with the surface normal.
                let onb = Onb::build_from_normal(record.normal);
                let h_world = onb.local_to_world(h_local);

                // Reflect wo about H to get wi.
                let wi = reflect(&wo, &h_world);

                // Discard samples that go below the surface.
                if wi.dot(&record.normal) <= 0.0 {
                    return None;
                }

                // Compute BRDF for the attenuation.
                let cos_o = wo.dot(&record.normal).max(0.0);
                let cos_i = wi.dot(&record.normal).max(0.0);
                let cos_h_o = wo.dot(&h_world).max(0.0);
                let cos_h_n = h_world.dot(&record.normal).max(0.0);

                let d = ggx_d(cos_h_n, alpha);
                let f = fresnel_schlick(cos_h_o, *ior);
                let g = geometry_schlick_ggx(cos_o, alpha) * geometry_schlick_ggx(cos_i, alpha);

                // Cook-Torrance: f_r = F * D * G / (4 * cos_o * cos_i)
                let denom = 4.0 * cos_o * cos_i;
                let brdf = if denom > 1e-12 {
                    f * d * g / denom
                } else {
                    0.0
                };

                // The PDF for this sample: D(H) * (n·H) / (4 * |wo·H|)
                let surface_pdf = Box::new(GgxSamplePDF::new(wo, record.normal, alpha));

                Some(Scatter::Diffuse {
                    // f_r * cos_i = the *integrand* for the Monte Carlo estimator.
                    attenuation: *albedo * brdf * cos_i,
                    surface_pdf,
                })
            }
            Material::Mix { a, b, weight } => {
                // Stochastic selection: pick a or b with probability weight.
                let chosen = if rng.random::<f64>() < *weight { b } else { a };
                chosen.scatter(ray, record, rng)
            }
            Material::Coated { substrate, coating } => {
                // Single-bounce approximation: choose coating vs substrate by
                // Fresnel. The coating's reflectance at this angle is evaluated
                // once and used to branch stochastically.
                //
                // We assume the coating is a thin dielectric with ior from
                // fresnel_schlick; the substrate handles the transmitted path.
                let wo = -ray.direction.unit_vector();
                let cos_o = wo.dot(&record.normal).abs();
                // Use a fixed ior=1.5 for the coat interface (clear-coat
                // assumption). A full implementation would let the coating
                // material carry its own ior.
                let coat_ior = 1.5;
                let f = fresnel_schlick(cos_o, coat_ior);
                if rng.random::<f64>() < f {
                    // Reflect off the coating.
                    coating.scatter(ray, record, rng)
                } else {
                    // Transmit through the coating to the substrate.
                    substrate.scatter(ray, record, rng)
                }
            }
            Material::DiffuseLight { .. } => None,
        }
    }

    /// Returns the emitted light color at the hit point.
    pub fn emitted(&self, hit_rec: &HitRecord) -> Color3 {
        match self {
            Material::DiffuseLight { emit, tex } if hit_rec.front_face => tex
                .as_ref()
                .map(|t| t.value(&hit_rec.texture_coords()))
                .unwrap_or(*emit),
            _ => Color3::from(0., 0., 0.),
        }
    }

    /// Evaluates the material's scattering PDF for a given incoming ray, hit record,
    /// and scattered direction. Used by the integrator in the Monte Carlo
    /// estimator: `attenuation * scattering_pdf / sampling_pdf`.
    pub fn scattering_pdf(&self, ray_in: &Ray, record: &HitRecord, scattered: &Ray) -> f64 {
        match self {
            Material::Lambertian { .. } => {
                let cos_theta = record.normal.dot(&scattered.direction.unit_vector());
                if cos_theta < 0.0 { 0.0 } else { cos_theta / PI }
            }
            Material::Metal { fuzz, .. } if *fuzz > 0.0 => {
                let reflected = reflect(&ray_in.direction.unit_vector(), &record.normal);
                let cos_alpha = reflected.dot(&scattered.direction.unit_vector());
                if cos_alpha < 0.0 { 0.0 } else { cos_alpha / PI }
            }
            Material::Isotropic { .. } => 1.0 / (4.0 * PI),
            Material::Glossy { roughness, .. } => {
                let alpha = (roughness * roughness).clamp(0.001, 1.0);
                let wo = -ray_in.direction.unit_vector();
                let wi = scattered.direction.unit_vector();
                let h = (wo + wi).unit_vector();
                let cos_h_n = h.dot(&record.normal).max(0.0);
                let cos_h_o = wo.dot(&h).max(0.0);
                if cos_h_o <= 0.0 {
                    return 0.0;
                }
                let d = ggx_d(cos_h_n, alpha);
                d * cos_h_n / (4.0 * cos_h_o)
            }
            Material::Mix { a, b, weight } => {
                // Average the PDFs weighted by selection probability.
                let pdf_a = a.scattering_pdf(ray_in, record, scattered);
                let pdf_b = b.scattering_pdf(ray_in, record, scattered);
                (1.0 - weight) * pdf_a + weight * pdf_b
            }
            Material::Coated { substrate, coating } => {
                // Same Fresnel-based split as scatter: return the weighted
                // sum of the two PDFs. (For a single-bounce coat
                // approximation.)
                let wo = -ray_in.direction.unit_vector();
                let cos_o = wo.dot(&record.normal).abs();
                let f = fresnel_schlick(cos_o, 1.5);
                let pdf_coat = coating.scattering_pdf(ray_in, record, scattered);
                let pdf_sub = substrate.scattering_pdf(ray_in, record, scattered);
                f * pdf_coat + (1.0 - f) * pdf_sub
            }
            _ => 0.0,
        }
    }
}

impl Material {
    /// Lambertian diffuse material from a solid color.
    pub fn lambertian_color(r: f64, g: f64, b: f64) -> Self {
        Self::Lambertian {
            albedo: Color3::from(r, g, b),
            tex: None,
        }
    }

    /// Lambertian diffuse material with a texture for spatial variation.
    pub fn lambertian(tex: Arc<dyn Texture>) -> Self {
        Self::Lambertian {
            albedo: Color3::ZERO,
            tex: Some(tex),
        }
    }

    /// Metal mirror with optional roughness (fuzz).
    pub fn metal(albedo: Color3, fuzz: f64) -> Self {
        Self::Metal { albedo, fuzz }
    }

    /// Glass / dielectric material with refractive index.
    pub fn dielectric(ior: f64) -> Self {
        Self::Dielectric {
            refractive_idx: ior,
        }
    }

    /// Area light emitting a constant color.
    pub fn light(emit: Color3) -> Self {
        Self::DiffuseLight { emit, tex: None }
    }

    /// Area light with a texture for spatial emission variation.
    pub fn light_textured(tex: Arc<dyn Texture>) -> Self {
        Self::DiffuseLight {
            emit: Color3::ZERO,
            tex: Some(tex),
        }
    }

    /// Isotropic scattering medium with a uniform albedo.
    pub fn isotropic(albedo: Color3) -> Self {
        Self::Isotropic { albedo, tex: None }
    }

    /// Isotropic scattering medium with a textured albedo.
    pub fn isotropic_texture(tex: Arc<dyn Texture>) -> Self {
        Self::Isotropic {
            albedo: Color3::ZERO,
            tex: Some(tex),
        }
    }

    /// Glossy microfacet BRDF (GGX).
    pub fn glossy(albedo: Color3, roughness: f64, ior: f64) -> Self {
        Self::Glossy {
            albedo,
            roughness,
            ior,
        }
    }

    /// Stochastic mix of two materials.
    ///
    /// `weight` ∈ [0, 1]: probability of choosing `b`. Use 0.5 for a 50/50
    /// blend.
    pub fn mix(self, other: Material, weight: f64) -> Self {
        let weight = weight.clamp(0.0, 1.0);
        Self::Mix {
            a: Box::new(self),
            b: Box::new(other),
            weight,
        }
    }

    /// Coat this material with a clear-coat layer (thin dielectric).
    pub fn coated(self, coat: Material) -> Self {
        Self::Coated {
            substrate: Box::new(self),
            coating: Box::new(coat),
        }
    }
}

impl Material {
    /// Flatten this material tree into a GPU-friendly buffer.
    ///
    /// The CPU material tree is a recursive enum. The GPU sees a flat array
    /// of [`GpuMaterialNode`]s; composition variants reference children by
    /// index. The shader's switch on `material_type` mirrors the CPU's
    /// match.
    pub fn to_gpu_buffer(&self) -> GpuMaterialBuffer {
        let mut buf = GpuMaterialBuffer::new();
        write_node(self, &mut buf);
        buf
    }
}

/// GGX/Trowbridge-Reitz normal distribution function.
pub fn ggx_d(cos_theta_h: f64, alpha: f64) -> f64 {
    if cos_theta_h <= 0.0 {
        return 0.0;
    }
    let a2 = alpha * alpha;
    let denom = cos_theta_h * cos_theta_h * (a2 - 1.0) + 1.0;
    a2 / (PI * denom * denom)
}

/// Smith's geometry function (Schlick-GGX approximation).
/// Accounts for self-shadowing/masking of microfacets.
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
        let mat = Material::Lambertian {
            albedo: Color3::from(0.5, 0.5, 0.5),
            tex: Some(Arc::new(SolidColor::new(Color3::from(0.7, 0.3, 0.1)))),
        };
        let buf = mat.to_gpu_buffer();
        assert_eq!(buf.nodes.len(), 1);
        // The GPU buffer should use the fallback albedo, not the texture's
        // color.
        assert_eq!(buf.params.len(), 12);
    }
}
