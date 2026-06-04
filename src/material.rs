//! Material models and scattering behavior for path tracing.
//!
//! A material maps an incoming ray + surface hit into either:
//! - a scattered ray with attenuation, or
//! - absorption (`None`).

use std::f64::consts::PI;
use std::sync::Arc;

use rand::RngExt;

use crate::hittable::HitRecord;
use crate::pdf::{CosinePDF, PDF, UniformSpherePDF};
use crate::ray::Ray;
use crate::texture::{Texture, TextureCoords};
use crate::vec3::{Color3, random_unit_vector_with_rng, reflect, refract};

/// Result of material sampling for a single bounce.
///
/// Diffuse materials return only the surface PDF — the integrator builds the
/// mixture with light sampling and generates the direction. Specular materials
/// return the fully-determined scattered ray.
pub enum Scatter<'a> {
    /// Diffuse path: integrator samples direction from a mixture of the
    /// surface PDF and light PDF.
    Diffuse {
        /// Multiplicative color throughput for this bounce.
        attenuation: Color3,
        /// Material's surface sampling PDF (e.g. cosine-weighted hemisphere).
        /// The integrator combines this with light sampling into a mixture PDF.
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
    /// Diffuse (Lambertian) surface using a texture for albedo.
    Lambertian { tex: Arc<dyn Texture> },
    /// Mirror-like reflection with optional roughness (`fuzz`).
    Metal { albedo: Color3, fuzz: f64 },
    /// Dielectric transmission/reflection controlled by refractive index.
    Dielectric { refractive_idx: f64 },
    /// Light emitting surface
    DiffuseLight { tex: Arc<dyn Texture> },
    /// Isotropic scattering medium for fog, mist, and homogeneous volumes.
    Isotropic { tex: Arc<dyn Texture> },
}

impl Material {
    /// Samples this material for a given incoming ray and hit record.
    ///
    /// For diffuse materials, returns a [`Scatter::Diffuse`] containing only
    /// the surface PDF. The integrator is responsible for building the mixture
    /// PDF with light sampling and generating the scattered direction.
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
            Material::Lambertian { tex } => {
                let attenuation = tex.value(&record.texture_coords());
                let surface_pdf = Box::new(CosinePDF::new(record.normal));
                Some(Scatter::Diffuse {
                    attenuation,
                    surface_pdf,
                })
            }
            Material::Metal { albedo, fuzz } => {
                let reflected = reflect(&ray.direction.unit_vector(), &record.normal);
                let direction = reflected + (*fuzz * random_unit_vector_with_rng(rng));
                // Guard against near-zero direction from opposing fuzz vector,
                // which would produce NaN in intersection (0/0 in quadratic).
                let length_sq = direction.length_squared();
                if length_sq < 1e-12 {
                    return None;
                }
                let scattered =
                    Ray::new_with_time(record.point, direction / length_sq.sqrt(), ray.time);
                if scattered.direction.dot(&record.normal) > 0.0 {
                    if *fuzz > 0.0 {
                        // Fuzzed metal: use cosine-weighted hemisphere PDF
                        // centered on the mirror direction. The integrator
                        // divides by the PDF, so this correctly normalizes the
                        // contribution (unlike Specular which assumes pdf=1).
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
                let direction =
                    if ri * sin_theta > 1.0 || self.reflectance(cos_theta) > rng.random::<f64>() {
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
            Material::Isotropic { tex } => {
                let attenuation = tex.value(&TextureCoords::new(
                    record.u,
                    record.v,
                    record.point,
                    record.mapping_point,
                    record.geometry_normal,
                ));
                let surface_pdf = Box::new(UniformSpherePDF::new());
                Some(Scatter::Diffuse {
                    attenuation,
                    surface_pdf,
                })
            }
            Material::DiffuseLight { tex: _ } => None,
        }
    }

    /// Returns the emitted light color at the hit point.
    ///
    /// Only `DiffuseLight` materials emit; all others return black.
    /// Emission is evaluated from the material's texture using hit record coordinates.
    pub fn emitted(&self, hit_rec: &HitRecord) -> Color3 {
        match self {
            Material::DiffuseLight { tex } if hit_rec.front_face => tex.value(&TextureCoords::new(
                hit_rec.u,
                hit_rec.v,
                hit_rec.point,
                hit_rec.mapping_point,
                hit_rec.geometry_normal,
            )),
            _ => Color3::from(0., 0., 0.),
        }
    }

    /// Evaluates the material's BRDF (bidirectional reflectance distribution
    /// function) for a given incoming ray, hit record, and scattered direction.
    ///
    /// Used by the integrator in the Monte Carlo estimator:
    /// `attenuation * scattering_pdf / sampling_pdf`.
    pub fn scattering_pdf(&self, ray_in: &Ray, record: &HitRecord, scattered: &Ray) -> f64 {
        match self {
            Material::Lambertian { tex: _ } => {
                let cos_theta = record.normal.dot(&scattered.direction.unit_vector());
                if cos_theta < 0.0 { 0.0 } else { cos_theta / PI }
            }
            Material::Metal { fuzz, .. } if *fuzz > 0.0 => {
                // Cosine-weighted hemisphere around the mirror direction.
                // PDF = cos(alpha) / PI, where alpha is the angle between
                // the scattered direction and the mirror direction.
                let reflected = reflect(&ray_in.direction.unit_vector(), &record.normal);
                let cos_alpha = reflected.dot(&scattered.direction.unit_vector());
                if cos_alpha < 0.0 { 0.0 } else { cos_alpha / PI }
            }
            Material::Isotropic { tex: _ } => 1.0 / (4.0 * PI),
            _ => 0.0,
        }
    }

    /// Schlick approximation for Fresnel reflectance.
    ///
    /// Used by dielectric materials to probabilistically choose reflection
    /// vs refraction near grazing angles.
    fn reflectance(&self, cosine: f64) -> f64 {
        match self {
            Material::Dielectric { refractive_idx } => {
                let r0 = (1.0 - refractive_idx) / (1.0 + refractive_idx);
                let r0 = r0 * r0;
                r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
            }
            _ => 0.0,
        }
    }
}
