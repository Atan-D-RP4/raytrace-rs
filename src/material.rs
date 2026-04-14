//! Material models and scattering behavior for path tracing.
//!
//! A material maps an incoming ray + surface hit into either:
//! - a scattered ray with attenuation, or
//! - absorption (`None`).

use std::sync::Arc;

use rand::RngExt;

use crate::hittable::HitRecord;
use crate::ray::Ray;
use crate::texture::{Texture, TextureCoords};
use crate::vec3::{Color3, dot, random_unit_vector_with_rng, reflect, refract, unit_vector};

/// Result of material sampling for a single bounce.
pub struct Scatter {
    /// Multiplicative color throughput for this bounce.
    pub attenuation: Color3,
    /// Outgoing ray sampled by the material.
    pub scattered: Ray,
}

impl Scatter {
    /// Builds a scatter result from attenuation and outgoing ray.
    pub fn new(attenuation: Color3, scattered: Ray) -> Self {
        Self {
            attenuation,
            scattered,
        }
    }
}

#[derive(Clone)]
/// Supported material models.
pub enum Material {
    /// Diffuse (Lambertian) surface using a texture for albedo.
    Lambertian { tex: Arc<dyn Texture> },
    /// Mirror-like reflection with optional roughness (`fuzz`).
    Metal { albedo: Color3, fuzz: f64 },
    /// Dielectric transmission/reflection controlled by refractive index.
    Dielectric { refractive_idx: f64 },
    /// Light emitting surface
    DiffuseLight { tex: Arc<dyn Texture> },
    ///
    Isotropic { tex: Arc<dyn Texture> },
}

impl Material {
    /// Samples this material for a given incoming ray and hit record.
    ///
    /// Returns `Some(Scatter)` when the ray continues, or `None` when the
    /// path is absorbed. Ray time is preserved across bounces for motion blur.
    pub fn scatter<R: rand::Rng + ?Sized>(
        &self,
        ray: &Ray,
        record: &HitRecord,
        rng: &mut R,
    ) -> Option<Scatter> {
        match self {
            Material::Lambertian { tex } => {
                let mut scatter_direction = record.normal + random_unit_vector_with_rng(rng);
                if scatter_direction.near_zero() {
                    scatter_direction = record.normal;
                }

                let scattered_ray = Ray::new_with_time(record.point, scatter_direction, ray.time);
                let attenuation = tex.value(&record.texture_coords());
                Some(Scatter::new(attenuation, scattered_ray))
            }
            Material::Metal { albedo, fuzz } => {
                let reflected = reflect(&ray.direction.unit_vector(), &record.normal);
                let scattered_ray = Ray::new_with_time(
                    record.point,
                    reflected + (*fuzz * random_unit_vector_with_rng(rng)),
                    ray.time,
                );
                if dot(&scattered_ray.direction, &record.normal) > 0.0 {
                    Some(Scatter::new(*albedo, scattered_ray))
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
                let unit_dir = unit_vector(ray.direction);

                let cos_theta = dot(&(-unit_dir), &record.normal).min(1.0);
                let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

                let direction =
                    if ri * sin_theta > 1.0 || self.reflectance(cos_theta) > rng.random::<f64>() {
                        reflect(&unit_dir, &record.normal)
                    } else {
                        refract(&unit_dir, &record.normal, ri)
                    };

                let scattered_ray = Ray::new_with_time(record.point, direction, ray.time);

                Some(Scatter::new(attenuation, scattered_ray))
            }
            Material::Isotropic { tex } => {
                let scattered_ray =
                    Ray::new_with_time(record.point, random_unit_vector_with_rng(rng), ray.time);
                let attenuation = tex.value(&TextureCoords::new(
                    record.u,
                    record.v,
                    record.point,
                    record.mapping_point,
                    record.geometry_normal,
                ));
                Some(Scatter::new(attenuation, scattered_ray))
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
