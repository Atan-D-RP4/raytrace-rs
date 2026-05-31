//! Material models and scattering behavior for path tracing.
//!
//! A material maps an incoming ray + surface hit into either:
//! - a scattered ray with attenuation, or
//! - absorption (`None`).

use std::f64::consts::PI;
use std::sync::Arc;

use rand::RngExt;

use crate::hittable::HitRecord;
use crate::onb::Onb;
use crate::ray::Ray;
use crate::texture::{Texture, TextureCoords};
use crate::vec3::{Color3, random_cosine_direction, random_unit_vector_with_rng, reflect, refract};

/// Result of material sampling for a single bounce.
pub struct Scatter {
    /// Multiplicative color throughput for this bounce.
    pub attenuation: Color3,
    /// Outgoing ray sampled by the material.
    pub scattered: Ray,
    /// PDF value for the sampled ray direction, used by materials with non-uniform scattering (e.g.
    /// Lambertian cosine-weighted hemisphere sampling).
    pub pdf: f64,
}

impl Scatter {
    /// Builds a scatter result from attenuation and outgoing ray.
    pub fn new(attenuation: Color3, scattered: Ray, pdf: f64) -> Self {
        Self {
            attenuation,
            scattered,
            pdf,
        }
    }
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
                // let mut scatter_direction = record.normal + random_unit_vector_with_rng(rng);
                // let mut scatter_direction = random_on_hemisphere(rng, record.normal);

                let onb = Onb::build_from_normal(record.normal);
                let scatter_direction = onb.local_to_world(random_cosine_direction(rng));

                // if scatter_direction.near_zero() {
                //     scatter_direction = record.normal;
                // }

                let scattered_ray = Ray::new_with_time(record.point, scatter_direction, ray.time);
                let attenuation = tex.value(&record.texture_coords());
                let pdf = onb.w.dot(&scatter_direction) / PI;
                Some(Scatter::new(attenuation, scattered_ray, pdf))
            }
            Material::Metal { albedo, fuzz } => {
                let reflected = reflect(&ray.direction.unit_vector(), &record.normal);
                let scattered_ray = Ray::new_with_time(
                    record.point,
                    reflected + (*fuzz * random_unit_vector_with_rng(rng)),
                    ray.time,
                );
                // Perfectly specular reflection has a delta distribution, so PDF is
                // 1 for the reflected direction and 0 elsewhere.
                let pdf = 1.0;

                if scattered_ray.direction.dot(&record.normal) > 0.0 {
                    Some(Scatter::new(*albedo, scattered_ray, pdf))
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
                let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

                let direction =
                    if ri * sin_theta > 1.0 || self.reflectance(cos_theta) > rng.random::<f64>() {
                        reflect(&unit_dir, &record.normal)
                    } else {
                        refract(&unit_dir, &record.normal, ri)
                    };

                let scattered_ray = Ray::new_with_time(record.point, direction, ray.time);
                let pdf = 1.0;

                Some(Scatter::new(attenuation, scattered_ray, pdf))
            }
            Material::Isotropic { tex } => {
                // Scatters incoming rays uniformly in all directions (unit-sphere directions).
                // Used to model translucent fog volumes and atmospheric effects
                // where no preferential scattering direction exists. The albedo is
                // given by the attached texture evaluated at hit coordinates.
                let scattered_ray =
                    Ray::new_with_time(record.point, random_unit_vector_with_rng(rng), ray.time);
                let attenuation = tex.value(&TextureCoords::new(
                    record.u,
                    record.v,
                    record.point,
                    record.mapping_point,
                    record.geometry_normal,
                ));
                let pdf = 1.0 / (4.0 * PI);
                Some(Scatter::new(attenuation, scattered_ray, pdf))
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

    pub fn scattering_pdf(&self, _ray_in: &Ray, record: &HitRecord, scattered: &Ray) -> f64 {
        match self {
            Material::Lambertian { tex: _ } => {
                let cos_theta = record.normal.dot(&scattered.direction.unit_vector());
                if cos_theta < 0. {
                    0.0
                } else {
                    cos_theta / std::f64::consts::PI
                }
            }

            Material::Isotropic { tex: _ } => 1.0 / (4.0 * PI),
            _ => 0.,
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
