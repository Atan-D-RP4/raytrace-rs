use crate::hittable::HitRecord;
use crate::ray::Ray;
use crate::vec3::{Color3, dot, random_unit_vector, reflect, refract, unit_vector};

pub struct Scatter {
    pub attenuation: Color3,
    pub scattered: Ray,
}

impl Scatter {
    pub fn new(attenuation: Color3, scattered: Ray) -> Self {
        Self {
            attenuation,
            scattered,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Material {
    Lambertian { albedo: Color3 },
    Metal { albedo: Color3, fuzz: f64 },
    Dielectric { refractive_idx: f64 },
}

impl Material {
    pub fn scatter(&self, ray: &Ray, record: &HitRecord) -> Option<Scatter> {
        match self {
            Material::Lambertian { albedo } => {
                let mut scatter_direction = record.normal + random_unit_vector();
                if scatter_direction.near_zero() {
                    scatter_direction = record.normal;
                }

                let scattered_ray = Ray::new(record.point, scatter_direction);
                Some(Scatter::new(*albedo, scattered_ray))
            }
            Material::Metal { albedo, fuzz } => {
                let reflected = reflect(&ray.direction.unit_vector(), &record.normal);
                let scattered_ray =
                    Ray::new(record.point, reflected + (*fuzz * random_unit_vector()));
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

                let direction = if ri * sin_theta > 1.0
                    || self.reflectance(cos_theta, ri) > rand::random::<f64>()
                {
                    reflect(&unit_dir, &record.normal)
                } else {
                    refract(&unit_dir, &record.normal, ri)
                };

                let scattered = Ray::new(record.point, direction);

                Some(Scatter::new(attenuation, scattered))
            }
        }
    }

    fn reflectance(&self, cosine: f64, refractive_idx: f64) -> f64 {
        // Schlick's approximation for reflectance
        let r0 = (1.0 - refractive_idx) / (1.0 + refractive_idx);
        let r0 = r0 * r0;
        r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
    }
}
