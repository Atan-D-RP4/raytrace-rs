use crate::hittable::HitRecord;
use crate::ray::Ray;
use crate::vec3::{Color3, Vec3, dot, random_unit_vector, reflect};

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
            Material::Dielectric { refractive_idx: _ } => Some(Scatter::new(Vec3::new(), *ray)),
        }
    }
}
