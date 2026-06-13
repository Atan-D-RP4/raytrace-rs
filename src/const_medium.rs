use std::sync::Arc;

use rand::RngExt;

use crate::aabb::Aabb;
use crate::hittable::{Bounded, Hit, Intersectable, MaterialHit};
use crate::interval::Interval;
use crate::material::Material;
use crate::ray::Ray;
use crate::texture::Texture;
use crate::vec3::Vec3;

pub struct ConstantMedium<T: Intersectable> {
    /// The boundary defines the shape of the volume (e.g., a sphere or box).
    boundary: Arc<T>,
    /// The phase function determines how light scatters within the volume.
    phase_fn: Arc<Material>,
    /// The negative inverse of the density is precomputed for efficient sampling of scattering events.
    neg_inv_density: f64,
}

impl<T: Intersectable> ConstantMedium<T> {
    pub fn new(boundary: Arc<T>, density: f64, phase_fn: Arc<Material>) -> Self {
        Self {
            boundary,
            neg_inv_density: -1.0 / density,
            phase_fn,
        }
    }

    /// Construct a constant medium with a textured phase function (isotropic
    /// scattering — correct for volumes).
    pub fn new_texture(boundary: Arc<T>, density: f64, tex: Arc<dyn Texture>) -> Self {
        Self {
            boundary,
            neg_inv_density: -1.0 / density,
            phase_fn: Arc::new(Material::isotropic_texture(tex)),
        }
    }

    /// Construct a constant medium with a uniform albedo (isotropic scattering) for pure uniform-sphere phase.
    pub fn new_albedo(boundary: Arc<T>, density: f64, albedo: Vec3) -> Self {
        Self {
            boundary,
            neg_inv_density: -1.0 / density,
            phase_fn: Arc::new(Material::isotropic(albedo)),
        }
    }
}

impl<T: Intersectable + Bounded> Intersectable for ConstantMedium<T> {
    fn intersect<'a>(&'a self, ray: &Ray, ray_t: Interval) -> Option<MaterialHit<'a>> {
        let rec1 = self.boundary.intersect(ray, Interval::UNIVERSE)?;
        let rec2 = self
            .boundary
            .intersect(ray, Interval::from(rec1.hit.time + 0.0001, f64::INFINITY))?;

        let t_min = rec1.hit.time.max(ray_t.min);
        let t_max = rec2.hit.time.min(ray_t.max);

        if t_min >= t_max {
            return None;
        }

        let t_min = t_min.max(0.);

        let ray_length = ray.direction.length();
        let dist_inside_boundary = (t_max - t_min) * ray_length;
        let mut rng = rand::rng();
        let hit_dist = self.neg_inv_density * rng.random::<f64>().max(1e-12).ln();

        if hit_dist > dist_inside_boundary {
            return None;
        }

        let new_time = t_min + hit_dist / ray_length;
        let point = ray.at(new_time);

        Some(MaterialHit {
            hit: Hit {
                time: new_time,
                point,
                geometric_normal: Vec3::from(0., 0., 0.),
                uv: None,
            },
            material: &self.phase_fn,
        })
    }
}

impl<T: Intersectable> Bounded for ConstantMedium<T> {
    fn bounding_box(&self) -> Aabb {
        self.boundary.bounding_box()
    }
}
