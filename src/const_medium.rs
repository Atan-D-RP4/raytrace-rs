use std::sync::Arc;

use rand::RngExt;

use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable};
use crate::interval::Interval;
use crate::material::Material;
use crate::ray::Ray;
use crate::texture::Texture;
use crate::vec3::Vec3;

pub struct ConstantMedium {
    /// The boundary defines the shape of the volume (e.g., a sphere or box).
    boundary: Arc<dyn Hittable>,
    /// The phase function determines how light scatters within the volume.
    phase_fn: Arc<Material>,
    /// The negative inverse of the density is precomputed for efficient sampling of scattering events.
    neg_inv_density: f64,
}

impl ConstantMedium {
    pub fn new(boundary: Arc<dyn Hittable>, density: f64, phase_fn: Arc<Material>) -> Self {
        Self {
            boundary,
            neg_inv_density: -1.0 / density,
            phase_fn,
        }
    }

    /// Construct a constant medium with a textured phase function (isotropic
    /// scattering — correct for volumes).
    pub fn new_texture(boundary: Arc<dyn Hittable>, density: f64, tex: Arc<dyn Texture>) -> Self {
        Self {
            boundary,
            neg_inv_density: -1.0 / density,
            phase_fn: Arc::new(Material::isotropic_texture(tex)),
        }
    }

    /// Construct a constant medium with a uniform albedo (isotropic scattering) for pure uniform-sphere phase.
    pub fn new_albedo(boundary: Arc<dyn Hittable>, density: f64, albedo: Vec3) -> Self {
        Self {
            boundary,
            neg_inv_density: -1.0 / density,
            phase_fn: Arc::new(Material::Isotropic { albedo, tex: None }),
        }
    }
}

impl Hittable for ConstantMedium {
    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord<'_>> {
        let mut rec1 = self
            .boundary
            .hit(ray, Interval::from(0.001, f64::INFINITY))?;
        let mut rec2 = self
            .boundary
            .hit(ray, Interval::from(rec1.time + 0.0001, f64::INFINITY))?;

        rec1.time = rec1.time.max(ray_t.min);
        rec2.time = rec2.time.min(ray_t.max);

        if rec1.time >= rec2.time {
            return None;
        }

        rec1.time = rec1.time.max(0.);

        let ray_length = ray.direction.length();
        let dist_inside_boundary = (rec2.time - rec1.time) * ray_length;
        let mut rng = rand::rng();
        let hit_dist = self.neg_inv_density * rng.random::<f64>().max(1e-12).ln();

        if hit_dist > dist_inside_boundary {
            return None;
        }

        let new_time = rec1.time + hit_dist / ray_length;
        let point = ray.at(new_time);
        // mapping_point = world position is correct for 3D procedural textures (Perlin, marble).
        // For image textures on volumes, this would be wrong - but that's unusual.
        let mut new_rec = HitRecord::new(
            new_time,
            point,
            point,
            // Volume has no real surface - normal is arbitrary, geometry_normal intentionally zero.
            Vec3::from(0., 0., 0.),
            self.phase_fn.as_ref(),
        );
        new_rec.front_face = true;

        Some(new_rec)
    }

    fn bounding_box(&self) -> Aabb {
        self.boundary.bounding_box()
    }
}
