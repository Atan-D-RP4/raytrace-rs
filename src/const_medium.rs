use std::sync::Arc;

use crate::aabb::Aabb;
use crate::hittable::{Bounded, Hit, Intersectable, MaterialHit};
use crate::interval::Interval;
use crate::material::Material;
use crate::ray::Ray;
use crate::sampler;
use crate::texture::Texture;
use crate::vec3::Vec3;

/// Dedicated dimension for volume scattering distance.
const VOLUME_DIM: u32 = 4096;

pub struct ConstantMedium<T: Intersectable> {
    /// The boundary defines the shape of the volume (e.g., a sphere or box).
    boundary: Arc<T>,
    /// The phase function determines how light scatters within the volume.
    phase_fn: Arc<Material>,
    /// The negative inverse of the density is precomputed for efficient sampling of scattering events.
    neg_inv_density: f64,
    /// Deterministic seed for volume QMC sampling.
    vol_seed: u64,
}

impl<T: Intersectable> ConstantMedium<T> {
    pub fn new(boundary: Arc<T>, density: f64, phase_fn: Arc<Material>) -> Self {
        // Derive deterministic seed from unique properties of this medium.
        let seed = sampler::splitmix64(
            density
                .to_bits()
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(Arc::as_ptr(&phase_fn) as usize as u64),
        );
        Self {
            boundary,
            neg_inv_density: -1.0 / density,
            phase_fn,
            vol_seed: seed,
        }
    }

    /// Construct a constant medium with a textured phase function (isotropic
    /// scattering — correct for volumes).
    pub fn new_texture(boundary: Arc<T>, density: f64, tex: Arc<dyn Texture>) -> Self {
        let seed = sampler::splitmix64(density.to_bits() ^ 0x7E57A35E5EED);
        Self {
            boundary,
            neg_inv_density: -1.0 / density,
            phase_fn: Arc::new(Material::isotropic_texture(tex)),
            vol_seed: seed,
        }
    }

    /// Construct a constant medium with a uniform albedo (isotropic scattering) for pure uniform-sphere phase.
    pub fn new_albedo(boundary: Arc<T>, density: f64, albedo: Vec3) -> Self {
        let seed = sampler::splitmix64(
            density
                .to_bits()
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(albedo.x.to_bits()),
        );
        Self {
            boundary,
            neg_inv_density: -1.0 / density,
            phase_fn: Arc::new(Material::isotropic(albedo)),
            vol_seed: seed,
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

        // Deterministic QMC sample for volume scattering distance.
        // Derive n from ray direction so the same ray always gets the same
        // scattering distance (reproducible), while different rays vary.
        let n = sampler::splitmix64(
            ray.direction
                .x
                .to_bits()
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(ray.direction.y.to_bits()),
        ) as u32;
        let qmc_sample = sampler::hash_sample(n, VOLUME_DIM, self.vol_seed);
        let hit_dist = self.neg_inv_density * qmc_sample.max(1e-12).ln();

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
