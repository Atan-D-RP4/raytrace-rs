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

pub struct ConstantMedium<T: Intersectable, const SURFACE: bool = true> {
    /// The boundary defines the shape of the volume (e.g., a sphere or box).
    boundary: T,
    /// The phase function determines how light scatters within the volume.
    phase_fn: Material,
    /// The negative inverse of the density is precomputed for efficient sampling of scattering events.
    neg_inv_density: f64,
    /// Deterministic seed for volume QMC sampling.
    vol_seed: u64,
}

impl<T: Intersectable> ConstantMedium<T> {
    pub fn new(boundary: T, density: f64, phase_fn: Material) -> Self {
        // Derive deterministic seed from density and the phase function's identity.
        // Uses the material's type tag bits rather than a heap pointer, so the seed
        // is reproducible across runs (not affected by ASLR / allocator state).
        let phase_tag = match &phase_fn {
            Material::Lambertian(_) => 0x01,
            Material::Metal(_) => 0x02,
            Material::Dielectric(_) => 0x03,
            Material::DiffuseLight(_) => 0x04,
            Material::Isotropic(_) => 0x05,
            Material::Glossy(_) => 0x06,
            Material::Mix { .. } => 0x07,
            Material::Coated { .. } => 0x08,
            Material::Custom(_) => 0x09,
        };
        let seed = sampler::splitmix64(
            density
                .to_bits()
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(phase_tag),
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
    pub fn new_texture(boundary: T, density: f64, tex: Arc<dyn Texture>) -> Self {
        let seed = sampler::splitmix64(density.to_bits() ^ 0x7E57A35E5EED);
        Self {
            boundary,
            neg_inv_density: -1.0 / density,
            phase_fn: Material::isotropic_texture(tex),
            vol_seed: seed,
        }
    }

    /// Construct a constant medium with a uniform albedo (isotropic scattering) for pure uniform-sphere phase.
    pub fn new_albedo(boundary: T, density: f64, albedo: Vec3) -> Self {
        let seed = sampler::splitmix64(
            density
                .to_bits()
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(albedo.x.to_bits()),
        );
        Self {
            boundary,
            neg_inv_density: -1.0 / density,
            phase_fn: Material::isotropic(albedo),
            vol_seed: seed,
        }
    }
}

impl<T: Intersectable + Bounded, const SURFACE: bool> Intersectable for ConstantMedium<T, SURFACE> {
    fn intersect<'a>(&'a self, ray: &Ray, ray_t: Interval) -> Option<MaterialHit<'a>> {
        // Always find the nearest boundary crossing (could be in front or behind ray)
        let rec1 = self.boundary.intersect(ray, Interval::UNIVERSE)?;

        // Outside the boundary → entry surface hit
        if rec1.hit.time >= ray_t.min && SURFACE {
            return self.boundary.intersect(ray, ray_t);
        }

        // Inside the boundary → sample volume scattering distance and check if it occurs before exiting.
        let rec2 = self
            .boundary
            .intersect(ray, Interval::from(rec1.hit.time + 0.0001, f64::INFINITY))?;

        // Compute the valid interval of ray parameter inside the boundary.
        let t_min = rec1.hit.time.max(ray_t.min);
        let t_max = rec2.hit.time.min(ray_t.max);

        // If the interval is invalid, no hit occurs.
        if t_min >= t_max {
            return None;
        }

        // If the ray starts inside the boundary, we need to clamp t_min to 0 to get the correct distance inside.
        let t_min = t_min.max(0.);

        // Compute the distance the ray travels inside the boundary.
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

        // Sample a scattering distance based on the medium's density using inverse transform sampling.
        let hit_dist = self.neg_inv_density * qmc_sample.max(1e-12).ln();

        // If the sampled scattering distance exceeds the distance inside the boundary, no hit occurs.
        if hit_dist > dist_inside_boundary {
            return None;
        }

        // Otherwise, we have a valid scattering event at distance hit_dist along the ray inside the medium.
        let new_time = t_min + hit_dist / ray_length;
        let point = ray.at(new_time);

        Some(MaterialHit {
            hit: Hit {
                time: new_time,
                point,
                mapping_point: point,
                geometric_normal: Vec3::from(0., 0., 0.),
                uv: None,
            },
            material: &self.phase_fn,
        })
    }
}

impl<T: Intersectable, const SURFACE: bool> Bounded for ConstantMedium<T, SURFACE> {
    fn bounding_box(&self) -> Aabb {
        self.boundary.bounding_box()
    }
}
