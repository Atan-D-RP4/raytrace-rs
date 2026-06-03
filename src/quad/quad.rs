use rand::RngExt;

use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable};
use crate::interval::Interval;
use crate::material::Material;
use crate::quad::{PlanarPatch, Region2D};
use crate::ray::Ray;
use crate::vec3::Vec3;

#[derive(Clone)]
#[allow(non_snake_case)]
pub struct Quad {
    patch: PlanarPatch,
}

impl Quad {
    #[allow(non_snake_case)]
    pub fn new(
        Q: crate::vec3::Point3,
        u: crate::vec3::Vec3,
        v: crate::vec3::Vec3,
        material: Material,
    ) -> Self {
        Self {
            patch: PlanarPatch::new(Q, u, v, material),
        }
    }
}

impl Region2D for Quad {
    fn contains(a: f64, b: f64) -> bool {
        let unit = Interval::from(0., 1.);
        unit.contains(a) && unit.contains(b)
    }
}

impl Hittable for Quad {
    fn hit<'a>(&'a self, ray: &Ray, ray_t: Interval) -> Option<HitRecord<'a>> {
        let hit = self.patch.hit_plane(ray, ray_t)?;
        if !Self::contains(hit.a, hit.b) {
            return None;
        }

        let (u, v) = Self::uv(hit.a, hit.b);
        Some(self.patch.make_hit_record(ray, hit, u, v))
    }

    fn bounding_box(&self) -> Aabb {
        self.patch.bounding_box()
    }

    fn pdf_value(&self, origin: Vec3, direction: Vec3) -> f64 {
        if let Some(hit) = self.hit(
            &Ray::new(origin, direction),
            Interval::from(0., f64::INFINITY),
        ) {
            let distance_squared = hit.time * hit.time * direction.length_squared();
            let cosine = hit.normal.dot(&direction.unit_vector()).abs();

            distance_squared / (cosine * self.patch.area)
        } else {
            0.0
        }
    }

    fn random(&self, origin: Vec3, rng: &mut dyn rand::Rng) -> Vec3 {
        let random_point = self.patch.corner
            + (self.patch.side_a * rng.random_range(0.0..1.0))
            + (self.patch.side_b * rng.random_range(0.0..1.0));

        random_point - origin
    }
}
