use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable};
use crate::interval::Interval;
use crate::material::Material;
use crate::ray::Ray;

use super::{PlanarPatch, Region2D};

#[derive(Clone)]
pub struct Tri {
    patch: PlanarPatch,
}

impl Tri {
    pub fn new(
        corner: crate::vec3::Point3,
        side_a: crate::vec3::Vec3,
        side_b: crate::vec3::Vec3,
        material: Material,
    ) -> Self {
        Self {
            patch: PlanarPatch::new(corner, side_a, side_b, material),
        }
    }
}

impl Region2D for Tri {
    fn contains(a: f64, b: f64) -> bool {
        a >= 0.0 && b >= 0.0 && (a + b) <= 1.0
    }
}

impl Hittable for Tri {
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
}
