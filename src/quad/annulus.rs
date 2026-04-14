use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable};
use crate::interval::Interval;
use crate::material::Material;
use crate::ray::Ray;

use super::{PlanarPatch, Region2D};

#[derive(Clone)]
pub struct Annulus {
    patch: PlanarPatch,
    inner: f64,
}

impl Annulus {
    pub fn new(
        center: crate::vec3::Point3,
        side_a: crate::vec3::Vec3,
        side_b: crate::vec3::Vec3,
        inner: f64,
        material: Material,
    ) -> Self {
        Self {
            patch: PlanarPatch::new(center, side_a, side_b, material),
            inner,
        }
    }
}

impl Region2D for Annulus {
    fn contains(a: f64, b: f64) -> bool {
        (a * a + b * b) <= 1.0
    }

    fn uv(a: f64, b: f64) -> (f64, f64) {
        (a * 0.5 + 0.5, b * 0.5 + 0.5)
    }
}

impl Hittable for Annulus {
    fn hit<'a>(&'a self, ray: &Ray, ray_t: Interval) -> Option<HitRecord<'a>> {
        let hit = self.patch.hit_plane(ray, ray_t)?;
        let radius = (hit.a * hit.a + hit.b * hit.b).sqrt();
        if radius < self.inner || radius > 1.0 {
            return None;
        }

        let (u, v) = Self::uv(hit.a, hit.b);
        Some(self.patch.make_hit_record(ray, hit, u, v))
    }

    fn bounding_box(&self) -> Aabb {
        self.patch.bounding_box()
    }
}
