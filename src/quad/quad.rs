use crate::hittable::{HitRecord, Hittable};
use crate::interval::Interval;
use crate::material::Material;
use crate::ray::Ray;

use super::{PlanarPatch, Region2D};
use crate::aabb::Aabb;

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
}
