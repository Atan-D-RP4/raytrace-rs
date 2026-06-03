use std::sync::Arc;

use crate::aabb::Aabb;
use crate::hittable::HitRecord;
use crate::interval::Interval;
use crate::material::Material;
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

mod annulus;
mod r#box;
mod ellipse;
mod quad;
mod tri;

pub use annulus::Annulus;
pub use r#box::box3d;
pub use ellipse::Ellipse;
pub use quad::Quad;
pub use tri::Tri;

#[derive(Clone)]
pub(crate) struct PlanarPatch {
    corner: Point3,
    side_a: Vec3,
    side_b: Vec3,
    w: Vec3,
    material: Arc<Material>,
    bbox: Aabb,
    normal: Vec3,
    d: f64,
    area: f64,
}

#[derive(Clone, Copy)]
pub(crate) struct PlanarHit {
    t: f64,
    point: Point3,
    a: f64,
    b: f64,
}

impl PlanarPatch {
    pub(crate) fn new(corner: Point3, side_a: Vec3, side_b: Vec3, material: Material) -> Self {
        let bbox_diagonal1 = Aabb::from_points(&corner, &(corner + side_a + side_b));
        let bbox_diagonal2 = Aabb::from_points(&(corner + side_a), &(corner + side_b));

        let n = side_a.cross(&side_b);
        let normal = n.unit_vector();

        Self {
            corner,
            side_a,
            side_b,
            w: n / n.dot(&n),
            material: Arc::new(material),
            bbox: bbox_diagonal1.merge(bbox_diagonal2),
            normal,
            d: normal.dot(&corner),
            area: n.length(),
        }
    }

    pub(crate) fn hit_plane(&self, ray: &Ray, ray_t: Interval) -> Option<PlanarHit> {
        let denom = self.normal.dot(&ray.direction);

        if denom.abs() < 1e-8 {
            return None;
        }

        let t = (self.d - self.normal.dot(&ray.origin)) / denom;
        if !ray_t.contains(t) {
            return None;
        }

        let point = ray.at(t);
        let planar_hit_point_vector = point - self.corner;
        let a = self.w.dot(&planar_hit_point_vector.cross(&self.side_b));
        let b = self.w.dot(&self.side_a.cross(&planar_hit_point_vector));

        Some(PlanarHit { t, point, a, b })
    }

    pub(crate) fn material(&self) -> &Material {
        self.material.as_ref()
    }

    pub(crate) fn normal(&self) -> &Vec3 {
        &self.normal
    }

    pub(crate) fn bounding_box(&self) -> Aabb {
        self.bbox
    }

    pub(crate) fn make_hit_record(
        &self,
        ray: &Ray,
        hit: PlanarHit,
        u: f64,
        v: f64,
    ) -> HitRecord<'_> {
        let mut hit_rec = HitRecord::new(hit.t, hit.point, hit.point, Vec3::new(), self.material());
        hit_rec.set_face_normal(ray, self.normal());
        hit_rec.u = u;
        hit_rec.v = v;
        hit_rec
    }
}

pub(crate) trait Region2D {
    fn contains(a: f64, b: f64) -> bool;

    fn uv(a: f64, b: f64) -> (f64, f64) {
        (a, b)
    }
}
