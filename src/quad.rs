use std::sync::Arc;

use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable};
use crate::interval::Interval;
use crate::material::Material;
use crate::vec3::{Point3, Vec3, cross, dot};

pub struct Quad {
    Q: Point3,
    u: Vec3,
    v: Vec3,
    material: Arc<Material>,
    bbox: Aabb,
    normal: Vec3,
    D: f64,
    w: Vec3,
}

impl Quad {
    pub fn new(Q: Point3, u: Vec3, v: Vec3, material: Material) -> Self {
        let bbox_diagonal1 = Aabb::from_points(&Q, &(Q + u + v));
        let bbox_diagonal2 = Aabb::from_points(&(Q + u), &(Q + v));

        let n = cross(&u, &v);
        let normal = Vec3::unit_vector(&n);

        Self {
            Q,
            u,
            v,
            material: Arc::new(material),
            bbox: bbox_diagonal1.merge(bbox_diagonal2),
            normal,
            D: dot(&normal, &Q),
            w: n / dot(&n, &n),
        }
    }

    /// Given the hit point in plane coordinates, return false if it is outside the
    /// primitive, otherwise return true.
    pub fn is_interior(a: f64, b: f64) -> bool {
        let unit_interval = Interval::from(0., 1.);
        unit_interval.contains(a) && unit_interval.contains(b)
    }
}

impl Hittable for Quad {
    fn hit(
        &self,
        ray: &crate::ray::Ray,
        ray_t: crate::interval::Interval,
    ) -> Option<crate::hittable::HitRecord> {
        let denom = dot(&self.normal, &ray.direction);

        if denom.abs() < 1e-8 {
            return None;
        }

        let t = (self.D - dot(&self.normal, &ray.origin)) / denom;
        if !ray_t.contains(t) {
            return None;
        }

        let intersection = ray.at(t);
        let planar_hit_point_vector = intersection - self.Q;
        let alpha = dot(&self.w, &cross(&planar_hit_point_vector, &self.v));
        let beta = dot(&self.w, &cross(&self.u, &planar_hit_point_vector));

        let mut hit_rec = HitRecord::new(
            t,
            intersection,
            Vec3::new(),
            Vec3::new(),
            self.material.clone(),
        );
        hit_rec.set_face_normal(ray, &self.normal);

        if !Self::is_interior(alpha, beta) {
            return None;
        } else {
            hit_rec.u = alpha;
            hit_rec.v = beta;
        }

        Some(hit_rec)
    }

    fn bounding_box(&self) -> Aabb {
        self.bbox
    }
}
