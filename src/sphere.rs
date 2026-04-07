use std::f64::consts::PI;
use std::sync::Arc;

use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable};
use crate::interval::Interval;
use crate::material::Material;
use crate::ray::Ray;
use crate::vec3::{dot, Point3, Vec3};

pub struct Sphere {
    center: Ray,
    pub radius: f64,
    pub material: Arc<Material>,
    bbox: Aabb,
}

impl Sphere {
    pub fn new(center: &Point3, radius: f64, mat: Arc<Material>) -> Self {
        let rvec = Vec3::from(radius, radius, radius);
        Self {
            center: Ray::new(*center, Vec3::from(0., 0., 0.)),
            radius: radius.max(0.0),
            material: mat.clone(),
            bbox: Aabb::from_points(&(*center - rvec), &(*center + rvec)),
        }
    }

    pub fn new_moving(
        center_start: &Point3,
        center_end: &Point3,
        radius: f64,
        mat: Arc<Material>,
    ) -> Self {
        let rvec = Vec3::from(radius, radius, radius);
        let center = Ray::new(*center_start, *center_end - *center_start);
        let box1 = Aabb::from_points(&(center.at(0.) - rvec), &(center.at(0.) + rvec));
        let box2 = Aabb::from_points(&(center.at(1.) - rvec), &(center.at(1.) + rvec));
        Self {
            center,
            radius: radius.max(0.0),
            material: mat,
            bbox: Aabb::merge(box1, box2),
        }
    }

    pub fn get_sphere_uv(&self, point: &Point3) -> (f64, f64) {
        let theta = (-point.y).acos();
        let phi = -point.z.atan2(point.x) + PI;

        let u = phi / (2.0 * PI);
        let v = theta / PI;

        (u, v)
    }
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord> {
        let current_center = self.center.at(ray.time);
        let origin_center = current_center - ray.origin;
        let a = ray.direction.length_squared(); // Simplified from: `dot(&ray.direction, &ray.direction)`, which equals current `a`
        let h = dot(&ray.direction, &origin_center); // if b = -2h
        let c = origin_center.length_squared() - (self.radius * self.radius); // Simplified same as `a`

        let discriminant = (h * h) - (a * c); // Simplified form (-b ± sqrt(b*b - 4*a*c)) / 2*a

        if discriminant < 0.0 {
            return None;
        };

        let sqrtd = discriminant.sqrt();

        let mut root = (h - sqrtd) / a;
        if ray_t.max <= root || root <= ray_t.min {
            root = (h + sqrtd) / a;
            if ray_t.max <= root || root <= ray_t.min {
                return None;
            }
        }

        let point = ray.at(root);
        let outward_normal = (point - current_center) / self.radius;

        let mut hit_rec = HitRecord::new(
            root,
            point,
            outward_normal,
            outward_normal,
            self.material.clone(),
        );
        hit_rec.set_face_normal(ray, &outward_normal);

        (hit_rec.u, hit_rec.v) = self.get_sphere_uv(&outward_normal);

        Some(hit_rec)
    }

    fn bounding_box(&self) -> Aabb {
        self.bbox
    }
}
