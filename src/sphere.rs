use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable};
use crate::interval::Interval;
use crate::material::Material;
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3, dot};

pub struct Sphere {
    center: Ray,
    pub radius: f64,
    pub material: Material,
    bbox: Aabb,
}

impl Sphere {
    // A Static sphere
    pub fn new(center: &Point3, radius: f64, mat: &Material) -> Self {
        let rvec = Vec3::from(radius, radius, radius);
        Self {
            center: Ray::new(*center, Vec3::from(0., 0., 0.)),
            radius: radius.max(0.0),
            material: *mat,
            bbox: Aabb::from_points(&(*center - rvec), &(*center + rvec)),
        }
    }

    // A Moving sphere
    pub fn new_moving(
        center_start: &Point3,
        center_end: &Point3,
        radius: f64,
        mat: &Material,
    ) -> Self {
        let rvec = Vec3::from(radius, radius, radius);
        let center = Ray::new(*center_start, *center_end - *center_start);
        let box1 = Aabb::from_points(&(center.at(0.) - rvec), &(center.at(0.) + rvec));
        let box2 = Aabb::from_points(&(center.at(1.) - rvec), &(center.at(1.) + rvec));
        Self {
            center,
            radius: radius.max(0.0),
            material: mat.clone(),
            bbox: Aabb::merge(box1, box2),
        }
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

        let mut hit_rec = HitRecord::new(root, point, outward_normal, self.material.clone().into());
        hit_rec.set_face_normal(ray, &outward_normal);

        Some(hit_rec)
    }

    fn bounding_box(&self) -> Aabb {
        self.bbox
    }
}
