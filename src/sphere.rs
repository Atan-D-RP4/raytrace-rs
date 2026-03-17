use crate::hittable::{HitRecord, Hittable};
use crate::interval::Interval;
use crate::material::Material;
use crate::vec3::{dot, Point3, Vec3};

pub struct Sphere {
    pub center: Vec3,
    pub radius: f64,
    pub material: Material,
}

impl Sphere {
    pub fn new(center: &Point3, radius: f64, mat: &Material) -> Self {
        Self {
            center: *center,
            radius: radius.max(0.0),
            material: *mat,
        }
    }
}

impl Hittable for Sphere {
    fn hit(&self, ray: &crate::ray::Ray, ray_t: Interval) -> Option<HitRecord> {
        let origin_center = self.center - ray.origin;
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
        let outward_normal = (point - self.center) / self.radius;

        let mut hit_rec = HitRecord::new(root, point, outward_normal, &self.material);
        hit_rec.set_face_normal(ray, &outward_normal);

        Some(hit_rec)
    }
}
