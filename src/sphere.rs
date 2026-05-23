use std::f64::consts::PI;
use std::sync::Arc;

use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable};
use crate::interval::Interval;
use crate::material::Material;
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3, dot};

/// Sphere primitive (static or linearly moving over ray time).
///
/// Motion is represented as a ray where `origin` is center at t=0 and
/// `direction` is center delta to t=1.
#[derive(Clone)]
pub struct Sphere {
    center: Ray,
    pub radius: f64,
    pub material: Arc<Material>,
    bbox: Aabb,
}

impl Sphere {
    /// Creates a static sphere.
    pub fn new(center: &Point3, radius: f64, mat: Material) -> Self {
        let rvec = Vec3::from(radius, radius, radius);
        Self {
            center: Ray::new(*center, Vec3::from(0., 0., 0.)),
            radius: radius.max(0.0),
            material: Arc::new(mat),
            bbox: Aabb::from_points(&(*center - rvec), &(*center + rvec)),
        }
    }

    /// Creates a moving sphere with linear center interpolation over time [0, 1].
    pub fn new_moving(
        center_start: &Point3,
        center_end: &Point3,
        radius: f64,
        mat: Material,
    ) -> Self {
        let rvec = Vec3::from(radius, radius, radius);
        let center = Ray::new(*center_start, *center_end - *center_start);
        let box1 = Aabb::from_points(&(center.at(0.) - rvec), &(center.at(0.) + rvec));
        let box2 = Aabb::from_points(&(center.at(1.) - rvec), &(center.at(1.) + rvec));
        Self {
            center,
            radius: radius.max(0.0),
            material: Arc::new(mat),
            bbox: box1.merge(box2),
        }
    }

    /// Converts a unit-sphere point into UV coordinates.
    ///
    /// Input `point` is expected on a unit sphere centered at origin.
    /// UV conventions follow RTIOW spherical mapping.
    pub fn get_sphere_uv(&self, point: &Point3) -> (f64, f64) {
        let theta = (-point.y).acos();
        let phi = -point.z.atan2(point.x) + PI;

        let u = phi / (2.0 * PI);
        let v = theta / PI;

        (u, v)
    }
}

impl Hittable for Sphere {
    /// Intersects a ray with the sphere and returns the nearest valid hit.
    ///
    /// Uses the quadratic root form optimized with `h = dot(d, oc)` and checks
    /// near root first, then far root within the supplied `ray_t` interval.
    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord<'_>> {
        let current_center = self.center.at(ray.time);
        let origin_center = current_center - ray.origin;
        let a = ray.direction.length_squared();
        let h = dot(&ray.direction, &origin_center);
        let c = origin_center.length_squared() - (self.radius * self.radius);

        let discriminant = (h * h) - (a * c);

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

        let mut hit_rec =
            HitRecord::new(root, point, outward_normal, outward_normal, &self.material);
        hit_rec.set_face_normal(ray, &outward_normal);

        (hit_rec.u, hit_rec.v) = self.get_sphere_uv(&outward_normal);

        Some(hit_rec)
    }

    fn bounding_box(&self) -> Aabb {
        self.bbox
    }
}
