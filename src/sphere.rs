use std::f64::consts::PI;
use std::sync::Arc;

use crate::aabb::Aabb;
use crate::hittable::{Bounded, Hit, Intersectable, MaterialHit, Sampleable};
use crate::interval::Interval;
use crate::material::Material;
use crate::onb::Onb;
use crate::ray::{ParametricCurve, Ray};
use crate::sampler::{DimCursor, Sampler};
use crate::vec3::{Point3, Vec3};

/// Sphere primitive (static or linearly moving over time).
///
/// Motion is represented as a [`ParametricCurve`] where `origin` is center at t=0
/// and `velocity` is center delta from t=0 to t=1.
#[derive(Clone)]
pub struct Sphere {
    center: ParametricCurve,
    pub radius: f64,
    pub material: Arc<Material>,
    bbox: Aabb,
}

impl Sphere {
    /// Creates a static sphere.
    pub fn new(center: &Point3, radius: f64, mat: Material) -> Self {
        let rvec = Vec3::from(radius, radius, radius);
        Self {
            center: ParametricCurve::new(*center, Vec3::ZERO),
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
        let center = ParametricCurve::new(*center_start, *center_end - *center_start);
        let box1 = Aabb::from_points(&(center.at(0.) - rvec), &(center.at(0.) + rvec));
        let box2 = Aabb::from_points(&(center.at(1.) - rvec), &(center.at(1.) + rvec));
        Self {
            center,
            radius: radius.max(0.0),
            material: Arc::new(mat),
            bbox: box1.merge(&box2),
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

    fn random_to_sphere(&self, distance_squared: f64, r1: f64, r2: f64) -> Vec3 {
        let z = 1. + r2 * ((1. - (self.radius * self.radius) / distance_squared).sqrt() - 1.);

        let phi = 2. * PI * r1;
        let x = phi.cos() * (1. - z * z).sqrt();
        let y = phi.sin() * (1. - z * z).sqrt();

        Vec3::from(x, y, z)
    }
}

impl Intersectable for Sphere {
    /// Intersects a ray with the sphere and returns the nearest valid hit.
    ///
    /// Uses the quadratic root form optimized with `h = dot(d, oc)` and checks
    /// near root first, then far root within the supplied `ray_t` interval.
    fn intersect<'a>(&'a self, ray: &Ray, ray_t: Interval) -> Option<MaterialHit<'a>> {
        let current_center = self.center.at(ray.time);
        let origin_center = current_center - ray.origin;
        let a = ray.direction.length_squared();
        let h = ray.direction.dot(&origin_center);
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
        let (u, v) = self.get_sphere_uv(&outward_normal);

        Some(MaterialHit {
            hit: Hit {
                time: root,
                point,
                mapping_point: outward_normal,
                geometric_normal: outward_normal,
                uv: Some((u, v)),
            },
            material: &self.material,
        })
    }
}

impl Bounded for Sphere {
    fn bounding_box(&self) -> Aabb {
        self.bbox
    }
}

impl<S: Sampler> Sampleable<S> for Sphere {
    fn pdf_value(&self, origin: Vec3, direction: Vec3) -> f64 {
        // Inline discriminant-only hit test — avoids constructing a temporary
        // Ray (3 divisions for `inverse_direction` that is never used) and
        // the full intersection (~50 ops) when we only need a boolean hit.
        let current_center = self.center.at(0.);
        let oc = current_center - origin;
        let a = direction.length_squared();
        let h = direction.dot(&oc);
        let c = oc.length_squared() - (self.radius * self.radius);
        let discriminant = h * h - a * c;

        if discriminant < 0.0 {
            return 0.0;
        }

        let sqrtd = discriminant.sqrt();
        let root = (h - sqrtd) / a;
        if root <= 0.001 {
            let root2 = (h + sqrtd) / a;
            if root2 <= 0.001 {
                return 0.0;
            }
        }

        let distance_squared = (current_center - origin).length_squared();
        let cos_theta_max = (1. - (self.radius * self.radius) / distance_squared).sqrt();
        let solid_angle = 2. * PI * (1. - cos_theta_max);
        1. / solid_angle
    }

    fn random(&self, origin: Vec3, dim_offset: &mut DimCursor<S>) -> Vec3 {
        let direction_to_center = self.center.at(0.0) - origin;
        let distance_squared = direction_to_center.length_squared();
        let uvw = Onb::build_from_normal(direction_to_center);

        let r1 = dim_offset.next_sample();
        let r2 = dim_offset.next_sample();
        uvw.local_to_world(self.random_to_sphere(distance_squared, r1, r2))
    }
}
