use std::borrow::Borrow;
use std::f64::consts::PI;

use crate::aabb::Aabb;
use crate::hittable::Hit;
use crate::interval::Interval;
use crate::material::Material;
use crate::onb::Onb;
use crate::ray::{ParametricCurve, Ray};
use crate::vec3::{Point3, Vec3};

use super::{Shape3D, ShapeObject};

/// Sphere geometry defined by a linearly-moving center and radius.
///
/// Stationary when `center.velocity` is zero. The center is given in local
/// (object) space — wrap in [`TransformObject`] for scene placement.
///
/// [TransformObject]: crate::transform::TransformObject
pub struct SphereShape {
    /// Center position at t=0; `velocity` is the delta from t=0 to t=1
    /// (zero for stationary spheres).
    pub center: ParametricCurve,
    pub radius: f64,
}

impl SphereShape {
    /// Creates a stationary sphere at `center` with given `radius`.
    pub fn new(center: Point3, radius: f64) -> Self {
        Self {
            center: ParametricCurve::new(center, Vec3::ZERO),
            radius,
        }
    }

    /// Creates a moving sphere that interpolates its center over ray time [0, 1].
    pub fn new_moving(center_start: Point3, center_end: Point3, radius: f64) -> Self {
        Self {
            center: ParametricCurve::new(center_start, center_end - center_start),
            radius,
        }
    }

    /// Converts a unit-sphere direction into UV coordinates.
    ///
    /// Convention follows RTIOW: u ∈ [0,1) (longitude), v ∈ [0,1] (latitude).
    pub fn get_sphere_uv(p: &Point3) -> (f64, f64) {
        let theta = (-p.y).acos();
        let phi = -p.z.atan2(p.x) + PI;
        let u = phi / (2.0 * PI);
        let v = theta / PI;
        (u, v)
    }

    /// Samples a direction within the cone subtended by this sphere, given
    /// the squared distance from origin to sphere center.
    ///
    /// Returns a local-space unit vector where +z is toward the sphere center.
    /// Uses the RTIOW cone-sampling derivation for uniform solid-angle PDF.
    fn random_to_sphere(&self, distance_squared: f64, r1: f64, r2: f64) -> Vec3 {
        let radius = self.radius;
        let phi = 2.0 * PI * r1;
        let z = 1.0 + r2 * ((1.0 - (radius * radius) / distance_squared).sqrt() - 1.0);
        let r = (1.0 - z * z).sqrt();
        Vec3::from(r * phi.cos(), r * phi.sin(), z)
    }
}

impl Shape3D for SphereShape {
    fn intersect_shape(&self, ray: &Ray, ray_t: Interval) -> Option<Hit> {
        // Quadratic form with h = dot(d, oc). Near root first, far root fallback.
        let current_center = self.center.at(ray.time);
        let oc = current_center - ray.origin;
        let a = ray.direction.length_squared();
        let h = ray.direction.dot(&oc);
        let c = oc.length_squared() - (self.radius * self.radius);
        let discriminant = h * h - a * c;

        if discriminant < 0.0 {
            return None;
        }

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
        let (u, v) = Self::get_sphere_uv(&outward_normal);

        Some(Hit {
            time: root,
            point,
            mapping_point: outward_normal,
            geometric_normal: outward_normal,
            uv: Some((u, v)),
        })
    }

    fn bounding_box(&self) -> Aabb {
        let rvec = Vec3::from(self.radius, self.radius, self.radius);
        let local = Aabb::from_points(&(-rvec), &(rvec));
        self.center.sweep_aabb(&local)
    }

    fn area(&self) -> f64 {
        4.0 * PI * self.radius * self.radius
    }

    fn sample(&self, u: f64, v: f64) -> (Point3, Vec3) {
        // Uniform area sampling on the sphere surface at t=0 center.
        // Standard z = 1 - 2v, θ = 2πu parameterization.
        let center = self.center.at(0.0);
        let theta = 2.0 * PI * u;
        let z = 1.0 - 2.0 * v;
        let r = (1.0 - z * z).sqrt();
        let normal = Vec3::from(r * theta.cos(), r * theta.sin(), z);
        (center + normal * self.radius, normal)
    }

    fn sample_direction(&self, origin: Vec3, u: f64, v: f64) -> Vec3 {
        // Sphere-specific solid-angle-uniform sampling via cone projection.
        // Less noisy for small spheres than the default area-based sampling.
        let center = self.center.at(0.0);
        let direction_to_center = center - origin;
        let distance_squared = direction_to_center.length_squared();
        let uvw = Onb::build_from_normal(direction_to_center);
        uvw.local_to_world(self.random_to_sphere(distance_squared, u, v))
    }

    fn pdf_direction(&self, origin: Vec3, direction: Vec3) -> f64 {
        // Sphere-specific: uniform solid-angle PDF = 1 / Ω where
        // Ω = 2π(1 - cos θ_max) is the solid angle subtended by the sphere.
        let current_center = self.center.at(0.0);
        let oc = current_center - origin;
        let a = direction.length_squared();
        let h = direction.dot(&oc);
        let c = oc.length_squared() - (self.radius * self.radius);
        let discriminant = h * h - a * c;

        // No intersection → hit misses sphere → 0 PDF
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
        let cos_theta_max = (1.0 - (self.radius * self.radius) / distance_squared).sqrt();
        let solid_angle = 2.0 * PI * (1.0 - cos_theta_max);
        1.0 / solid_angle
    }
}

/// A material-wrapped sphere. The material is stored separately from the shape to allow for
/// instancing and reuse of geometry with different materials.
pub type Sphere<M> = ShapeObject<SphereShape, M>;

/// Creates a stationary sphere at `center` with `radius` and `material`.
pub fn sphere<M: Borrow<Material>>(center: Point3, radius: f64, material: M) -> Sphere<M> {
    ShapeObject::new(SphereShape::new(center, radius), material)
}

/// Creates a moving sphere that interpolates over ray time [0, 1].
pub fn moving_sphere<M: Borrow<Material>>(
    center_start: Point3,
    center_end: Point3,
    radius: f64,
    material: M,
) -> Sphere<M> {
    ShapeObject::new(
        SphereShape::new_moving(center_start, center_end, radius),
        material,
    )
}
