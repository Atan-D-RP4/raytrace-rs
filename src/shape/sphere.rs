use std::borrow::Borrow;
use std::f32::consts::PI;

use glam::Vec3;

use crate::aabb::Aabb;
use crate::hittable::{Hit, LightSample};
use crate::interval::Interval;
use crate::material::Material;
use crate::onb::Onb;
use crate::ray::{ParametricCurve, Ray};
use crate::shape::{Shape3D, ShapeObject};
use crate::texture::UVDifferentiable;
use crate::vec3::{Color3, Direction3, Point3};

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
    pub radius: f32,
}

impl SphereShape {
    /// Creates a stationary sphere at `center` with given `radius`.
    pub fn new(center: Point3, radius: f32) -> Self {
        Self {
            center: ParametricCurve::new(center, Vec3::ZERO),
            radius,
        }
    }

    /// Creates a moving sphere that interpolates its center over ray time [0, 1].
    pub fn new_moving(center_start: Point3, center_end: Point3, radius: f32) -> Self {
        Self {
            center: ParametricCurve::new(center_start, center_end - center_start),
            radius,
        }
    }

    /// Converts a unit-sphere direction into UV coordinates.
    ///
    /// Convention follows RTIOW: u ∈ [0,1) (longitude), v ∈ [0,1] (latitude).
    pub fn get_sphere_uv(p: &Point3) -> (f32, f32) {
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
    fn random_to_sphere(&self, distance_squared: f32, r1: f32, r2: f32) -> Vec3 {
        let radius = self.radius;
        let phi = 2.0 * PI * r1;
        let (sin_phi, cos_phi) = phi.sin_cos();
        let z = 1.0 + r2 * ((1.0 - (radius * radius) / distance_squared).sqrt() - 1.0);
        let r = (1.0 - z * z).sqrt();
        Vec3::new(r * cos_phi, r * sin_phi, z)
    }
}

impl UVDifferentiable for SphereShape {
    /// Returns (∂u/∂p, ∂v/∂p) where `p` is the **world-space** hit position.
    ///
    /// `mapping_point` is the unit-sphere direction `(p - center)/r`, so the chain
    /// rule contributes a `1/r` scale: `∂u/∂p_world = ∂u/∂p_mapping / r`. Without it
    /// the texture-space footprint is overstated by `r` (e.g. 2× for the earth sphere),
    /// pushing mip selection too coarse and blurring the image.
    fn uv_gradient(&self, mapping_point: &Point3) -> (Vec3, Vec3) {
        let (x, y, z) = (mapping_point.x, mapping_point.y, mapping_point.z);
        let xz2 = (x * x + z * z).max(1e-12);

        // u = (atan2(-z, x) + π) / 2π
        // ∂u/∂x = +z / (2π·xz²)
        // ∂u/∂z = -x / (2π·xz²)
        let du_dp = Vec3::new(
            z / (2.0 * PI * xz2),  // ∂u/∂x
            0.,                    // ∂u/∂y
            -x / (2.0 * PI * xz2), // ∂u/∂z
        );

        // v = acos(-y) / π  ->  d/dy acos(-y) = 1/√(1-y²)
        let sin_theta = (1.0 - y * y).sqrt().max(1e-12);
        let dv_dp = Vec3::new(0., (PI * sin_theta).recip(), 0.);

        // Scale unit-sphere (mapping) gradients to world space: d/dp_world = d/dp_mapping / r.
        let r_inv = 1.0 / self.radius;
        (du_dp * r_inv, dv_dp * r_inv)
    }
}

impl Shape3D for SphereShape {
    fn intersect_shape(&self, ray: &Ray, ray_t: Interval) -> Option<Hit> {
        // Quadratic form with h = dot(d, oc). Near root first, far root fallback.
        let current_center = self.center.at(ray.time);
        let oc = current_center - ray.origin;
        let a = ray.direction.length_squared();
        let h = ray.direction.dot(oc);
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
        let (u, v) = Self::get_sphere_uv(&Point3(outward_normal));

        Some(Hit::new(
            root,
            point,
            Point3(outward_normal),
            Direction3(outward_normal),
            Some((u, v)),
            None,
        ))
    }

    fn bounding_box(&self) -> Aabb {
        let rvec = Vec3::new(self.radius, self.radius, self.radius);
        let local = Aabb::from_points(&Point3(-rvec), &Point3(rvec));
        self.center.sweep_aabb(&local)
    }

    fn area(&self) -> f32 {
        4.0 * PI * self.radius * self.radius
    }

    fn sample(&self, u: f32, v: f32, time: f32) -> (Point3, Vec3) {
        // Uniform area sampling on the sphere surface at t=0 center.
        // Standard z = 1 - 2v, θ = 2πu parameterization.
        let center = self.center.at(time);
        let theta = 2.0 * PI * u;
        let (sin_theta, cos_theta) = theta.sin_cos();
        let z = 1.0 - 2.0 * v;
        let r = (1.0 - z * z).sqrt();
        let normal = Vec3::new(r * cos_theta, r * sin_theta, z);
        (center + normal * self.radius, normal)
    }

    fn sample_direction(&self, origin: Vec3, u: f32, v: f32, time: f32) -> Vec3 {
        // Sphere-specific solid-angle-uniform sampling via cone projection.
        // Less noisy for small spheres than the default area-based sampling.
        let center = self.center.at(time);
        let direction_to_center = center - origin;
        let distance_squared = direction_to_center.length_squared();
        let uvw = Onb::build_from_normal(direction_to_center.into_inner());
        uvw.local_to_world(self.random_to_sphere(distance_squared, u, v))
    }

    fn sample_light(&self, origin: Vec3, u: f32, v: f32, time: f32) -> LightSample {
        // sample_direction returns a unit vector via cone projection.
        let direction = self.sample_direction(origin, u, v, time);
        let center = self.center.at(time);

        // Compute actual ray-sphere intersection along the sampled direction.
        // The unit direction tells us WHERE to look; the quadratic tells us HOW FAR.
        let oc = center - origin;
        let h = direction.dot(oc.into_inner());
        let c = oc.length_squared() - self.radius * self.radius;
        let discriminant = (h * h - c).max(0.0);
        let sqrtd = discriminant.sqrt();
        let distance = (h - sqrtd).max(0.001);

        let hit_point = origin + direction * distance;
        let normal = (hit_point - center.into_inner()) / self.radius;

        // Area PDF: convert from solid-angle PDF (1/Ω) to area measure.
        // p_A(q) = p_ω(ω) · |cos θ_l| / d²
        let cos_theta = normal.dot(-direction).abs();
        let distance_squared = (center - origin).length_squared();
        let cos_theta_max = (1.0 - (self.radius * self.radius) / distance_squared)
            .sqrt()
            .min(1.0);
        let solid_angle = 2.0 * PI * (1.0 - cos_theta_max);
        let area_pdf = if solid_angle > 1e-20 {
            (cos_theta / (solid_angle * distance * distance)).max(0.0)
        } else {
            0.0
        };

        LightSample {
            direction: Direction3(direction * distance), // displacement from surface to light
            normal: Direction3(normal),
            distance,
            pdf: area_pdf,
            emission: Color3::ZERO, // filled in by ShapeObject::sample_light via material
        }
    }

    fn pdf_direction(&self, origin: Vec3, direction: Vec3, time: f32) -> f32 {
        // Sphere-specific: uniform solid-angle PDF = 1 / Ω where
        // Ω = 2π(1 - cos θ_max) is the solid angle subtended by the sphere.
        let current_center = self.center.at(time);
        let oc = current_center - origin;
        let a = direction.length_squared();
        let h = direction.dot(oc.into_inner());
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
pub fn sphere<M: Borrow<Material>>(center: Point3, radius: f32, material: M) -> Sphere<M> {
    ShapeObject::new(SphereShape::new(center, radius), material)
}

/// Creates a moving sphere that interpolates over ray time [0, 1].
pub fn moving_sphere<M: Borrow<Material>>(
    center_start: Point3,
    center_end: Point3,
    radius: f32,
    material: M,
) -> Sphere<M> {
    ShapeObject::new(
        SphereShape::new_moving(center_start, center_end, radius),
        material,
    )
}
