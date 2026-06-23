use crate::aabb::Aabb;
use crate::hittable::{Bounded, Hit, Intersectable, MaterialHit, Sampleable};
use crate::interval::Interval;
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

/// A geometric transform that can map rays, hit records, and bounds.
///
/// TODO(optional): add rotation/scale variants once needed by scene builders.
/// TODO(feat): add a macro DSL for ergonomic transform chaining.
pub trait Transform: Send + Sync {
    /// Transform a ray from world space to object space. This is the inverse of the forward transform.
    fn ray(&self, ray: &Ray) -> Ray;

    /// Transform a hit record from object space to world space. This is the forward transform.
    fn hit(&self, hit: &mut Hit);

    /// Transform an axis-aligned bounding box from object space to world space.
    fn bbox(&self, bbox: Aabb) -> Aabb;

    /// Transform a direction vector from object space to world space.
    /// For rigid transforms this is the inverse rotation applied to the direction.
    /// Default assumes identity (direction unchanged), which is correct for Translate.
    /// For non-rigid transforms (e.g. scale, shear), this may not be correct and should be overridden.
    fn object_to_world_direction(&self, dir: Vec3) -> Vec3 {
        dir
    }

    /// Transform a point from world space to object space.
    ///
    /// Default implementation uses a ray to map the point, which is correct for rigid transforms (rotation + translation).
    /// For non-rigid transforms (e.g. scale, shear), this may not be correct and should be overridden.
    fn world_to_object_point(&self, point: Point3) -> Point3 {
        let ray = Ray::new_with_time(point, Vec3::X, 0.0);
        self.ray(&ray).origin
    }
}

/// A zero-cost wrapper that applies a transform to any intersectable object.
///
/// The transform and object stay generic, so the compiler can inline through
/// the whole stack without trait-object dispatch.
pub struct TransformObject<T, O: Intersectable> {
    transform: T,
    object: O,
    bbox: Aabb,
}

impl<T, O> TransformObject<T, O>
where
    T: Transform,
    O: Intersectable,
{
    pub fn new(transform: T, object: O) -> Self {
        let bbox = transform.bbox(object.bounding_box());

        Self {
            transform,
            object,
            bbox,
        }
    }
}

impl<T, O> Intersectable for TransformObject<T, O>
where
    T: Transform,
    O: Intersectable,
{
    fn intersect<'a>(&'a self, ray: &Ray, ray_t: Interval) -> Option<MaterialHit<'a>> {
        let transformed_ray = self.transform.ray(ray);
        let mut mat_hit = self.object.intersect(&transformed_ray, ray_t)?;

        self.transform.hit(&mut mat_hit.hit);

        Some(mat_hit)
    }
}

impl<T, O> Bounded for TransformObject<T, O>
where
    T: Transform,
    O: Intersectable,
{
    fn bounding_box(&self) -> Aabb {
        self.bbox
    }
}

impl<T, O> Sampleable for TransformObject<T, O>
where
    T: Transform,
    O: Sampleable,
{
    fn pdf_value(&self, origin: Vec3, direction: Vec3, time: f64) -> f64 {
        // Transform the ray into object space and delegate to the inner object.
        // For rigid transforms (rotation + translation), the Jacobian of the
        // area mapping is 1, so the solid-angle PDF is preserved.
        let ray = Ray::new_with_time(origin, direction, time);
        let transformed = self.transform.ray(&ray);
        self.object
            .pdf_value(transformed.origin, transformed.direction, time)
    }

    fn random_direction(&self, origin: Vec3, u: f64, v: f64, time: f64) -> Vec3 {
        let to_obj_origin = self.transform.world_to_object_point(origin);

        // Sample a direction in object space.
        let dir = self.object.random_direction(to_obj_origin, u, v, time);
        // Transform direction back to world space using the inverse rotation.
        self.transform.object_to_world_direction(dir)
    }
}

/// Translation transform for wrapping hittables without runtime dispatch.
pub struct Translate {
    offset: Vec3,
}

impl Translate {
    pub fn new(offset: Vec3) -> Self {
        Self { offset }
    }
}

impl Transform for Translate {
    fn ray(&self, ray: &Ray) -> Ray {
        Ray::new_with_time(ray.origin - self.offset, ray.direction, ray.time)
    }

    fn hit(&self, hit: &mut Hit) {
        hit.point += self.offset;
        hit.mapping_point += self.offset;
    }

    fn bbox(&self, bbox: Aabb) -> Aabb {
        bbox.translate(self.offset)
    }

    fn world_to_object_point(&self, point: Point3) -> Point3 {
        point - self.offset
    }
}

pub struct RotateY {
    sin_theta: f64,
    cos_theta: f64,
}

impl RotateY {
    pub fn new(angle: f64) -> Self {
        let radians = angle.to_radians();
        Self {
            sin_theta: radians.sin(),
            cos_theta: radians.cos(),
        }
    }
}

impl Transform for RotateY {
    fn ray(&self, ray: &Ray) -> Ray {
        let origin = Point3::from(
            (self.cos_theta * ray.origin.x) - (self.sin_theta * ray.origin.z),
            ray.origin.y,
            (self.sin_theta * ray.origin.x) + (self.cos_theta * ray.origin.z),
        );
        let direction = Vec3::from(
            (self.cos_theta * ray.direction.x) - (self.sin_theta * ray.direction.z),
            ray.direction.y,
            (self.sin_theta * ray.direction.x) + (self.cos_theta * ray.direction.z),
        );
        Ray::new_with_time(origin, direction, ray.time)
    }

    fn hit(&self, hit: &mut Hit) {
        hit.point = Point3::from(
            (self.cos_theta * hit.point.x) + (self.sin_theta * hit.point.z),
            hit.point.y,
            (-self.sin_theta * hit.point.x) + (self.cos_theta * hit.point.z),
        );
        hit.mapping_point = Point3::from(
            (self.cos_theta * hit.mapping_point.x) + (self.sin_theta * hit.mapping_point.z),
            hit.mapping_point.y,
            (-self.sin_theta * hit.mapping_point.x) + (self.cos_theta * hit.mapping_point.z),
        );
        hit.set_geometric_normal(Vec3::from(
            (self.cos_theta * hit.geometric_normal().x)
                + (self.sin_theta * hit.geometric_normal().z),
            hit.geometric_normal().y,
            (-self.sin_theta * hit.geometric_normal().x)
                + (self.cos_theta * hit.geometric_normal().z),
        ));
    }

    fn bbox(&self, bbox: Aabb) -> Aabb {
        let mut min = Point3::from(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut max = Point3::from(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);

        (0..2)
            .flat_map(|i| (0..2).flat_map(move |j| (0..2).map(move |k| (i, j, k))))
            .for_each(|(i, j, k)| {
                let i = i as f64;
                let j = j as f64;
                let k = k as f64;

                let x = i * bbox.x.max + (1. - i) * bbox.x.min;
                let y = j * bbox.y.max + (1. - j) * bbox.y.min;
                let z = k * bbox.z.max + (1. - k) * bbox.z.min;

                let newx = self.cos_theta * x + self.sin_theta * z;
                let newz = -self.sin_theta * x + self.cos_theta * z;

                let tester = Vec3::from(newx, y, newz);
                (0..=2).for_each(|c| {
                    min[c] = min[c].min(tester[c]);
                    max[c] = max[c].max(tester[c]);
                });
            });
        Aabb::from_points(&min, &max)
    }

    fn object_to_world_direction(&self, dir: Vec3) -> Vec3 {
        // Inverse of the forward rotation: transpose the matrix (negate sin_theta).
        Vec3::from(
            (self.cos_theta * dir.x) + (self.sin_theta * dir.z),
            dir.y,
            (-self.sin_theta * dir.x) + (self.cos_theta * dir.z),
        )
    }

    fn world_to_object_point(&self, point: Point3) -> Point3 {
        // Inverse of the forward rotation: transpose the matrix (negate sin_theta).
        Point3::from(
            (self.cos_theta * point.x) - (self.sin_theta * point.z),
            point.y,
            (self.sin_theta * point.x) + (self.cos_theta * point.z),
        )
    }
}

// TODO(optional): implement RotateX / RotateZ with cached sin/cos.
// TODO(optional): support transform composition helpers (e.g. nested wrappers or a chain type).
// TODO(feat): add scene-builder helpers/macros that lower into `TransformObject<T, O>`.
