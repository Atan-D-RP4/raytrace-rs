use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable};
use crate::interval::Interval;
use crate::ray::Ray;
use crate::sampler::Sampler;
use crate::vec3::{Point3, Vec3};

/// A geometric transform that can map rays, hit records, and bounds.
///
/// TODO(optional): add rotation/scale variants once needed by scene builders.
/// TODO(feat): add a macro DSL for ergonomic transform chaining.
pub trait Transform: Send + Sync {
    fn ray(&self, ray: &Ray) -> Ray;

    fn hit(&self, hit: &mut HitRecord<'_>);

    fn bbox(&self, bbox: Aabb) -> Aabb;

    /// Transform a direction vector from object space to world space.
    /// For rigid transforms this is the inverse rotation applied to the direction.
    /// Default assumes identity (direction unchanged), which is correct for Translate.
    fn object_to_world_direction(&self, dir: Vec3) -> Vec3 {
        dir
    }
}

/// A zero-cost wrapper that applies a transform to any hittable object.
///
/// The transform and object stay generic, so the compiler can inline through
/// the whole stack without trait-object dispatch.
pub struct TransformObject<T, O> {
    transform: T,
    object: O,
    bbox: Aabb,
}

impl<T, O> TransformObject<T, O>
where
    T: Transform,
    O: Hittable,
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

impl<T, O> Hittable for TransformObject<T, O>
where
    T: Transform,
    O: Hittable,
{
    fn hit<'a>(&'a self, ray: &Ray, ray_t: Interval) -> Option<HitRecord<'a>> {
        let transformed_ray = self.transform.ray(ray);
        let mut hit = self.object.hit(&transformed_ray, ray_t)?;

        self.transform.hit(&mut hit);

        Some(hit)
    }

    fn bounding_box(&self) -> Aabb {
        self.bbox
    }

    fn pdf_value(&self, origin: Vec3, direction: Vec3) -> f64 {
        // Transform the ray into object space and delegate to the inner object.
        // For rigid transforms (rotation + translation), the Jacobian of the
        // area mapping is 1, so the solid-angle PDF is preserved.
        let ray = Ray::new_with_time(origin, direction, 0.0);
        let transformed = self.transform.ray(&ray);
        self.object
            .pdf_value(transformed.origin, transformed.direction)
    }

    fn random(
        &self,
        origin: Vec3,
        sampler: &dyn Sampler,
        sample_index: u32,
        dim_offset: &mut u32,
    ) -> Vec3 {
        // Transform origin to object space via a dummy ray.
        let to_obj = self
            .transform
            .ray(&Ray::new_with_time(origin, Vec3::ZERO, 0.0));
        // Sample a direction in object space.
        let dir = self
            .object
            .random(to_obj.origin, sampler, sample_index, dim_offset);
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

    fn hit(&self, hit: &mut HitRecord<'_>) {
        hit.point += self.offset;
        hit.mapping_point += self.offset;
    }

    fn bbox(&self, bbox: Aabb) -> Aabb {
        bbox.translate(self.offset)
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

    fn hit(&self, hit: &mut HitRecord<'_>) {
        hit.point = Point3::from(
            (self.cos_theta * hit.point.x) + (self.sin_theta * hit.point.z),
            hit.point.y,
            (-self.sin_theta * hit.point.x) + (self.cos_theta * hit.point.z),
        );
        hit.normal = Vec3::from(
            (self.cos_theta * hit.normal.x) + (self.sin_theta * hit.normal.z),
            hit.normal.y,
            (-self.sin_theta * hit.normal.x) + (self.cos_theta * hit.normal.z),
        );
        hit.mapping_point = Vec3::from(
            (self.cos_theta * hit.mapping_point.x) + (self.sin_theta * hit.mapping_point.z),
            hit.mapping_point.y,
            (-self.sin_theta * hit.mapping_point.x) + (self.cos_theta * hit.mapping_point.z),
        );
        hit.geometry_normal = Vec3::from(
            (self.cos_theta * hit.geometry_normal.x) + (self.sin_theta * hit.geometry_normal.z),
            hit.geometry_normal.y,
            (-self.sin_theta * hit.geometry_normal.x) + (self.cos_theta * hit.geometry_normal.z),
        );
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
}

// TODO(optional): implement RotateX / RotateY / RotateZ with cached sin/cos.
// TODO(optional): support transform composition helpers (e.g. nested wrappers or a chain type).
// TODO(feat): add scene-builder helpers/macros that lower into `TransformObject<T, O>`.
