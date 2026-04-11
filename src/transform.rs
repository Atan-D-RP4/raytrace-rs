use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable};
use crate::interval::Interval;
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

/// A geometric transform that can map rays, hit records, and bounds.
///
/// TODO(optional): add rotation/scale variants once needed by scene builders.
/// TODO(feat): add a macro DSL for ergonomic transform chaining.
pub trait Transform: Send + Sync {
    fn ray(&self, ray: &Ray) -> Ray;

    fn hit(&self, hit: &mut HitRecord);

    fn bbox(&self, bbox: Aabb) -> Aabb;
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
    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord> {
        let transformed_ray = self.transform.ray(ray);
        let mut hit = self.object.hit(&transformed_ray, ray_t)?;

        self.transform.hit(&mut hit);

        Some(hit)
    }

    fn bounding_box(&self) -> Aabb {
        self.bbox
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

    fn hit(&self, hit: &mut HitRecord) {
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

    fn hit(&self, hit: &mut HitRecord) {
        hit.point = Point3::from(
            (self.cos_theta * hit.point.x) - (self.sin_theta * hit.point.z),
            hit.point.y,
            (self.sin_theta * hit.point.x) + (self.cos_theta * hit.point.z),
        );
        hit.normal = Vec3::from(
            (self.cos_theta * hit.normal.x) - (self.sin_theta * hit.normal.z),
            hit.normal.y,
            (self.sin_theta * hit.normal.x) + (self.cos_theta * hit.normal.z),
        );
        hit.mapping_point = Vec3::from(
            (self.cos_theta * hit.mapping_point.x) - (self.sin_theta * hit.mapping_point.z),
            hit.mapping_point.y,
            (self.sin_theta * hit.mapping_point.x) + (self.cos_theta * hit.mapping_point.z),
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
}

// TODO(optional): implement RotateX / RotateY / RotateZ with cached sin/cos.
// TODO(optional): support transform composition helpers (e.g. nested wrappers or a chain type).
// TODO(feat): add scene-builder helpers/macros that lower into `TransformObject<T, O>`.
