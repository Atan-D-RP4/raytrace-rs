use std::sync::Arc;

use crate::bvh::aabb::Aabb;
use crate::intersect::interaction::MaterialHit;
use crate::math::interval::Interval;
use crate::ray::Ray;

pub mod interaction;

pub trait Intersectable: Send + Sync + Bounded {
    /// Returns the closest hit inside `[ray_t.min, ray_t.max]`, if any,
    /// along with a reference to the intersected material.
    fn intersect<'a>(&'a self, ray: &Ray, ray_t: Interval) -> Option<MaterialHit<'a>>;

    // Returns bool and short-circuits the moment any primitive/node reports a hit inside the interval,
    // rather than tightening best_t and continuing
    fn occluded(&self, ray: &Ray, ray_t: Interval) -> bool {
        self.intersect(ray, ray_t).is_some()
    }
}

impl<T: Intersectable> Intersectable for Vec<T> {
    fn intersect<'a>(&'a self, ray: &Ray, ray_t: Interval) -> Option<MaterialHit<'a>> {
        let mut closest = ray_t.max;
        let mut result = None;

        for object in self {
            if let Some(mat_hit) = object.intersect(ray, Interval::from(ray_t.min, closest)) {
                closest = mat_hit.hit.time;
                result = Some(mat_hit);
            }
        }

        result
    }
}

impl<T: Intersectable + ?Sized> Intersectable for Arc<T> {
    fn intersect<'a>(&'a self, ray: &Ray, ray_t: Interval) -> Option<MaterialHit<'a>> {
        (**self).intersect(ray, ray_t)
    }
}

pub trait Bounded: Send + Sync {
    /// Returns a conservative world-space AABB for acceleration structures.
    fn bounding_box(&self) -> Aabb;
}

impl<T: Bounded> Bounded for Vec<T> {
    fn bounding_box(&self) -> Aabb {
        self.iter()
            .fold(Aabb::empty(), |acc, obj| acc.merge(&obj.bounding_box()))
    }
}

impl<T: Bounded + ?Sized> Bounded for Arc<T> {
    fn bounding_box(&self) -> Aabb {
        (**self).bounding_box()
    }
}
