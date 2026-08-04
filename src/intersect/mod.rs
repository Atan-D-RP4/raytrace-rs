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

macro_rules! impl_intersectable_for {
    ($($wrapper:ty),+ $(,)?) => {
        $(
            impl<T: ?Sized + Intersectable> Intersectable for $wrapper {
                fn intersect<'a>(&'a self, ray: &Ray, ray_t: Interval) -> Option<MaterialHit<'a>> {
                    (**self).intersect(ray, ray_t)
                }
                fn occluded(&self, ray: &Ray, ray_t: Interval) -> bool {
                    (**self).occluded(ray, ray_t)
                }
            }
        )+
    };
}
impl_intersectable_for!(Arc<T>, Box<T>, &T);

macro_rules! impl_bounded_for {
    ($($wrapper:ty),+ $(,)?) => {
        $(
            impl<T: ?Sized + Bounded> Bounded for $wrapper {
                fn bounding_box(&self) -> Aabb { (**self).bounding_box() }
            }
        )+
    };
}

impl_bounded_for!(Arc<T>, Box<T>, &T);
