use std::sync::Arc;

use crate::bvh::aabb::Aabb;
use crate::intersect::interaction::MaterialHit;
use crate::math::interval::Interval;
use crate::ray::RayPacked;

pub mod interaction;

pub trait Intersectable: Send + Sync + Bounded {
    /// Scalar object-safe intersection entry point.
    ///
    /// This is the dynamic-dispatch escape hatch for custom primitives. The
    /// packet entry point below is intentionally restricted to sized types so
    /// that `Arc<dyn Intersectable>` remains object-safe.
    fn intersect_scalar<'a>(
        &'a self,
        ray: &RayPacked<1>,
        ray_t: Interval<1>,
    ) -> Option<MaterialHit<'a>>;

    /// Scalar object-safe occlusion entry point.
    fn occluded_scalar(&self, ray: &RayPacked<1>, ray_t: Interval<1>) -> bool {
        self.intersect_scalar(ray, ray_t).is_some()
    }

    /// Returns the closest hit inside `[ray_t.min, ray_t.max]`, if any,
    /// along with a reference to the intersected material.
    ///
    /// `N` is the ray-pack width: `N = 1` is the scalar path, `N > 1` a
    /// packet of rays. Implementations may use lane 0 only (scalar leaf
    /// intersection) or operate on all lanes (packet traversal).
    fn intersect<'a, const N: usize>(
        &'a self,
        ray: &RayPacked<N>,
        ray_t: Interval<N>,
    ) -> [Option<MaterialHit<'a>>; N]
    where
        Self: Sized,
    {
        let lanes: [RayPacked<1>; N] = (*ray).into();
        core::array::from_fn(|i| self.intersect_scalar(&lanes[i], ray_t.lane(i)))
    }

    // Returns bool and short-circuits the moment any primitive/node reports a hit inside the interval,
    // rather than tightening best_t and continuing
    fn occluded<const N: usize>(&self, ray: &RayPacked<N>, ray_t: Interval<N>) -> [bool; N]
    where
        Self: Sized,
    {
        self.intersect(ray, ray_t).map(|hit| hit.is_some())
    }
}

impl<T: Intersectable> Intersectable for Vec<T> {
    fn intersect_scalar<'a>(
        &'a self,
        ray: &RayPacked<1>,
        ray_t: Interval<1>,
    ) -> Option<MaterialHit<'a>> {
        let mut closest = ray_t.max_value();
        let mut result = None;

        for object in self {
            // Tighten the interval to the current closest hit so later
            // primitives can cull hits beyond it (mirrors the packet path).
            if let Some(mat_hit) =
                object.intersect_scalar(ray, Interval::from(ray_t.min_value(), closest))
                && mat_hit.hit.time < closest
            {
                closest = mat_hit.hit.time;
                result = Some(mat_hit);
            }
        }

        result
    }

    fn intersect<'a, const N: usize>(
        &'a self,
        ray: &RayPacked<N>,
        ray_t: Interval<N>,
    ) -> [Option<MaterialHit<'a>>; N] {
        let mut closest = ray_t.max();
        let mut result = [None; N];

        for object in self {
            let hits = object.intersect(ray, Interval::from_array(ray_t.min(), closest));
            for i in 0..N {
                if let Some(mat_hit) = hits[i]
                    && mat_hit.hit.time < closest[i]
                {
                    closest[i] = mat_hit.hit.time;
                    result[i] = Some(mat_hit);
                }
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
            impl<T: Intersectable> Intersectable for $wrapper {
                fn intersect_scalar<'a>(
                    &'a self,
                    ray: &RayPacked<1>,
                    ray_t: Interval<1>,
                ) -> Option<MaterialHit<'a>> {
                    (**self).intersect_scalar(ray, ray_t)
                }

                fn intersect<'a, const N: usize>(
                    &'a self,
                    ray: &RayPacked<N>,
                    ray_t: Interval<N>,
                ) -> [Option<MaterialHit<'a>>; N] {
                    (**self).intersect(ray, ray_t)
                }
                fn occluded<const N: usize>(&self, ray: &RayPacked<N>, ray_t: Interval<N>) -> [bool; N] {
                    (**self).occluded(ray, ray_t)
                }
            }
        )+
    };
}
impl_intersectable_for!(Arc<T>, Box<T>, &T);

// Dynamic custom primitives use the object-safe scalar bridge. The packet
// default on the trait then fans that scalar call out across the lanes.
impl Intersectable for Arc<dyn Intersectable> {
    fn intersect_scalar<'a>(
        &'a self,
        ray: &RayPacked<1>,
        ray_t: Interval<1>,
    ) -> Option<MaterialHit<'a>> {
        (**self).intersect_scalar(ray, ray_t)
    }
}

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
