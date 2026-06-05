use std::sync::Arc;

use rand::RngExt;

use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable};
use crate::interval::Interval;
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};
use tracing::{info, trace};

/// A Hittable that never registers a hit. Used as a sentinel child in empty BVH nodes.
struct AlwaysMiss;

impl Hittable for AlwaysMiss {
    fn hit(&self, _ray: &Ray, _ray_t: Interval) -> Option<HitRecord<'_>> {
        None
    }

    fn bounding_box(&self) -> Aabb {
        Aabb::new()
    }

    fn random(&self, _origin: Vec3, _rng: &mut dyn rand::Rng) -> Vec3 {
        Vec3::from(1., 0., 0.)
    }

    fn pdf_value(&self, _origin: Vec3, _direction: Vec3) -> f64 {
        0.0
    }
}

/// A binary BVH node for accelerating ray-scene intersection queries.
pub struct BvhNode {
    /// Left child subtree or leaf primitive.
    left: Arc<dyn Hittable>,
    /// Right child subtree or leaf primitive.
    right: Arc<dyn Hittable>,
    /// World-space bounds enclosing both children.
    bbox: Aabb,
}

impl BvhNode {
    /// Builds a BVH subtree from `objects[start..end]`.
    ///
    /// Strategy:
    /// - compute merged bounds for this span,
    /// - choose longest axis,
    /// - sort by min bound on that axis,
    /// - recurse on two halves.
    pub fn new(objects: &mut Vec<Arc<dyn Hittable>>, start: usize, end: usize) -> Self {
        let obj_span = end - start;
        let is_root = start == 0 && end == objects.len();
        if is_root {
            info!(object_count = obj_span, "building bvh");
        }

        let mut bbox = Aabb::new();
        let mut centroids: Vec<(Arc<dyn Hittable>, Point3)> = Vec::with_capacity(obj_span);

        (start..end).for_each(|idx| {
            let object_bbox = objects[idx].bounding_box();
            bbox = bbox.merge(object_bbox);
            centroids.push((objects[idx].clone(), object_bbox.centroid()));
        });

        let (left, right): (Arc<dyn Hittable>, Arc<dyn Hittable>) = match obj_span {
            0 => {
                trace!("bvh empty");
                let sentinel = Arc::new(AlwaysMiss);
                (sentinel.clone(), sentinel)
            }
            1 => {
                trace!(object_count = obj_span, "bvh leaf");
                (centroids[0].0.clone(), Arc::new(AlwaysMiss))
            }
            2 => {
                trace!(object_count = obj_span, "bvh leaf");
                (centroids[0].0.clone(), centroids[1].0.clone())
            }
            _ => {
                // Surface Area Heuristic (SAH) for optimal BVH construction.
                let mut best_cost = f64::INFINITY;
                let mut best_axis = 0;
                let mut best_split = 0;

                for axis in 0..3 {
                    // Sort by centroid along this axis, then sweep from left and right to compute
                    // SAH cost of each split.
                    centroids.sort_by(|(_, a), (_, b)| a[axis].partial_cmp(&b[axis]).unwrap());

                    // Precompute surface areas of left and right bounding boxes for each split
                    // point.
                    let mut left_areas = Vec::with_capacity(obj_span);
                    let mut right_areas = Vec::with_capacity(obj_span);

                    // Sweep from left to right, keeping track of the bounding box for the left side
                    // of the split.
                    let mut left_bbox = Aabb::new();
                    for (object, _centroid) in &centroids {
                        left_bbox = left_bbox.merge(object.bounding_box());
                        left_areas.push(left_bbox.surface_area());
                    }

                    // Sweep from right to left, keeping track of the bounding box for the right
                    // side of the split.
                    let mut right_bbox = Aabb::new();
                    for (object, _centroid) in centroids.iter().rev() {
                        right_bbox = right_bbox.merge(object.bounding_box());
                        right_areas.push(right_bbox.surface_area());
                    }
                    // Reverse right areas to align with split points: right_areas[i] is the area of
                    // the right side if we split after the i-th object.
                    right_areas.reverse();

                    // Compute SAH cost for each split point and update best split if we find a
                    // cheaper one.
                    for i in 0..obj_span - 1 {
                        let cost = left_areas[i] * (i as f64 + 1.)
                            + right_areas[i + 1] * ((obj_span - i - 1) as f64);
                        if cost < best_cost {
                            best_cost = cost;
                            best_axis = axis;
                            best_split = i + 1;
                        }
                    }
                }

                trace!(
                    object_count = obj_span,
                    best_axis, best_split, "splitting bvh node with SAH"
                );
                // Sort objects by centroid along the best axis, then split at the best point.
                centroids
                    .sort_by(|(_, a), (_, b)| a[best_axis].partial_cmp(&b[best_axis]).unwrap());
                // Copy sorted objects back to the original slice for recursive construction.
                for (slot, (object, _)) in centroids.iter().enumerate() {
                    objects[start + slot] = object.clone();
                }

                // Recurse on the two halves to build child nodes.
                let mid = start + best_split;
                let left: Arc<dyn Hittable> = Arc::new(BvhNode::new(objects, start, mid));
                let right: Arc<dyn Hittable> = Arc::new(BvhNode::new(objects, mid, end));
                (left, right)
            }
        };

        if is_root {
            info!(object_count = obj_span, "bvh built");
        }

        Self { left, right, bbox }
    }
}

impl Hittable for BvhNode {
    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord<'_>> {
        // Prune entire subtree if node bounds are missed.
        if !self.bbox.hit(ray, ray_t) {
            return None;
        }

        // Hit left first, then clamp right traversal to nearest left hit time.
        let hit_left = self.left.hit(ray, ray_t);
        let hit_right = self.right.hit(
            ray,
            Interval::from(ray_t.min, hit_left.as_ref().map_or(ray_t.max, |h| h.time)),
        );

        hit_right.or(hit_left)
    }

    fn bounding_box(&self) -> Aabb {
        self.bbox
    }

    fn pdf_value(&self, origin: Vec3, direction: Vec3) -> f64 {
        0.5 * self.left.pdf_value(origin, direction) + 0.5 * self.right.pdf_value(origin, direction)
    }

    fn random(&self, origin: Vec3, rng: &mut dyn rand::Rng) -> Vec3 {
        if rng.random_bool(0.5) {
            self.left.random(origin, rng)
        } else {
            self.right.random(origin, rng)
        }
    }
}
