use std::sync::Arc;

use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable};
use crate::interval::Interval;
use crate::ray::Ray;
use crate::vec3::Point3;
use tracing::{info, trace};

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

        for idx in start..end {
            let object_bbox = objects[idx].bounding_box();
            bbox = bbox.merge(object_bbox);
            centroids.push((objects[idx].clone(), object_bbox.centroid()));
        }

        let (left, right): (Arc<dyn Hittable>, Arc<dyn Hittable>) = match obj_span {
            1 => {
                trace!(object_count = obj_span, "bvh leaf");
                (centroids[0].0.clone(), centroids[0].0.clone())
            }
            2 => {
                trace!(object_count = obj_span, "bvh leaf");
                (centroids[0].0.clone(), centroids[1].0.clone())
            }
            _ => {
                let axis = bbox.longest_axis() as usize;
                trace!(object_count = obj_span, axis, "splitting bvh node");
                centroids.sort_by(|(_, a), (_, b)| a[axis].partial_cmp(&b[axis]).unwrap());
                for (slot, (object, _)) in centroids.iter().enumerate() {
                    objects[start + slot] = object.clone();
                }
                let mid = start + obj_span / 2;
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
}
