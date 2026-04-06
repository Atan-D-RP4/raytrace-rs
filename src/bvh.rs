use std::sync::Arc;

use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable};
use crate::interval::Interval;
use crate::ray::Ray;

pub struct BvhNode {
    left: Arc<dyn Hittable>,
    right: Arc<dyn Hittable>,
    bbox: Aabb,
}

impl BvhNode {
    pub fn new(objects: &mut Vec<Arc<dyn Hittable>>, start: usize, end: usize) -> Self {
        let obj_span = end - start;
        let mut bbox = Aabb::new();
        (start..end).for_each(|idx| bbox = Aabb::merge(bbox, objects[idx].bounding_box()));

        let (left, right): (Arc<dyn Hittable>, Arc<dyn Hittable>) = match obj_span {
            1 => {
                // Clone the Arc, no removal needed
                (objects[start].clone(), objects[start].clone())
            }
            2 => (objects[start].clone(), objects[start + 1].clone()),
            _ => {
                let axis = bbox.longest_axis();
                objects[start..end].sort_by(|a, b| {
                    let a_min = a.bounding_box().axis_interval(axis).min;
                    let b_min = b.bounding_box().axis_interval(axis).min;
                    a_min.partial_cmp(&b_min).unwrap()
                });
                let mid = start + obj_span / 2;
                let left: Arc<dyn Hittable> = Arc::new(BvhNode::new(objects, start, mid));
                let right: Arc<dyn Hittable> = Arc::new(BvhNode::new(objects, mid, end));
                (left, right)
            }
        };

        Self { left, right, bbox }
    }
}

impl Hittable for BvhNode {
    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord> {
        if !self.bbox.hit(ray, ray_t) {
            return None;
        }

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
