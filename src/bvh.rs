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
    const BIN_SIZE: usize = 32;

    /// Builds a BVH subtree from `objects` (a mutable slice).
    ///
    /// Strategy:
    /// - compute merged bounds for all objects,
    /// - bin centroids on each axis and evaluate SAH cost,
    /// - split at cheapest partition, recurse.
    pub fn new(objects: &mut [Arc<dyn Hittable>]) -> Self {
        info!(object_count = objects.len(), "building bvh");
        let obj_span = objects.len();

        let mut bbox = Aabb::new();
        let mut centroids: Vec<(Arc<dyn Hittable>, Point3)> = Vec::with_capacity(obj_span);

        for idx in 0..obj_span {
            let object_bbox = objects[idx].bounding_box();
            bbox = bbox.merge(object_bbox);
            centroids.push((objects[idx].clone(), object_bbox.centroid()));
        }

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
                // Binned Surface Area Heuristic (SAH) for optimal BVH construction.
                let mut best_cost = f64::INFINITY;
                let mut best_axis = 0;
                let mut best_split = 0;

                for axis in 0..3 {
                    // Find Centroid range along the axis
                    let (min_c, max_c) = centroids.iter().fold(
                        (f64::INFINITY, f64::NEG_INFINITY),
                        |(min, max), (_, centroid)| {
                            (min.min(centroid[axis]), max.max(centroid[axis]))
                        },
                    );

                    // Create the Bins
                    let mut bin_count = [0; Self::BIN_SIZE];
                    let mut bin_bbox = [Aabb::new(); Self::BIN_SIZE];

                    let range = max_c - min_c;
                    if range < 1e-10 {
                        continue; // Degenerate on this axis — skip it
                    }

                    // Bin the objects
                    for (object, centroid) in centroids.iter() {
                        let t = (centroid[axis] - min_c) / range;
                        let b = (t * Self::BIN_SIZE as f64)
                            .floor()
                            .clamp(0., Self::BIN_SIZE as f64 - 1.)
                            as usize;
                        bin_count[b] += 1;
                        bin_bbox[b] = bin_bbox[b].merge(object.bounding_box());
                    }

                    // Precompute suffix AABBs and counts.
                    // suffix_bbox[b] = AABB of bins[b..B-1], suffix_count[b] = #objects in those bins.
                    let mut suffix_bbox = [Aabb::new(); Self::BIN_SIZE];
                    let mut suffix_count = [0usize; Self::BIN_SIZE];
                    {
                        let mut bbox = Aabb::new();
                        let mut count = 0;
                        for b in (0..Self::BIN_SIZE).rev() {
                            bbox = bbox.merge(bin_bbox[b]);
                            count += bin_count[b];
                            suffix_bbox[b] = bbox;
                            suffix_count[b] = count;
                        }
                    }

                    // Sweep from left to right, using precomputed suffix for the right side.
                    let mut left_bbox = Aabb::new();
                    let mut left_count = 0;
                    for b in 0..Self::BIN_SIZE - 1 {
                        left_bbox = left_bbox.merge(bin_bbox[b]);
                        left_count += bin_count[b];
                        let right_bbox = suffix_bbox[b + 1];
                        let right_count = suffix_count[b + 1];

                        if left_count == 0 || right_count == 0 {
                            continue; // Skip empty splits
                        }

                        let cost = left_count as f64 * left_bbox.surface_area()
                            + right_count as f64 * right_bbox.surface_area();

                        if cost < best_cost {
                            best_cost = cost;
                            best_axis = axis;
                            best_split = left_count; // Object count, not bin index
                        }
                    }
                }

                trace!(
                    object_count = obj_span,
                    best_axis, best_split, "splitting bvh node with SAH"
                );

                // Sort objects by centroid along the best axis, then split at the best point.
                centroids.select_nth_unstable_by(best_split, |a, b| {
                    a.1[best_axis]
                        .partial_cmp(&b.1[best_axis])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                // Copy sorted objects back to the original slice for recursive construction.
                for (slot, (object, _)) in centroids.iter().enumerate() {
                    objects[slot] = object.clone();
                }

                // Recurse on the two halves to build child nodes.
                let (left_half, right_half) = objects.split_at_mut(best_split);
                let (left, right) = rayon::join(
                    || Arc::new(Self::new(left_half)),
                    || Arc::new(Self::new(right_half)),
                );
                (left, right)
            }
        };

        info!(object_count = objects.len(), "bvh built");
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
