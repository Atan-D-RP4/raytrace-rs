use std::sync::Arc;

use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable};
use crate::interval::Interval;
use crate::ray::Ray;
use crate::sampler::{DimCursor, Sampler};
use crate::vec3::{Point3, Vec3};
use tracing::{info, trace};

const BVH_BIN_SIZE: usize = 32;

/// A binary BVH node for accelerating ray-scene intersection queries.
pub enum BvhNode<S: Sampler> {
    Empty,
    Interior {
        left: Box<BvhNode<S>>,
        right: Box<BvhNode<S>>,
        bbox: Aabb,
    },
    Leaf {
        object: Arc<dyn Hittable<S>>,
        bbox: Aabb,
    },
}

impl<S: Sampler> BvhNode<S> {
    /// Builds a BVH subtree from `objects` (a mutable slice).
    ///
    /// Strategy:
    /// - compute merged bounds for all objects,
    /// - bin centroids on each axis and evaluate SAH cost,
    /// - split at cheapest partition, recurse.
    pub fn new(objects: &mut [Arc<dyn Hittable<S>>]) -> Self {
        info!(object_count = objects.len(), "building bvh");
        let obj_span = objects.len();

        let mut centroids: Vec<(Arc<dyn Hittable<S>>, Point3)> = Vec::with_capacity(obj_span);
        let mut bbox = Aabb::new();

        for object in objects.iter() {
            let object_bbox = object.bounding_box();
            bbox = bbox.merge(object_bbox);
            centroids.push((object.clone(), object_bbox.centroid()));
        }

        let result = match obj_span {
            0 => {
                trace!("bvh empty");
                Self::Empty
            }
            1 => {
                trace!(object_count = obj_span, "bvh leaf");
                Self::Leaf {
                    object: centroids[0].0.clone(),
                    bbox: centroids[0].0.bounding_box(),
                }
            }
            2 => {
                trace!(object_count = obj_span, "bvh leaf");
                let left = Box::new(Self::Leaf {
                    object: centroids[0].0.clone(),
                    bbox: centroids[0].0.bounding_box(),
                });
                let right = Box::new(Self::Leaf {
                    object: centroids[1].0.clone(),
                    bbox: centroids[1].0.bounding_box(),
                });
                Self::Interior { left, right, bbox }
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
                    let mut bin_count = [0; BVH_BIN_SIZE];
                    let mut bin_bbox = [Aabb::new(); BVH_BIN_SIZE];

                    let range = max_c - min_c;
                    if range < 1e-10 {
                        continue; // Degenerate on this axis — skip it
                    }

                    // Bin the objects
                    for (object, centroid) in centroids.iter() {
                        let t = (centroid[axis] - min_c) / range;
                        let b = (t * BVH_BIN_SIZE as f64)
                            .floor()
                            .clamp(0., BVH_BIN_SIZE as f64 - 1.)
                            as usize;
                        bin_count[b] += 1;
                        bin_bbox[b] = bin_bbox[b].merge(object.bounding_box());
                    }

                    // Precompute suffix AABBs and counts.
                    // suffix_bbox[b] = AABB of bins[b..B-1], suffix_count[b] = #objects in those bins.
                    let mut suffix_bbox = [Aabb::new(); BVH_BIN_SIZE];
                    let mut suffix_count = [0usize; BVH_BIN_SIZE];
                    {
                        let mut bbox = Aabb::new();
                        let mut count = 0;
                        for b in (0..BVH_BIN_SIZE).rev() {
                            bbox = bbox.merge(bin_bbox[b]);
                            count += bin_count[b];
                            suffix_bbox[b] = bbox;
                            suffix_count[b] = count;
                        }
                    }

                    // Sweep from left to right, using precomputed suffix for the right side.
                    let mut left_bbox = Aabb::new();
                    let mut left_count = 0;
                    for b in 0..BVH_BIN_SIZE - 1 {
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
                    || Box::new(Self::new(left_half)),
                    || Box::new(Self::new(right_half)),
                );
                Self::Interior { left, right, bbox }
            }
        };

        info!(object_count = objects.len(), "bvh built");
        result
    }

    /// Returns the number of leaf objects in this subtree.
    fn leaf_count(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::Leaf { .. } => 1,
            Self::Interior { left, right, .. } => left.leaf_count() + right.leaf_count(),
        }
    }
}

impl<S: Sampler> Hittable<S> for BvhNode<S> {
    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord<'_>> {
        match self {
            Self::Empty => None,
            Self::Leaf { object, .. } => object.hit(ray, ray_t),
            Self::Interior { left, right, bbox } => {
                if !bbox.hit(ray, ray_t) {
                    return None;
                }
                let hit_left = left.hit(ray, ray_t);
                let hit_right = right.hit(
                    ray,
                    Interval::from(ray_t.min, hit_left.as_ref().map_or(ray_t.max, |h| h.time)),
                );
                hit_right.or(hit_left)
            }
        }
    }

    fn bounding_box(&self) -> Aabb {
        match self {
            Self::Empty => Aabb::new(),
            Self::Leaf { bbox, .. } | Self::Interior { bbox, .. } => *bbox,
        }
    }

    fn pdf_value(&self, origin: Vec3, direction: Vec3) -> f64 {
        match self {
            Self::Empty => 0.0,
            Self::Leaf { object, .. } => object.pdf_value(origin, direction),
            Self::Interior { left, right, .. } => {
                let left_count = left.leaf_count() as f64;
                let right_count = right.leaf_count() as f64;
                let total = left_count + right_count;
                (left_count / total) * left.pdf_value(origin, direction)
                    + (right_count / total) * right.pdf_value(origin, direction)
            }
        }
    }

    fn random(&self, origin: Vec3, dim_offset: &mut DimCursor<S>) -> Vec3 {
        match self {
            Self::Empty => Vec3::from(1., 0., 0.),
            Self::Leaf { object, .. } => object.random(origin, dim_offset),
            Self::Interior { left, right, .. } => {
                let left_count = left.leaf_count() as f64;
                let right_count = right.leaf_count() as f64;
                let total = left_count + right_count;
                let u = dim_offset.next_sample();
                if u < left_count / total {
                    left.random(origin, dim_offset)
                } else {
                    right.random(origin, dim_offset)
                }
            }
        }
    }
}
