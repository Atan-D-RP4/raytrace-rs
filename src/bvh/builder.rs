use std::cmp::Ordering;
use std::sync::Arc;

use tracing::trace;

use crate::bvh::aabb::Aabb;
use crate::hittable::{Bounded, Intersectable, MaterialHit};
use crate::interval::Interval;
use crate::ray::Ray;

/// The number of bins to use for binned SAH BVH construction. More bins gives a more accurate SAH
/// estimate, but increases the build cost.
const BVH_BIN_SIZE: usize = 32;
/// The threshold number of objects in a BVH node above which we will build child nodes in parallel
/// using rayon.
const BVH_PARALLEL_THRESHOLD: usize = 64;
/// The threshold number of objects in a BVH node below which we will create a leaf node instead
/// of splitting further. This is a tradeoff between tree depth and leaf size.
const BVH_LEAF_THRESHOLD: usize = 4;

/// A binary BVH node for accelerating ray-scene intersection queries.
///
/// No generic parameter needed — leaf objects are trait-object slices
/// that provide both `Intersectable` and `Bounded`.
pub enum TreeBuilder {
    /// An empty BVH node, used for empty scenes or empty child nodes.
    Empty,
    /// An interior BVH node with two child nodes and a bounding box that encloses both children.
    Interior {
        /// The left child node.
        left: Box<TreeBuilder>,
        /// The right child node.
        right: Box<TreeBuilder>,
        /// The bounding box that encloses both child nodes.
        bbox: Aabb,
    },
    /// A leaf BVH node that contains multiple objects and a bounding box that encloses all of them.
    LeafN {
        /// The objects contained in this leaf node.
        objects: [Arc<dyn Intersectable>; BVH_LEAF_THRESHOLD],
        /// The number of objects contained in this leaf node.
        count: usize,
        /// The bounding box that encloses all objects in this leaf node.
        bbox: Aabb,
    },
    /// A leaf BVH node that contains a single object and a bounding box that encloses it.
    Leaf {
        /// The object contained in this leaf node.
        object: Arc<dyn Intersectable>,
        /// The bounding box that encloses the object in this leaf node.
        bbox: Aabb,
    },
}

impl TreeBuilder {
    /// Builds a BVH subtree from `objects` (a mutable slice).
    ///
    /// Strategy:
    /// - compute merged bounds for all objects,
    /// - bin centroids on each axis and evaluate SAH cost,
    /// - split at cheapest partition, recurse.
    pub fn new(objects: &mut [Arc<dyn Intersectable>]) -> Self {
        let obj_span = objects.len();

        let mut centroids = objects
            .iter()
            .map(|object| {
                let object_bbox = object.bounding_box();
                (object.clone(), object_bbox, object_bbox.centroid())
            })
            .collect::<Vec<_>>();
        let root_bbox = centroids
            .iter()
            .fold(Aabb::empty(), |acc, (_, bbox, _)| acc.merge(bbox));

        match obj_span {
            0 => {
                trace!("bvh empty");
                Self::Empty
            }
            1 => {
                trace!(object_count = obj_span, "bvh leaf");
                Self::Leaf {
                    object: centroids[0].0.clone(),
                    bbox: centroids[0].1,
                }
            }
            2..BVH_LEAF_THRESHOLD => {
                trace!(object_count = obj_span, "bvh leaf");
                let mut leaf_objects = core::array::from_fn(|_| {
                    Arc::new(TreeBuilder::Empty) as Arc<dyn Intersectable>
                });
                for (i, (object, _, _)) in centroids.iter().enumerate() {
                    leaf_objects[i] = object.clone();
                }
                Self::LeafN {
                    objects: leaf_objects,
                    count: obj_span,
                    bbox: root_bbox,
                }
            }
            _ => {
                // Binned Surface Area Heuristic (SAH) for optimal BVH construction.
                let mut best_cost = f32::INFINITY;
                let mut best_axis = 0;
                let mut best_split = 0;

                for axis in 0..3 {
                    // Find Centroid range along the axis
                    let (min_c, max_c) = centroids.iter().fold(
                        (f32::INFINITY, f32::NEG_INFINITY),
                        |(min, max), (_, _, centroid)| {
                            let centroid_val = centroid[axis][0];
                            (min.min(centroid_val), max.max(centroid_val))
                        },
                    );

                    // Create the Bins
                    let mut bin_count = [0; BVH_BIN_SIZE];
                    let mut bin_bbox = [Aabb::empty(); BVH_BIN_SIZE];

                    let range = max_c - min_c;
                    if range < 1e-10 {
                        continue; // Degenerate on this axis — skip it
                    }

                    // Bin the objects
                    for (_, bbox, centroid) in centroids.iter() {
                        let centroid_val = centroid[axis][0];
                        let t = (centroid_val - min_c) / range;
                        let b = (t * BVH_BIN_SIZE as f32)
                            .floor()
                            .clamp(0., BVH_BIN_SIZE as f32 - 1.)
                            as usize;
                        bin_count[b] += 1;
                        bin_bbox[b] = bin_bbox[b].merge(bbox);
                    }

                    // Precompute suffix AABBs and counts.
                    let mut suffix_bbox = [Aabb::empty(); BVH_BIN_SIZE];
                    let mut suffix_count = [0usize; BVH_BIN_SIZE];
                    {
                        let mut bbox = Aabb::empty();
                        let mut count = 0;
                        for b in (0..BVH_BIN_SIZE).rev() {
                            bbox = bbox.merge(&bin_bbox[b]);
                            count += bin_count[b];
                            suffix_bbox[b] = bbox;
                            suffix_count[b] = count;
                        }
                    }

                    // Sweep from left to right, using precomputed suffix for the right side.
                    let mut left_bbox = Aabb::empty();
                    let mut left_count = 0;
                    for b in 0..BVH_BIN_SIZE - 1 {
                        left_bbox = left_bbox.merge(&bin_bbox[b]);
                        left_count += bin_count[b];
                        let right_bbox = suffix_bbox[b + 1];
                        let right_count = suffix_count[b + 1];

                        if left_count == 0 || right_count == 0 {
                            continue; // Skip empty splits
                        }

                        let cost = left_count as f32 * left_bbox.surface_area()[0]
                            + right_count as f32 * right_bbox.surface_area()[0];

                        if cost < best_cost {
                            best_cost = cost;
                            best_axis = axis;
                            best_split = left_count; // Object count, not bin index
                        }
                    }
                }

                let root_sa = root_bbox.surface_area();
                let trav_cost = root_sa[0] * 0.5;
                let leaf_cost = root_sa[0] * obj_span as f32;

                if best_cost.is_finite() && best_cost + trav_cost < leaf_cost {
                    trace!(
                        object_count = obj_span,
                        best_cost, best_axis, best_split, "splitting bvh node with SAH"
                    );
                } else {
                    trace!(
                        object_count = obj_span,
                        best_cost, best_axis, best_split, "not splitting bvh node with SAH"
                    );
                    // Not worth splitting — pack into a multi-object leaf.
                    // Only pack if we can fit all objects; otherwise force split below.
                    if obj_span <= BVH_LEAF_THRESHOLD {
                        let mut leaf_objects = core::array::from_fn(|_| {
                            Arc::new(TreeBuilder::Empty) as Arc<dyn Intersectable>
                        });
                        for (i, (object, _, _)) in centroids.iter().enumerate() {
                            leaf_objects[i] = object.clone();
                        }
                        return Self::LeafN {
                            objects: leaf_objects,
                            count: obj_span,
                            bbox: root_bbox,
                        };
                    }
                }

                trace!(
                    object_count = obj_span,
                    best_axis, best_split, "splitting bvh node with SAH"
                );

                // Sort objects by centroid along the best axis, then split at the best point.
                centroids.select_nth_unstable_by(best_split, |a, b| {
                    a.2[best_axis]
                        .partial_cmp(&b.2[best_axis])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                // Copy sorted objects back to the original slice for recursive construction.
                for (slot, (object, _, _)) in centroids.iter().enumerate() {
                    objects[slot] = object.clone();
                }

                // Recurse on the two halves to build child nodes.
                let (left_half, right_half) = objects.split_at_mut(best_split);
                let (left, right) = match obj_span.cmp(&BVH_PARALLEL_THRESHOLD) {
                    Ordering::Less => {
                        // For small node sizes, build sequentially to avoid thread overhead.
                        (
                            Box::new(Self::new(left_half)),
                            Box::new(Self::new(right_half)),
                        )
                    }
                    Ordering::Greater | Ordering::Equal => {
                        // For larger node sizes, build in parallel using rayon.
                        rayon::join(
                            || Box::new(Self::new(left_half)),
                            || Box::new(Self::new(right_half)),
                        )
                    }
                };
                Self::Interior {
                    left,
                    right,
                    bbox: root_bbox,
                }
            }
        }
    }
}

impl Intersectable for TreeBuilder {
    fn intersect<'a>(&'a self, ray: &Ray, ray_t: Interval) -> Option<MaterialHit<'a>> {
        match self {
            Self::Empty => None,
            Self::Interior { left, right, bbox } => {
                if !bbox.hit_single(ray, &ray_t) {
                    return None;
                }
                let hit_left = left.intersect(ray, ray_t);
                let hit_right = right.intersect(
                    ray,
                    Interval::from(
                        ray_t.min,
                        hit_left.as_ref().map_or(ray_t.max, |h| h.hit.time),
                    ),
                );
                hit_right.or(hit_left)
            }
            Self::Leaf { object, .. } => object.intersect(ray, ray_t),
            Self::LeafN {
                objects,
                count,
                bbox,
                ..
            } => {
                if !bbox.hit_single(ray, &ray_t) {
                    return None;
                }
                let mut closest_hit: Option<MaterialHit> = None;
                let mut closest_time = ray_t.max;
                for object in objects[..*count].iter() {
                    if let Some(hit) =
                        object.intersect(ray, Interval::from(ray_t.min, closest_time))
                    {
                        closest_time = hit.hit.time;
                        closest_hit = Some(hit);
                    }
                }
                closest_hit
            }
        }
    }
}

impl Bounded for TreeBuilder {
    fn bounding_box(&self) -> Aabb {
        match self {
            Self::Empty => Aabb::empty(),
            Self::Interior { bbox, .. } | Self::Leaf { bbox, .. } => *bbox,
            Self::LeafN { bbox, .. } => *bbox,
        }
    }
}
