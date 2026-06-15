//! Cache-friendly flat BVH for fast ray-scene intersection.
//!
//! [`FlatBvh`] is a linearised version of [`BvhNode`] stored as a contiguous
//! array of [`FlatBvhNode`]s. The layout is optimised for traversal:
//!
//! - Each node is exactly 64 bytes (one cache line on x86-64/ARM64).
//! - f64 AABB fields avoid precision loss at the BVH/primitive boundary.
//! - Interior nodes store child indices (not pointers) so the array is
//!   serialisable and trivially convertible from the tree [`BvhNode`].
//! - Leaf nodes store a range into a separate primitive index array.
//! - Children are ordered near-first (smaller t-bound first) for early
//!   termination during iterative traversal.
//!
//! # Traversal
//!
//! Traversal is iterative with an explicit stack (no recursion). The stack
//! depth is fixed at 64 entries — sufficient for any practical BVH depth.

/*
References for optimizing BVH traversal and flat layouts on the CPU:
1. Embree architecture: Wald et al., "Embree: A Kernel Framework for Efficient CPU Ray Tracing," SIGGRAPH 2014 — embree.org/papers/2014-Siggraph-Embree.pdf (https://www.embree.org/papers/2014-Siggraph-Embree.pdf)
2. Embree source: kernels/bvh/bvh.h and kernels/bvh/bvh_node_aabb.h — github.com/RenderKit/embree (https://github.com/RenderKit/embree)
3. WiVe algorithm: Fuetterling et al., "Accelerated Single-Ray Tracing for Wide Vector Units," HPG 2017
4. PBRT BVH chapter: pbr-book.org/4ed/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies (https://www.pbr-book.org/4ed/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies)
5. Psychopath BVH4 analysis: psychopath.io/post/2017_08_03_bvh4_without_simd (https://psychopath.io/post/2017_08_03_bvh4_without_simd)
6. tinybvh: github.com/jbikker/tinybvh (https://github.com/jbikker/tinybvh) — includes BVH4_CPU, BVH8_CPU, CWBVH implementations
7. CLPT paper (coherent packets for BVH4): jcgt.org/published/0004/04/05/
8. Stackless BVH traversal: Hapala et al., "Efficient Stack-less BVH Traversal for Ray Tracing," SCCG 2011
9. DRST (dynamic ray stream tracing): Barringer & Akenine-Möller, 2014
*/

use std::sync::Arc;

use crate::aabb::Aabb;
use crate::bvh::BvhNode;
use crate::hittable::{Bounded, Intersectable, MaterialHit};
use crate::interval::Interval;
use crate::ray::Ray;

/// Maximum traversal stack depth. 64 handles BVHs with up to 2^64 primitives.
const MAX_STACK: usize = 64;

/// A flat (array-of-structs) BVH node. Aligned to 64 bytes for cache efficiency.
///
/// Layout is `repr(C)` for deterministic padding and cache-line alignment.
/// Interior nodes: left/right are indices into the flat node array. Leaf nodes: prim_offset/count
/// index into the primitive index array.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FlatBvhNode {
    /// [0..3] min AABB: 3 x f64 = 24 bytes
    pub min: [f64; 3], // min_x, min_y, min_z
    /// [4..6] max AABB: 3 x f64 = 24 bytes
    pub max: [f64; 3], // max_x, max_y, max_z
    /// [48] child_or_count: u32  (interior: left child index; leaf: prim count)
    pub child_or_count: u32,
    /// [52] right_or_unused: u32 (interior: right child index; leaf: 0)
    pub right_or_unused: u32,
    /// [56] prim_offset: u32 (leaf: start index; interior: 0)
    pub prim_offset: u32,
    /// [60] flags: u8    (0 = interior, 1 = leaf)
    pub flags: u8,
    /// [61..63] _pad: [u8; 3]
    _pad: [u8; 3],
}

impl FlatBvhNode {
    const INTERIOR: u8 = 0;
    const LEAF: u8 = 1;

    fn interior(min: [f64; 3], max: [f64; 3], left: u32, right: u32) -> Self {
        Self {
            min,
            max,
            child_or_count: left,
            right_or_unused: right,
            prim_offset: 0,
            flags: Self::INTERIOR,
            _pad: [0; 3],
        }
    }

    fn leaf(min: [f64; 3], max: [f64; 3], prim_offset: u32, prim_count: u32) -> Self {
        Self {
            min,
            max,
            child_or_count: prim_count,
            right_or_unused: 0,
            prim_offset,
            flags: Self::LEAF,
            _pad: [0; 3],
        }
    }

    #[inline]
    fn is_leaf(&self) -> bool {
        self.flags == Self::LEAF
    }

    #[inline]
    fn left_child(&self) -> u32 {
        debug_assert!(!self.is_leaf(), "left_child() on leaf");
        self.child_or_count
    }

    #[inline]
    fn right_child(&self) -> u32 {
        debug_assert!(!self.is_leaf(), "right_child() on leaf");
        self.right_or_unused
    }

    #[inline]
    fn prim_start(&self) -> usize {
        debug_assert!(self.is_leaf(), "prim_start() on interior");
        self.prim_offset as usize
    }

    #[inline]
    fn prim_count(&self) -> usize {
        debug_assert!(self.is_leaf(), "prim_count() on interior");
        self.child_or_count as usize
    }
}

/// Flat BVH container: array-of-nodes + flat primitive list.
pub struct FlatBvh {
    /// Contiguous array of flat BVH nodes in DFS (pre-order) layout.
    nodes: Vec<FlatBvhNode>,
    /// Scene primitives in the order they appear in the flat leaf nodes.
    primitives: Vec<Arc<dyn Intersectable>>,
}

impl From<BvhNode> for FlatBvh {
    /// Builds a flat BVH from a tree BVH.
    ///
    /// Traverses the tree in depth-first pre-order, collecting leaf
    /// primitives and emitting flat nodes. Interior children are stored
    /// by index; the DFS ordering guarantees children are emitted
    /// immediately after their parent.
    fn from(bvh: BvhNode) -> Self {
        let mut flat_nodes = Vec::new();
        let mut primitives = Vec::new();

        Self::flatten_node(bvh, &mut flat_nodes, &mut primitives);

        FlatBvh {
            nodes: flat_nodes,
            primitives,
        }
    }
}

impl FlatBvh {
    /// Recursively flattens a tree BVH node into the flat array.
    ///
    /// Takes ownership of the tree node and its children, moving leaf
    /// primitives into the flat primitive list.
    ///
    /// Returns the index of the emitted node.
    fn flatten_node(
        node: BvhNode,
        flat_nodes: &mut Vec<FlatBvhNode>,
        primitives: &mut Vec<Arc<dyn Intersectable>>,
    ) -> u32 {
        match node {
            BvhNode::Empty => {
                let idx = flat_nodes.len() as u32;
                // Degenerate leaf with zero primitives — never hits.
                flat_nodes.push(FlatBvhNode::leaf([0.0; 3], [0.0; 3], 0, 0));
                idx
            }
            BvhNode::Leaf { object, bbox } => {
                let prim_offset = primitives.len() as u32;
                primitives.push(object);
                let idx = flat_nodes.len() as u32;
                flat_nodes.push(FlatBvhNode::leaf(
                    [bbox.x.min, bbox.y.min, bbox.z.min],
                    [bbox.x.max, bbox.y.max, bbox.z.max],
                    prim_offset,
                    1,
                ));
                idx
            }
            BvhNode::Interior { left, right, bbox } => {
                let idx = flat_nodes.len() as u32;
                // Reserve slot; children will be emitted next, then we patch.
                flat_nodes.push(FlatBvhNode::interior(
                    [bbox.x.min, bbox.y.min, bbox.z.min],
                    [bbox.x.max, bbox.y.max, bbox.z.max],
                    // placeholder indices
                    0,
                    0,
                ));

                let left_idx = Self::flatten_node(*left, flat_nodes, primitives);
                let right_idx = Self::flatten_node(*right, flat_nodes, primitives);

                // Patch reserved slot with real child indices.
                flat_nodes[idx as usize].child_or_count = left_idx;
                flat_nodes[idx as usize].right_or_unused = right_idx;

                idx
            }
        }
    }

    /// Returns the number of flat nodes (for diagnostics).
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the number of primitives (for diagnostics).
    pub fn primitive_count(&self) -> usize {
        self.primitives.len()
    }
}

impl Intersectable for FlatBvh {
    fn intersect<'a>(&'a self, ray: &Ray, ray_t: Interval) -> Option<MaterialHit<'a>> {
        if self.nodes.is_empty() {
            return None;
        }

        let mut best_t = ray_t.max;
        let mut best_hit: Option<MaterialHit<'a>> = None;

        // Iterative stack-based traversal — no recursion, no allocation.
        let mut stack = [0u32; MAX_STACK];
        let mut sp = 0usize;
        stack[sp] = 0; // root
        sp += 1;

        // Precompute ray components — hoisted out of the loop to avoid
        // repeated field access through pointer indirection.
        let ox = ray.origin.x;
        let oy = ray.origin.y;
        let oz = ray.origin.z;
        let idx = ray.inverse_direction.x;
        let idy = ray.inverse_direction.y;
        let idz = ray.inverse_direction.z;
        let dx = ray.direction.x;
        let dy = ray.direction.y;
        let dz = ray.direction.z;
        let tmin = ray_t.min;

        while sp > 0 {
            sp -= 1;
            let node = &self.nodes[stack[sp] as usize];

            // Inline slab AABB test — avoids Interval construction and
            // array-of-tuples iteration per node visit.
            let mut lo = tmin;
            let mut hi = best_t;

            let t0 = (node.min[0] - ox) * idx;
            let t1 = (node.max[0] - ox) * idx;
            lo = lo.max(t0.min(t1));
            hi = hi.min(t0.max(t1));
            if hi <= lo {
                continue;
            }

            let t0 = (node.min[1] - oy) * idy;
            let t1 = (node.max[1] - oy) * idy;
            lo = lo.max(t0.min(t1));
            hi = hi.min(t0.max(t1));
            if hi <= lo {
                continue;
            }

            let t0 = (node.min[2] - oz) * idz;
            let t1 = (node.max[2] - oz) * idz;
            lo = lo.max(t0.min(t1));
            hi = hi.min(t0.max(t1));
            if hi <= lo {
                continue;
            }

            if node.is_leaf() {
                let start = node.prim_start();
                let count = node.prim_count();
                for i in start..start + count {
                    if let Some(mat_hit) =
                        self.primitives[i].intersect(ray, Interval::from(tmin, best_t))
                        && mat_hit.hit.time < best_t
                    {
                        best_t = mat_hit.hit.time;
                        best_hit = Some(mat_hit);
                    }
                }
            } else {
                let left_idx = node.left_child();
                let right_idx = node.right_child();

                // Near-first ordering via centroid projection — cheaper than
                // full t_bound (6 multiplies + 6 adds vs 36 ops per pair).
                // The 0.5 factor cancels in the comparison, so we skip it.
                let left = &self.nodes[left_idx as usize];
                let right = &self.nodes[right_idx as usize];

                let lcx = left.min[0] + left.max[0];
                let lcy = left.min[1] + left.max[1];
                let lcz = left.min[2] + left.max[2];
                let rcx = right.min[0] + right.max[0];
                let rcy = right.min[1] + right.max[1];
                let rcz = right.min[2] + right.max[2];

                let ld = (lcx - ox) * dx + (lcy - oy) * dy + (lcz - oz) * dz;
                let rd = (rcx - ox) * dx + (rcy - oy) * dy + (rcz - oz) * dz;

                let (near_idx, far_idx) = if ld <= rd {
                    (left_idx, right_idx)
                } else {
                    (right_idx, left_idx)
                };

                // Push far child first so near is popped first (stack LIFO).
                if sp < MAX_STACK {
                    stack[sp] = far_idx;
                    sp += 1;
                }
                if sp < MAX_STACK {
                    stack[sp] = near_idx;
                    sp += 1;
                }
            }
        }

        best_hit
    }
}

impl Bounded for FlatBvh {
    fn bounding_box(&self) -> Aabb {
        if self.nodes.is_empty() {
            return Aabb::new();
        }
        let root = &self.nodes[0];
        Aabb::from_intervals(
            crate::interval::Interval::from(root.min[0], root.max[0]),
            crate::interval::Interval::from(root.min[1], root.max[1]),
            crate::interval::Interval::from(root.min[2], root.max[2]),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hittable::{Bounded, Intersectable};
    use crate::material::Material;
    use crate::sphere::Sphere;
    use crate::vec3::Vec3;

    /// Number of bytes per flat BVH node. Chosen to fit one cache line (64B).
    const NODE_SIZE: usize = 64;

    #[test]
    fn flat_bvh_node_size() {
        assert_eq!(std::mem::size_of::<FlatBvhNode>(), NODE_SIZE);
    }

    #[test]
    fn flat_bvh_empty() {
        let bvh: BvhNode = BvhNode::Empty;
        let flat = FlatBvh::from(bvh);
        let ray = Ray::new_with_time(Vec3::ZERO, Vec3::from(0., 0., -1.), 0.0);
        assert!(
            flat.intersect(&ray, Interval::from(0.001, f64::INFINITY))
                .is_none()
        );
    }

    #[test]
    fn flat_bvh_single_sphere() {
        let sphere: Arc<dyn Intersectable> = Arc::new(Sphere::new(
            &Vec3::from(0., 0., -2.),
            0.5,
            Material::lambertian_color(0.8, 0.2, 0.2),
        ));
        let bbox = sphere.bounding_box();
        let bvh: BvhNode = BvhNode::Leaf {
            object: sphere.clone(),
            bbox,
        };
        let flat = FlatBvh::from(bvh);
        assert_eq!(flat.primitive_count(), 1);
        assert_eq!(flat.node_count(), 1);

        // Ray toward the sphere.
        let ray = Ray::new_with_time(Vec3::ZERO, Vec3::from(0., 0., -1.), 0.0);
        assert!(
            flat.intersect(&ray, Interval::from(0.001, f64::INFINITY))
                .is_some()
        );

        // Ray missing the sphere.
        let ray = Ray::new_with_time(Vec3::ZERO, Vec3::from(10., 0., -1.), 0.0);
        assert!(
            flat.intersect(&ray, Interval::from(0.001, f64::INFINITY))
                .is_none()
        );
    }

    #[test]
    fn flat_bvh_two_spheres() {
        let s1: Arc<dyn Intersectable> = Arc::new(Sphere::new(
            &Vec3::from(-1., 0., -2.),
            0.5,
            Material::lambertian_color(1.0, 0.0, 0.0),
        ));
        let s2: Arc<dyn Intersectable> = Arc::new(Sphere::new(
            &Vec3::from(1., 0., -2.),
            0.5,
            Material::lambertian_color(0.0, 1.0, 0.0),
        ));

        let bbox1 = s1.bounding_box();
        let bbox2 = s2.bounding_box();
        let merged_bbox = bbox1.merge(bbox2);

        let interior: BvhNode = BvhNode::Interior {
            left: Box::new(BvhNode::Leaf {
                object: s1.clone(),
                bbox: bbox1,
            }),
            right: Box::new(BvhNode::Leaf {
                object: s2.clone(),
                bbox: bbox2,
            }),
            bbox: merged_bbox,
        };

        let flat = FlatBvh::from(interior);
        assert_eq!(flat.primitive_count(), 2);
        assert_eq!(flat.node_count(), 3); // 1 interior + 2 leaves

        // Hit left sphere (at -1, 0, -2).
        let ray = Ray::new_with_time(Vec3::ZERO, Vec3::from(-1., 0., -2.).unit_vector(), 0.0);
        assert!(
            flat.intersect(&ray, Interval::from(0.001, f64::INFINITY))
                .is_some()
        );

        // Hit right sphere (at 1, 0, -2).
        let ray = Ray::new_with_time(Vec3::ZERO, Vec3::from(1., 0., -2.).unit_vector(), 0.0);
        assert!(
            flat.intersect(&ray, Interval::from(0.001, f64::INFINITY))
                .is_some()
        );

        // Hit neither.
        let ray = Ray::new_with_time(Vec3::ZERO, Vec3::from(0., 10., -1.), 0.0);
        assert!(
            flat.intersect(&ray, Interval::from(0.001, f64::INFINITY))
                .is_none()
        );
    }
}
