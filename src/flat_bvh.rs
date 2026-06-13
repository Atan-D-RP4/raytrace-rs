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

use std::sync::Arc;

use crate::aabb::Aabb;
use crate::bvh::BvhNode;
use crate::hittable::{HitRecord, Hittable};
use crate::interval::Interval;
use crate::ray::Ray;
use crate::sampler::Sampler;

/// Maximum traversal stack depth. 64 handles BVHs with up to 2^64 primitives.
const MAX_STACK: usize = 64;

/// A flat (array-of-structs) BVH node.
///
/// Layout is `repr(C)` for deterministic padding and cache-line alignment.
/// Interior nodes: left/right are indices into the flat node array.
/// Leaf nodes: prim_offset/count index into the primitive index array.
///
/// # Memory Layout (64 bytes)
///
/// ```text
/// [0..6]   min/max AABB: 6 x f64 = 48 bytes
/// [48]     child_or_count: u32  (interior: left child index; leaf: prim count)
/// [52]     right_or_unused: u32 (interior: right child index; leaf: 0)
/// [56]     prim_offset: u32     (leaf: start index; interior: 0)
/// [60]     flags: u8            (0 = interior, 1 = leaf)
/// [61..63] _pad: [u8; 3]
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FlatBvhNode {
    pub min: [f64; 3], // min_x, min_y, min_z
    pub max: [f64; 3], // min_x, min_y, min_z
    pub child_or_count: u32,
    pub right_or_unused: u32,
    pub prim_offset: u32,
    pub flags: u8,
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

    /// Slab-method AABB-ray intersection test.
    #[inline]
    fn hit_aabb(&self, ray: &Ray, ray_t: &mut Interval) -> bool {
        for (origin, inv_d, ax_min, ax_max) in [
            (
                ray.origin.x,
                ray.inverse_direction.x,
                self.min[0], // min_x
                self.max[0], // max_x
            ),
            (
                ray.origin.y,
                ray.inverse_direction.y,
                self.min[1], // min_y
                self.max[1], // max_y
            ),
            (
                ray.origin.z,
                ray.inverse_direction.z,
                self.min[2],
                self.max[2],
            ),
        ] {
            let t0 = (ax_min - origin) * inv_d;
            let t1 = (ax_max - origin) * inv_d;
            ray_t.min = ray_t.min.max(t0.min(t1));
            ray_t.max = ray_t.max.min(t0.max(t1));
            if ray_t.max <= ray_t.min {
                return false;
            }
        }
        true
    }
}

/// Flat BVH container: array-of-nodes + flat primitive list.
pub struct FlatBvh<S: Sampler> {
    /// Contiguous array of flat BVH nodes in DFS (pre-order) layout.
    nodes: Vec<FlatBvhNode>,
    /// Scene primitives in the order they appear in the flat leaf nodes.
    primitives: Vec<Arc<dyn Hittable<S>>>,
}

impl<S: Sampler> FlatBvh<S> {
    /// Builds a flat BVH from a tree BVH.
    ///
    /// Traverses the tree in depth-first pre-order, collecting leaf
    /// primitives and emitting flat nodes. Interior children are stored
    /// by index; the DFS ordering guarantees children are emitted
    /// immediately after their parent.
    pub fn from_bvh(bvh: BvhNode<S>) -> Self {
        let mut flat_nodes = Vec::new();
        let mut primitives = Vec::new();

        Self::flatten_node(bvh, &mut flat_nodes, &mut primitives);

        FlatBvh {
            nodes: flat_nodes,
            primitives,
        }
    }

    /// Recursively flattens a tree BVH node into the flat array.
    ///
    /// Takes ownership of the tree node and its children, moving leaf
    /// primitives into the flat primitive list.
    ///
    /// Returns the index of the emitted node.
    fn flatten_node(
        node: BvhNode<S>,
        flat_nodes: &mut Vec<FlatBvhNode>,
        primitives: &mut Vec<Arc<dyn Hittable<S>>>,
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

    /// Estimates the t-value at which a ray enters a node's AABB.
    /// Used for near-first child ordering during traversal.
    #[inline]
    fn t_bound(ray: &Ray, node: &FlatBvhNode) -> f64 {
        let mut t_min = f64::NEG_INFINITY;
        for (origin, inv_d, ax_min, ax_max) in [
            (
                ray.origin.x,
                ray.inverse_direction.x,
                node.min[0],
                node.max[0],
            ),
            (
                ray.origin.y,
                ray.inverse_direction.y,
                node.min[1],
                node.max[1],
            ),
            (
                ray.origin.z,
                ray.inverse_direction.z,
                node.min[2],
                node.max[2],
            ),
        ] {
            let t0 = (ax_min - origin) * inv_d;
            let t1 = (ax_max - origin) * inv_d;
            t_min = t_min.max(t0.min(t1));
        }
        t_min
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

impl<S: Sampler> Hittable<S> for FlatBvh<S> {
    fn hit<'a>(&'a self, ray: &Ray, ray_t: Interval) -> Option<HitRecord<'a>> {
        if self.nodes.is_empty() {
            return None;
        }

        let mut best_t = ray_t.max;
        let mut best_hit: Option<HitRecord> = None;

        // Iterative stack-based traversal — no recursion, no allocation.
        let mut stack = [0u32; MAX_STACK];
        let mut sp = 0usize;
        stack[sp] = 0; // root
        sp += 1;

        while sp > 0 {
            sp -= 1;
            let node_idx = stack[sp] as usize;
            let node = &self.nodes[node_idx];

            // AABB test with the current best-t to prune distant hits.
            let mut current_t = Interval::from(ray_t.min, best_t);
            if !node.hit_aabb(ray, &mut current_t) {
                continue;
            }

            if node.is_leaf() {
                let start = node.prim_start();
                let count = node.prim_count();
                for i in start..start + count {
                    if let Some(hit) =
                        self.primitives[i].hit(ray, Interval::from(ray_t.min, best_t))
                        && hit.time < best_t
                    {
                        best_t = hit.time;
                        best_hit = Some(hit);
                    }
                }
            } else {
                let left_idx = node.left_child();
                let right_idx = node.right_child();

                // Near-first ordering: push far child first so near is popped first.
                let left_node = &self.nodes[left_idx as usize];
                let right_node = &self.nodes[right_idx as usize];
                let left_t = Self::t_bound(ray, left_node);
                let right_t = Self::t_bound(ray, right_node);

                if left_t <= right_t {
                    // Push right (far) first, then left (near).
                    if sp < MAX_STACK {
                        stack[sp] = right_idx;
                        sp += 1;
                    }
                    if sp < MAX_STACK {
                        stack[sp] = left_idx;
                        sp += 1;
                    }
                } else {
                    if sp < MAX_STACK {
                        stack[sp] = left_idx;
                        sp += 1;
                    }
                    if sp < MAX_STACK {
                        stack[sp] = right_idx;
                        sp += 1;
                    }
                }
            }
        }

        best_hit
    }

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
    use crate::material::Material;
    use crate::sampler::SobolQmcSampler;
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
        let bvh = BvhNode::<SobolQmcSampler>::Empty;
        let flat = FlatBvh::from_bvh(bvh);
        let ray = Ray::new_with_time(Vec3::ZERO, Vec3::from(0., 0., -1.), 0.0);
        assert!(
            flat.hit(&ray, Interval::from(0.001, f64::INFINITY))
                .is_none()
        );
    }

    #[test]
    fn flat_bvh_single_sphere() {
        let sphere: Arc<dyn Hittable<SobolQmcSampler>> = Arc::new(Sphere::new(
            &Vec3::from(0., 0., -2.),
            0.5,
            Material::lambertian_color(0.8, 0.2, 0.2),
        ));
        let bbox = sphere.bounding_box();
        let bvh = BvhNode::Leaf {
            object: sphere.clone(),
            bbox,
        };
        let flat = FlatBvh::from_bvh(bvh);
        assert_eq!(flat.primitive_count(), 1);
        assert_eq!(flat.node_count(), 1);

        // Ray toward the sphere.
        let ray = Ray::new_with_time(Vec3::ZERO, Vec3::from(0., 0., -1.), 0.0);
        assert!(
            flat.hit(&ray, Interval::from(0.001, f64::INFINITY))
                .is_some()
        );

        // Ray missing the sphere.
        let ray = Ray::new_with_time(Vec3::ZERO, Vec3::from(10., 0., -1.), 0.0);
        assert!(
            flat.hit(&ray, Interval::from(0.001, f64::INFINITY))
                .is_none()
        );
    }

    #[test]
    fn flat_bvh_two_spheres() {
        let s1: Arc<dyn Hittable<SobolQmcSampler>> = Arc::new(Sphere::new(
            &Vec3::from(-1., 0., -2.),
            0.5,
            Material::lambertian_color(1.0, 0.0, 0.0),
        ));
        let s2: Arc<dyn Hittable<SobolQmcSampler>> = Arc::new(Sphere::new(
            &Vec3::from(1., 0., -2.),
            0.5,
            Material::lambertian_color(0.0, 1.0, 0.0),
        ));

        let bbox1 = s1.bounding_box();
        let bbox2 = s2.bounding_box();
        let merged_bbox = bbox1.merge(bbox2);

        let interior = BvhNode::Interior {
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

        let flat = FlatBvh::from_bvh(interior);
        assert_eq!(flat.primitive_count(), 2);
        assert_eq!(flat.node_count(), 3); // 1 interior + 2 leaves

        // Hit left sphere (at -1, 0, -2).
        let ray = Ray::new_with_time(Vec3::ZERO, Vec3::from(-1., 0., -2.).unit_vector(), 0.0);
        assert!(
            flat.hit(&ray, Interval::from(0.001, f64::INFINITY))
                .is_some()
        );

        // Hit right sphere (at 1, 0, -2).
        let ray = Ray::new_with_time(Vec3::ZERO, Vec3::from(1., 0., -2.).unit_vector(), 0.0);
        assert!(
            flat.hit(&ray, Interval::from(0.001, f64::INFINITY))
                .is_some()
        );

        // Hit neither.
        let ray = Ray::new_with_time(Vec3::ZERO, Vec3::from(0., 10., -1.), 0.0);
        assert!(
            flat.hit(&ray, Interval::from(0.001, f64::INFINITY))
                .is_none()
        );
    }
}
