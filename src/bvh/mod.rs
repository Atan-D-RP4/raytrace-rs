//! Cache-friendly BVH for fast ray-scene intersection.
//!
//! [`Bvh`] is a linearised version of [`BvhNode`] stored as a contiguous
//! array of nodes. The layout is optimised for traversal:
//!
//! - Each node is cache-line-aligned: W=2 → 64 bytes (1 CL), W=4 → 128 bytes (2 CL).
//! - f32 AABB fields avoid precision loss at the BVH/primitive boundary.
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
//!
//! References for optimizing BVH traversal and flat layouts on the CPU:
//! 1. Embree architecture: Wald et al., "Embree: A Kernel Framework for Efficient CPU Ray Tracing," SIGGRAPH 2014 — embree.org/papers/2014-Siggraph-Embree.pdf (https://www.embree.org/papers/2014-Siggraph-Embree.pdf)
//! 2. Embree source: kernels/bvh/bvh.h and kernels/bvh/bvh_node_aabb.h — github.com/RenderKit/embree (https://github.com/RenderKit/embree)
//! 3. WiVe algorithm: Fuetterling et al., "Accelerated Single-Ray Tracing for Wide Vector Units," HPG 2017
//! 4. PBRT BVH chapter: pbr-book.org/4ed/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies (https://www.pbr-book.org/4ed/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies)
//! 5. Psychopath BVH4 analysis: psychopath.io/post/2017_08_03_bvh4_without_simd (https://psychopath.io/post/2017_08_03_bvh4_without_simd)
//! 6. tinybvh: github.com/jbikker/tinybvh (https://github.com/jbikker/tinybvh) — includes BVH4_CPU, BVH8_CPU, CWBVH implementations
//! 7. CLPT paper (coherent packets for BVH4): jcgt.org/published/0004/04/05/
//! 8. Stackless BVH traversal: Hapala et al., "Efficient Stack-less BVH Traversal for Ray Tracing," SCCG 2011
//! 9. DRST (dynamic ray stream tracing): Barringer & Akenine-Möller, 2014

use std::simd::num::SimdFloat;
use std::simd::prelude::*;
use std::simd::{Mask, Simd};
use std::sync::Arc;

use tracing::info;

use crate::intersect::interaction::MaterialHit;
use crate::intersect::{Bounded, Intersectable};
use crate::math::interval::Interval;
use crate::ray::Ray;

pub mod aabb;
pub mod builder;
#[cfg(test)]
mod tests;

use aabb::{Aabb, AabbPacked};
use builder::TreeBuilder;

/// Maximum traversal stack depth. 64 handles BVHs with up to 2^64 primitives.
const MAX_STACK: usize = 64;

/// Inline slab AABB test. Returns true if the ray segment [tmin, tmax] intersects the AABB.
///
/// Computes for all 3 axes unconditionally to avoid branching and allow the compiler to vectorize.
#[inline]
fn slab_aabb_test(min: [f32; 3], max: [f32; 3], ray: &Ray, tmin: f32, tmax: f32) -> bool {
    let mut lo = tmin;
    let mut hi = tmax;

    let ox = ray.origin.x();
    let oy = ray.origin.y();
    let oz = ray.origin.z();
    let idx = ray.inverse_direction.x();
    let idy = ray.inverse_direction.y();
    let idz = ray.inverse_direction.z();

    // X slab
    let t0 = (min[0] - ox) * idx;
    let t1 = (max[0] - ox) * idx;

    // Update lo/hi with the intersection interval of the X slab.
    lo = lo.max(t0.min(t1));
    hi = hi.min(t0.max(t1));

    // Y slab
    let t0 = (min[1] - oy) * idy;
    let t1 = (max[1] - oy) * idy;

    // Update lo/hi with the intersection interval of the Y slab.
    lo = lo.max(t0.min(t1));
    hi = hi.min(t0.max(t1));

    // Z slab
    let t0 = (min[2] - oz) * idz;
    let t1 = (max[2] - oz) * idz;
    // Update lo/hi with the intersection interval of the Z slab.
    lo = lo.max(t0.min(t1));
    hi = hi.min(t0.max(t1));

    // If the intersection interval is empty, skip this node.
    hi > lo
}

/// N-wide BVH node. Layout is parametric on W.
///
/// For W=2 (binary):
///   64 bytes — 1 cache line, direction-sign traversal tests 1 AABB at a time.
///
/// For W=4 (wide):
///   128 bytes — 2 cache lines (per Embree BVH4Node), per-child AABB testing.
///
/// For W=8 (ultra-wide):
///   256 bytes — 4 cache lines.
///
/// Alignment: 64-byte cache line. The compiler automatically adds trailing padding so that
/// `Vec<BvhNode<W>>` has stride = 64B for W=2, 128B for W=4, 256B for W=8.
///
/// # Field layout (W=2 shown)
///
/// | Offset | Size | Field |
/// |--------|------|-------|
/// | 0      | 24   | min: [[f32;3]; W]   — child/leaf AABBs |
/// | 24     | 24   | max: [[f32;3]; W]   |
/// | 48     | 8    | child_offset: [u32; W] — wide node index or prim_start |
/// | 56     | 4    | leaf_info: [u16; W]  — prim_count for leaf children |
/// | 60     | 2    | leaf_mask: u16       — bit i → child i is a leaf |
/// | 62     | 1    | child_count: u8      — number of valid children (0..W) |
/// | 63     | 1    | split_axis: u8       — split axis for direction-sign ordering |
/// | **64** |      | Total (align(64) pads to 64) |
#[repr(C, align(64))]
#[derive(Debug, Clone)]
pub struct BvhNode<const W: usize> {
    /// SoA AABBs: W bounding boxes stored component-wise for SIMD gather.
    /// min[0] = first child's xmin, min[1] = second child's xmin, etc.
    pub bbox: AabbPacked<W>, // 24 * 2 * W
    /// W child indices or primitive offsets.
    /// For interior children: child_offset[i] = wide node index into `self.nodes`.
    /// For leaf children: child_offset[i] = prim_start into `self.primitives`.
    pub child_offset: [u32; W], // 4 * W
    /// Per-slot primitive count for leaf children.
    /// Interior children have leaf_info[i] = 0, and `self.child_count` stores the
    /// number of valid children. Leaf children use this for the inline primitive test.
    pub leaf_info: [u16; W], // 2 * W
    /// Bitmask: bit i = 1 → child i is a leaf node.
    pub leaf_mask: u16, // 2
    /// Number of valid children in this node (0..=W).
    /// For W=2 leaf nodes, this is the primitive count (≤ BVH_LEAF_THRESHOLD).
    /// For W≥4, this is the number of valid child slots (some may be unused).
    pub child_count: u8, // 1
    /// Axis along which this node was split (0=x, 1=y, 2=z).
    split_axis: u8, // 1
}

impl<const W: usize> BvhNode<W> {
    const ASSERT: () = assert!(
        W >= 2 && W.is_power_of_two(),
        "BvhNode width must be a power of two >= 2"
    );

    /// Bottom W bits all set — "every child is a leaf".
    const ALL_LEAVES: u16 = (!0u16) >> (u16::BITS as usize - W);

    /// True if this node is a leaf (all W children are primitives).
    /// For W=2: equivalent to `leaf_mask != 0` since both bits are always the same.
    /// For W≥4: true only when ALL children are leaves.
    #[inline]
    fn is_leaf(&self) -> bool {
        self.leaf_mask == Self::ALL_LEAVES
    }

    /// True if child `i` is a leaf. Used by the wide traversal (W≥4)
    /// to decide per-child: intersect primitives vs push onto stack.
    #[inline]
    #[allow(dead_code)] // only used in wide traversal (W≥4)
    fn is_child_leaf(&self, i: usize) -> bool {
        (self.leaf_mask & (1 << i)) != 0
    }

    #[inline]
    fn child(&self, i: usize) -> u32 {
        self.child_offset[i]
    }

    #[inline]
    fn prim_start(&self) -> usize {
        self.child_offset[0] as usize
    }

    #[inline]
    fn prim_count(&self) -> usize {
        self.child_count as usize
    }

    /// Creates a leaf node. `prim_start` is the index into `self.primitives`.
    fn leaf(bbox: AabbPacked<W>, prim_start: u32, prim_count: u16) -> Self {
        Self {
            bbox,
            child_offset: {
                let mut arr = [0; W];
                arr[0] = prim_start;
                arr
            },
            leaf_info: {
                let mut arr = [0; W];
                arr[0] = prim_count;
                arr
            },
            leaf_mask: Self::ALL_LEAVES,   // all W bits set
            child_count: prim_count as u8, // BVH_LEAF_THRESHOLD ≤ 4, safe truncation
            split_axis: 0,                 // unused for leaf nodes
        }
    }

    /// Creates an interior node. Children are filled in after creation.
    fn interior() -> Self {
        Self {
            bbox: AabbPacked::empty(),
            child_offset: [0; W],
            leaf_info: [0; W],
            leaf_mask: 0, // no bits set = all children are interior nodes
            child_count: 0,
            split_axis: 0, // will be set later
        }
    }
}

/// Flat BVH container: array-of-nodes + flat primitive list.
pub struct Bvh<const W: usize> {
    /// Contiguous array of flat BVH nodes in DFS (pre-order) layout.
    nodes: Vec<BvhNode<W>>,
    /// Scene primitives in the order they appear in the flat leaf nodes.
    primitives: Vec<Arc<dyn Intersectable>>,
}

impl<const W: usize> Bvh<W> {
    /// Creates a BVH from a list of scene objects.
    pub fn new(objects: &mut Vec<Arc<dyn Intersectable>>) -> Self {
        // Force const evaluation of BvhNode<W>'s width assertion.
        BvhNode::<W>::ASSERT;
        info!(object_count = objects.len(), "building bvh");
        let tree = TreeBuilder::new(objects);
        let bvh = Bvh::from(tree);
        info!(object_count = objects.len(), "bvh built");
        bvh
    }
}

impl<const W: usize> From<TreeBuilder> for Bvh<W> {
    /// Builds a flat BVH from a tree BVH.
    ///
    /// Traverses the tree in depth-first pre-order, collecting leaf
    /// primitives and emitting flat nodes. Interior children are stored
    /// by index; the DFS ordering guarantees children are emitted
    /// immediately after their parent.
    fn from(bvh: TreeBuilder) -> Self {
        let mut flat_nodes = Vec::new();
        let mut primitives = Vec::new();

        Self::flatten_node(bvh, &mut flat_nodes, &mut primitives);
        BvhNode::<W>::ASSERT;

        Bvh {
            nodes: flat_nodes,
            primitives,
        }
    }
}

impl<const W: usize> Bvh<W> {
    /// Recursively flattens a tree BVH node into the flat array.
    ///
    /// Takes ownership of the tree node and its children, moving leaf
    /// primitives into the flat primitive list.
    ///
    /// Returns the index of the emitted node.
    fn flatten_node(
        node: TreeBuilder,
        flat_nodes: &mut Vec<BvhNode<W>>,
        primitives: &mut Vec<Arc<dyn Intersectable>>,
    ) -> u32 {
        match node {
            TreeBuilder::Empty => {
                let idx = flat_nodes.len() as u32;
                // Degenerate leaf with zero primitives — never hits.
                flat_nodes.push(BvhNode::leaf(AabbPacked::empty(), 0, 0));
                idx
            }
            TreeBuilder::Interior { left, right, .. } => {
                // Extract correct bounding boxes BEFORE consuming children.
                // child_aabb(0) on a flat interior node only gives the first
                // grandchild's AABB — not the union — so we must capture the
                // TreeBuilder's own bbox (the true union) before flattening.
                let left_bbox = left.bounding_box();
                let right_bbox = right.bounding_box();

                let idx = flat_nodes.len() as u32;
                // Reserve slot; children will be emitted next, then we patch.
                flat_nodes.push(BvhNode::interior());

                let left_idx = Self::flatten_node(*left, flat_nodes, primitives);
                let right_idx = Self::flatten_node(*right, flat_nodes, primitives);

                // Patch reserved slot with real child indices.
                flat_nodes[idx as usize].child_offset[0] = left_idx;
                flat_nodes[idx as usize].child_offset[1] = right_idx;
                flat_nodes[idx as usize].child_count = 2;

                // Determine split axis from child centroid separation.
                let lct = left_bbox.centroid_point();
                let rct = right_bbox.centroid_point();
                let cdx = (rct.x() - lct.x()).abs();
                let cdy = (rct.y() - lct.y()).abs();
                let cdz = (rct.z() - lct.z()).abs();
                let split_axis = if cdx >= cdy && cdx >= cdz {
                    0
                } else if cdy >= cdz {
                    1
                } else {
                    2
                };
                flat_nodes[idx as usize].split_axis = split_axis;

                // Patch child AABBs using the correct union bounding boxes
                // extracted from the TreeBuilder (not child_aabb(0) which
                // only returns the first grandchild's AABB for interior nodes).
                // SoA layout: [axis][child_slot]
                let lb = left_bbox;
                let rb = right_bbox;
                // Slot 0 = left child
                flat_nodes[idx as usize].bbox.min[0][0] = lb.min[0][0];
                flat_nodes[idx as usize].bbox.min[1][0] = lb.min[1][0];
                flat_nodes[idx as usize].bbox.min[2][0] = lb.min[2][0];
                flat_nodes[idx as usize].bbox.max[0][0] = lb.max[0][0];
                flat_nodes[idx as usize].bbox.max[1][0] = lb.max[1][0];
                flat_nodes[idx as usize].bbox.max[2][0] = lb.max[2][0];
                // Slot 1 = right child
                flat_nodes[idx as usize].bbox.min[0][1] = rb.min[0][0];
                flat_nodes[idx as usize].bbox.min[1][1] = rb.min[1][0];
                flat_nodes[idx as usize].bbox.min[2][1] = rb.min[2][0];
                flat_nodes[idx as usize].bbox.max[0][1] = rb.max[0][0];
                flat_nodes[idx as usize].bbox.max[1][1] = rb.max[1][0];
                flat_nodes[idx as usize].bbox.max[2][1] = rb.max[2][0];
                idx
            }
            TreeBuilder::LeafN {
                objects,
                count,
                bbox,
                ..
            } => {
                let prim_offset = primitives.len() as u32;
                let prim_count = count as u16;
                primitives.extend(objects.iter().take(count).cloned());
                let idx = flat_nodes.len() as u32;
                flat_nodes.push(BvhNode::leaf(
                    AabbPacked::from(&bbox),
                    prim_offset,
                    prim_count,
                ));
                idx
            }
            TreeBuilder::Leaf { object, bbox } => {
                let prim_offset = primitives.len() as u32;
                primitives.push(object);
                let idx = flat_nodes.len() as u32;
                flat_nodes.push(BvhNode::leaf(AabbPacked::from(&bbox), prim_offset, 1));
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

// ---------------------------------------------------------------------------
// Wide BVH collapse — SAH-aware conversion from Bvh<2> to Bvh<W>
//
// The collapse algorithm processes the binary tree top-down, collecting
// up to W children per wide node. Each wide node corresponds to
// log2(W) levels of the binary tree, preserving the SAH quality of
// the original binary tree by only collapsing within those levels.
// ---------------------------------------------------------------------------

impl Bvh<2> {
    /// Widen this binary BVH to a wide BVH of width W.
    ///
    /// Collapses consecutive levels of the binary tree so that each wide
    /// node stores up to W children. Interior children below collapse depth
    /// become their own wide subtrees recursively.
    ///
    /// The result has the same primitives (shared via `Arc` clones) and
    /// produces identical intersection results.
    pub fn widen<const W: usize>(self) -> Bvh<W> {
        let collapse_depth = W.ilog2() as usize;
        let mut wide_nodes = Vec::new();

        if self.nodes.is_empty() {
            return Bvh {
                nodes: wide_nodes,
                primitives: self.primitives,
            };
        }

        Self::collapse_subtree::<W>(
            &self.nodes,
            0, // root index
            0, // current depth
            collapse_depth,
            &mut wide_nodes,
        );

        Bvh {
            nodes: wide_nodes,
            primitives: self.primitives,
        }
    }

    /// Recursively build a wide subtree from the binary node at `node_idx`.
    ///
    /// * Binary leaf → wide leaf node (ALL_LEAVES, single child in slot 0).
    /// * Binary interior at depth < collapse_depth → collect up to W children
    ///   by descending the binary tree, then emit a wide interior node.
    /// * Binary interior at depth ≥ collapse_depth → becomes an interior child
    ///   of the parent wide node; its own subtree is widened recursively.
    fn collapse_subtree<const W: usize>(
        binary: &[BvhNode<2>],
        node_idx: usize,
        depth: usize,
        collapse_depth: usize,
        wide_nodes: &mut Vec<BvhNode<W>>,
    ) -> u32 {
        let bn = &binary[node_idx];

        // Binary leaf → wide leaf with one pinned leaf child.
        if bn.is_leaf() {
            let mut wn = BvhNode::<W>::interior();
            wn.leaf_mask = BvhNode::<W>::ALL_LEAVES;
            wn.child_offset[0] = bn.child_offset[0]; // prim_start
            wn.leaf_info[0] = bn.child_count as u16; // prim_count
            let (bbox_min, bbox_max) = bn.bbox.child_aabb(0);
            for i in 0..W {
                wn.bbox.min[0][i] = bbox_min[0];
                wn.bbox.min[1][i] = bbox_min[1];
                wn.bbox.min[2][i] = bbox_min[2];
                wn.bbox.max[0][i] = bbox_max[0];
                wn.bbox.max[1][i] = bbox_max[1];
                wn.bbox.max[2][i] = bbox_max[2];
            }
            wn.child_count = 1;
            wide_nodes.push(wn);
            return (wide_nodes.len() - 1) as u32;
        }

        // Interior: reserve a wide node, then collect children.
        let wn_idx = wide_nodes.len() as u32;
        wide_nodes.push(BvhNode::<W>::interior());

        let mut offsets = [0u32; W];
        let mut leaf_counts = [0u16; W];
        let mut mins = [[0.0; 3]; W];
        let mut maxs = [[0.0; 3]; W];
        let mut leaf_mask: u16 = 0;
        let mut count: usize = 0;

        Self::collect_wide_children::<W>(
            binary,
            node_idx,
            depth,
            collapse_depth,
            wide_nodes,
            &mut offsets,
            &mut leaf_counts,
            &mut mins,
            &mut maxs,
            &mut leaf_mask,
            &mut count,
        );

        // Fill the wide node slots.
        let wn = &mut wide_nodes[wn_idx as usize];
        for i in 0..count {
            wn.child_offset[i] = offsets[i];
            wn.leaf_info[i] = leaf_counts[i];
            wn.bbox.min[0][i] = mins[i][0];
            wn.bbox.min[1][i] = mins[i][1];
            wn.bbox.min[2][i] = mins[i][2];
            wn.bbox.max[0][i] = maxs[i][0];
            wn.bbox.max[1][i] = maxs[i][1];
            wn.bbox.max[2][i] = maxs[i][2];
            if leaf_counts[i] > 0 {
                wn.leaf_mask |= 1u16 << i;
            }
        }
        wn.child_count = count as u8;
        wn.split_axis = bn.split_axis;

        wn_idx
    }

    /// Recursively collect up to W children from the binary subtree at `node_idx`.
    ///
    /// Each collected child is either:
    /// * A binary leaf → stored as a leaf child (prim_start, prim_count) with
    ///   `leaf_mask` bit set.
    /// * A binary interior at collapse depth → its subtree becomes a new wide
    ///   node; the child is stored as an interior child with the wide node's index.
    /// * A binary interior above collapse depth → recursed into to collect its
    ///   own children instead.
    fn collect_wide_children<const W: usize>(
        binary: &[BvhNode<2>],
        node_idx: usize,
        depth: usize,
        collapse_depth: usize,
        wide_nodes: &mut Vec<BvhNode<W>>,
        offsets: &mut [u32; W],
        leaf_counts: &mut [u16; W],
        mins: &mut [[f32; 3]; W],
        maxs: &mut [[f32; 3]; W],
        leaf_mask: &mut u16,
        count: &mut usize,
    ) {
        if *count >= W {
            return;
        }

        let node = &binary[node_idx];
        let slot = *count;

        if node.is_leaf() || depth >= collapse_depth {
            // --- Terminal: this node becomes a child of the wide node ---

            if node.is_leaf() {
                offsets[slot] = node.child_offset[0]; // prim_start
                leaf_counts[slot] = node.child_count as u16; // prim_count
                *leaf_mask |= 1u16 << slot;
                // Leaf AABB = node's own AABB (stored in min[0]/max[0]).
                let (child_min, child_max) = node.bbox.child_aabb(0);
                mins[slot] = child_min;
                maxs[slot] = child_max;
            } else {
                // Interior at collapse depth → build a wide subtree.
                let child_wide_idx = Self::collapse_subtree::<W>(
                    binary,
                    node_idx,
                    0, // restart depth for the new subtree
                    collapse_depth,
                    wide_nodes,
                );
                offsets[slot] = child_wide_idx;
                leaf_counts[slot] = 0; // interior child

                // Use the binary node's own child AABBs, which already encode
                // the full subtree bounds (including any deeper wide nodes).
                // The wide node's bbox only covers its direct children, so it
                // underestimates when those children are interior.
                let (min0, max0) = node.bbox.child_aabb(0);
                let (min1, max1) = node.bbox.child_aabb(1);
                for a in 0..3 {
                    mins[slot][a] = min0[a].min(min1[a]);
                    maxs[slot][a] = max0[a].max(max1[a]);
                }
            }

            *count += 1;
        } else {
            // --- Above collapse depth → descend the binary tree ---

            // Recursively process left child, then right child.
            // The recursion collects children into `offsets/leaf_counts/mins/maxs`
            // and advances `count`, up to W.
            Self::collect_wide_children::<W>(
                binary,
                node.child_offset[0] as usize,
                depth + 1,
                collapse_depth,
                wide_nodes,
                offsets,
                leaf_counts,
                mins,
                maxs,
                leaf_mask,
                count,
            );
            Self::collect_wide_children::<W>(
                binary,
                node.child_offset[1] as usize,
                depth + 1,
                collapse_depth,
                wide_nodes,
                offsets,
                leaf_counts,
                mins,
                maxs,
                leaf_mask,
                count,
            );
        }
    }
}

impl<const W: usize> Intersectable for Bvh<W>
where
    Simd<f32, W>: SimdPartialOrd + SimdFloat,
    <Simd<f32, W> as SimdPartialEq>::Mask: Into<Mask<i32, W>>,
{
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

        let tmin = ray_t.min;
        let dx = ray.direction.x();
        let dy = ray.direction.y();
        let dz = ray.direction.z();

        while sp > 0 {
            sp -= 1;
            let node = &self.nodes[stack[sp] as usize];

            // --- Compact leaf path (W=2 only) ---
            // Binary leaves store all primitives contiguously from child_offset[0].
            // This path does a single slab test then iterates the primitive range.
            if W == 2 && node.is_leaf() {
                let (child_min, child_max) = node.bbox.child_aabb(0);
                if !slab_aabb_test(child_min, child_max, ray, tmin, best_t) {
                    continue;
                }

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
                continue;
            }

            // --- Wide path (W ≥ 4): branchless per-child test ---
            if W >= 4 {
                let mut hits = node.bbox.hit_mask(ray, tmin, best_t);
                // Mask unused slots — empty sentinel AABBs always report a
                // hit, which defeats the early-exit check.
                let valid = ((1u32 << node.child_count) - 1) as u16;
                hits &= valid;
                if hits == 0 {
                    continue;
                }

                for i in 0..node.child_count as usize {
                    if (hits & (1 << i)) == 0 {
                        continue;
                    }

                    if node.is_child_leaf(i) {
                        // Inline leaf test — primitives for this child are pinned
                        // directly in the wide node (no separate leaf node).
                        let prim_start = node.child_offset[i];
                        let prim_count = node.leaf_info[i] as usize;
                        for p in &self.primitives[prim_start as usize..][..prim_count] {
                            if let Some(mat_hit) = p.intersect(ray, Interval::from(tmin, best_t))
                                && mat_hit.hit.time < best_t
                            {
                                best_t = mat_hit.hit.time;
                                best_hit = Some(mat_hit);
                            }
                        }
                    } else {
                        // Interior child — push for later traversal.
                        debug_assert!(sp < MAX_STACK, "Bvh traversal stack overflow");
                        stack[sp] = node.child_offset[i];
                        sp += 1;
                    }
                }
                continue;
            }

            // --- Binary interior (W = 2, !is_leaf): direction-sign ordering ---
            let hits = node.bbox.hit_mask(ray, tmin, best_t);
            let hit0 = (hits & 1) != 0;
            let hit1 = (hits & 2) != 0;
            if !hit0 && !hit1 {
                continue;
            }

            let sign = match node.split_axis {
                0 => (dx.to_bits() >> 31) as usize,
                1 => (dy.to_bits() >> 31) as usize,
                _ => (dz.to_bits() >> 31) as usize,
            };
            let near_idx = node.child(sign);
            let far_idx = node.child(1 - sign);
            let near_hit = if sign == 0 { hit0 } else { hit1 };
            let far_hit = if sign == 0 { hit1 } else { hit0 };

            debug_assert!(sp < MAX_STACK, "Bvh traversal stack overflow");
            if far_hit {
                stack[sp] = far_idx;
                sp += 1;
            }
            if near_hit {
                stack[sp] = near_idx;
                sp += 1;
            }
        }

        best_hit
    }

    fn occluded(&self, ray: &Ray, ray_t: Interval) -> bool {
        if self.nodes.is_empty() {
            return false;
        }

        let mut stack = [0u32; MAX_STACK];
        let mut sp = 0usize;
        stack[sp] = 0;
        sp += 1;

        let tmin = ray_t.min;
        let tmax = ray_t.max;
        let dx = ray.direction.x();
        let dy = ray.direction.y();
        let dz = ray.direction.z();

        while sp > 0 {
            sp -= 1;
            let node = &self.nodes[stack[sp] as usize];

            // --- Compact leaf path (W=2 only) ---
            if W == 2 && node.is_leaf() {
                let (child_min, child_max) = node.bbox.child_aabb(0);
                if !slab_aabb_test(child_min, child_max, ray, tmin, tmax) {
                    continue;
                }

                let start = node.prim_start();
                let count = node.prim_count();
                if self.primitives[start..start + count]
                    .iter()
                    .any(|p| p.occluded(ray, Interval::from(tmin, tmax)))
                {
                    return true;
                }
                continue;
            }

            // --- Wide path (W ≥ 4): branchless per-child test ---
            if W >= 4 {
                let mut hits = node.bbox.hit_mask(ray, tmin, tmax);
                // Mask unused slots — empty sentinel AABBs always report a
                // hit, which defeats the early-exit check.
                let valid = ((1u32 << node.child_count) - 1) as u16;
                hits &= valid;
                if hits == 0 {
                    continue;
                }

                for i in 0..node.child_count as usize {
                    if (hits & (1 << i)) == 0 {
                        continue;
                    }

                    if node.is_child_leaf(i) {
                        let prim_start = node.child_offset[i];
                        let prim_count = node.leaf_info[i] as usize;
                        if self.primitives[prim_start as usize..][..prim_count]
                            .iter()
                            .any(|p| p.occluded(ray, Interval::from(tmin, tmax)))
                        {
                            return true;
                        }
                    } else {
                        debug_assert!(sp < MAX_STACK, "Bvh traversal stack overflow");
                        stack[sp] = node.child_offset[i];
                        sp += 1;
                    }
                }
                continue;
            }

            // --- Binary interior (W = 2): direction-sign ordering ---
            let hits = node.bbox.hit_mask(ray, tmin, tmax);
            let hit0 = (hits & 1) != 0;
            let hit1 = (hits & 2) != 0;

            if !hit0 && !hit1 {
                continue;
            }

            let sign = match node.split_axis {
                0 => (dx.to_bits() >> 31) as usize,
                1 => (dy.to_bits() >> 31) as usize,
                _ => (dz.to_bits() >> 31) as usize,
            };
            let near_idx = node.child(sign);
            let far_idx = node.child(1 - sign);
            let near_hit = if sign == 0 { hit0 } else { hit1 };
            let far_hit = if sign == 0 { hit1 } else { hit0 };

            debug_assert!(sp < MAX_STACK, "Bvh traversal stack overflow");
            if far_hit {
                stack[sp] = far_idx;
                sp += 1;
            }
            if near_hit {
                stack[sp] = near_idx;
                sp += 1;
            }
        }
        false
    }
}

impl<const W: usize> Bounded for Bvh<W> {
    fn bounding_box(&self) -> Aabb {
        if self.nodes.is_empty() {
            return Aabb::empty();
        }
        // Phase AABB layout: each slot of each node stores individual child/leaf AABBs. The root's
        // W children partition the entire scene, so their union = scene bounds.
        let root = &self.nodes[0];
        let aabbs: [Aabb; W] = (&root.bbox).into();
        aabbs
            .iter()
            .fold(Aabb::empty(), |acc, aabb| acc.merge(aabb))
    }
}
