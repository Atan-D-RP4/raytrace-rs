use crate::bvh::BvhNode;

/// Index of a triangle within a mesh's index buffer.
///
/// This is the BVH primitive for the mesh policy: `MeshPolicy` maps a
/// `TriangleIndex` to its vertices via `MeshData::indices` and returns a
/// [`MeshHit`](crate::intersect::interaction::MeshHit). The primitive
/// identity survives the BLAS so a later material layer can resolve
/// `MeshData::per_tri_material` without coupling the geometry BVH to
/// materials.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TriangleIndex(pub u32);

pub struct MeshBvh {
    nodes: Vec<BvhNode<2>>,
    tri_indices: Vec<u32>,
}
