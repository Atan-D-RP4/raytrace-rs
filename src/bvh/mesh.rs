use crate::bvh::BvhNode;

pub struct MeshBvh {
    nodes: Vec<BvhNode<2>>,
    tri_indices: Vec<u32>,
}
