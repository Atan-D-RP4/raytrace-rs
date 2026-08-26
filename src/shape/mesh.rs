use std::sync::Arc;

use crate::bvh::aabb::Aabb;
use crate::intersect::interaction::Hit;
use crate::intersect::Bounded;
use crate::material::Material;
use crate::math::interval::Interval;
use crate::math::vec3::{Direction3, Point3};
use crate::ray::RayPacked;
use crate::shape::Shape3D;
use crate::texture::UVDifferentiable;

/// Mesh data structure for triangle meshes
pub struct MeshData {
    /// The positions of the vertices in the mesh
    pub positions: Vec<Point3>,
    /// The normals of the vertices in the mesh
    pub normals: Vec<Direction3>, // per-vertex, may be empty
    /// The UV coordinates of the vertices in the mesh
    pub uvs: Vec<(f32, f32)>, // per-vertex UVs, may be empty
    /// The indices of the triangles in the mesh
    pub indices: Vec<[u32; 3]>, // triangle vertex indices
    /// The materials of the triangles in the mesh
    pub per_tri_material: Option<Vec<Arc<Material>>>, // per-triangle materials
}

/// A mesh shape. Pure geometry — no material.
///
/// Material comes from the ShapeObject wrapper:
///   let mesh: Arc<dyn Intersectable> = Arc::new(ShapeObject::new(
///       MeshShape::from_data(data),
///       material,
///   ));
///
/// Or, for instancing, wrap Arc<MeshShape> directly:
///   let shape: Arc<dyn Intersectable> = Arc::new(MeshShape::from_data(data));
#[derive(Clone)]
pub struct MeshShape {
    data: Arc<MeshData>,
    bbox: Aabb,
    area: f32,
}

impl Bounded for MeshShape {
    fn bounding_box(&self) -> Aabb {
        self.bbox
    }
}

impl UVDifferentiable for MeshShape {
    fn uv_gradient(&self, _mapping_point: &Point3) -> (Direction3, Direction3) {
        // Phase 1 fallback: UV gradients are computed by the mesh policy with
        // the winning triangle's barycentrics (M3/M4). Without a resolved
        // triangle, the deliberate fallback is zero — the same fallback a
        // degenerate/UV-less mesh produces.
        (Direction3::ZERO, Direction3::ZERO)
    }
}

impl Shape3D for MeshShape {
    fn intersect_shape<const N: usize>(
        &self,
        ray: &RayPacked<N>,
        ray_t: Interval<N>,
    ) -> Option<Hit> {
        // Phase 1: no BLAS yet. Mesh intersection arrives with the mesh BVH
        // (M4) via `MeshBvh`; until then a mesh shape misses every ray.
        let _ = (ray, ray_t);
        None
    }
}
