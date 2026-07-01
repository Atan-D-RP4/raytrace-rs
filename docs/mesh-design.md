# Mesh Feature: Shape3D + Internal BVH

## Design v1 — Single-Material Mesh with Geometry-Only Internal Acceleration

______________________________________________________________________

## Changelog

- **v1 (2026-06-29)** — Initial design after Shape3D/ShapeObject refactor.
  `Sampleable` is non-generic, `Shape3D::sample` takes `time: f64`.
  No mesh infrastructure exists. Design targets single-material meshes
  (most common case) with extension path for per-face materials.
- **v2 (2026-06-29)** — Bi-directional audit against 4 existing design docs
  (denoiser.md, adaptive-sampling.md, ARCH_HYBRID.md, samplestream-refactor.md).
  Fixed scene integration (Arc-based dual reg), added cross-doc refs,
  documented rasterizer tension as deferred integration point.

______________________________________________________________________

## 0. Problem Statement

The renderer supports spheres (`SphereShape`) and planar primitives
(`PlanarPatch<R>` for quads, triangles, ellipses, etc.). It has no way to
represent a **triangle mesh** — a collection of thousands of triangles sharing
vertex data and (typically) a single material.

A mesh is architecturally different from a sphere or a quad:

| Property | Sphere / Quad | Mesh |
|---|---|---|
| **Geometry** | Single parametric surface | Collection of triangles |
| **Intersection** | One analytic test | BVH traversal + per-triangle Möller–Trumbore |
| **Scale** | 1 primitive | 1K–1M+ triangles |
| **Material** | Single (always) | Typically single, occasionally per-face |
| **Data sharing** | None | Vertex/index buffers shared across all triangles |

A mesh cannot be a `PlanarPatch<TriRegion>` — that represents a single
triangle on a plane, not a collection with shared acceleration.

______________________________________________________________________

## 1. pbrt-v4 Reference

pbrt-v4 separates mesh geometry into three layers:

```
TriangleMesh
  └── Shared vertex data: positions, normals, UVs, indices
  └── One allocation, referenced by all triangles via global pointer array

Triangle (8 bytes: meshIndex + triIndex)
  └── Lightweight index pair referencing TriangleMesh vertex data
  └── Implements Shape interface: Intersect(), Bounds(), Area(), Sample()
  └── Intersect() fetches 3 vertices from mesh and runs Möller–Trumbore

GeometricPrimitive (Triangle + Material + Light + Medium)
  └── Wraps each Triangle Shape into a Primitive (Shape + Material)
  └── All primitives inserted into the scene BVH (BVHAggregate)

Shape :: TaggedPointer<Sphere, Cylinder, Disk, Triangle, BilinearPatch, Curve>
Primitive :: TaggedPointer<SimplePrimitive, GeometricPrimitive, ...BVHAggregate>
```

Key insights from pbrt-v4:

1. **Mesh is not a Shape.** The mesh is shared data. Individual triangles are
   Shapes. This enables per-triangle materials (though uncommon).

2. **No separate mesh-internal BVH.** Triangles go directly into the scene
   BVH. This works because `Triangle` is 8 bytes (fits in a tagged pointer),
   and `GeometricPrimitive` stores `Shape` (16 bytes) + `Material` (16 bytes)
   inline — no heap allocation per triangle.

3. **`CreateTriangles()` produces `Vec<Shape>`.** Each triangle is constructed
   from mesh + index and stored inline in the BVH leaf.

______________________________________________________________________

## 2. Current Architecture Audit

### 2.1 What We Have

```
Shape3D trait (src/shape/mod.rs)
  │   intersect_shape(&self, ray, ray_t) -> Option<Hit>
  │   bounding_box(&self) -> Aabb
  │   area(&self) -> f64
  │   sample(&self, u, v, time) -> (Point3, Vec3)
  │   sample_direction(&self, origin, u, v, time) -> Vec3       [default]
  │   pdf_direction(&self, origin, direction, time) -> f64      [default]
  │
  └── SphereShape (src/shape/sphere.rs) — example Shape3D impl

ShapeObject<Sh: Shape3D, M: Borrow<Material>>
  │   Wraps Sh + M, derives Intersectable + Bounded + Sampleable
  │   Intersectable::intersect → shape.intersect_shape() + borrow material
  │   Sampleable::random_direction → shape.sample_direction()
  │   Sampleable::pdf_value → shape.pdf_direction()
  │
  └── Sphere<M> = ShapeObject<SphereShape, M>
  └── Mesh<M>   = ShapeObject<MeshShape, M>  [future]

PlanarPatch<R: Region2D, M> (src/planar/mod.rs)
  │   Implements Intersectable, Bounded, Sampleable directly
  │   (not via ShapeObject — it predates the Shape3D trait)
  │
  └── Tri<M>  = PlanarPatch<TriRegion, M>   — single triangle on a plane
  └── Quad<M> = PlanarPatch<QuadRegion, M>

BvhNode / FlatBvh (src/bvh.rs, src/flat_bvh.rs)
  │   Stores Arc<dyn Intersectable> at leaves
  │   FlatBvh: cache-line-optimized iterative traversal, 64B nodes
  │   BvhNode: SAH-binned construction with rayon parallelism
  │
  └── Scene BVH: Arc<dyn Intersectable> = Sphere | Quad | Mesh | ...
```

### 2.2 Key Constraints

- **`FlatBvh` leaf primitives are `Arc<dyn Intersectable>`**, returning
  `MaterialHit` (geometry + material reference). For mesh-internal BVH
  we need geometry-only intersection (`Hit`, no material).

- **`Hit::geometric_normal` is private** — must use `Hit::new()` or
  `set_geometric_normal()`.

- **`Shape3D::sample()` and friends take `time: f64`** — important for
  animated meshes later (TODO), but not required for static meshes.

- **`Sampleable` is non-generic** (no `S: Sampler`) — uses raw
  `(u, v, time)` parameters.

- **Scene's `add_*` pattern** — emissive objects register both an
  `Arc<dyn Intersectable>` for the object list and an
  `Arc<dyn Sampleable>` for the light list (same geometry, two Arcs).

### 2.3 Non-Goal: Individual Mesh Triangles as Shape3Ds

We will NOT create a pbrt-style `MeshTriangle` index-pair that implements
`Shape3D` and gets individually wrapped in `ShapeObject`. The reasons:

1. **`Arc<dyn Intersectable>` overhead.** Each triangle would need a heap
   allocation for the `ShapeObject` + `Arc`. At 100K triangles this is
   prohibitive — memory pressure, cache misses, BVH construction time.

2. **No per-triangle material use case yet.** The codebase always assigns
   one material per shape. Per-face materials can be added later via a
   material-index buffer + `Sampleable` override.

3. **Existing BVH uses `Arc<dyn Intersectable>`.** The `BvhNode` / `FlatBvh`
   are designed for scene-level objects (hundreds), not mesh-level triangles
   (hundreds of thousands). A mesh needs its own internal accelerator.

______________________________________________________________________

## 3. Design: Mesh as Shape3D with Internal BVH

### 3.1 Data Structures

```rust
// ─── src/mesh/data.rs ───

/// Shared mesh vertex/index data. Cheaply cloneable via Arc.
pub struct MeshData {
    pub positions: Vec<Point3>,
    pub normals: Vec<Vec3>,      // per-vertex, may be empty
    pub uvs: Vec<(f64, f64)>,    // per-vertex UVs, may be empty
    pub indices: Vec<[u32; 3]>,  // triangle vertex indices
}
```

`MeshData` is the `TriangleMesh` equivalent from pbrt. Stored in an `Arc`
and shared by all references to the mesh geometry. Immutable after
construction.

```rust
// ─── src/mesh/bvh.rs ───

/// Internal flat BVH over mesh triangles. Geometry-only.
///
/// Differs from the scene-level FlatBvh:
///   - Leaf primitives are triangle index ranges, not Arc<dyn Intersectable>
///   - Intersection returns Option<MeshHit> not Option<MaterialHit>
///   - Triangle data is fetched from MeshData at intersection time
///   - No material, no vtable dispatch
pub struct MeshBvh {
    nodes: Vec<FlatBvhNode>,    // same node format as FlatBvh (64B cache line)
    tri_indices: Vec<u32>,      // triangle indices in traversal order
    mesh_data: Arc<MeshData>,   // source vertex data
}
```

The `FlatBvhNode` format is reused (64 bytes, same layout) — the leaf stores
`(tri_offset, tri_count)` where `tri_indices[tri_offset..tri_offset+tri_count]`
are triangle indices. At leaf traversal, each triangle index is fetched, its
three vertex positions are extracted from `mesh_data`, and
Möller–Trumbore intersection is computed on-the-fly.

This avoids per-triangle heap allocation entirely. A 100K-triangle mesh
with a 64K-node BVH costs ~4 MB for nodes + ~400 KB for tri_indices.

```rust
// ─── src/mesh/shape.rs ───

/// A mesh shape with internal BVH acceleration. Pure geometry — no material.
///
/// Material comes from the ShapeObject wrapper:
///   let mesh: Arc<dyn Intersectable> = Arc::new(ShapeObject::new(
///       MeshShape::from_data(data),
///       material,
///   ));
pub struct MeshShape {
    data: Arc<MeshData>,
    bvh: MeshBvh,
    bbox: Aabb,          // precomputed from all triangle AABBs
    area: f64,            // precomputed sum of triangle areas
}
```

### 3.2 Shape3D Implementation

```rust
impl Shape3D for MeshShape {
    fn intersect_shape(&self, ray: &Ray, ray_t: Interval) -> Option<Hit> {
        // Traverse MeshBvh:
        //   1. Iterative stack traversal over FlatBvhNode array
        //   2. At leaf, for each tri_index:
        //      a. Fetch 3 vertex positions from self.data
        //      b. Run Möller–Trumbore intersection
        //      c. On hit, compute barycentric-interpolated normal + UVs
        //      d. Return closest Hit
        self.bvh.intersect(ray, ray_t)
    }

    fn bounding_box(&self) -> Aabb {
        self.bbox
    }

    fn area(&self) -> f64 {
        self.area
    }

    fn sample(&self, u: f64, v: f64, time: f64) -> (Point3, Vec3) {
        // Uniform triangle sampling:
        //   1. Pick triangle i by area-weighted distribution
        //   2. Sample barycentric coords via sqrt(u), sqrt(v) method
        //   3. Interpolate vertex positions + normal
        //   4. Return (point, normal)
    }

    // sample_direction  — uses default (area-based) ⇒ fine for meshes
    // pdf_direction     — uses default (area-to-solid-angle) ⇒ fine for meshes
}
```

### 3.3 `MeshHit` Struct (Internal)

```rust
/// Intermediate hit result produced by MeshBvh traversal.
/// Converted to `Hit` before returning from `intersect_shape`.
struct MeshHit {
    t: f64,
    point: Point3,
    normal: Vec3,       // geometric normal (unit length)
    uv: (f64, f64),     // barycentric → texture UV
    tri_index: u32,
}
```

### 3.4 Möller–Trumbore Integration

The triangle intersection routine needs to handle:

- **Watertight intersection** near edges (double-precision fallback when
  barycentric coordinates are near-zero, following pbrt-v4's approach).
- **Back-face culling** is NOT performed — the `set_face_normal` logic
  in `SurfaceInteraction` handles front-face determination.
- **UV interpolation** when `mesh.normals` / `mesh.uvs` are present.
- **Shading normal interpolation** from per-vertex normals when available.

The intersection function is:

```rust
fn intersect_triangle(
    ray: &Ray, t_max: f64,
    p0: Point3, p1: Point3, p2: Point3,
) -> Option<(f64, f64, f64, f64)>  // (t, b0, b1, b2) barycentric coords
```

### 3.5 BVH Construction

The mesh BVH construction mirrors `BvhNode::new` (SAH binning) but:

- Primitives are triangle index ranges, not `Arc<dyn Intersectable>`
- AABB computation fetches vertex positions from `MeshData`
- Centroid computation uses the triangle's centroid (average of 3 vertices)
- Uses `rayon::join` for parallel construction (same as scene BVH)

The output is a `MeshBvh` (flat BVH node array + triangle index array).

### 3.6 Scene Integration

```rust
// In src/shape/mod.rs — blanket impl so Arc<MeshShape> can be used in ShapeObject

impl Shape3D for Arc<MeshShape> {
    fn intersect_shape(&self, ray: &Ray, ray_t: Interval) -> Option<Hit> {
        self.as_ref().intersect_shape(ray, ray_t)
    }
    fn bounding_box(&self) -> Aabb { self.as_ref().bounding_box() }
    fn area(&self) -> f64 { self.as_ref().area() }
    fn sample(&self, u: f64, v: f64, time: f64) -> (Point3, Vec3) {
        self.as_ref().sample(u, v, time)
    }
    fn sample_direction(&self, origin: Point3, u: f64, v: f64, time: f64) -> Vec3 {
        self.as_ref().sample_direction(origin, u, v, time)
    }
    fn pdf_direction(&self, origin: Point3, direction: Vec3, time: f64) -> f64 {
        self.as_ref().pdf_direction(origin, direction, time)
    }
}
```

This enables dual registration without deep-copying the mesh BVH:

```rust
// In src/scene.rs — following add_sphere pattern, shares BVH via Arc

impl Scene {
    pub fn add_mesh(&mut self, data: Arc<MeshData>, material: Material) {
        let mesh_shape = Arc::new(MeshShape::from_data(data));
        let material = Arc::new(material);
        // First ShapeObject consumes mesh_shape.clone() (cheap Arc increment)
        let obj: Arc<dyn Intersectable> = Arc::new(
            ShapeObject::new(mesh_shape.clone(), material.clone()),
        );
        if material.is_emissive() {
            // Second ShapeObject shares the same MeshShape + BVH via second Arc increment
            let light: Arc<dyn Sampleable> = Arc::new(
                ShapeObject::new(mesh_shape, material),
            );
            self.add_intersectable(obj, Some(light));
        } else {
            self.add_intersectable(obj, None);
        }
    }
}
```

Key points:

- `MeshShape::from_data` builds the BVH exactly once.
- `Arc::clone(&mesh_shape)` is a cheap atomic ref-count increment.
- `Arc<MeshShape>: Shape3D` via the delegation impl above.
- Two `ShapeObject<Arc<MeshShape>, Arc<Material>>` instances share the same
  `MeshShape` and BVH arrays — no deep-copy.

### 3.7 TransformObject Compatibility

`Mesh` wraps naturally:

```rust
TransformObject<Translate, ShapeObject<MeshShape, Arc<Material>>>
```

`TransformObject` implements `Intersectable` by:

1. Transform ray → object space via `world_to_object_point`
2. Intersect mesh in object space
3. Transform hit back via `Transform::hit()`

This works unchanged because `ShapeObject<MeshShape, M>` implements
`Intersectable` (which `TransformObject` requires).

______________________________________________________________________

## 4. Option Comparison

| Criterion | Individual Tri Shapes (pbrt) | MeshShape + Internal BVH (ours) |
|---|---|---|
| **Per-triangle material** | ✅ Natural | ❌ Single material (can add later) |
| **Per-triangle memory** | 8 bytes (Triangle) + Shape (16B) + Material (16B) = ~40B/tri | 0 bytes/tri (data in shared MeshData) |
| **Vtable dispatch** | Per-triangle via TaggedPointer (fast) | Per-mesh via Shape3D (faster) |
| **BVH construction** | Scene BVH handles all (tens of K objects) | Two-level: scene BVH over meshes, mesh BVH over tris |
| **Cache behavior** | Triangles scattered in scene BVH leaves | Mesh triangles in compact MeshBvh = better locality |
| **Implementation cost** | Moderate (needs Shape3D triangle, global mesh registry) | Moderate (needs MeshBvh, Möller–Trumbore) |
| **Scene building speed** | O(N log N) scene BVH over N triangles | O(N log N) mesh BVH + O(M log M) scene BVH over M meshes |

The internal BVH approach is preferred because:

1. **Memory efficiency** dominates at 100K+ triangles — no per-triangle
   allocations, no Arc overhead, no vtable pointers.

2. **Cache-friendly traversal** — the mesh BVH stores triangle indices
   compactly, and the FlatBvhNode layout is 64B cache lines.

3. **Fits the Shape3D abstraction** — `MeshShape` is just another shape
   to `ShapeObject`. No new trait, no new wrapper, no architecture break.

4. **Two-level BVH is standard practice** — Embree, OptiX, and most
   production renderers use two-level acceleration (scene BVH over
   instance BVHs per mesh).

______________________________________________________________________

## 5. Files to Create / Modify

### New files

| File | Contents |
|---|---|
| `src/mesh/mod.rs` | Module root: re-exports from data, bvh, shape |
| `src/mesh/data.rs` | `MeshData` struct + OBJ/PLY parsing |
| `src/mesh/bvh.rs` | `MeshBvh` struct + flat BVH traversal + SAH construction |
| `src/mesh/shape.rs` | `MeshShape` struct + `Shape3D` impl + free `mesh()` constructor |

### Modified files

| File | Change |
|---|---|
| `src/shape/mod.rs` | Add `impl Shape3D for Arc<MeshShape>` (delegation blanket — 6 methods). Required for shared-BVH dual registration. |
| `src/lib.rs` | Add `pub mod mesh;` |
| `src/scene.rs` | Add `add_mesh()` with Arc-based BVH sharing for emissive meshes |

### No changes needed

| File | Reason |
|---|---|
| `src/hittable.rs` | Traits unchanged. `Hit`, `MaterialHit` unchanged. |
| `src/bvh.rs` | Scene BVH unchanged — builds over `Arc<dyn Intersectable>` as before. |
| `src/flat_bvh.rs` | Scene flat BVH unchanged. MeshBvh reuses `FlatBvhNode` layout. |
| `src/planar/mod.rs` | Individual `Tri<M>` via `PlanarPatch<TriRegion, M>` unchanged. |

______________________________________________________________________

## 6. Implementation Phases

### Phase 1 — Core Geometry (this PR)

1. `MeshData` — positions, normals, uvs, indices. OBJ file format parser.
2. `MeshBvh` — SAH construction, flat-node iterative traversal.
3. `MeshShape: Shape3D` — intersection via internal BVH + Möller–Trumbore.
4. `mesh()` constructor + `add_mesh()` scene method.
5. Integration test: Cornell box mesh variant with a single quad/tri mesh.

### Phase 2 — Sampling + Light Integration

1. `MeshShape::sample()` — area-weighted triangle sampling.
2. `MeshShape::area()` — precomputed sum.
3. Verify emissive mesh integration via `Sampleable` (default delegation
   via `ShapeObject` already works).

### Phase 3 — Normals + UV Interpolation

1. Per-vertex normal interpolation with barycentric coordinates.
2. Per-vertex UV interpolation.
3. Auto-generated smooth normals when mesh has no normals.

### Phase 4 — File Format Support

1. OBJ parser with material groups.
2. PLY parser (binary + ASCII).
3. Optional: material-index-per-face for per-face materials.

______________________________________________________________________

## 7. Open Questions

1. **`FlatBvhNode` reuse.** Should `MeshBvh` own a separate copy of the
   `FlatBvhNode` struct (same layout), or should `FlatBvhNode` be extracted
   into a shared module? Current code is in `src/flat_bvh.rs`. Extracting
   avoids duplication but broadens the API surface.

2. ~~**`MeshShape::clone()` for dual light registration.**~~ **RESOLVED in v2.**
   The scene code (§3.6) uses `impl Shape3D for Arc<MeshShape>` so an
   `Arc::clone` (cheap atomic increment) shares the BVH between two
   `ShapeObject` instances. No deep-copy of the BVH array. Requires the
   delegation impl in `src/shape/mod.rs`.

3. **Two-level BVH vs single flat list.** For scenes with one mesh, a
   single BVH is sufficient. For scenes with many meshes, two-level is
   essential. The current design assumes two-level (scene BVH of meshes).
   If a scene has only meshes (no spheres/quads), is two-level overhead
   acceptable? Yes — each mesh's BVH is traversed once per ray, and the
   scene BVH culls non-hit meshes.

4. **Watertight intersection.** pbrt-v4 uses double-precision fallback
   for near-zero barycentric coordinates. Required for production quality
   but can be deferred to Phase 2.

5. **ARCH_HYBRID TriangleRasterizer ↔ mesh triangle access.**
   ARCH_HYBRID.md §2 describes a `TriangleRasterizer` that iterates all
   world triangles by vertex transform → clip → rasterize. A `MeshShape`
   (single `Shape3D`) is opaque to per-triangle iteration — the rasterizer
   cannot extract vertex data from it. This is a deferred integration point:
   either (a) `MeshShape` exposes a `triangles()` iterator, (b) the
   rasterizer uses intersection-based rasterization (ray-per-pixel), or
   (c) the rasterizer requires scene geometry to be decomposed into
   individual triangles (contradicts our mesh design). Not blocking Phase 1.

______________________________________________________________________

## 8. Cross-Document References

### Existing design docs (bi-directional audit v2)

| Doc | Relationship to Mesh | Status |
|---|---|---|
| `ARCH_HYBRID.md` §2, §9 | `SampleableEnum` needs `Mesh` variant (additive). `TriangleRasterizer` needs mesh triangle access (deferred, see §7.5). Primitive registration pattern matches. | ✅ Compatible, 2 tensions documented |
| `denoiser.md` | Denoiser post-processes film output. Orthogonal to geometry. No shared interfaces. | ✅ No conflict |
| `adaptive-sampling.md` | Variance estimation + convergence criteria. Orthogonal to geometry types. | ✅ No conflict |
| `samplestream-refactor.md` | `SampleStreamEnum` replaces `DimCursor` in integrator signatures. Mesh uses `Sampleable` (non-generic, raw params). | ✅ No conflict |
| `CORE_THESIS.md` §4 | SpatialDomain pattern, leaf sovereignty. (External reference, not in docs/). | ✅ Compatible |

### Codebase references

- `src/shape/mod.rs` — Shape3D trait, ShapeObject wrapper.
- `src/flat_bvh.rs` — FlatBvhNode layout (64 bytes), iterative traversal.
- `src/planar/mod.rs` and `src/planar/tri.rs` — TriRegion (existing single-triangle primitive).
