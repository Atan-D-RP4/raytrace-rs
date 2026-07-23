# Mesh Feature: Shape3D + Internal BVH

## Design v7 — Codebase Reconciliation: BVH, Sampleable, glam/f32

______________________________________________________________________

## Changelog

- **v7 (2026-07-23)** — Codebase reconciliation: BVH restructure, Sampleable decoupling, glam/f32.
  - **BVH module restructured.** `src/bvh.rs` + `src/flat_bvh.rs` merged into
    `src/bvh/` with `mod.rs`, `builder.rs`, `aabb.rs`, `tests.rs`. The old
    `FlatBvh` and `FlatBvhNode` types are gone. Replaced by generic `Bvh<W>`
    and `BvhNode<W>` (parametric width W=2,4,8) with SoA layout and SIMD
    `hit_mask` traversal.
  - **`Sampleable` decoupled from `Intersectable.` Now `Sampleable: Send + Sync`
    only. Area lights participate in both lists; non-geometric lights (environment)
    only in `important_objects`. `add_importance_target` no longer pushes to `objects`.
  - **Glam migration complete.** Custom Vec3 replaced with `glam::Vec3`. `Point3`,
    `Direction3`, `Color3` are newtype wrappers. TransformObject still uses
    `Translate`/`RotateY` (not yet `Affine3A`).
  - **f32 migration complete.** All type signatures use `f32`, not `f64`.
  - **MeshBvh design updated.** `Bvh<W>` stores `Vec<Arc<dyn Intersectable>>`
    and can't be directly reused for mesh (needs geometry-only triangle indices).
    `MeshBvh` will use `BvhNode<W>` node format but with `Vec<u32>` tri_indices
    and separate construction from `TreeBuilder`.
- **v5 (2026-07-20)** — PlanarPatch deleted, PlanarShape unification completed.
  - **`src/planar/` module removed entirely.** `PlanarPatch<R, M>`, `PlanarHit`,
    and all `PlanarPatch` type aliases deleted. No deprecation period.
  - **§2.4 rewritten:** Fork analysis updated — PlanarShape unification resolved
    the structural fork; `PlanarShape<R>` now lives in `src/shape/` and replaces
    PlanarPatch's geometry role. "When to Use Which" table revised.
  - **§3.8 updated:** `BoxShape` implemented without generic scalar parameter.
    Code examples match actual `BoxShape::new(p_min, p_max)` API.
  - **§5 rewritten:** Files list reflects `src/planar/` deletion,
    `src/shape/constructors.rs`, `src/shape/regions/` module.
  - **§6 updated:** Phase 1 Step 7 (BoxShape) completed. Phase 5 (Tessellatable)
    deferred. PlanarPatch bridge sections marked as superseded.
  - **§7 OQ6 resolved:** Primitive fork closed — PlanarShape unification means
    all planar types now go through Shape3D → ShapeObject, sharing one trait
    with Sphere and Box. Only the future `MeshShape` and `PlanarShape` categories
    needed in Primitive.
  - **§8 updated:** Cross-references updated for PlanarPatch removal.
- **v4 (2026-07-20)** — Adversarial review audit against actual codebase state.
  Caught and corrected:
  - **§2.4 added:** Object model fork analysis — `Shape3D+ShapeObject` vs
    `Region2D+PlanarPatch` are structurally parallel with zero shared
    implementation below the `Intersectable`/`Bounded`/`Sampleable` scene
    traits. Per-shape-category use-case guidance matrix.
  - **§3.8 added:** `BoxShape<T>: Shape3D` — uniform-material axis-aligned box
    as a procedural `Shape3D` implementor. Opt-in fast path when per-face
    materials aren't needed. Coexists with `box3d()` (6-`PlanarPatch` path).
  - **§3.9 added:** `Tessellatable` trait — universal trait over `Region2D` with
    mandatory `max_error` parameter. Every `Region2D` impl gets `tessellate()`;
    straight-edged regions do it exactly (zero error), curved regions do it
    approximately at the requested tolerance. Single coherent `From` impl,
    no negative trait bounds, no nightly features.
  - **OQ6 added:** `Primitive`/`GeoPrimitive` fork concern — `Shape3D` objects
    and `PlanarPatch` objects don't share a trait below `Intersectable`, so a
    closed `Primitive` enum (from `renderer_arch.md`) structurally can't hold
    both without either a second arm or a unificiation retrofit. Noted as the
    forcing function for any future `Shape3D` retrofit on `PlanarPatch`.
  - **§8 updated:** Cross-ref to `renderer_arch.md` Primitive enum, flagged as
    long-term structural tension.
  - Corrected `FunctionRegion`/height-field conflation (§2.4). Corrected
    `From<PlanarPatch<R>>` blanket scope (§3.9).
- **v3 (2026-07-10)** — Resolved Open Questions.
  - OQ1 resolved: merge `src/bvh.rs` + `src/flat_bvh.rs` into `src/bvh/` module.
    FlatBvhNode shared. MeshBvh lives in `src/bvh/mesh.rs`.
  - OQ4 updated: watertight intersection = permute+shear ray transform (primary)
    + double-precision fallback at edges (secondary). Not deferred.
  - OQ5 resolved: `MeshShape` exposes `triangles()` iterator for rasterizer.
  - §3.7 updated: `TransformObject` uses `Arc<MeshShape>` for instancing.
  - Added glam migration note (§2.2).
- **v2 (2026-06-29)** — Bi-directional audit against 4 existing design docs
  (denoiser.md, adaptive-sampling.md, renderer_arch.md, samplestream-refactor.md).
  Fixed scene integration (Arc-based dual reg), added cross-doc refs,
  documented rasterizer tension as deferred integration point.
- **v1 (2026-06-29)** — Initial design after Shape3D/ShapeObject refactor.
  `Sampleable` is non-generic (raw `(u, v, time)` params), decoupled from `Intersectable`.
  No mesh infrastructure exists. Design targets single-material meshes
  (most common case) with extension path for per-face materials.

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
  │   area(&self) -> f32
  │   sample(&self, u, v, time) -> (Point3, Direction3)
  │   sample_direction(&self, origin, u, v, time) -> Direction3  [default]
  │   pdf_direction(&self, origin, direction, time) -> f32       [default]
  │
  └── SphereShape, PlanarShape<R>, BoxShape — existing Shape3D impls
  └── MeshShape [future]

ShapeObject<Sh: Shape3D, M: Borrow<Material>>
  │   Wraps Sh + M, derives Intersectable + Bounded + Sampleable
  │   Intersectable::intersect → shape.intersect_shape() + borrow material
  │   Sampleable::random_direction → shape.sample_direction()
  │   Sampleable::pdf_value → shape.pdf_direction()
  │
  └── Sphere<M>   = ShapeObject<SphereShape, M>
  └── Quad<M>     = ShapeObject<PlanarShape<QuadRegion>, M>
  └── Mesh<M>     = ShapeObject<MeshShape, M>  [future]

Bvh<W> (src/bvh/mod.rs)
  │   Parametric wide BVH: W=2 (binary), W=4, W=8
  │   BvhNode<W>: SoA AABB layout, SIMD hit_mask traversal
  │   TreeBuilder (src/bvh/builder.rs): SAH-binned construction with rayon
  │   Stores Vec<Arc<dyn Intersectable>> at leaves
  │
  └── Scene BVH: Bvh<2> or Bvh<4> → Arc<dyn Intersectable> = Sphere | Quad | Box | Mesh | ...
```

### 2.2 Key Constraints

- **`Bvh<W>` leaf primitives are `Arc<dyn Intersectable>`**, returning
  `MaterialHit` (geometry + material reference). For mesh-internal BVH
  we need geometry-only intersection (`Hit`, no material). `Bvh<W>` stores
  `Vec<Arc<dyn Intersectable>>` and can't be directly reused — `MeshBvh`
  uses the same `BvhNode<W>` node format but with `Vec<u32>` tri_indices
  and its own construction pipeline (not `TreeBuilder`).

- **`Hit::geometric_normal` is `pub(crate)`** — must use `Hit::new()` or
  `set_geometric_normal()`.

- **`Shape3D::sample()` and friends take `time: f32`** — important for
  animated meshes later (TODO), but not required for static meshes.

- **`Sampleable` is decoupled from `Intersectable`** (now `Sampleable: Send + Sync`).
  Area lights register in both lists (same shape, two trait casts). Non-geometric
  lights (environment) only in `important_objects`.

- **Glam migration complete.** Custom `Vec3` replaced with `glam::Vec3` (0.33.2).
  `Point3`, `Direction3`, `Color3` are newtype wrappers in `src/vec3.rs`.
  `MeshData.positions` stores `Point3` (wraps `glam::Vec3`).

### 2.3 Non-Goal: Individual Mesh Triangles as Shape3Ds

We will NOT create a pbrt-style `MeshTriangle` index-pair that implements
`Shape3D` and gets individually wrapped in `ShapeObject`. The reasons:

1. **`Arc<dyn Intersectable>` overhead.** Each triangle would need a heap
   allocation for the `ShapeObject` + `Arc`. At 100K triangles this is
   prohibitive — memory pressure, cache misses, BVH construction time.

2. **No per-triangle material use case yet.** The codebase always assigns
   one material per shape. Per-face materials can be added later via a
   material-index buffer + `Sampleable` override.

3. **`Bvh<W>` stores `Vec<Arc<dyn Intersectable>>`.** The parametric wide BVH
   is designed for scene-level objects (hundreds), not mesh-level triangles
   (hundreds of thousands). Each triangle needs geometry-only intersection
   without vtable overhead. A mesh needs its own internal accelerator using
   the same `BvhNode<W>` format but with triangle indices.

______________________________________________________________________

### 2.4 PlanarPatch / Region2D → PlanarShape Unification

**(This fork has been resolved.)** The renderer originally had two structurally
separate object models that converged at the scene level but shared zero
implementation beneath it. The `Region2D` + `PlanarPatch<R, M>` system handled
planar primitives; `Shape3D` + `ShapeObject<Sh, M>` handled analytic
(SphereShape) and future mesh shapes. The `PlanarShape<R>` type was created
to bridge them: it takes the geometry of `PlanarPatch` (corner, side_a, side_b,
region) and implements `Shape3D`, so `ShapeObject<PlanarShape<R>, M>` replaces
`PlanarPatch<R, M>` entirely.

`PlanarPatch` has been **deleted** (no deprecation period). `Region2D` and all
region implementations (`QuadRegion`, `TriRegion`, etc.) now live in
`src/shape/regions/` alongside `PlanarShape` in `src/shape/planar.rs`.

The unification diagram was:

```
Scene BVH: Arc<dyn Intersectable> / Arc<dyn Sampleable>
                    │
            ┌───────┴───────┐
            │               │
       Shape3D trait    Region2D trait (implementation detail of PlanarShape)
            │
    ShapeObject<Sh, M>
            │
       ┌────┴────┐
       │         │
   SphereShape  PlanarShape<R>
                (all Region2D types)
```

Both ends up as `Arc<dyn Intersectable>` in the scene BVH via `ShapeObject`,
so at the *scene* level they remain interchangeable.

The trait-level relationship after unification:

| Layer | Shape3D + ShapeObject | Region2D (inside PlanarShape) |
|---|---|---|
| **Intersection strategy** | Analytic (sphere/box) or BVH + Möller–Trumbore (mesh, future) | Plane intersection + `R::contains(a,b)` containment test |
| **Shape definition** | Parametric surface or vertex/index buffer | Parametric `Region2D` trait with `(a,b) → bool` |
| **UV mapping** | Analytic (sphere/box) or barycentric (mesh) | `R::uv(a,b)` from plane basis vectors |
| **UV gradients** | Analytic (constant for planar/box) or barycentric (mesh) | Analytic from `side_a`/`side_b` (exact, smooth, free) |
| **Normals** | Per-vertex (mesh) or analytic (sphere/box) | Plane normal (constant — flat shading) |
| **Memory** | Per-shape allocation | Per-patch allocation, no sharing |
| **Scale** | 1 shape (potentially 1M+ triangles) | 1 shape (exactly 1 primitive) |

#### When to Use Which

| Shape Category | Example | Route | Reason |
|---|---|---|---|
| **Large flat surface** | Wall, floor, ceiling | `PlanarShape<QuadRegion>` → `ShapeObject` | Single plane intersection vs 2 mesh triangles + BVH. Faster, no approximation. |
| **Curved planar boundary** | Ellipse, annulus, rounded rect, superellipse | `PlanarShape<R>` → `ShapeObject` | Exact mathematical containment. Mesh requires tessellation → approximation error. |
| **Arbitrary boolean region** | Logo silhouette, fractal, function-defined mask | `PlanarShape<FunctionRegion>` → `ShapeObject` | Zero discretization error — evaluates closure at intersection time. |
| **Single procedural triangle** | Custom triangle in code | `PlanarShape<TriRegion>` → `ShapeObject` | One triangle, no index buffer, no BVH overhead. |
| **Complex concave polygon** | Cookie-cutter mask, architectural detail | `PlanarShape<PolygonRegion>` → `ShapeObject` | Exact ear-clipping intersection, no tessellation needed. |
| **Per-face materials** | Box with different materials per face | 6× `quad()` calls (via `box3d()`) | Face-level granularity. Mesh has one material per triangle group. |
| **File-loaded geometry** | glTF/OBJ scene, 100K triangles | `MeshShape` | Cannot construct by hand. Shared vertex buffers, two-level BVH. |
| **Complex surfaces** | Furniture, characters, organic | `MeshShape` | Smooth normals via per-vertex interpolation. Barycentric UVs. |
| **Instancing (many copies)** | Forest, cityscape | `Arc<MeshShape>` + `TransformObject` | One BVH shared by all instances via `Arc`. Planar shapes duplicate all data. |
| **Large procedural batches** | Voxel structure, grid of identical elements | `MeshShape` (with procedural `MeshData::*` constructor) | Single scene-BVH entry per batch. Internal BVH at mesh granularity. |

#### The `FunctionRegion` Clarification

`FunctionRegion` is a **boolean containment predicate** — `(a, b) → bool` —
not a **height field** — `z = f(x, y)`. These are fundamentally different
mathematical objects:

| | FunctionRegion | Height Field |
|---|---|---|
| **Signature** | `(a, b) → bool` | `z = f(x, y)` |
| **Defines** | Which (a,b) points are "in" on a flat plane | A displaced surface via z-offset per (x,y) |
| **Mesh representation** | Requires marching-squares boundary approximation (lossy) | Well-represented by grid tessellation (exact at sample points) |
| **Best representation** | Keep as `PlanarPatch<FunctionRegion>` (exact) | Tessellated mesh (standard approach, used by pbrt Heightfield → triangle mesh conversion) |

The conclusion (keep `FunctionRegion` as `PlanarPatch`) is correct; the
height-field case cuts the other way (a height field is well-represented as
a mesh). The `FunctionRegion` case is actually the *strongest* argument for
keeping the planar system — an arbitrary boolean boundary (possibly fractal,
discontinuous, non-analytic) gets exact containment evaluation at
intersection time, which no finite triangle tessellation can match.

#### Summary

`PlanarPatch<R>` stays as the procedural/analytical/primitive system.
`MeshShape` is the file-loaded/complex/instanced system. They overlap only
at the scene traits and should not share code below that — forcing them
to unify would be a net loss of expressiveness or performance for both.

## 3. Design: Mesh as Shape3D with Internal BVH

### 3.1 Data Structures

```rust
// ─── src/mesh/data.rs ───

/// Shared mesh vertex/index data. Cheaply cloneable via Arc.
pub struct MeshData {
    pub positions: Vec<Point3>,
    pub normals: Vec<Direction3>,  // per-vertex, may be empty
    pub uvs: Vec<(f32, f32)>,     // per-vertex UVs, may be empty
    pub indices: Vec<[u32; 3]>,  // triangle vertex indices
}
```

`MeshData` is the `TriangleMesh` equivalent from pbrt. Stored in an `Arc`
and shared by all references to the mesh geometry. Immutable after
construction.

```rust
// ─── src/bvh/mesh.rs ───

/// Internal wide BVH over mesh triangles. Geometry-only.
///
/// Uses the same `BvhNode<W>` node format as the scene-level `Bvh<W>`,
/// but stores triangle indices instead of `Arc<dyn Intersectable>`.
/// Has its own SAH construction (not `TreeBuilder`, which is typed
/// on `Arc<dyn Intersectable>`).
///
/// Differs from the scene-level `Bvh<W>`:
///   - Leaf primitives are triangle index ranges, not `Arc<dyn Intersectable>`
///   - Intersection returns `Option<MeshHit>` not `Option<MaterialHit>`
///   - Triangle data is fetched from MeshData at intersection time
///   - No material, no vtable dispatch
pub struct MeshBvh {
    nodes: Vec<BvhNode<2>>,      // same BvhNode<W> format as scene BVH
    tri_indices: Vec<u32>,       // triangle indices in traversal order
    mesh_data: Arc<MeshData>,    // source vertex data
}
```

The `BvhNode<W>` format is reused — for mesh BVH, `W=2` (binary) gives
64-byte cache-aligned nodes. Leaf nodes store `(tri_offset, tri_count)` via
`child_offset[0]` and `leaf_info[0]`. At leaf traversal, each triangle index
is fetched, its three vertex positions are extracted from `mesh_data`, and
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
    area: f32,            // precomputed sum of triangle areas
}
```

### 3.2 Shape3D Implementation

```rust
impl Shape3D for MeshShape {
    fn intersect_shape(&self, ray: &Ray, ray_t: Interval) -> Option<Hit> {
        // Traverse MeshBvh:
        //   1. Iterative stack traversal over BvhNode<W> array
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

    fn area(&self) -> f32 {
        self.area
    }

    fn sample(&self, u: f32, v: f32, time: f32) -> (Point3, Direction3) {
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
    t: f32,
    point: Point3,
    normal: Direction3,  // geometric normal (unit length)
    uv: (f32, f32),      // barycentric → texture UV
    tri_index: u32,
}
```

### 3.4 Möller–Trumbore Integration

The triangle intersection routine needs to handle:

- **Watertight intersection** via pbrt-v4's two-layer approach:
  1. **Primary: ray-space permute + shear transform.** Find the ray's dominant
     axis (`kz`), permute vertices so `kz` is the shear axis, then shear the
     transformed 2D coordinates so the ray is axis-aligned. This uses
     `DifferenceOfProducts` (a fused multiply-subtract that reduces catastrophic
     cancellation) for edge function evaluation.
  2. **Secondary: double-precision fallback.** When edge coefficients `e0`, `e1`,
     or `e2` are exactly zero in single precision, re-evaluate the problematic
     terms using double-precision arithmetic.
  
  The key insight: the permute+shear transform itself makes intersection robust
  by simplifying the math. The double-precision fallback is only triggered at
  edges where single precision produces exact zero. This is required for
  production quality — **do not defer** to Phase 2.
- **Back-face culling** is NOT performed — the `set_face_normal` logic
  in `SurfaceInteraction` handles front-face determination.
- **UV interpolation** when `mesh.normals` / `mesh.uvs` are present.
- **Shading normal interpolation** from per-vertex normals when available.

The intersection function is:

```rust
fn intersect_triangle(
    ray: &Ray, t_max: f32,
    p0: Point3, p1: Point3, p2: Point3,
) -> Option<(f32, f32, f32, f32)>  // (t, b0, b1, b2) barycentric coords
```

### 3.5 BVH Construction

The mesh BVH construction uses SAH binning (same algorithm as `TreeBuilder`) but:

- Primitives are triangle index ranges, not `Arc<dyn Intersectable>`
- AABB computation fetches vertex positions from `MeshData`
- Centroid computation uses the triangle's centroid (average of 3 vertices)
- Uses `rayon::join` for parallel construction (same as scene BVH). A separate `MeshTreeBuilder` or equivalent handles triangle AABBs without `Arc<dyn Intersectable>` overhead.

The output is a `MeshBvh` (flat BVH node array + triangle index array).

### 3.6 Scene Integration

```rust
// In src/shape/mod.rs — blanket impl so Arc<MeshShape> can be used in ShapeObject

impl Shape3D for Arc<MeshShape> {
    fn intersect_shape(&self, ray: &Ray, ray_t: Interval) -> Option<Hit> {
        self.as_ref().intersect_shape(ray, ray_t)
    }
    fn bounding_box(&self) -> Aabb { self.as_ref().bounding_box() }
    fn area(&self) -> f32 { self.as_ref().area() }
    fn sample(&self, u: f32, v: f32, time: f32) -> (Point3, Direction3) {
        self.as_ref().sample(u, v, time)
    }
    fn sample_direction(&self, origin: Point3, u: f32, v: f32, time: f32) -> Direction3 {
        self.as_ref().sample_direction(origin, u, v, time)
    }
    fn pdf_direction(&self, origin: Point3, direction: Direction3, time: f32) -> f32 {
        self.as_ref().pdf_direction(origin, direction, time)
    }
}
```

This enables dual registration without deep-copying the mesh BVH:

```rust
// In src/scene.rs — following add_sphere pattern, shares BVH via Arc

impl Scene {
    pub fn add_mesh(&mut self, data: Arc<MeshData>, material: impl Into<Material>) {
        let material = Arc::new(material.into());
        let mesh_shape = Arc::new(MeshShape::from_data(data));
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

### 3.7 Transform Sharing via Arc\<MeshShape\>

Multiple transforms can share the same mesh geometry via `Arc<MeshShape>`:

```rust
// Build once
let mesh_shape = Arc::new(MeshShape::from_data(data));

// Use at multiple transforms — cheap Arc::clone(), no BVH duplication
TransformObject<Translate, ShapeObject<Arc<MeshShape>, Arc<Material>>>
TransformObject<RotateY,   ShapeObject<Arc<MeshShape>, Arc<Material>>>
```

This enables forest scenes, cityscapes, etc. where one mesh appears many times.

`ShapeObject<Arc<MeshShape>, Arc<Material>>` works because:
- `Arc<MeshShape>: Shape3D` via the delegation impl (§3.6).
- `Arc<Material>: Borrow<Material>` via standard library impl.
- `ShapeObject` types with `Arc<MeshShape>` implement `Intersectable`,
  so `TransformObject<...>` accepts them.

`TransformObject` implements `Intersectable` by:

1. Transform ray → object space via `world_to_object_point`
2. Intersect mesh in object space
3. Transform hit back via `Transform::hit()`

______________________________________________________________________

### 3.8 BoxShape — Procedural Shape3D for Boxes

`BoxShape` provides a single-entry uniform-material axis-aligned box as a
`Shape3D` implementor. It currently coexists with the `box3d()` helper
(which returns 6 independent `Arc<dyn Intersectable>` for per-face materials).

#### BoxShape Design

```rust
// ─── src/shape/box3d.rs ───

/// Axis-aligned box shape. Six faces, single material, analytic slab intersection.
/// Precomputes face areas for area-weighted sampling.
pub struct BoxShape {
    min: Point3,
    max: Point3,
}

impl Shape3D for BoxShape {
    fn intersect_shape(&self, ray: &Ray, ray_t: Interval) -> Option<Hit> {
        // Standard AABB ray-slab intersection on all 3 axes.
        // t_enter = entry point (potential hit)
        // t_exit = exit point
        // The entry face is determined by which axis produced t_enter,
        // giving the correct outward normal.
        // UV coordinates computed per-face from the two tangential axes.
    }
    fn bounding_box(&self) -> Aabb { Aabb::new(self.min, self.max) }
    fn area(&self) -> f32 { /* 2(dx*dy + dy*dz + dx*dz) */ }
    fn sample(&self, u: f32, v: f32, _time: f32) -> (Point3, Direction3) {
        // Face-area-weighted selection, then uniform on the chosen face.
    }
}
```

`BoxShape` uses a single type (no generic scalar parameter — always `f32`).
The intersection uses the standard ray-slab method: compute `t_near` and `t_far`
for each axis, take the overlap interval, and the entry face determines the
normal and UV mapping. Face detection for UV gradients is done by checking
which coordinate of the hit point is at the min/max boundary.

#### Scene Integration

```rust
// In src/scene.rs

impl Scene {
    pub fn add_box(&mut self, a: Point3, b: Point3, material: impl Into<Material>) {
        let material = Arc::new(material.into());
        let box_3d: Box3D = shape_box3d(a, b, material.clone());
        if material.is_emissive() {
            self.add_intersectable(
                Arc::new(box_3d),
                Some(Arc::new(shape_box3d(a, b, material))),
            );
        } else {
            self.add_intersectable(Arc::new(box_3d), None);
        }
    }
}
```

#### Relationship to `box3d()`

Both coexist:

| | `box3d()` (6× quad calls) | `Scene::add_box` (BoxShape) |
|---|---|---|
| **Materials** | Per-face (six independent materials) | Single uniform material for whole box |
| **Scene BVH entries** | 6 per box | 1 per box |
| **Intersection cost** | 6 plane-intersection tests | 6 plane-intersection tests (identical) |
| **Per-face material flexibility** | ✅ Full | ❌ None |
| **Best for** | Multi-material boxes (decorative, mixed surfaces) | Uniform-material boxes (crates, walls, architectural volumes) |

Use `box3d()` when you need different materials on different faces. Use
`add_box()` / `BoxShape` for the common uniform-material case as an
optimization that also strengthens the `Shape3D` pattern.

______________________________________________________________________

### 3.9 Tessellatable Trait — Bridging Planar Shapes to Mesh Data

**(Not yet implemented — deferred.)** This section references the old `PlanarPatch<Self, ()>`
type which has been deleted. The design needs updating to use `PlanarShape<R>` / `PlanarShapeGeometry`
before implementation. The two object models (§2.4) can be
bridged for one specific operation: converting a `PlanarShape<R>` into mesh
triangles for use in a `MeshShape`.
This is useful for:

- Combining procedural and file-loaded geometry in the same mesh BVH
- Applying mesh-only operations (e.g., simplification, displacement) to
  procedural shapes
- Feeding procedural shapes into the rasterizer pipeline via
  `MeshShape::triangles()`

#### Design

Rather than a conditional `From` impl (which would require unstable negative
trait bounds or specialization), every `Region2D` impl gets a universal
`tessellate()` method. The distinction between exact and approximate
conversion is handled by a **mandatory `max_error` parameter** that is simply
ignored by regions that tessellate exactly:

```rust
// ─── src/planar/tessellate.rs (new) ───

/// Maximum sagitta (boundary deviation) for curved-region tessellation.
/// Ignored by regions that tessellate exactly.
///
/// Lower values produce more segments, increasing triangle count but
/// reducing approximation error. A reasonable default for production
/// is 0.01 × the patch's longest bounding dimension.
const DEFAULT_MAX_ERROR: f32 = 0.01;

/// A planar region that can be tessellated into mesh triangles.
pub trait Tessellatable: Region2D {
    /// Returns mesh triangles approximating (or exactly representing)
    /// this region, bounded by the given max_error.
    fn tessellate(
        &self,
        patch: &PlanarPatch<Self, ()>,
        max_error: f32,
    ) -> Vec<MeshTriangle>;
}
```

Every existing `Region2D` implementor implements `Tessellatable`:

```rust
impl Tessellatable for TriRegion {
    fn tessellate(&self, _patch: &PlanarPatch<Self, ()>, _max_error: f32) -> Vec<MeshTriangle> {
        // Exact: one triangle from the patch's corner/side_a/side_b.
        // Barycentric (a,b) = (0,0), (1,0), (0,1) map to the 3 vertices.
        // UVs are the analytic region-space coords.
    }
}

impl Tessellatable for QuadRegion {
    fn tessellate(&self, patch: &PlanarPatch<Self, ()>, _max_error: f32) -> Vec<MeshTriangle> {
        // Exact: two triangles from the 4 quad corners via side_a/side_b.
        // (0,0), (1,0), (0,1) → tri 1; (1,0), (1,1), (0,1) → tri 2.
    }
}

impl Tessellatable for PolygonRegion {
    fn tessellate(&self, patch: &PlanarPatch<Self, ()>, _max_error: f32) -> Vec<MeshTriangle> {
        // Exact: ear-clipping fan of the polygon's actual vertices.
    }
}

impl Tessellatable for EllipseRegion {
    fn tessellate(&self, patch: &PlanarPatch<Self, ()>, max_error: f32) -> Vec<MeshTriangle> {
        // Approximate: segment count derived from sagitta formula
        // given max_error and the ellipse's semi-axes.
    }
}

impl Tessellatable for AnnulusRegion {
    fn tessellate(&self, patch: &PlanarPatch<Self, ()>, max_error: f32) -> Vec<MeshTriangle> {
        // Approximate: concentric rings + radial segments, sagitta-bounded.
    }
}

impl Tessellatable for SuperellipseRegion {
    fn tessellate(&self, patch: &PlanarPatch<Self, ()>, max_error: f32) -> Vec<MeshTriangle> {
        // Approximate: marching-squares boundary sampling.
    }
}

impl Tessellatable for FunctionRegion {
    fn tessellate(&self, patch: &PlanarPatch<Self, ()>, max_error: f32) -> Vec<MeshTriangle> {
        // Approximate: marching-squares or Monte Carlo boundary sampling.
        // The boolean predicate is sampled on a grid of resolution derived
        // from max_error, then the isosurface is extracted.
    }
}

impl Tessellatable for RoundedRectRegion {
    fn tessellate(&self, patch: &PlanarPatch<Self, ()>, max_error: f32) -> Vec<MeshTriangle> {
        // Approximate: straight sides are exact (quads), corners are
        // circular arcs tessellated at sagitta-bounded segment count.
    }
}
```

#### From Impl with Mandatory Error Parameter

The `From` conversion takes `max_error` as an explicit parameter to force
the caller to consciously acknowledge the potential approximation:

```rust
impl<R: Tessellatable, M: Borrow<Material> + Clone> PlanarPatch<R, M> {
    /// Convert this patch into mesh data, tessellating the region
    /// with the given max_error bound.
    pub fn to_mesh_data(&self, max_error: f32) -> MeshData {
        let triangles = self.region().tessellate(&self.without_material(), max_error);
        let positions: Vec<Point3> = triangles.iter().flat_map(|t| t.positions()).collect();
        let normals: Vec<Direction3> = triangles.iter().flat_map(|t| t.normals()).collect();
        let uvs: Vec<(f32, f32)> = triangles.iter().flat_map(|t| t.uvs()).collect();
        let indices: Vec<[u32; 3]> = (0..triangles.len())
            .map(|i| [i as u32 * 3, i as u32 * 3 + 1, i as u32 * 3 + 2])
            .collect();
        MeshData { positions, normals, uvs, indices }
    }
}
```

There is no blanket `From<PlanarPatch<R, M>> for MeshData` — the conversion
always requires an explicit `max_error` parameter, making approximation
visible at the call site. This avoids the `!Tessellatable` coherence problem
entirely (no negative trait bounds needed) and ensures the caller always
acknowledges the approximation risk, even for regions that turn out to be
exact (where the parameter is silently ignored).

If compile-time prevention of approximate conversions is needed later, the
idiomatic pattern is an `Exact<R>` / `Approximate<R>` newtype split:

```rust
pub struct Exact<R>(R);           // only implemented for Tri/Quad/Polygon
pub struct Approximate<R>(R);     // implemented for all regions

impl<R: Tessellatable> From<PlanarPatch<Exact<R>, M>> for MeshData { /* infallible */ }
impl<R: Tessellatable> From<PlanarPatch<Approximate<R>, M>> for MeshData { /* with max_error */ }
```

This compiles on stable Rust — no coherence conflict, since `Exact<R>` and
`Approximate<R>` are genuinely different concrete types. Only worth reaching
for if the mandatory `max_error` parameter is found to be too easy to ignore
in practice.

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
   compactly, and the BvhNode<W> layout is cache-line-aligned (64B for W=2).

3. **Fits the Shape3D abstraction** — `MeshShape` is just another shape
   to `ShapeObject`. No new trait, no new wrapper, no architecture break.

4. **Two-level BVH is standard practice** — Embree, OptiX, and most
   production renderers use two-level acceleration (scene BVH over
   instance BVHs per mesh).

______________________________________________________________________

## 5. Files to Create / Modify

### Deleted (PlanarPatch removal — Phase 0)

| File | Reason |
|---|---|
| `src/planar/mod.rs` | Entire planar module deleted. PlanarPatch, PlanarHit, type aliases removed. |
| `src/planar/box.rs` | `box3d()` moved to `src/shape/constructors.rs`. |

### BVH module restructure (COMPLETED)

The BVH module has already been restructured into `src/bvh/`:

| File | Contents |
|---|---|
| `src/bvh/mod.rs` | `Bvh<W>` (wide BVH), `BvhNode<W>` (SoA layout, SIMD hit_mask) |
| `src/bvh/builder.rs` | `TreeBuilder` (SAH-binned binary construction with rayon) |
| `src/bvh/aabb.rs` | `Aabb`, `AabbPacked<W>` (SIMD-ready SoA bounding boxes) |
| `src/bvh/tests.rs` | BVH tests |

`MeshBvh` will live in `src/bvh/mesh.rs` as a parallel structure using
`BvhNode<W>` nodes with triangle-index leaves.

### New mesh files

| File | Contents |
|---|---|
| `src/mesh/mod.rs` | Module root: re-exports from data, shape |
| `src/mesh/data.rs` | `MeshData` struct + OBJ/PLY parsing + `MeshShape::triangles()` |
| `src/mesh/shape.rs` | `MeshShape` struct + `Shape3D` impl + free `mesh()` constructor |

### New shape files

| File | Contents |
|---|---|
| `src/shape/box3d.rs` | `BoxShape` struct + `Shape3D` impl for uniform-material AABB. `shape_box3d()` constructor + `Box3D` type alias. |
| `src/shape/planar.rs` | `PlanarShape<R>` struct + `Shape3D` impl for planar primitives (replaces PlanarPatch's geometry role). |
| `src/shape/constructors.rs` | Free construction functions: `quad()`, `ellipse()`, `tri()`, etc. + `box3d()` per-face helper. |
| `src/shape/regions/mod.rs` | Module root: declares and re-exports all 8 region types from `src/shape/regions/`. |

### Modified files

| File | Change |
|---|---|
| `src/lib.rs` | Add `pub mod mesh;`. BVH module restructuring already done. |
| `src/shape/mod.rs` | Add `impl Shape3D for Arc<MeshShape>` (delegation blanket). Add `mod box3d;`, `mod constructors;`, mod re-exports. Region2D trait lives here. |
| `src/scene.rs` | Add `add_mesh()` with Arc-based BVH sharing. Add `add_box()` for uniform-material BoxShape. `add_quad()` + friends transitively use PlanarShape via constructors module. |
| All files importing `crate::bvh` | Update imports to `crate::bvh::*`. (Already done.) |
| All files importing `crate::planar::quad` | Update to `crate::shape::quad` (constructors module). |

### No changes needed

| File | Reason |
|---|---|
| `src/hittable.rs` | Traits: `Sampleable` decoupled from `Intersectable`. `Hit`, `MaterialHit` unchanged. |
| `src/vec3.rs` | Point3, Direction3, Color3 unchanged. |

______________________________________________________________________

## 6. Implementation Phases

### Phase 0 — PlanarShape Unification (completed)

0. **Prerequisite: Region2D migration.** Move `Region2D` trait and all 8 region
   types from `src/planar/` into `src/shape/regions/`. Update all imports.
1. **PlanarShape creation.** Extract PlanarPatch's geometry fields into
   `PlanarShape<R>` implementing `Shape3D`. `src/shape/planar.rs`.
2. **BoxShape implementation.** `BoxShape` as Shape3D for uniform-material AABB.
   `src/shape/box3d.rs`.
3. **Construction functions.** Move `quad()`, `ellipse()`, `tri()`, etc. and
   `box3d()` into `src/shape/constructors.rs`. All return
   `ShapeObject<PlanarShape<R>, Material>`.
4. **PlanarPatch deletion.** Remove `src/planar/` entirely. All callers updated
   to import from `crate::shape::*`.

### Phase 1 — Core Geometry (pending)

Phase 1 steps 1–6 are the main mesh feature and remain unchanged:

0. **Prerequisite: BVH module restructure.** ✅ COMPLETED. `src/bvh/` module
   with `BvhNode<W>`, `Bvh<W>`, `TreeBuilder`, `AabbPacked<W>`. MeshBvh will
   reuse `BvhNode<W>` node format in `src/bvh/mesh.rs`.
1. `MeshData` — positions, normals, uvs, indices. OBJ file format parser.
2. `MeshBvh` — SAH construction, flat-node iterative traversal. `src/bvh/mesh.rs`.
3. `MeshShape: Shape3D` — intersection via internal BVH + Möller–Trumbore
   with watertight permute+shear + double-precision fallback.
4. `impl Shape3D for Arc<MeshShape>` — delegation impl for dual registration
   and transform sharing. `src/shape/mod.rs`.
5. `mesh()` constructor + `add_mesh()` scene method.
6. Integration test: Cornell box mesh variant with a single quad/tri mesh.

Phase 0 step 7 (BoxShape) has been completed and moved up.

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

### Phase 5 — PlanarPatch Bridge (parallel, independent)

1. `Tessellatable` trait — `src/planar/tessellatable.rs`.
2. `impl Tessellatable for` every existing `Region2D` type.
3. `PlanarPatch::to_mesh_data(max_error)` conversion method.
4. Integration: procedural planar shape → MeshData → MeshShape conversion
   in scene construction.

______________________________________________________________________

## 7. Open Questions

1. ~~**BVH module restructure / BvhNode reuse.**~~ **RESOLVED.** BVH module
   restructured into `src/bvh/{mod.rs, builder.rs, aabb.rs, tests.rs}`.
   `FlatBvhNode` replaced by `BvhNode<W>` (parametric wide BVH with SoA
   layout and SIMD hit_mask). `MeshBvh` will reuse `BvhNode<W>` format in
   `src/bvh/mesh.rs` but with `Vec<u32>` tri_indices instead of
   `Vec<Arc<dyn Intersectable>>`. Own SAH construction, not `TreeBuilder`.

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

4. ~~**Watertight intersection.**~~ **UPDATED in v3.** Verified against pbrt-v4
   source. The technique has two layers:
   - **Primary: ray-space permute + shear transform.** Align the ray with its
     dominant axis (`kz`), permute vertices so `kz` maps to z, then shear the
     transformed 2D coordinates so the ray is axis-aligned. Edge functions use
     `DifferenceOfProducts` (fused multiply-subtract) to reduce cancellation.
   - **Secondary: double-precision fallback.** When `e0`, `e1`, or `e2` are
     exactly zero in single precision, re-evaluate with `double`.
   
   The import: the permute+shear transform is the primary robustness mechanism,
   not the double-precision fallback. The fallback only fires at exact-zero
   edges. **This is required for production quality — do not defer.** Implement
   in Phase 1 alongside Möller–Trumbore.

5. ~~**renderer_arch TriangleRasterizer ↔ mesh triangle access.**~~
   **RESOLVED in v3.** `MeshShape` exposes a `triangles()` accessor/iterator
   that yields triangle index + vertex positions. The rasterizer calls
   `mesh.triangles()` to iterate, transform vertices, and rasterize.
   This preserves `MeshShape` as a single `Shape3D` while enabling
   rasterization. Not blocking Phase 1 — implement when rasterizer is built.

6. **`Primitive` / `GeoPrimitive` enum fork (from `renderer_arch.md`).**
    **RESOLVED in v5 by PlanarShape unification.** The `PlanarPatch` ↔ `Shape3D`
    fork that made a closed `Primitive` enum impossible has been resolved by
    deleting `PlanarPatch` and routing all planar geometry through
    `PlanarShape<R>: Shape3D` → `ShapeObject<PlanarShape<R>, M>`. Now every
    scene object (spheres, boxes, planars) goes through `Shape3D` + `ShapeObject`,
    so `Primitive` only needs variants per shape *category* (`Sphere`, `Box`,
    `Planar`, `Mesh`) rather than per region *type*. The `Custom(Arc<dyn Intersectable>)`
    escape hatch remains available as a safety net but is no longer needed for
    the planar subsystem.

______________________________________________________________________

## 8. Cross-Document References

### Existing design docs (bi-directional audit v2)

| Doc | Relationship to Mesh | Status |
|---|---|---|
| `renderer_arch.md` §2, §9 | `LightPrimitive` needs `Mesh` variant (additive). `TriangleRasterizer` uses `MeshShape::triangles()` (§7.5 — resolved in v3). Primitive registration pattern matches. | ✅ Compatible |
| | **⚠ OQ6 resolved in v5.** PlanarShape unification means all scene objects route through Shape3D → ShapeObject, so Primitive only needs 4 variants (Sphere, Box, Planar, Mesh) instead of one per region type. | ✅ Resolved |
| `denoiser.md` | Denoiser post-processes film output. Orthogonal to geometry. No shared interfaces. | ✅ No conflict |
| `adaptive-sampling.md` | Variance estimation + convergence criteria. Orthogonal to geometry types. | ✅ No conflict |
| `samplestream-refactor.md` | `SampleStreamEnum` replaces `DimCursor` in integrator signatures. Mesh uses `Sampleable` (non-generic, raw params). | ✅ No conflict |
| `CORE_THESIS.md` §4 | SpatialDomain pattern, leaf sovereignty. (External reference, not in docs/). | ✅ Compatible |

### Codebase references

- `src/shape/mod.rs` — Shape3D trait, ShapeObject wrapper, Region2D trait.
- `src/shape/regions/` — All 8 region type implementations (QuadRegion, TriRegion, ...).
- `src/shape/planar.rs` — PlanarShape\<R: Region2D\> implementing Shape3D (replaces PlanarPatch's geometry role).
- `src/shape/constructors.rs` — Free construction functions (quad(), ellipse(), tri(), ..., box3d()).
- `src/shape/box3d.rs` — BoxShape (AABB via slab intersection, uniform material).
- `src/bvh/` — BVH module: `mod.rs` (`Bvh<W>`, `BvhNode<W>` with SoA layout), `builder.rs` (SAH `TreeBuilder`), `aabb.rs` (`Aabb`, `AabbPacked<W>`), `tests.rs`.
