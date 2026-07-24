# Mesh Feature: MeshShape + Internal BVH

## Design v9 — Architecture Restored: MeshShape + Internal BVH with v8 Improvements

______________________________________________________________________

## Changelog

- **v9 (2026-07-24)** — MeshShape + internal BVH restored. v8's Triangle-as-Shape
  over-corrected: our `Bvh<W>` stores `Arc<dyn Intersectable>` at leaves (not inline
  8-byte index pairs like pbrt-v4), making per-triangle `Arc` allocation + vtable
  dispatch expensive. MeshShape wraps shared `MeshData` with an always-on
  `MeshBvh` (`BvhNode<2>` over triangle AABBs) for tight spatial clustering and
  one scene entry per mesh (not per triangle). v8 improvements retained:
  From impls (Tessellatable → MeshData), per-triangle materials buffer, clean
  `src/mesh/{mod,data,shape}.rs` structure, transform sharing via `Arc<MeshShape>`.

- **v8 (2026-07-24)** — Architecture pivot: triangles in scene BVH, no internal mesh BVH.
  - **Mesh is shared data, not a Shape.** Individual `TriangleShape` instances
    (8 bytes: `Arc<MeshData>` + tri_index) implement `Shape3D` and go directly
    into the scene BVH. No `MeshShape`, no `MeshBvh`.
  - **No separate mesh-internal BVH.** Triangles live in the scene BVH like
    any other shape. This matches pbrt-v4's `Triangle` + `TriangleMesh` pattern
    and eliminates the two-level BVH complexity.
  - **From impls for Tessellatable.** `PlanarShape<R>` collections and
    `Tessellatable` regions convert into `MeshData` via `From` impls, enabling
    procedural → mesh conversion for batching, instancing, or file export.
  - **Per-triangle materials.** `MeshData` gains an optional `materials: Vec<Arc<Material>>`
    buffer. When present, each triangle references its own material.
  - **All `PlanarPatch` references removed from body** (kept in changelog only).
  - **§2.3 flipped:** Individual triangles as Shapes is now the *goal*, not a non-goal.
  - **§3 rewritten:** `TriangleShape` replaces `MeshShape` + `MeshBvh`.
  - **§4 updated:** Comparison table now favors Triangle-as-Shape.
  - **§5/§6/§7/§8 updated** for new architecture.

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
(`PlanarShape<R>` for quads, triangles, ellipses, etc.). It has no way to
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

A single triangle on a plane is a `PlanarShape<TriRegion>` — that's a
self-contained shape with no shared data. A mesh is different: thousands of
triangles sharing vertex/index buffers, referenced by lightweight index pairs.

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

- **`Bvh<W>` leaf primitives are `Arc<dyn Intersectable>`** — mesh is
  a single `MeshShape` in the scene BVH, not per-triangle `Arc` entries.
  Triangles are an internal detail of the mesh's `MeshBvh`.

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

### 2.3 Non-Goal: Individual Triangles as Shape3Ds

Unlike pbrt-v4, individual mesh triangles are **not** registered as separate
`Shape3D` instances in the scene BVH. This is an architectural consequence of
our BVH storing `Arc<dyn Intersectable>` at leaves — pbrt-v4's `Triangle`
works as a separate `Shape` because it stores 8-byte index pairs inline in a
flat `GeometricPrimitive` array. Our `Bvh<W>` allocates one heap `Arc` per
primitive, making 100K triangles incur ~1.6 MB of `Arc` overhead plus vtable
dispatch per intersection. Not acceptable for the core mesh path.

Instead:

1. **Mesh is a single Shape3D (`MeshShape`).** Contains an always-on internal
   BVH (`MeshBvh`) over its triangle AABBs. Triangles are an implementation
   detail — not exposed as public `Shape3D` instances.

2. **One scene entry per mesh.** `ShapeObject<MeshShape, M>` registers once
   in the scene BVH, in both `Intersectable` and `Sampleable` lists if
   emissive. No per-triangle `Arc` overhead.

3. **Per-triangle materials via buffer, not individual shapes.** `MeshData`
   carries an optional `per_tri_material: Vec<Arc<Material>>` buffer. The
   material is selected at intersection time within `MeshShape`'s traversal —
   no `Sampleable` override needed.

4. **MeshBvh is always-on** — even for small meshes the overhead is
   negligible (a 2-triangle quad produces ~3 BVH nodes × 64 bytes ~192 bytes).
   The single `Bvh<2>` (binary BVH) provides tighter spatial clustering than
   distributing triangles across scene BVH leaves.

The `Tessellatable`/`PlanarShape` → `MeshData` conversion path (v8 addition)
is retained: procedural shapes can be converted into shared mesh data, and
the resulting `MeshShape` is registered as a single scene entry.

______________________________________________________________________

### 2.4 PlanarShape vs Mesh — When to Use Which

`PlanarShape<R>` and mesh triangles serve different use cases. They share the
`Shape3D` → `ShapeObject` pipeline and are interchangeable at the scene level.

The trait-level relationship:

| Layer | Shape3D + ShapeObject | Region2D (inside PlanarShape) |
|---|---|---|
| **Intersection strategy** | Analytic (sphere/box) or BVH + Möller–Trumbore (mesh) | Plane intersection + `R::contains(a,b)` containment test |
| **Shape definition** | Parametric surface or vertex/index buffer | Parametric `Region2D` trait with `(a,b) → bool` |
| **UV mapping** | Analytic (sphere/box) or barycentric (mesh) | `R::uv(a,b)` from plane basis vectors |
| **UV gradients** | Analytic (constant for planar/box) or barycentric (mesh) | Analytic from `side_a`/`side_b` (exact, smooth, free) |
| **Normals** | Per-vertex (mesh) or analytic (sphere/box) | Plane normal (constant — flat shading) |
| **Memory** | Per-shape allocation | Per-patch allocation, no sharing |
| **Scale** | 1 shape (potentially 1M+ triangles) | 1 shape (exactly 1 primitive) |

#### When to Use Which

| Shape Category | Example | Route | Reason |
|---|---|---|---|
| **Large flat surface** | Wall, floor, ceiling | `PlanarShape<QuadRegion>` → `ShapeObject` | Single plane intersection vs 2 mesh triangles. Faster, exact. |
| **Curved planar boundary** | Ellipse, annulus, rounded rect | `PlanarShape<R>` → `ShapeObject` | Exact mathematical containment. Mesh requires tessellation. |
| **Arbitrary boolean region** | Logo silhouette, fractal mask | `PlanarShape<FunctionRegion>` → `ShapeObject` | Zero discretization error — evaluates closure at intersection time. |
| **Single procedural triangle** | Custom triangle in code | `PlanarShape<TriRegion>` → `ShapeObject` | One triangle, no index buffer, no sharing overhead. |
| **Per-face materials** | Box with different materials per face | 6× `quad()` calls (via `box3d()`) | Face-level granularity. |
| **File-loaded geometry** | glTF/OBJ scene, 100K triangles | `MeshShape` via `add_mesh()` | Cannot construct by hand. Shared vertex buffers. |
| **Complex surfaces** | Furniture, characters, organic | `MeshShape` via `add_mesh()` | Smooth normals via per-vertex interpolation. Barycentric UVs. |
| **Instancing (many copies)** | Forest, cityscape | `Arc<MeshShape>` + `TransformObject` | One `MeshShape` shared by all instances. |
| **Large procedural batches** | Voxel structure, grid of elements | `MeshShape` via `add_mesh()` | Many triangles, single scene entry. |

#### The `FunctionRegion` Clarification

`FunctionRegion` is a **boolean containment predicate** — `(a, b) → bool` —
not a **height field** — `z = f(x, y)`. These are fundamentally different
mathematical objects:

| | FunctionRegion | Height Field |
|---|---|---|
| **Signature** | `(a, b) → bool` | `z = f(x, y)` |
| **Defines** | Which (a,b) points are "in" on a flat plane | A displaced surface via z-offset per (x,y) |
| **Mesh representation** | Requires marching-squares boundary approximation (lossy) | Well-represented by grid tessellation (exact at sample points) |
| **Best representation** | Keep as `PlanarShape<FunctionRegion>` (exact) | Tessellated mesh (standard approach, used by pbrt Heightfield → triangle mesh conversion) |

The conclusion (keep `FunctionRegion` as `PlanarShape`) is correct; the
height-field case cuts the other way (a height field is well-represented as
a mesh). The `FunctionRegion` case is actually the *strongest* argument for
keeping the planar system — an arbitrary boolean boundary (possibly fractal,
discontinuous, non-analytic) gets exact containment evaluation at
intersection time, which no finite triangle tessellation can match.

#### Summary

`PlanarShape<R>` stays as the procedural/analytical/primitive system.
`MeshShape` is the file-loaded/complex/instanced system. They overlap only
at the scene traits and should not share code below that — forcing them
to unify would be a net loss of expressiveness or performance for both.

## 3. Design: MeshShape with Internal BVH

### 3.1 Data Structures

```rust
// ─── src/mesh/data.rs ───

/// Shared mesh vertex/index data. Cheaply cloneable via Arc.
pub struct MeshData {
    pub positions: Vec<Point3>,
    pub normals: Vec<Direction3>,       // per-vertex, may be empty
    pub uvs: Vec<(f32, f32)>,          // per-vertex UVs, may be empty
    pub indices: Vec<[u32; 3]>,        // triangle vertex indices
    pub per_tri_material: Option<Vec<Arc<Material>>>,  // per-triangle materials
}
```

`MeshData` is the `TriangleMesh` equivalent from pbrt. Stored in an `Arc`
and shared by all triangles in the mesh. Immutable after construction.

```rust
// ─── src/bvh/bvh.rs ─── (or src/mesh/bvh.rs)

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
    leaf_counts: Vec<u32>,       // primitive count per leaf node
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
///
/// Or, for instancing, wrap Arc<MeshShape> directly:
///   let shape: Arc<dyn Intersectable> = Arc::new(MeshShape::from_data(data));
pub struct MeshShape {
    data: Arc<MeshData>,
    bvh: MeshBvh,
    bbox: Aabb,            // precomputed from all triangle AABBs
    area: f32,              // precomputed sum of triangle areas
}

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

    fn bounding_box(&self) -> Aabb { self.bbox }

    fn area(&self) -> f32 { self.area }

    fn sample(&self, u: f32, v: f32, time: f32) -> (Point3, Direction3) {
        // Uniform triangle sampling:
        //   1. Pick triangle i by area-weighted distribution
        //   2. Sample barycentric coords via sqrt(u), sqrt(v) method
        //   3. Interpolate vertex positions + normal
        //   4. Return (point, normal)
    }

    // Uses default area-to-solid-angle conversion — fine for meshes.
    // Shape-specific ONB-based sampling can override for lower variance
    // on emissive meshes, but the default is correct for all shapes.
}
```

`MeshShape` bundles all mesh data:
- **`Arc<MeshData>`** — shared vertex data, cloneable for instancing.
- **`MeshBvh`** — a `Bvh<2>` (binary BVH) over triangle AABBs, constructed
  once during `MeshShape::new()`. Always-on: even a 2-triangle quad needs
  only ~3 nodes × 64 bytes ≈ 192 bytes.
- **`bbox` / `area`** — precomputed at construction for O(1) access.

### 3.2 MeshBvh — Internal Acceleration

`MeshBvh` is a `Bvh<2>` built from triangle AABBs during `MeshShape` construction.
`TreeBuilder` (scene BVH) is not reusable — it stores `Arc<dyn Intersectable>` at
leaves. `MeshBvh` has its own SAH construction over triangle-index AABBs:

- **Primitive**: triangle index range, not `Arc<dyn Intersectable>`.
- **AABB computation**: fetches vertex positions from `MeshData`.
- **Centroid**: triangle centroid (average of 3 vertices).
- **Parallel**: `rayon::join` for recursive splitting (same pattern as scene BVH).
- **Output**: flat `BvhNode<2>` array + triangle index array.

Internal intersection returns an intermediate result before material resolution:

```rust
/// Intermediate hit from MeshBvh traversal. Converted to Hit by MeshShape.
struct MeshHit {
    t: f32,
    point: Point3,
    normal: Direction3,  // geometric normal (unit length)
    uv: (f32, f32),      // barycentric → texture UV
    tri_index: u32,      // for per-triangle material lookup
}
```

```rust
// ─── src/mesh/bvh.rs ───

/// Geometry-only BVH traversal — returns (t, tri_index, barycentric).
/// Struct defined in §3.1. See that section for fields.
impl MeshBvh {
    fn intersect(&self, ray: &Ray, ray_t: Interval) -> Option<MeshHit> { ... }
}
```

Traversal:
```
1. Node stack traversal (max depth ≈ 2·log₂(N))
2. Interior node: test ray against AABB of each child (SIMD via AabbPacked<2>)
3. Push hit children onto stack, closest-first
4. Leaf node: iterate tri_indices[offset..offset + count]
5. For each triangle: fetch vertices from MeshData, run Moller–Trumbore
6. Track closest t, interpolate normal/UV
```

Key choices:
- **Binary (`W=2`)** — mesh triangles are cheap to intersect; wide BVH SIMD
  benefits are outweighed by leaf decompression overhead.
- **Geometry-only** — returns `(t, tri_index, barycentric)`, not `MaterialHit`.
  Material resolved after traversal from `ShapeObject`'s material or
  `MeshData.per_tri_material`.
- **Always-on** — ~3 nodes for a 2-triangle quad; ~200K nodes for 100K
  triangles (~12.8 MB). No threshold needed.

### 3.3 Moller–Trumbore Integration

Each leaf triangle uses pbrt-v4 watertight intersection:

```rust
fn intersect_triangle(
    ray: &Ray, t_max: f32,
    p0: Point3, p1: Point3, p2: Point3,
) -> Option<(f32, f32, f32, f32)>  // (t, b0, b1, b2) barycentric coords
```

Two-layer watertight approach:

1. **Primary: ray-space permute + shear.** Find the ray's dominant axis (`kz`),
   permute vertices so `kz` is the shear axis, then shear the transformed 2D
   coordinates so the ray is axis-aligned. Edge function evaluation uses
   `DifferenceOfProducts` (a fused multiply-subtract that reduces catastrophic
   cancellation at shared edges).

2. **Secondary: double-precision fallback** — when edge coefficients `e0`, `e1`,
   or `e2` are exactly zero in single precision, re-evaluate the problematic
   terms using double precision.

The key insight: the permute+shear transform itself makes intersection robust
by simplifying the math. The double-precision fallback only fires when single
precision produces exact zero at edges — it is NOT the primary technique.
Production renderers require both layers for watertight results — **do not defer** to Phase 2.

No back-face culling — `set_face_normal` handles orientation. UV interpolation
from `mesh.uvs` and shading normal from `mesh.normals` when present.

### 3.4 Scene Integration

`add_mesh()` creates one `ShapeObject<MeshShape>` per mesh. For single
meshes (no instancing), the `MeshShape` is owned directly by `ShapeObject`:

```rust
impl Scene {
    pub fn add_mesh(
        &mut self,
        data: Arc<MeshData>,
        material: impl Into<Material>,
    ) {
        let material = Arc::new(material.into());
        let mesh = MeshShape::new(data);
        let so: Arc<dyn Intersectable> =
            Arc::new(ShapeObject::new(mesh, material));
        self.add_intersectable(so, None);
    }
}
```

One `Arc<dyn Intersectable>` per mesh. The material goes into `ShapeObject`,
not `MeshShape`. `MeshShape` stays pure geometry.

If emissive, the same `Arc<dyn Intersectable>` serves both lists (box pattern):

```rust
let so = Arc::new(ShapeObject::new(MeshShape::new(data), material));
self.add_intersectable(so.clone(), Some(so));
```

For instancing (multiple meshes sharing one `MeshData`), see §3.5.

### 3.5 Transform Sharing via Arc\<MeshShape\>

Multiple transforms share the same mesh geometry via `Arc<MeshShape>`:

```rust
let mesh_data = Arc::new(MeshData::from_obj(path));
let mesh_shape = Arc::new(MeshShape::new(mesh_data));

// Each TransformObject gets its own ShapeObject, but both share
// the same Arc<MeshShape> — no deep-copy of the MeshBvh:
let so1 = ShapeObject::new(mesh_shape.clone(), material.clone());
let so2 = ShapeObject::new(mesh_shape, material);

let tree1 = TransformObject::new(
    Translate(glam::vec3(0.0, 0.0, 5.0)), so1,
);
let tree2 = TransformObject::new(
    Translate(glam::vec3(2.0, 0.0, 5.0)), so2,
);
```

The `Arc<MeshShape>` is shared. Each `TransformObject` is a single scene
BVH entry — no per-triangle overhead for instancing.

`TransformObject` delegates through the mesh generically:
1. Transform ray → object space via `Transform::ray()` / `world_to_object_point`.
2. Intersect the child in object space (ShapeObject → MeshShape → MeshBvh).
3. Transform hit back to world space via `Transform::hit()`.

This works for any `Transform` implementation (`Translate`, `RotateY`,
future `Affine3A`) — the mesh shape is agnostic to the transform.

`MeshShape` is pure geometry — material always comes from `ShapeObject`,
whether single (`ShapeObject<MeshShape, M>`) or instanced
(`ShapeObject<Arc<MeshShape>, M>`).

### 3.6 From Impls — Tessellatable → MeshData

`Tessellatable` converts a planar region into triangles. The conversion
chain is: region → `MeshData` → `MeshShape` → scene BVH.

```rust
/// A geometry-only triangle from tessellation (no MeshData reference yet).
pub struct TriangleGeometry {
    pub positions: [Point3; 3],
    pub normals: [Direction3; 3],
    pub uvs: [(f32, f32); 3],
}
```

`Tessellatable::tessellate()` returns `Vec<TriangleGeometry>`:

```rust
pub trait Tessellatable: Region2D {
    fn tessellate(&self, geometry: &PlanarShapeGeometry, max_error: f32)
        -> Vec<TriangleGeometry>;
}
```

From impls collect tessellated triangles into `MeshData`:

```rust
impl From<Vec<TriangleGeometry>> for MeshData {
    fn from(tris: Vec<TriangleGeometry>) -> Self {
        // Collect unique vertices via hash dedup
        // Build positions, normals, uvs, indices
    }
}

impl<R: Tessellatable> PlanarShape<R> {
    pub fn to_mesh_data(&self, max_error: f32) -> MeshData {
        let tris = self.region.tessellate(&self.geometry(), max_error);
        MeshData::from(tris)
    }
}
```

### 3.7 Per-Triangle Materials

The `MeshData` struct (§3.1) carries an optional per-triangle material buffer.
When `per_tri_material` is `Some`, `MeshShape::intersect_shape` selects the material from the
buffer at intersection time (after MeshBvh returns `tri_index`). When
`None`, `ShapeObject`'s single `M: Borrow<Material>` is used — no
per-triangle overhead for the common case.

For emissive per-triangle materials, emission is resolved through the
selected material's `emitted()` at hit time — no `Sampleable` override
needed.

### 3.8 BoxShape — Procedural Shape3D for Boxes

`BoxShape` provides a single-entry uniform-material axis-aligned box as a
`Shape3D` implementor. Coexists with `box3d()` (6 `Arc<dyn Intersectable>`
via 6× `quad()` calls for per-face materials).

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
        // t_enter = entry point on box (potential hit)
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

| | `box3d()` (6× quad) | `Scene::add_box` (BoxShape) |
|---|---|---|
| **Materials** | Per-face (six independent materials) | Single uniform material |
| **Scene BVH entries** | 6 per box | 1 per box |
| **Intersection cost** | 6 plane-intersection tests | 6 plane-intersection tests (identical) |
| **Per-face material flexibility** | ✓ Full | ✗ None |
| **Best for** | Multi-material boxes | Uniform boxes (crates, walls) |

Use `box3d()` when you need different materials on different faces. Use
`BoxShape` / `Scene::add_box()` for the common uniform-material case as an
optimization that also strengthens the `Shape3D` pattern.

### 3.9 Tessellatable Trait — Bridging Planar Shapes to Mesh Data

**(Not yet implemented — deferred.)** Converts a `PlanarShape<R>` into
mesh triangles via the `Tessellatable` trait on `Region2D`. Output feeds
into `MeshData` → `MeshShape` → scene BVH.

Useful for combining procedural and file-loaded geometry in the same
acceleration structure; applying mesh-only operations (simplification,
displacement) to procedural shapes; and feeding the rasterizer pipeline.

Rather than a conditional `From` impl (which would need unstable negative
trait bounds or specialization), every `Region2D` impl gets a universal
`tessellate()` method. The distinction between exact and approximate
conversion uses a **mandatory `max_error` parameter** that is simply
ignored by regions that tessellate exactly:

```rust
// ─── src/shape/tessellate.rs (new) ───

/// Maximum sagitta (boundary deviation) for curved-region tessellation.
/// Ignored by exact regions. Reasonable default: 0.01 × patch's longest dimension.
const DEFAULT_MAX_ERROR: f32 = 0.01;

// TriangleGeometry struct defined in §3.6.

/// A planar region that can be tessellated into mesh triangles.
pub trait Tessellatable: Region2D {
    fn tessellate(
        &self,
        geometry: &PlanarShapeGeometry,
        max_error: f32,
    ) -> Vec<TriangleGeometry>;
}
```

Each `Region2D` implementor implements `Tessellatable` with its own strategy:

```rust
impl Tessellatable for TriRegion {
    fn tessellate(&self, geometry: &PlanarShapeGeometry, _max_error: f32) -> Vec<TriangleGeometry> {
        // Exact: one triangle from the patch's corner/side_a/side_b.
        // Barycentric (a,b) = (0,0), (1,0), (0,1) map to the 3 vertices.
    }
}

impl Tessellatable for QuadRegion {
    fn tessellate(&self, geometry: &PlanarShapeGeometry, _max_error: f32) -> Vec<TriangleGeometry> {
        // Exact: two triangles from the 4 quad corners via side_a/side_b.
    }
}

impl Tessellatable for PolygonRegion {
    fn tessellate(&self, geometry: &PlanarShapeGeometry, _max_error: f32) -> Vec<TriangleGeometry> {
        // Exact: ear-clipping fan of the polygon's actual vertices.
    }
}

impl Tessellatable for EllipseRegion {
    fn tessellate(&self, geometry: &PlanarShapeGeometry, max_error: f32) -> Vec<TriangleGeometry> {
        // Approximate: segment count from sagitta formula + semi-axes.
    }
}

impl Tessellatable for AnnulusRegion {
    fn tessellate(&self, geometry: &PlanarShapeGeometry, max_error: f32) -> Vec<TriangleGeometry> {
        // Approximate: concentric rings + radial segments, sagitta-bounded.
    }
}

impl Tessellatable for SuperellipseRegion {
    fn tessellate(&self, geometry: &PlanarShapeGeometry, max_error: f32) -> Vec<TriangleGeometry> {
        // Approximate: marching-squares boundary sampling.
    }
}

impl Tessellatable for FunctionRegion {
    fn tessellate(&self, geometry: &PlanarShapeGeometry, max_error: f32) -> Vec<TriangleGeometry> {
        // Approximate: marching-squares or Monte Carlo boundary sampling.
        // Grid resolution derived from max_error, then isosurface extraction.
    }
}

impl Tessellatable for RoundedRectRegion {
    fn tessellate(&self, geometry: &PlanarShapeGeometry, max_error: f32) -> Vec<TriangleGeometry> {
        // Approximate: straight sides exact (quads), corners are circular arcs
        // tessellated at sagitta-bounded segment count.
    }
}
```

The `PlanarShape<R>::to_mesh_data(max_error)` conversion is defined in §3.6.

The `max_error` parameter is mandatory — approximation is always explicit at
the call site. If compile-time prevention of approximate conversions is needed
later, an `Exact<R>` / `Approximate<R>` newtype split is the standard pattern:

```rust
pub struct Exact<R>(R);           // only for Tri/Quad/Polygon
pub struct Approximate<R>(R);     // for all regions

impl<R: Tessellatable> From<Exact<R>> for MeshData { /* infallible */ }
impl<R: Tessellatable> From<Approximate<R>> for MeshData { /* requires max_error */ }
```

This compiles on stable Rust — no coherence conflict, since `Exact<R>` and
`Approximate<R>` are genuinely different concrete types.

## 4. Option Comparison

| Criterion | MeshShape + Internal BVH (v9, chosen) | Triangle-as-Shape (v8, rejected) |
|---|---|---|
| **Per-triangle material** | ✅ Buffer per-triangle, resolves at intersect time | ✅ Natural — `MeshData.per_tri_material` |
| **Per-triangle memory** | 0 bytes/tri (data in shared MeshData + MeshBvh) | ~16B (TriangleShape: Arc + u32) + Arc for ShapeObject |
| **Vtable dispatch** | Per-mesh via Shape3D (single dispatch) | Per-triangle via Arc<dyn Intersectable> (100K dispatches) |
| **BVH structure** | Two-level: scene BVH over meshes, MeshBvh over tris | Single-level: scene BVH over all triangles (SAH over N) |
| **Cache behavior** | Mesh triangles in compact MeshBvh = better locality | Triangles distributed in scene BVH leaves |
| **Implementation cost** | Moderate: MeshShape + MeshBvh + SAH construction | Low: TriangleShape (100 LOC) + add_mesh() loop |
| **Scene building** | O(N log N) mesh BVH + O(M log M) scene BVH | O(N log N) scene BVH |
| **Memory efficiency** | ~4 MB for nodes + ~400 KB for tri_indices at 100K tri | ~1.6 MB Arc overhead per 100K triangles |

The **MeshShape + Internal BVH** approach is chosen because:

1. **No per-triangle `Arc` overhead.** Our `Bvh<W>` stores `Arc<dyn Intersectable>`
   at leaves. Each TriangleShape would need its own heap allocation + vtable
   dispatch — 100K extra allocations for a 100K triangle mesh. MeshShape has one.

2. **Tighter spatial clustering.** The MeshBvh is built from ONLY the mesh's
   triangles — its SAH produces tighter nodes than distributing triangles across
   the scene BVH (which must mix mesh triangles with spheres, quads, and boxes).

3. **Better cache behavior.** MeshBvh stores triangle indices contiguously in
   `Vec<u32>`, and `MeshData` vertices are compact. Traversing a small subset of
   triangles within one mesh stays hot in cache vs jumping across scene BVH leaves.

4. **Always-on internal BVH is cheap.** A 2-triangle quad produces ~3 BVH nodes
   (192 bytes). At 100K triangles, ~4 MB for nodes + ~400 KB for tri_indices. No
   threshold needed.

5. **Per-triangle materials via buffer, not individual shapes.** The
   `MeshData.per_tri_material` buffer works with either architecture. In v9,
   the material is selected within `MeshShape::intersect_shape` after the
   MeshBvh returns the tri_index — no extra allocations.

**When to reconsider:** If a future BVH refactor stores primitives inline
(like pbrt-v4's flat `GeometricPrimitive` array) rather than `Arc<dyn>`,
Triangle-as-Shape becomes viable. Until then, MeshShape + internal BVH is
the right architecture for our BVH design.

______________________________________________________________________

## 5. Files to Create / Modify

### New mesh files

| File | Contents |
|---|---|
| `src/mesh/mod.rs` | Module root: re-exports |
| `src/mesh/data.rs` | `MeshData` struct + `From<TriangleGeometry>` + OBJ/PLY parsing |
| `src/mesh/shape.rs` | `MeshShape` struct + `Shape3D` impl. Internal MeshBvh traversal. |
| `src/mesh/bvh.rs` | `MeshBvh` — `Bvh<2>` over triangle AABBs, SAH construction, traversal. |

### New shape files

| File | Contents |
|---|---|
| `src/shape/box3d.rs` | `BoxShape` struct + `Shape3D` impl for uniform-material AABB. `shape_box3d()` constructor + `Box3D` type alias. |
| `src/shape/planar.rs` | `PlanarShape<R>` struct + `Shape3D` impl for planar primitives (replaces PlanarPatch's geometry role). |
| `src/shape/constructors.rs` | Free construction functions: `quad()`, `tri()`, `annulus()`, `ellipse()`, `rounded_rect()`, `superellipse()`, `polygon()`, `function_patch()`, `shape_box3d()` |
| `src/shape/regions/mod.rs` | Module root: declares and re-exports all 8 region types from `src/shape/regions/`. |

### Modified files

| File | Change |
|---|---|
| `src/lib.rs` | Add `pub mod mesh;` |
| `src/scene.rs` | Add `add_mesh()` — creates one `ShapeObject<MeshShape>` per mesh. Also `add_box()` for uniform-material BoxShape. |
| `src/shape/mod.rs` | Shape3D trait, ShapeObject, Region2D already defined. No changes needed for mesh. `impl Shape3D for Arc<MeshShape>` lives in `src/mesh/shape.rs` alongside the struct. |
| All files importing `crate::planar::quad` | Update to `crate::shape::quad` (constructors module). (Already done.) |

### No changes needed

| File | Reason |
|---|---|
| `src/hittable.rs` | Traits unchanged. `Sampleable` already decoupled. |
| `src/vec3.rs` | Point3, Direction3, Color3 unchanged. |
| `src/shape/mod.rs` | No changes — MeshShape is in `src/mesh/`, not a new `Shape3D` impl in `src/shape/`. The `Shape3D` trait is used, not modified. |
| `src/bvh/` | `BvhNode<2>` and `AabbPacked<2>` reused via `use crate::bvh::{...}` in `src/mesh/bvh.rs`. No changes to the scene BVH module. |

______________________________________________________________________

## 6. Implementation Phases

### Phase 0 — PlanarShape Unification (completed)

Already done: PlanarShape, BoxShape, Region2D migration, constructors module.

### Phase 1 — Core Geometry

1. **`MeshData`** — positions, normals, uvs, indices. Freeze the struct layout.
   `From<Vec<TriangleGeometry>>` for dedup vertex construction.
2. **`MeshBvh`** — `Bvh<2>` over triangle AABBs, SAH construction, node traversal.
   `src/mesh/bvh.rs`.
3. **`MeshShape`** — wraps `MeshData` + `MeshBvh`, implements `Shape3D`.
   Watertight Möller–Trumbore per triangle within MeshBvh traversal.
   `src/mesh/shape.rs`.
4. **`add_mesh()`** — scene method that creates one `ShapeObject<MeshShape>`,
   registers in scene BVH. `src/scene.rs`.
5. **Integration test** — Cornell box mesh variant with a single quad/tri mesh
   loaded as a mesh (2 triangles, 4 vertices).

### Phase 2 — Sampling + Light Integration

1. `MeshShape::sample()` — face-area-weighted triangle selection, then uniform
   sampling on the chosen face.
2. `MeshShape::area()` — precomputed from triangle area sum.
3. Verify emissive mesh via `add_mesh()` with emissive material → single
   `ShapeObject<MeshShape>` registered in both `Intersectable` and `Sampleable` lists.

### Phase 3 — Normals + UV Interpolation

1. Per-vertex normal interpolation with barycentric coordinates.
2. Per-vertex UV interpolation.
3. Auto-generated smooth normals when mesh has no normals (angle-weighted
   face normal averaging).

### Phase 4 — File Format Support

1. OBJ parser — faces, vertices, normals, UVs, material groups → `MeshData`.
2. PLY parser (binary + ASCII).
3. Optional: per-face material groups from OBJ MTL → `per_tri_material` buffer.

### Phase 5 — Tessellatable + PlanarShape Bridge

1. `Tessellatable` trait — `src/shape/tessellate.rs`.
2. `impl Tessellatable for` every existing `Region2D` type.
3. `PlanarShape<R>::to_mesh_data(max_error)` — conversion via Tessellatable.
4. Integration: procedural planar shape → `MeshData` → `MeshShape` → scene BVH.

______________________________________________________________________

## 7. Open Questions

1. ~~**BVH module restructure.**~~ **RESOLVED in v7.** `src/bvh/` module
   with `Bvh<W>`, `BvhNode<W>`, `TreeBuilder`, `AabbPacked<W>`. MeshBvh
   reuses `BvhNode<2>` node format in `src/mesh/bvh.rs`.

2. ~~**Dual light registration.**~~ **RESOLVED in v9.** One `Arc<MeshShape>`
   shared between `Intersectable` and `Sampleable` lists (box pattern).
   `Arc::clone` for the other list — same allocation, two trait casts.

3. ~~**Two-level BVH vs single flat list.**~~ **RESOLVED in v9.** Two-level:
   scene BVH over meshes, MeshBvh inside each MeshShape. The MeshBvh is
   always-on (negligible overhead for small meshes).

4. ~~**Watertight intersection.**~~ **UPDATED in v3.** pbrt-v4 permute+shear
   (primary) + double-precision edge fallback. **Required in Phase 1.**

5. ~~**TriangleRasterizer ↔ triangle access.**~~ **RESOLVED.** `MeshShape`
   exposes the internal `MeshData` — the rasterizer iterates `MeshData.indices`
   to access all triangles. `MeshData::triangles()` iterator yields
   `(tri_index, p0, p1, p2)`.

6. **`Primitive` / `LightPrimitive` enum.** PlanarShape unification (v5) already
   resolved the trait-level fork. With v9, every shape (Sphere, Planar, Mesh)
   goes through `Shape3D` → `ShapeObject`, so `Primitive` needs variants per
   *category* plus a `Custom(Arc<dyn Intersectable>)` escape hatch.

______________________________________________________________________

## 8. Cross-Document References

### Existing design docs

| Doc | Relationship to Mesh | Status |
|---|---|---|
| `renderer_arch.md` §2, §9 | `LightPrimitive` needs `Mesh` variant (additive). Rasterizer iterates `MeshData.indices` for triangle access. | ✅ Compatible |
| `denoiser.md` | Denoiser post-processes film. Orthogonal. | ✅ No conflict |
| `adaptive-sampling.md` | Variance estimation. Orthogonal. | ✅ No conflict |
| `samplestream-refactor.md` | `Sampleable` is non-generic (raw params). Unchanged. | ✅ No conflict |
| `CORE_THESIS.md` §4 | SpatialDomain pattern, leaf sovereignty. Compatible. | ✅ Compatible |

### Codebase references

- `src/shape/mod.rs` — Shape3D trait, ShapeObject wrapper, Region2D trait.
- `src/shape/regions/` — All 8 region types.
- `src/shape/planar.rs` — PlanarShape\<R: Region2D\>.
- `src/shape/constructors.rs` — Free construction functions.
- `src/shape/box3d.rs` — BoxShape.
- `src/bvh/` — `Bvh<W>`, `BvhNode<W>`, `TreeBuilder`, `AabbPacked<W>`.
- `src/mesh/` — **New module**: `data.rs`, `shape.rs`, `bvh.rs`.
