# Mesh Capstone — Implementation Plan (Policy-backed BVH + Phase 1 Core Geometry)

**Reference:** `docs/mesh-design.md` (design v9, 2026-07-24)
**Codebase state:** evolved since v9 — packet-SIMD `Shape3D`, closed `Primitive` enum, packed `Aabb<W>`, `BvhNode<W>` SoA nodes
**Mode:** learner-owned. This plan outlines *what* to build and *why*; you write every line.

---

## 0. Current State (verified 2026-08-17, after the learner's initial mesh start)

| Item | State |
|---|---|
| `src/bvh/mesh.rs` | Present as an 8-line `MeshBvh` stub with `nodes`/`tri_indices`; this will become a thin wrapper around the policy-backed generic `Bvh`, not a second traversal implementation |
| `src/shape/mesh.rs` | Present with `MeshData` and a partial `MeshShape`; `uv_gradient` and `intersect_shape` are still empty/incomplete, and `MeshShape` has no BVH field or constructor |
| `src/shape/mod.rs` | `mesh` is declared/re-exported; `ShapeObject` now conditionally fills UV gradients via `get_or_insert` (a correct prerequisite for mesh gradients); the stale `mod tests;` declaration has been removed in the current staged diff |
| `src/intersect/interaction.rs` | `MeshHit { tri_index: u32, hit: Hit }` is present; retain this geometry-plus-identity shape, preferably with a `TriangleIndex` newtype |
| `src/primitives.rs` | `TriangleMesh(ShapeObject<MeshShape, Arc<Material>>)` has been added, but exhaustive `Intersectable`/`Bounded`/`Sampleable` matches are not complete |
| `Cargo.toml` | Contains unrelated profile/toolchain edits in the current staged diff; keep those separate from the mesh/BVH work unless intentionally part of another change |
| `git diff` | The learner has started the mesh implementation; source changes are intentionally left untouched by this plan update |
| `cargo check` | **FAILS with 9 errors from the partial mesh start**: wrong `Shape3D::intersect_shape` return type, empty `uv_gradient`, and missing `TriangleMesh` match arms |
| `cargo test` | Blocked by the current `cargo check` failures; run it only after the compile baseline is restored |

**Baseline:** the checkout is now intentionally mid-implementation. The next action is to restore `cargo check`; do not paper over the errors with wildcard match arms or placeholder geometry. The former dangling test-module issue is already addressed in the current staged diff, but must be confirmed with `cargo test` after compilation succeeds.

---

## 1. Verified API Facts (the plan's ground truth)

### Shape3D trait (`src/shape/mod.rs`)
- Super-traits: `Bounded + UVDifferentiable + Send + Sync`
- **Only `intersect_shape` + `uv_gradient` need implementing.** `intersect_shape_packed`, `occluded_shape`, `occluded_shape_packed` all have **defaults that scatter to the scalar path**. The stub's empty overrides of the packed/occluded methods are wrong — delete them, use the defaults.
- Signature: `fn intersect_shape<const N: usize>(&self, ray: &RayPacked<N>, ray_t: Interval<N>) -> Option<Hit>`

### Hit (`src/intersect/interaction.rs`)
- `Hit::new(time, point, mapping_point, geometric_normal, uv, uv_gradients)` — `geometric_normal` is private, set only via constructor
- Fields: `time, point, mapping_point, uv: Option<(f32,f32)>, uv_gradients: Option<(Direction3, Direction3)>, curvature: f32, geometric_normal`
- `SurfaceInteraction::from_material_hit` (line 133): `shading_normal = geometric_normal`, then `set_face_normal(ray)` flips it by `ray·n < 0`. **The shading normal IS the geometric normal in this codebase** — no separate field (yet).

### ShapeObject (`src/shape/mod.rs`)
- `ShapeObject<Sh: Clone, M: Borrow<Material>>` — `new(shape, material)`, `shape()`, `material()`
- Intersectable impl (lines 251-274): calls `intersect_shape`, then uses `get_or_insert` for UV gradients. This preserves gradients computed by a mesh policy/shape and is already the correct behavior.

### BvhNode (`src/bvh/mod.rs`)
- `pub struct BvhNode<const W: usize>` — repr(C, align(64)), **pub fields**: `bbox: AabbPacked<W>`, `child_offset: [u32; W]`, `leaf_info: [u16; W]`, `leaf_mask: u16`, `child_count: u8`, `split_axis: u8`
- Constructors `leaf()`/`interior()` are private, but the policy-backed mesh path does not construct nodes directly; generic `Bvh` construction remains the sole owner of node invariants.
- Leaf encoding (W=2): `child_offset[0] = prim_start`, `leaf_info[0] = prim_count`, `child_count = prim_count`, `leaf_mask = ALL_LEAVES` (0b11)

### Aabb (`src/bvh/aabb.rs`)
- `Aabb = AabbPacked<1>`; `from_points(&[Point3; W])` (W=1 → single point — **not** for triangles), `from_corners(p1, p2)` (line 331), `hit_single<const N>(&self, ray: &RayPacked<N>, ray_t: &Interval<1>)` (line 345)
- Triangle AABB: compute min/max of the 3 vertices → `from_corners`

### TreeBuilder (`src/bvh/builder.rs`)
- Current bound: `P: Clone + Intersectable + Bounded`; this is the first contract to relax.
- Target: the builder needs `Clone` plus a policy-provided AABB, not scene intersection or material resolution. Preserve binned SAH, `rayon::join`, and the leaf threshold while adding a policy-aware constructor.
- Existing `TreeBuilder::new` remains a convenience for scene primitives through the default policy; mesh indices use `new_with`.
- Build-only storage/flattening/widening should require only `P: Clone`; scene-facing `Intersectable`/`Bounded` impls remain separately constrained to the default policy path.

### Primitive (`src/primitives.rs`)
- Closed enum, 11 variants + `Custom(Arc<dyn Intersectable>)` escape hatch; `impl_primitive_from_shape!` macro generates `From` impls
- Sampleable impl requires `Sh: ShapeSurfaceSampling` — SDFs are excluded at compile time (their arm returns zeros). **MeshShape in Phase 1 follows the Sdf precedent: no `ShapeSurfaceSampling` impl → `Primitive::Mesh` Sampleable arm returns zeros** (emissive meshes land in Phase 2)

### Scene (`src/scene.rs`)
- `add_intersectable(object: Primitive, importance_target: Option<LightPrimitive>)`, `add_object(object: impl Into<Primitive>)`. No `add_mesh` yet.

### Current intersection boundary (`src/intersect/mod.rs`)
- `Intersectable` is currently scene-facing and material-specific: scalar/packet methods return `MaterialHit<'a>`.
- This fixed return type is why a bare indexed triangle cannot be passed through the existing `Bvh<W, P>` without either baking material into each triangle adapter or introducing a separate acceleration policy.
- The accepted plan introduces a **static, policy-backed acceleration contract** beneath `Intersectable`; it does not immediately make `Intersectable` itself object-generic, preserving `Arc<dyn Intersectable>`.

### TransformObject (`src/transform.rs`)
- `TransformObject<O: Intersectable, T: Transform = StaticTransform>` — instancing works via `ShapeObject<Arc<MeshShape>, M>`, which requires `impl Shape3D for Arc<MeshShape>` (delegating)

### Ray / Interval
- `RayPacked<N>` SoA; scalar = `RayPacked<1>`; `at(t)`, lane accessors on lane 0
- `Interval<N>`: `from(min, max)`, `min()`, `max()`, `min_value()`, `max_value()`

---

## 2. Design Decisions (accepted direction)

### D1 — Generalize the BVH contract, not `Hit<T>`
The base `Hit` remains the geometry record. Do not parameterize it with material or primitive identity: that would spread a payload type through `SurfaceInteraction`, object-safe `Intersectable`, transforms, and packet code.

Instead, formalize the **acceleration policy** beneath the scene-facing `Intersectable` trait:

```rust
trait HitRecord: Copy {
    fn time(&self) -> f32;
}

trait BvhPolicy<P> {
    type Hit<'a>: HitRecord
    where
        P: 'a;

    fn bounds(&self, primitive: &P) -> Aabb;
    fn intersect<'a>(
        &self,
        primitive: &'a P,
        ray: &RayPacked<1>,
        ray_t: Interval<1>,
    ) -> Option<Self::Hit<'a>>
    where
        P: 'a;
    fn occluded(&self, primitive: &P, ray: &RayPacked<1>, ray_t: Interval<1>) -> bool;
}
```

This is a design sketch, not copy-paste code: the exact GAT lifetime bounds are part of M1. The key contract is that the BVH only needs a comparable `time()`, while the policy owns bounds, primitive intersection, and the hit payload.

- `DefaultBvhPolicy` forwards to the existing `Bounded`/`Intersectable` methods and returns `MaterialHit<'a>`, preserving the current scene behavior and `Arc<dyn Intersectable>` escape hatch.
- `MeshPolicy` receives `&MeshData`, intersects a `TriangleIndex`, and returns `MeshHit` without a material.
- Policies are statically dispatched and passed to build/intersection methods; do not make `Intersectable` itself generic or object-unsafe.
- Keep the public `Bvh<W, P>` shape for the default scene path. Add `new_with_policy`, `intersect_with`, and `occluded_with` rather than forcing policy types into every existing scene call site.

### D2 — Generalize `TreeBuilder` around bounds
`TreeBuilder` should store `P: Clone`, not require `P: Intersectable + Bounded` merely to build. Add a policy-aware construction entry point that obtains each primitive's AABB through `BvhPolicy::bounds`.

- Preserve `TreeBuilder::new`/`Bvh::new` as convenience APIs using `DefaultBvhPolicy` for existing scene primitives.
- Add `TreeBuilder::new_with`/`Bvh::new_with_policy` for indexed mesh primitives.
- Keep SAH binning, leaf threshold, rayon recursion, flat DFS emission, `BvhNode<W>`, and widening unchanged until policy behavior is proven.
- Refactor every scalar, packet, and occlusion leaf call from `primitive.intersect*()` to the policy, and every closest-hit comparison from `.hit.time` to `HitRecord::time()`.
- The outer packet traversal may continue invoking the policy one scalar lane at a time initially; that matches the current BVH leaf behavior and leaves packet policy specialization for a measured optimization.

### D3 — Reuse `Hit` in the mesh policy result
Keep the current direction, with an explicit triangle-index type:

```rust
#[derive(Copy, Clone)]
pub struct MeshHit {
    pub tri_index: TriangleIndex,
    pub hit: Hit,
}
```

`MeshHit` is the policy result: `Hit` carries all existing geometric fields, and `TriangleIndex` carries the mesh-specific identity. It does not carry a material. `MaterialHit` remains the scene policy's result.

### D4 — `MeshBvh` is a thin generic-BVH wrapper
Replace the current `{ nodes, tri_indices }` stub with a wrapper around `Bvh<2, TriangleIndex>` (or a type alias if no wrapper API is needed). It must not duplicate node storage, SAH construction, stack traversal, or occlusion traversal.

`MeshPolicy<'a>` borrows `MeshData` for the duration of build/intersection. `MeshShape` owns the immutable `Arc<MeshData>` and the built `MeshBvh`; the policy is recreated as a short-lived view when a ray is tested. This keeps the BLAS geometry-only and allows one mesh BVH to serve instances with different materials.

### D5 — UV gradients belong to the winning mesh hit
`ShapeObject` already uses `get_or_insert`, so gradients computed by the mesh policy are preserved. The mesh policy should compute triangle UVs and `dpdu`/`dpdv` while it has the winning triangle's indices and barycentrics; `MeshShape` returns the embedded `Hit` without a second point query.

If the mesh has no UVs or the UV system is degenerate, return the existing zero/`None` fallback deliberately. `MeshShape::uv_gradient(mapping_point)` must not attempt to rediscover a triangle through a second BVH traversal.

### D6 — Phase-1 normal and material scope
Use the face normal `normalize((p1−p0)×(p2−p0))` as `geometric_normal` in Phase 1; no back-face culling. Interpolated vertex normals and the geometric-vs-shading-normal split remain Phase 3 work.

Keep `MeshData::per_tri_material` in the data layout but do not resolve it in `MeshPolicy` or `MeshHit` during Phase 1. The primitive identity now survives the BLAS, so a later material-resolution design can select `per_tri_material[tri_index]` without coupling the geometry BVH to materials.

### D7 — Optimization target and honest boundary
The policy contract ensures the scene and mesh paths share the node format, SAH builder, scalar/packet traversal, occlusion traversal, and future traversal cleanup. It does **not** require every future storage optimization to use identical leaves:

- `Triangle4Block` can become a policy primitive and reuse the generic BVH traversal while running a SIMD triangle kernel.
- Inline triangle data inside leaf nodes (bvh-performance.md T3/E2) is a later internal layout specialization, justified only by benchmarks; it must not create a second public BVH API or a second independently maintained traversal algorithm.
- Apply builder work from `docs/bvh-performance.md` (B1/D1/B2) after correctness: one generic builder optimization should benefit both scene and mesh policies.

### D8 — File layout
Keep `src/shape/mesh.rs` for `MeshData`/`MeshShape`. Keep `src/bvh/mesh.rs` for `TriangleIndex`, `MeshPolicy`, the watertight triangle kernel, and the thin `MeshBvh` wrapper. The generic policy and traversal machinery belongs in `src/bvh/mod.rs` and `src/bvh/builder.rs`.

### D9 — Rejected `TriRef` shortcut
The smallest code change would be a `TriRef { Arc<MeshData>, tri_index, material }` implementing the existing scene-facing `Intersectable`, then `Bvh<2, TriRef>`. This is explicitly rejected as the primary design:

- The current `Intersectable` contract returns `MaterialHit<'a>`, so a `TriRef` must bake material lookup into each mesh primitive or build one mesh BVH per material.
- That breaks the intended pure-geometry `MeshShape`/shared-BLAS boundary and complicates instances that reuse one mesh with different materials.
- It also discards the useful `MeshHit`/primitive-identity result needed for deferred per-triangle material lookup.

`TriRef` remains a valid fallback if the policy migration proves disproportionate, but it is not the learner's chosen architecture.

---

## 3. Milestones (each ends with `cargo check` + `cargo test` green)

### M0 — Restore a compiling baseline
- Finish the learner's partial skeleton without adding placeholder wildcard arms: `Shape3D::intersect_shape` must return `Option<Hit>`, `uv_gradient` must return a deliberate fallback, and every new `TriangleMesh` enum arm must be explicit.
- Confirm the staged removal of the dangling `#[cfg(test)] mod tests;` declaration is intentional and run `cargo test` once compilation is restored.
- Do not add mesh acceleration or rendering behavior until `cargo check` and `cargo test` provide a known baseline.

### M1 — Define the preliminary-hit and policy contracts
- Introduce a `TriangleIndex(pub u32)` newtype and make `MeshHit { tri_index: TriangleIndex, hit: Hit }` a `Copy` value record. `Hit` remains geometry-only; `MaterialHit` remains the default scene result.
- Add a `HitRecord` contract exposing only the closest-hit comparison (`time()`), plus a policy contract for bounds, scalar intersection, and occlusion. Keep the exact GAT bounds minimal and let the compiler guide the lifetime design.
- Implement a default scene policy that forwards to `Bounded`/`Intersectable`, returning `MaterialHit<'a>`. Preserve the existing object-safe `Intersectable` and `Arc<dyn Intersectable>` escape hatch rather than making that trait generic.
- Add a small policy-only test primitive/result if needed to prove the contract without involving meshes.
- **Verify:** existing BVH and primitive tests remain behaviorally unchanged; `cargo check` and `cargo test` are green.

### M2 — Generalize TreeBuilder and Bvh without duplicating traversal
- Relax `TreeBuilder`'s construction dependency from scene intersection to policy-provided bounds. Add `new_with` while retaining the existing `new` convenience path.
- Make storage-only `Bvh<W, P>`/`TreeBuilder<P>` operations depend on `P: Clone`; keep the existing `Intersectable` impl as a default-policy adapter with its current bounds.
- Add policy-backed construction/intersection/occlusion entry points to `Bvh<W, P>`. The stored node layout, flat DFS representation, leaf packing, widening, explicit stack, direction-sign ordering, and `Bounded` implementation remain shared.
- Change scalar, packet, and occlusion leaf visits to call the policy. Closest-hit tracking uses `HitRecord::time()`, never a material-specific `.hit.time` path.
- Keep initial packet policy calls scalar per lane. A policy-specific packet kernel is a later measured optimization, not part of the contract migration.
- **Verify:** compare the default-policy BVH with the pre-refactor scene path on existing unit scenes; exercise intersect and occluded for W=2 and the current wide widths.

### M3 — Watertight triangle kernel and MeshPolicy
Standalone kernel in `src/bvh/mesh.rs`:
```rust
fn intersect_triangle(
    ray: &RayPacked<1>,
    t_max: f32,
    p0: Point3,
    p1: Point3,
    p2: Point3,
) -> Option<(f32, f32, f32, f32)> // (t, b0, b1, b2)
```
- **Primary:** pbrt-v4 permute+shear — dominant ray axis, permuted vertices, shear, edge functions, determinant, sign-consistent edge tests, scaled t, barycentrics, and interval rejection.
- **Secondary:** if an edge function is exactly zero/ambiguous, recompute the edge tests in `f64`; no back-face culling.
- `MeshPolicy` owns/borrows `MeshData`, maps `TriangleIndex` to vertices, computes triangle AABBs for the builder, runs the watertight kernel, and constructs `MeshHit` with `Hit::new` only for a candidate that becomes closest.
- Phase 1 uses the face normal. UVs and UV gradients use the same barycentric weights and the winning triangle's vertex data; degenerate UV systems get an explicit fallback.
- **Unit tests:** front/back hits, miss, shared-edge hit, degenerate triangle, t-range rejection, barycentric reconstruction, and the double-precision fallback path.
- **Verify:** `cargo test`; compare policy results against a brute-force reference loop before introducing any mesh scene wiring.

### M4 — MeshBvh wrapper and MeshShape
- Replace the current `{ nodes, tri_indices }` stub with a thin `MeshBvh` wrapper around `Bvh<2, TriangleIndex>` (or a type alias where a wrapper adds no value). It owns no second node format and no second traversal.
- `MeshBvh::new` builds through `Bvh::new_with_policy` using `MeshPolicy`; `MeshBvh::intersect` recreates a short-lived policy view over the immutable `Arc<MeshData>` and returns `Option<MeshHit>`.
- Add `bvh: MeshBvh` to `MeshShape`, plus `from_data(data: Arc<MeshData>)`, bounding-box union, area accumulation, and a triangle iterator/helper as needed by the policy.
- `Shape3D::intersect_shape` uses the scalar policy path and returns `MeshHit.hit`; `ShapeObject`'s existing `get_or_insert` preserves mesh-computed UV gradients.
- Add `impl Shape3D for Arc<MeshShape>` (delegating) so transforms can share one BLAS.
- **Verify:** 2-triangle quad tests, random-ray property test against brute-force MT, bbox/area checks, UV/barycentric checks, and `cargo test`.
- **Learning hooks:** barycentric-interp, normals-face-vs-vertex, mesh-bvh-why, mesh-bvh-leaf, and two-level-bvh.

### M5 — Scene integration
- Complete the `TriangleMesh(ShapeObject<MeshShape, Arc<Material>>)` match arms explicitly for intersection, bounds, sampling fallback, and any other exhaustive `Primitive` operations.
- Add `Scene::add_mesh` following the existing `add_object`/`ShapeObject` ownership pattern. Keep material outside `MeshBvh`; `per_tri_material` remains a later lookup design.
- **Verify:** `cargo check`, `cargo test`, exhaustive-match coverage, and a minimal scene that reaches the mesh through the scene `Bvh`.

### M6 — End-to-end and instancing evidence
- Render a Cornell-box variant with a two-triangle mesh wall and compare it against the analytic planar wall; measure mean pixel difference.
- Build one `Arc<MeshShape>` once, wrap it in two transformed objects, and verify both instances render while sharing the same BLAS. Test differing materials if the scene API permits it; this is specifically why material must remain outside the mesh policy.
- **Verify:** image comparison, shared-BVH construction evidence, and no regression in the existing analytic-shape scene.

### M7 — Shared BVH performance track (after correctness)
- Use `docs/bvh-performance.md` as the optimization order: builder quality first (B1), direct wide construction/reduced widening bookkeeping (D1/B2), then deduplicated intersect/occluded traversal (T7) around the policy visitor.
- Benchmark binary versus wide paths (D2) before changing leaf kernels. Record scene-level and mesh-heavy results separately.
- Try `Triangle4Block` as a policy primitive for SIMD MT so the generic traversal is still reused. Only pursue inline triangle data in leaf nodes (T3/E2) if measurements justify an internal layout specialization; do not create a second public BVH API or independently maintained traversal.
- **Verify:** benchmark deltas, default-policy regression tests, mesh brute-force equivalence, and a report of which optimization actually wins.

---

## 4. Verification Strategy (evidence path)

| Claim | Evidence |
|---|---|
| Baseline and contracts are sound | M0/M1: `cargo check`, `cargo test`, default-policy regression tests |
| Generic BVH behavior is preserved | M2: default-policy result/occlusion equivalence for W=2 and wide widths |
| MT and MeshPolicy are correct | M3 unit tests: known triangles, shared edges, degenerates, brute-force comparison |
| MeshShape integrates | M4: quad tests, bbox/area, UV/barycentric checks, `cargo test` green |
| Scene integration | M5 compile, exhaustive matches, and a mesh reached through the scene BVH |
| **End-to-end correctness** | M6 render comparison: mesh wall vs analytic quad wall, near-identical pixels |
| No regressions | `cargo test` after every milestone |

## 5. Risks

- **Policy/GAT lifetime design** — keep the policy static and separate from object-safe `Intersectable`; start with scalar policy calls, then add packet specialization only when the bounds are stable.
- **Default-policy regression** — split storage-only BVH bounds from scene-facing trait impls carefully; compare old and new results before mesh integration.
- **Watertight MT subtlety** — the double-precision fallback must fire on exact-zero edges only; test the edge case explicitly.
- **SAH builder** — generalize the existing TreeBuilder rather than creating a mesh-specific copy; preserve its invariants while removing only the unnecessary scene-intersection bound.
- **Primitive enum** — exhaustive matches in multiple impls; the compiler will find them all.
- **D9 (`TriRef` fallback temptation)** — it is mechanically easy but couples mesh acceleration to material ownership; do not silently revert to it to avoid finishing the contract migration.
- **Per-triangle materials** — identity survives Phase 1 through `MeshHit`; resolve `per_tri_material` only in a later material/shading layer.

## 6. Learning Map (concept → milestone)

| Session node | Where it lands |
|---|---|
| mesh-def, mesh-indexing | M4/M6 — index buffers, shared vertices |
| mt-review, watertight | M3 — triangle policy/kernel |
| barycentric-interp | M3/M4 — UV/normal blending with (w,u,v) |
| normals-face-vs-vertex | M4 (face), Phase 3 (vertex) |
| uv-mapping | M4 — barycentric UVs, seam duplication |
| mesh-bvh-why, mesh-bvh-leaf | M2/M4 — generic BLAS construction and mesh use |
| instancing, two-level-bvh | M4 (BLAS), M6 (instances) |
| file-parsing | Phase 4 (later) |
