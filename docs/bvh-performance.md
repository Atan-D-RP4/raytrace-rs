# BVH Performance Optimization Spec

## Changelog

- **v4 (2026-07-23)** — Traversal review + benchmark data. (a) Added Benchmark
  Results section with measured spp/s from 1024×576 @ 1024 spp, 8 threads, 1764
  objects. (b) Added pbrt-v4 and rs-pbrt to comparative analysis. (c) New items:
  P1 (L1 prefetch, P2), T6 (redundant W=2 leaf slab test), T7 (deduplicate
  intersect/occluded), T8 (Bounded::bounding_box O(N×W) bug), T9 (SAH traversal
  cost tuning for wide BVHs), S6 (robust AABB test with ULP rounding). (d)
  Updated W=16 regression analysis with root causes (cache, AVX2 register
  pressure, SAH degradation from over-collapsing). (e) Added spatial splits
  (SBVH) as B3 under builder quality.
- **v3 (2026-07-23)** — Design constraints and builder priority. (a) Added
  Design Constraints section: unified `Bvh<W>` interface (no split into
  BVH2/BVH4/BVH8), `std::simd` over intrinsics, const generics, future-friendly
  codegen. (b) Builder promoted to first-class optimization target — Phase 0 now
  includes B1 (builder quality) and B2 (reduce widening bookkeeping). (c) D4
  updated: specialized W=2/W=4 kernels are internal specializations within the
  unified API, not separate public types. (d) D6 updated: co-design is about
  internal kernel specialization, not breaking the API. (e) Added "what to keep"
  and "what to change" sections from review. (f) Bottom line: remove avoidable
  construction overhead, reduce leaf dispatch cost, tighten traversal kernel —
  while staying clean, portable, reference-worthy.
- **v2 (2026-07-23)** — Second-pass review. New findings: (a) wide-node collapse
  path (`collapse_subtree`/`collect_wide_children`) is expensive — build wide
  directly instead of binary→wide widening. (b) `hit_mask` constructs
  `Simd::from_array` on every call — pre-pack or reduce lane construction
  overhead. (c) Need specialized kernels for W=2 and W=4, not just one generic
  const-generic path. (d) Must benchmark binary vs wide separately to verify
  wide actually wins. (e) Child ordering for wider nodes needs more than
  near-first: sorted hit masks, direction-aware ordering. (f) Traversal kernel
  and node format should be co-designed, not independent. Added new items to
  Phase 3 and Phase 0 (direct wide build).
- **v1 (2026-07-23)** — Initial spec. Comparative analysis of raytrace-rs BVH vs
  Embree, tinybvh, and svenstaro/bvh. 14 improvements cataloged across 5 phases.
  Node layout audit (all W land exactly on cache-line boundaries). Identified
  stack compress after hit and inline SoA triangle data as highest-impact
  changes.

## Implementation Status

### Phase 0 — Build rearchitecture (PLANNED)
- [ ] Build wide BVH directly instead of binary→wide widening (D1)
- [ ] Benchmark binary vs wide separately — verify wide actually wins on real
  scenes (D2)
- [ ] Builder quality as first-class optimization target — SAH, build
  parallelism (B1)
- [ ] Reduce collapse/widen bookkeeping — fewer temporary arrays, preserve
  bounds directly (B2)
- [ ] Spatial splits (SBVH) for overlapping geometry (B3)

### Phase 1 — Traversal micro-optimizations
- [x] `debug_assert!` stack overflow check (T5)
- [x] Precomputed `org_rdir` per ray (T4/E4)
- [ ] Stack compress on hit (T1)
- [ ] Stack distance-sort on push (E3)
- [ ] Consistency assertions (S1)
- [ ] Reduce `Simd::from_array` construction in hot loop — pre-pack or cache
  lane data (D3)
- [ ] Remove redundant W=2 leaf AABB test (T6)
- [ ] Deduplicate intersect/occluded traversal code (T7)
- [x] Fix `Bounded::bounding_box()` O(N×W) bug — should be O(W) (T8)
- [ ] Tune SAH traversal cost for wide BVHs — scale with W (T9)
- [ ] L1 prefetch before child node descent (P1)

### Phase 2 — Memory bandwidth (PLANNED)
- [ ] Quantized AABB nodes — 8-bit relative coords (E1)
- [ ] Compact leaf references — encode prim count in pointer tag (E6)

### Phase 3 — Traversal specialization (PLANNED)
- [ ] Direction-sign template specialization — 8 variants via macro (T2)
- [ ] Entry/exit stackless traversal — Hapala-style (S5)
- [ ] Specialized traversal kernels for W=2 and W=4 (D4)
- [ ] Sorted hit masks / direction-aware child ordering for W≥4 (D5)
- [ ] Co-design traversal kernel with node format — make traversal "know" layout
  intimately (D6)

### Phase 4 — Geometry rearchitecture (PLANNED)
- [ ] Inline SoA triangle data in leaf (T3/E2)
- [ ] Node-count precomputation (S2)
- [ ] Shape→node backpointer for refit (S3)

### Phase 5 — Polish (PLANNED)
- [ ] ISA-specific compilation (E5)
- [ ] Point queries (S4)
- [ ] Robust AABB test with ULP rounding (S6)

---

## Benchmark Results

**Scene**: earth_sphere — 1764 objects, 1024×576 @ 1024 spp, progressive, 8 threads.

| Config | Total Time | spp/s | vs FlatBvh | vs Best | Notes |
|--------|-----------|-------|------------|---------|-------|
| FlatBvh (scalar binary, d284316) | 974.74s | 1.05 | baseline | 0.57× | Scalar slab test, iterative stack |
| `Bvh<2>` (SIMD binary) | 590.97s | 1.73 | **1.65×** | 0.93× | `hit_mask` SIMD slab, direction-sign ordering |
| `Bvh<8>` (SIMD wide) | 552.76s | **1.85** | **1.76×** | baseline | Wide nodes, no near-far sorting on children |
| `Bvh<16>` (SIMD ultra-wide) | 844.02s | 1.21 | 1.16× | 0.65× | Regression — see W=16 analysis below |

### Key observations

1. **SIMD binary alone gives 1.65×** — the `hit_mask` SIMD slab test +
   direction-sign ordering account for the entire gain over scalar. No wide-node
   features needed.

2. **W=2 → W=8 gives only 7%** — far less than the 30-50% Embree typically gets
   from BVH4/8 over BVH2. Root cause: the wide path (W≥4) iterates children in
   slot order, not by proximity. No near-far sorting, no prefetch, no FMA slab
   test.

3. **W=16 regresses 35%** — three compounding causes:
   - **Cache pressure**: 512B/node (8 cache lines) vs 128B/node for W=4.
     Traversal working set grows 4×, causing L1/L2 misses.
   - **AVX2 register pressure**: 16×f32 requires 2 SIMD registers for each of
     lo/hi, consuming 4 of 16 YMM registers for a single slab test. Compiler
     spills reduce throughput.
   - **SAH quality degradation**: `widen::<16>()` collapses 4 binary levels into
     one wide node. The SAH split tuned for binary produces suboptimal 16-way
     partitions — children overlap more, increasing wasted intersection tests.

4. **W=4 is the sweet spot for AVX2** — 128B nodes fit 2 cache lines, 4×f32 fits
in a single YMM register, and collapse depth is only 2 binary levels.

---

## Design Constraints

These are non-negotiable project-level constraints that shape every optimization
decision.

1. **Unified `Bvh<W>` interface** — Do not split into `Bvh2`/`Bvh4`/`Bvh8` types
or `intersect_bvh2`/`4`/`8` functions. The const-generic `Bvh<W>` stays as the
single public API. Internal specialization (e.g., hand-tuned `impl Bvh<2>`
alongside the generic path) is fine as long as the public interface remains
unified.

2. **`std::simd` over handwritten intrinsics** — Prefer `std::simd` for
cross-platform SIMD portability and maintainability. Accept that compiler
codegen may improve over time; stay future-friendly rather than overfitting to
one microarchitecture.

3. **Const generics for node width** — Already in use. Keep this approach. The
compiler may not produce optimal code for all widths, but the API clarity is
worth it. Internal specialization within `impl Bvh<W>` blocks is acceptable.

4. **Pure-Rust reference implementation** — The codebase should remain
expressive, type-safe, ergonomic, and clean enough to serve as a reference.
Optimize for the fastest *practical* pure-Rust BVH, not a full Rust port of
Embree.

5. **Measurable wins over aesthetic micro-optimizations** — Every optimization
must be benchmarkable. Don't chase Embree's architecture wholesale — only adopt
patterns that produce measurable improvements in real scenes.

---

## Architecture Decision

**Wide BVH with const-generic `W` (2, 4, 8, 16), SoA AABB layout, iterative DFS
traversal with direction-sign ordering.**

The current implementation builds a binary SAH tree, then widens to `Bvh<W>` in
a second pass. The wide node enables SIMD AABB testing via
`AabbPacked<W>::hit_mask` using `std::simd`. Traversal uses an explicit 64-entry
stack with direction-sign near/far ordering.

### Node layout

All widths land exactly on cache-line boundaries — no padding waste:

| W | Node size | Cache lines |
|---|-----------|-------------|
| 2 | 64B | 1 |
| 4 | 128B | 2 |
| 8 | 256B | 4 |
| 16 | 512B | 8 |

`BvhNode<W>` is `#[repr(C, align(64))]` with fields:
- `bbox: AabbPacked<W>` — SoA `[[f32; W]; 3]` for min and max (axis-major,
  child-minor)
- `child_offset: [u32; W]` — node indices into flat array
- `leaf_info: [u16; W]` — primitive start indices
- `leaf_mask: u16` — which children are leaves
- `child_count: u8` — number of valid children
- `split_axis: u8` — which axis the SAH split used

### AABB test (SIMD path)

```rust
// hit_mask: tests ray against all W children in parallel
let min_x = Simd::from_array(self.min[0]);  // load 8 floats
let max_x = Simd::from_array(self.max[0]);  // load 8 floats
let t0 = (min_x - ox) * idx;                // slab entry
let t1 = (max_x - ox) * idx;                // slab exit
lo = lo.simd_max(t0.simd_min(t1));          // max of near planes
hi = hi.simd_min(t0.simd_max(t1));          // min of far planes
```

Returns `SimdMask<W>` — bit mask of which children the ray hits.

### Traversal (direction-sign ordering)

```rust
let sign = (ray.dir[axis].to_bits() >> 31) as usize;
let near_first = node.child_offset[sign];
let near_second = node.child_offset[1 - sign];
// push far child first (tested second), near child second (tested first)
```

This gives the compiler better branch prediction for coherent rays (primary, shadow).

---

## Comparative Analysis

### What Embree does that we don't

| Feature | Embree | raytrace-rs | Gap |
|---------|--------|-------------|-----|
| Quantized AABB | 8-bit relative coords, 136B (BVH8) | Full f32, 256B (W=8) | 1.88× more bytes |
| Inline triangle SoA | `Triangle4`: 28B/tri, precomputed edges | `Arc<dyn Intersectable>` per triangle | 3-6× slower leaf intersection |
| Precomputed `org_rdir` | FMA slab test: `msub(bounds, rdir, org_rdir)` | `(bounds - org) * rdir` | 2 ops vs 1 FMA per axis |
| Stack compress | SIMD filter on hit, distance-sorted | No compress, no sort | Testing nodes past `best_t` |
| ISA compilation | 4-5 kernel variants, runtime dispatch | Portable `std::simd` | Compiler may miss FMA fusion |
| Compact leaf refs | 4-bit count in pointer tag | 8B per leaf (start+count) | Minor |

### What tinybvh does that we don't

| Feature | tinybvh | raytrace-rs | Gap |
|---------|---------|-------------|-----|
| Stack compress after hit | `__popc(mask & validMask)` ~8-16 cycles | None | Wasted node tests |
| Direction-sign specialization | 8 template variants, zero sign branches | Runtime branch | ~5-10% overhead |
| Inline SoA triangles | BVHTri4Leaf: 4 triangles in SoA MT | Virtual dispatch per triangle | Biggest leaf gap |
| Precomputed `org * rdir` | FMA slab test | Sub+mul slab test | ~2× more ops per axis |

### What pbrt-v4 does that we don't

| Feature | pbrt-v4 | raytrace-rs | Gap |
|---------|---------|-------------|-----|
| BVH builder | Binned SAH with spatial splits (SBVH) | Binned SAH, no spatial splits | Overlapping geometry overlap more |
| Embree backend | Optional — delegates to Embree for production | N/A | Feature gap (we are the accel, not a wrapper) |
| Primitive decomposition | Splits complex shapes into triangle soup at build time | `Arc<dyn Intersectable>` — any shape inline | More general, but no SoA leaf optimization |
| Memory layout | Linear BVH — nodes stored contiguously after build | Flat array with child indices | Same approach |
| Traversal | Scalar, iterative, near-first | SIMD `hit_mask` + iterative stack | We have wider SIMD; pbrt has robust mode |
| Robustness | Standard (Watertight ray-triangle) | No robust AABB rounding | See S6 |

**Net**: pbrt-v4's BVH is architecturally simpler (binary, scalar) but has
higher build quality (SBVH spatial splits) and robust intersection. Our SIMD
path is faster per-node, but pbrt-v4's tree quality may compensate on complex
scenes with overlapping geometry.

### What rs-pbrt does that we don't

| Feature | rs-pbrt | raytrace-rs | Gap |
|---------|---------|-------------|-----|
| BVH type | `BvhAccelerator` — recursive `BvhNode` tree | Flat `Bvh<W>` with iterative stack | We are structurally faster (flat + SIMD) |
| Builder | Recursive SAH | Binned SAH + rayon parallel | Our builder is more sophisticated |
| SIMD | None — fully scalar | `std::simd` via `hit_mask` | 1.65× advantage from SIMD slab test alone |
| Node layout | `Arc<dyn Primitive>` per node | `Arc<dyn Intersectable>` per leaf | Same trait-object overhead |
| Thread safety | `Sync + Send` | `Sync + Send` via `Arc` | Equivalent |

**Net**: rs-pbrt's BVH is a straightforward recursive implementation with no
SIMD. Our `Bvh<W>` is structurally ahead — the gap is entirely in our missing
features (robust AABB, spatial splits, prefetch) rather than rs-pbrt having
something we lack.

### What svenstaro/bvh does that we don't

| Feature | svenstaro | raytrace-rs | Gap |
|---------|-----------|-------------|-----|
| Consistency assertions | `assert_consistent`, `assert_tight` | None | Debugging time |
| Entry/exit traversal | Stackless, Hapala-style | Explicit stack | Simpler hot path |
| Point queries | `nearest_to`, iterators | None | Feature gap |
| Node-count precomputation | Pre-allocates exact flat array | Two-pass (tree→flatten) | Extra allocation |

---

## Detailed Improvement Catalog

### Phase 0 — Build rearchitecture

#### D1: Build wide BVH directly

**Source**: Architecture review (v2)
**Effort**: ~2-3 days
**Impact**: High (eliminates entire widen pass)

The current build path is binary SAH → `TreeBuilder` → `flatten_node` → `Bvh<2>`
→ `widen::<8>()`. The widen step (`collapse_subtree` + `collect_wide_children`)
repeatedly rebuilds wide subtrees, carries temporary arrays, and recomputes
bounds from binary children instead of preserving them directly.

**Build wide directly**: During SAH construction, when accumulating children for
a node, directly group up to `W` children into a wide node. This avoids the
entire binary→wide conversion pass and gives a node layout closer to what the
traversal kernel wants.

**Key constraint**: The SAH split still operates on binary partitions (two
subsets). The wide build collects up to `W` consecutive binary children into one
wide node, using the SAH cost to decide when to stop collecting and emit a wide
node.

#### D2: Benchmark binary vs wide separately

**Source**: Architecture review (v2) **Effort**: ~1 day **Impact**: Trust
(prevents wasted work)

It is common for a wide layout to look elegant and still lose on real scenes
because node fanout is not high enough to amortize the extra per-node work.
Measure:

- Binary BVH (`Bvh<2>`) traversal throughput (nodes/sec, rays/sec)
- Wide BVH (`Bvh<4>`, `Bvh<8>`) traversal throughput
- Crossover point: at what scene complexity does wide beat binary?

If wide doesn't win on your target scenes, the investment in Phase 3/4
specialized kernels may not pay off.

#### B1: Builder quality as first-class optimization target

**Source**: Architecture review (v3) **Effort**: ~3-5 days **Impact**: High
(mediocre tree makes even a good kernel look worse)

The builder is not just a preprocessing step — it's a first-class optimization
target. A strong BVH builder matters as much as traversal. Key areas:

- **SAH quality**: Current binning with 32 bins is decent. Consider increasing
  bin count for complex scenes, or adding a spatial-split pass (SBVH) for
  irregular meshes where objects overlap significantly.
- **Build parallelism**: Current `rayon::join` with a hardcoded threshold.
  Consider Embree's executor pattern — parallelize over independent subtrees
  with work-stealing.
- **Primitive ordering**: How primitives are ordered within leaves affects cache
  behavior during intersection. Sort triangles by spatial locality within each
  leaf.

#### B2: Reduce collapse/widen bookkeeping

**Source**: Architecture review (v3) **Effort**: ~1-2 days **Impact**: Medium
(construction cost, node quality)

The current `collapse_subtree` + `collect_wide_children` path:
- Recursively rebuilds wide subtrees
- Carries temporary arrays (`Vec<usize>`)
- Recomputes bounds from binary children instead of preserving them directly

This adds construction cost and can compromise final node quality (conservative
AABBs from combining binary children). If D1 (direct wide build) is implemented,
this path becomes unnecessary. If D1 is deferred, reduce the bookkeeping:
preserve binary child AABBs directly instead of recomputing, and avoid temporary
allocations.

#### B3: Spatial splits (SBVH)

**Source**: Embree + pbrt-v4 **Effort**: ~3-5 days **Impact**: High (~15-20% of
gap for complex scenes with overlapping geometry)

Standard object-split SAH assigns each primitive to exactly one side of the
split. Spatial splits (SBVH, Wald et al. 2007) allow primitives that straddle
the split plane to be duplicated — one copy in each child, with tighter
per-child AABBs. This dramatically reduces overlap for scenes where triangles
span split boundaries (e.g., a floor plane split across a BVH partition, or
instanced geometry with complex topology).

**How it works**:
1. During SAH evaluation, compute both object splits (standard) and spatial
splits (duplicate straddling primitives).
2. A spatial split clips the primitive's AABB at the split plane, creating two
tighter AABBs.
3. The SAH cost for spatial splits includes the duplication penalty: `dup_cost =
duplicated_primitives × leaf_cost`.
4. Choose whichever split (object or spatial) has the lowest total SAH cost.

**Key insight**: Embree's SBVH uses a separate pass for spatial splits — it
first tries object splits, and only tries spatial splits if object splits
produce significant overlap. This limits the duplication overhead (typically <5%
of primitives are duplicated).

**Trade-off**: Increases build time by ~30-50% and memory by ~10-20% (duplicated
primitives in leaves). But produces significantly better trees for complex
geometry — pbrt-v4 reports 10-30% traversal speedup on the Stanford Dragon and
similar meshes.

**When to implement**: After D1 (direct wide build) and when benchmarking on
complex triangle-mesh scenes (not just the current test scenes with 1764 convex
objects). The current scenes have low overlap, so SBVH impact would be minimal —
it matters for production-quality meshes.

---

### Phase 1 — Traversal micro-optimizations

These don't change data structures, just traversal efficiency.

#### T5: Remove hot-path stack overflow check

**Source**: tinybvh **Effort**: ~15 min **Impact**: Low (~1-2% of gap)

Replace `if sp >= MAX_STACK { tracing::warn!(...); break; }` with
`debug_assert!`. The branch is predictable-not-taken but still costs a
comparison + branch instruction per interior node. tinybvh uses a fixed large
stack and doesn't check at all.

**Current code** (src/bvh/mod.rs, line 678):
```rust
if sp >= MAX_STACK {
    tracing::warn!("Bvh traversal stack overflow");
    break;
}
```

**Change**:
```rust
debug_assert!(sp < MAX_STACK, "Bvh traversal stack overflow");
```

#### T4/E4: Precomputed `org_rdir` per ray

**Source**: tinybvh + Embree **Effort**: ~30 min **Impact**: Medium (~5-8% of
gap)

Precompute `org_rdir = ray.origin * inv_dir` once per ray. In the slab test,
replace `(bounds_min - origin) * inv_dir` with `bounds_min * inv_dir - org_rdir`
— one FMA instead of one sub + one mul.

**Where to precompute**: In `Ray::new()` or at the point where a ray is first
passed to the BVH. Store as a `Vec3` field on `Ray`.

**Where to use**: In `AabbPacked::hit_mask()` and the scalar `slab_aabb_test()`. Replace:
```rust
let t0 = (min_x - ox) * idx;
let t1 = (max_x - ox) * idx;
```
with:
```rust
let t0 = min_x * idx - org_rdir_x;
let t1 = max_x * idx - org_rdir_x;
```

#### T1: Stack compress after hit

**Source**: tinybvh **Effort**: ~1 day **Impact**: High (~15-20% of gap)

After finding a closer hit (`best_t` updated), filter the active stack to remove
entries whose minimum distance exceeds `best_t`. This prevents testing nodes
that can't contain a closer intersection.

**Implementation sketch**:
```rust
// After best_t is updated (inside the intersection test block):
stack[0..sp].retain(|&entry| {
    // entry is (node_idx, t_min) — compare t_min against new best_t
    entry.1 <= best_t
});
```

Or SIMD-accelerated (like tinybvh): load 4 stack entries at once, compare `dist
<= best_t` as a SIMD mask, compact survivors.

**Key insight**: The stack entries already have their AABB `t_min` stored (from
the direction-sign ordering push). So filtering is a simple comparison, not a
re-evaluation.

#### E3: Stack distance-sort on push

**Source**: Embree **Effort**: ~1 day **Impact**: Medium (~5-10% of gap)

Push hit children ordered by `tNear` distance (nearest first), not by traversal
order. This makes the stack a priority queue — the first pop always gives the
nearest unvisited node. This amplifies the benefit of T1 (compress), because
early hits produce tighter `best_t` bounds.

**Implementation**: After `hit_mask` returns a mask, iterate set bits in `tNear`
order (smallest first). Push each hit child to the stack in that order.

#### S1: Consistency assertions

**Source**: svenstaro/bvh
**Effort**: ~hours
**Impact**: Trust (debugging)

Add `assert_consistent()` and `assert_tight()` methods to `Bvh<W>`:
- Verify every child AABB is contained within its parent AABB
- Verify `prim_start + prim_count` fits in `self.primitives`
- Verify all node indices are in range
- Verify no detached subtrees (all nodes reachable from root)

These catch AABB patching bugs (e.g., the SAH centroid-bbox patching at lines 296-331 of mod.rs).

#### D3: Reduce `Simd::from_array` construction in hot loop

**Source**: Architecture review (v2)
**Effort**: ~1 day
**Impact**: Medium (~5-8% of gap)

`hit_mask` loads each axis into `Simd::from_array(self.min[0])` etc. on every
call. This is clean but leaves overhead from repeated lane construction, mask
extraction, and scalar-to-vector shuffling.

**Options**:
1. Pre-pack node data into `Simd`-compatible layout at build time (store
   `Simd<f32, W>` directly instead of `[f32; W]`)
2. Use `unsafe` transmute from aligned `[f32; W]` to `Simd<f32, W>` (guaranteed
   layout-compatible on nightly)
3. At minimum, ensure the compiler is optimizing away the copy (check assembly)

The traversal function should be arranged so the most probable miss path is as
cheap as possible — miss is the common case for primary rays in open scenes.

#### T6: Remove redundant W=2 leaf AABB test

**Source**: Traversal review (v4)
**Effort**: ~15 min
**Impact**: Low (~1-2% of gap, only W=2 path)

In the W=2 `intersect` path (mod.rs:627-631), after the parent interior node's
`hit_mask` already validated that a child AABB is hit, the leaf branch re-tests
the same AABB with a scalar `slab_aabb_test`. This is redundant — the mask
already confirmed the hit.

**Current code**:
```rust
// mod.rs:627-631 — W=2 leaf path
BvhNode2::Leaf { object, bbox, .. } => {
    if !bbox.slab_aabb_test(ray, ray_t) {  // ← redundant: parent hit_mask already passed
        return None;
    }
    object.intersect(ray, ray_t)
}
```

**Change**: Remove the `slab_aabb_test` for W=2 leaves. The parent's `hit_mask`
already validated this AABB. For W≥4, the leaf AABB test is still needed because
the wide node's mask covers all children simultaneously — a child marked as leaf
was hit, but we still need the tight scalar slab entry for the leaf's own AABB.

**Caveat**: If the child's AABB is tighter than the parent's (e.g., a
sub-region), the scalar test would give a tighter `ray_t` bound. For W=2 with
direction-sign ordering, the child AABBs are typically very close to the
parent's — the tightness difference is negligible.

#### T7: Deduplicate intersect/occluded traversal code

**Source**: Traversal review (v4)
**Effort**: ~1-2 days
**Impact**: Medium (~5-10% of gap, primarily maintainability + codegen)

The `intersect()` and `occluded()` methods share ~260 lines of near-duplicate
traversal logic (lines 625-850 of mod.rs). The only difference is `occluded`
returns `bool` (early exit on any hit) while `intersect` tracks `best_t` and
returns `Option<MaterialHit>`.

**Approach**: Extract a generic traversal function parameterized by a callback:
```rust
fn traverse_wide<W, F>(nodes: &[BvhNode<W>], ray: &Ray, ray_t: Interval, mut visit: F) -> bool
where
    F: FnMut(&dyn Intersectable, Interval) -> bool,  // return true to continue, false to early-exit
```

Or use a trait-based visitor pattern where the traversal kernel calls `visitor.on_hit(object, ray_t) -> Option<Interval>` which returns `Some(tighter_interval)` for intersect or `None` (early exit) for occluded.

This eliminates code divergence bugs (e.g., future changes to one path but not the other) and may improve compiler codegen by reducing code duplication.

#### T8: Fix `Bounded::bounding_box()` O(N×W) bug

**Source**: Traversal review (v4) **Effort**: ~5 min **Impact**: Low (build-time
only, but correctness)

`Bounded::bounding_box()` for `Bvh<W>` iterates every node in the BVH to compute
the scene bounding box — O(N×W). The root's children already cover the entire
scene. Should only examine the root node's AABB.

**Current code**:
```rust
impl<W: const usize> Bounded for Bvh<W> {
    fn bounding_box(&self) -> Aabb {
        // Iterates ALL nodes — O(N×W)
        self.nodes.iter().fold(Aabb::new(), |acc, node| acc.merge(&node.bounding_box()))
    }
}
```

**Change**:
```rust
impl<W: const usize> Bounded for Bvh<W> {
    fn bounding_box(&self) -> Aabb {
        // Root node already covers the entire scene — O(W)
        self.nodes.first().map_or(Aabb::new(), |root| root.bounding_box())
    }
}
```

#### T9: Tune SAH traversal cost for wide BVHs

**Source**: Traversal review (v4) **Effort**: ~1 hour **Impact**: Medium (~5-10%
of gap for W≥8)

The builder uses `trav_cost = root_sa × 0.5` (mod.rs:174) — a value tuned for
binary BVHs. For wider BVHs, each interior node tests more children, so the
traversal cost per node is proportionally higher. Using the binary traversal
cost for wide BVHs produces overly-deep trees that don't exploit the wider
fanout.

**Recommendation**: Scale traversal cost with W:
```rust
let trav_cost = root_sa * 0.5 * W as f32;  // W=2: same as before; W=8: 4× higher
```

Or calibrate empirically: run a binary build, measure the actual traversal cost
ratio (interior nodes × W tests vs leaf tests), and use that to set the wide
BVH's `trav_cost`. Higher `trav_cost` → shallower trees with more objects per
leaf → better cache utilization during traversal.

#### P1: L1 prefetch before child node descent

**Source**: Embree **Effort**: ~1 day **Impact**: High (~10-15% of gap,
especially for W≥4)

Embree prefetches 2-4 cache lines of the child node before descending into it.
This hides memory latency by issuing the load request before the CPU needs the
data. Critical for wide BVHs because each node is larger (128-512 bytes = 2-8
cache lines).

**Implementation**:
```rust
use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};

// After hit_mask determines which child to descend into:
let child_ptr = &self.nodes[child_idx] as *const BvhNode<W> as *const i8;
unsafe { _mm_prefetch(child_ptr, _MM_HINT_T0); }
// _MM_HINT_T0 = prefetch into all cache levels (L1/L2/L3)
// _MM_HINT_T1 = L2 only (less aggressive, lower cache pollution)
```

**Where**: Right before the `continue` that descends into a child node in both
the W=2 path and the wide traversal path.

**Note**: Use `_MM_HINT_T1` (L2 prefetch) instead of `_MM_HINT_T0` (L1) if cache
pollution is a concern. L2 is safer for wide BVHs where the large working set
may evict useful data from L1.

---

### Phase 2 — Memory bandwidth

These reduce per-node memory footprint, improving cache utilization.

#### E1: Quantized AABB nodes

**Source**: Embree **Effort**: ~2-3 days **Impact**: High (~20-25% of gap)

Replace full `f32` bounds with 8-bit relative coordinates. Each node stores:
- `origin: [f32; 3]` — the minimum corner of the bounding box
- `scale: [f32; 3]` — `(max - min) / 255.0`
- `lower: [[u8; W]; 3]` — quantized min per child per axis
- `upper: [[u8; W]; 3]` — quantized max per child per axis

Dequantization is one FMA: `dequant = origin + scale * (u8 as f32)`.

**Memory savings**:
- Current W=8: 256B per node
- Quantized W=8: ~136B per node (47% reduction)

**Conservative encoding**: The quantized bounds must always enclose the original
bounds. Embree's approach: after quantization, check if the dequantized bounds
are smaller than the originals, and expand if so.

```rust
// Quantize
let lower_q = ((child_min - origin) / scale).round().clamp(0.0, 255.0) as u8;
let upper_q = ((child_max - origin) / scale).round().clamp(0.0, 255.0) as u8;

// Dequantize and verify
let lower_deq = origin + scale * (lower_q as f32);
let upper_deq = origin + scale * (upper_q as f32);
// If lower_deq > child_min or upper_deq < child_max, expand the quantized bounds
```

**Performance trade-off**: Adds dequantization overhead (~6 FMA per node) but
saves cache misses. The FMA cost is amortized across 8 children — one dequant +
one slab test per child, vs one slab test per child with full f32. Net win is
cache-dependent: for deep trees with large working sets, this is significant.

#### E6: Compact leaf references

**Source**: Embree **Effort**: ~1 day **Impact**: Low (~3-5% of gap)

Encode primitive count in the node structure instead of separate fields.
Currently:
- `leaf_info: [u16; W]` — primitive start indices (W × 2 bytes)
- `leaf_mask: u16` — which children are leaves

If W ≤ 8 and primitive count ≤ 7 (Embree's limit), encode count directly in the
leaf_mask or a combined field. This saves 2 bytes per node but is minor.

---

### Phase 3 — Traversal specialization

These make the traversal loop branchless per node.

#### T2: Direction-sign template specialization

**Source**: tinybvh **Effort**: ~1 day **Impact**: Medium (~5-10% of gap)

Generate 8 explicit traversal function variants, one per ray octant:
```rust
macro_rules! define_traversal {
    ($name:ident, $px:expr, $py:expr, $pz:expr) => {
        fn $name(node: &BvhNode<W>, ray: &Ray, ...) -> ... {
            // All sign-dependent logic becomes compile-time constants
            const SIGN: usize = ($px as usize) | (($py as usize) << 1) | (($pz as usize) << 2);
            // Near/far ordering is a compile-time constant
            // AABB min/max selection is a compile-time constant
        }
    };
}
```

Dispatch once per ray at the BVH entry point:
```rust
let sign = (ray.dir.x.to_bits() >> 31) as usize
         | ((ray.dir.y.to_bits() >> 31) << 1) as usize
         | ((ray.dir.z.to_bits() >> 31) << 2) as usize;
match sign {
    0 => traverse::<true, true, true>(root, ray, ...),
    1 => traverse::<false, true, true>(root, ray, ...),
    // ...
    7 => traverse::<false, false, false>(root, ray, ...),
}
```

This eliminates all `if ray.dir.x >= 0.0 { ... } else { ... }` branches inside the hot loop. The compiler can then:
- Fold constant AABB min/max selection
- Eliminate dead code paths
- Potentially vectorize better

#### S5: Entry/exit stackless traversal

**Source**: svenstaro/bvh **Effort**: ~couple days **Impact**: Medium (simpler
hot path, no stack overflow path)

Replace the explicit stack with Hapala-style entry/exit indices. Each node
stores the "next node to visit after this subtree" as an implicit entry in the
traversal sequence. Traversal becomes:

```rust
let mut index = 0; // root
loop {
    let node = &self.nodes[index];
    if node.is_leaf() {
        // test primitives
        index = node.exit_index; // next node after this subtree
    } else if ray.intersects_aabb(&node.bbox) {
        index = node.child_offset[0]; // go deeper (near child)
    } else {
        index = node.exit_index; // skip subtree
    }
    if index == SENTINEL { break; }
}
```

No stack at all. The exit index is computed during build by walking the flat
array in DFS order.

**Trade-off**: Removes stack push/pop overhead and the overflow path entirely.
But the entry/exit encoding adds 4 bytes per node (the exit index). For W≥4,
this may not fit cleanly since each node already has W child offsets. The
entry/exit approach works best with binary trees — for wide BVH, the stackless
benefit is less clear.

#### D4: Specialized traversal kernels for W=2 and W=4

**Source**: Architecture review (v2, v3) **Effort**: ~1-2 days **Impact**:
Medium (~5-8% of gap)

Rust const generics are nice, but the compiler is not obligated to produce the
same quality of code as hand-specialized paths. For `W=2`, the node is 64B (one
cache line) — the entire node fits in a single load. For `W=4`, it's 128B (two
cache lines). The traversal logic for these narrow widths may benefit from
manual unrolling or specialized SIMD patterns that the generic `W` path doesn't
produce.

**Implementation**: Add `impl Bvh<2>` and `impl Bvh<4>` with hand-tuned
traversal functions alongside the generic `Bvh<W>` path. Use `#[inline(always)]`
and check assembly to verify the compiler is producing optimal code.

**Constraint**: This is an internal optimization only. The public API stays as
`Bvh<W>` — no separate `Bvh2`/`Bvh4` types or `intersect_bvh2`/`4` functions.
The specialized impls are transparent to callers.

#### D5: Sorted hit masks / direction-aware child ordering for W≥4

**Source**: Architecture review (v2) **Effort**: ~1-2 days **Impact**: Medium
(~5-10% of gap)

Near-first ordering works well for binary nodes, but for wider nodes (W≥4), the
hit mask from `hit_mask` contains multiple set bits. Instead of testing all
lanes then branching, sort the hit lanes by distance and test nearest first:

```rust
let mask = node.bbox.hit_mask(ray);
// Extract hit lanes sorted by tNear (nearest first)
while !mask.is_empty() {
    let lane = mask.iter_set_lanes().min_by_key(|&l| t_near[l]);
    // test child at lane
}
```

For Embree-style performance, the child test order should be tightly coupled to ray direction and node split axis. The `split_axis` metadata is already stored — use it to order children by expected hit probability, not just distance.

#### D6: Co-design traversal kernel with node format

**Source**: Architecture review (v2, v3) **Effort**: ~3-5 days **Impact**: High
(this is where Embree-class performance comes from)

Embree's edge is not just SIMD math — it's the tight integration between
traversal strategy and node format. The traversal kernel "knows" the layout
intimately: it uses byte-offset indexing into SOA nodes, precomputed ray info
that matches the node format exactly, and stack management that is tuned to the
specific node width.

**Current state**: Your node format is decent (SoA AABB, child indices, leaf
info), but the traversal logic still looks generic — it loads data from the
node, constructs SIMD types, runs the test, extracts results, then does scalar
traversal logic.

**Target state**: The traversal kernel should be written as one cohesive unit
with the node format, not as "load data → construct SIMD → test → extract →
traverse." This means:
- The traversal function is macro-generated for specific W values (W=2, 4, 8,
  16)
- The stack management is specialized for the specific node width
- The hit mask extraction is fused with the traversal logic (no intermediate
  SIMD type)
- The child ordering is integrated with the hit mask processing

**Constraint** (from v3 review): This is internal specialization within the
unified `Bvh<W>` API. No separate public types or functions. The macro generates
`impl Bvh<2>`, `impl Bvh<4>`, etc. with hand-tuned traversal, but callers still
use `Bvh<W>` generically.

This is the single highest-effort item but also the one that most directly
closes the gap with Embree.

---

### Phase 4 — Geometry rearchitecture

These change how primitives are stored and intersected.

#### T3/E2: Inline SoA triangle data in leaf

**Source**: tinybvh + Embree
**Effort**: ~1-2 weeks
**Impact**: Highest (~30-40% of gap)

This is the fundamental structural difference between raytrace-rs and
high-performance BVH implementations. Currently:

```rust
// Current: every primitive is a heap-allocated trait object
let primitives: Vec<Arc<dyn Intersectable>>;
// Leaf visit: N virtual dispatches + N scalar Moeller-Trumbore passes
for p in &self.primitives[start..][..count] {
    p.intersect(ray, ...); // vtable call
}
```

**Target**: Store triangle data inline in the BVH leaf, in SoA format, with
precomputed edges:

```rust
struct Triangle4 {
    v0: [Simd<f32, 4>; 3],  // base vertices, x/y/z across 4 triangles
    e1: [Simd<f32, 4>; 3],  // precomputed edges (v0 - v1)
    e2: [Simd<f32, 4>; 3],  // precomputed edges (v2 - v0)
    ng: [Simd<f32, 4>; 3],  // precomputed normals (cross(e2, e1))
    geom_ids: Simd<u32, 4>,
    prim_ids: Simd<u32, 4>,
}
// 112 bytes = 28 bytes per triangle (Embree's Triangle4)
```

The SIMD Moeller-Trumbore kernel tests all 4 triangles simultaneously:
```rust
fn intersect_triangle4(ray: &Ray, tri: &Triangle4) -> SimdMask<4> {
    let o = Simd::splat(ray.origin.x); // broadcast ray origin
    let d = Simd::splat(ray.dir.x);
    // ... full MT in SIMD, 4 results at once
}
```

**Key optimizations in Embree's MT kernel**:
1. Precomputed edges: no per-test `v1 - v0` subtraction
2. Precomputed normal `ng = cross(e2, e1)`: avoids one cross product per test
3. Sign masking: `U = dot(R, tri_e2) ^ sgnDen` — XOR flips sign for backface
   culling, no branch
4. Early-out: `none(valid)` check after each test stage

**Impact**: This is the single biggest performance gain available. For a
4-triangle leaf, you go from 4 virtual calls + 4 scalar MT passes to 1 SoA SIMD
MT pass — roughly 4-6× faster per leaf intersection.

**Implementation path**:
1. Create a `TriangleStore` that precomputes and stores `Triangle4` blocks
2. Add a mapping from BVH leaf index to `Triangle4` block index
3. In leaf intersection, load the `Triangle4` and run the SIMD kernel
4. Remove `Arc<dyn Intersectable>` from the BVH traversal path (keep it for
   general shapes)

**Prerequisite**: The BVH must know it's storing triangles. Add a `TriangleBvh`
type or a builder that precomputes the SoA layout during build.

#### S3: Shape→node backpointer

**Source**: svenstaro/bvh
**Effort**: ~hours
**Impact**: Low (enables future animated scenes)

Each primitive stores which BVH leaf it belongs to:
```rust
struct ShapeMeta {
    node_index: u32,  // which BvhNode<W> contains this shape
    prim_offset: u16, // offset within that node's leaf
}
```

Enables incremental AABB refit for animated scenes without full rebuild.

---

### Phase 5 — Polish

These are lower-priority improvements.

#### E5: ISA-specific compilation

**Source**: Embree **Effort**: ~1 week **Impact**: Low-Medium (~5-8% of gap,
only matters when geometry is the bottleneck)

Compile the traversal kernel multiple times for different ISA levels:
- SSE4.1 baseline
- AVX2 + FMA3
- AVX-512 (if available)

Runtime dispatch via CPU feature detection. This gives the compiler explicit ISA
constraints instead of relying on `std::simd` auto-vectorization.

**Trade-off**: Significant complexity. Only justified once Phase 4 (inline SoA)
is done and geometry intersection is the dominant cost. With `std::simd`, the
compiler does a reasonable job — this is diminishing returns.

#### S4: Point queries

**Source**: svenstaro/bvh **Effort**: ~half-day **Impact**: Features (not
performance)

Add `nearest_to(point: Point3) -> Option<&Arc<dyn Intersectable>>` to `Bvh<W>`.
Uses the same iterative traversal with AABB minimum-distance pruning.

**Use cases**: Photon mapping, ambient occlusion point queries, proximity-based
effects, debug visualization.

#### S6: Robust AABB test with ULP rounding

**Source**: Embree **Effort**: ~1 day **Impact**: Medium (correctness + prevents
rare visual artifacts)

Embree uses a robust AABB test that rounds bounds outward by 1-3 ULPs (units in
the last place) to avoid watertight intersection issues at BVH/primitive
boundaries. Without this, rays that clip the edge of a primitive's bounding box
may miss due to floating-point rounding, causing cracks at mesh boundaries or
between adjacent BVH leaves.

**Implementation**:
```rust
/// Expand AABB bounds by `ulp_count` ULPs on each side for robust slab testing.
fn expand_ulps(&self, ulp_count: i32) -> Aabb {
    Aabb {
        min: Point3::new(
            add_ulps(self.min.x, -ulp_count),
            add_ulps(self.min.y, -ulp_count),
            add_ulps(self.min.z, -ulp_count),
        ),
        max: Point3::new(
            add_ulps(self.max.x, ulp_count),
            add_ulps(self.max.y, ulp_count),
            add_ulps(self.max.z, ulp_count),
        ),
    }
}

fn add_ulps(f: f32, ulps: i32) -> f32 {
    let bits = f.to_bits() as i32;
    f32::from_bits((bits + ulps) as u32)
}
```

Apply expansion to node AABBs at build time (in the `BvhNode<W>` constructor)
rather than at traversal time, so the cost is amortized over all rays.

**When to implement**: After Phase 4 (inline SoA triangles), when mesh boundary
cracks become visible. Not urgent for current scenes (all convex shapes), but
essential for triangle-mesh scenes.

---

## What to Keep

These are already strong enough to preserve:

- Flat, contiguous node storage with child indices instead of pointers
- Unified const-generic `Bvh<W>` interface (no type explosion)
- `std::simd` as the portable SIMD abstraction
- Iterative traversal with explicit stack
- Near-first child ordering for coherent rays
- Packed AABB representation (SoA, axis-major)
- Readability and correctness as optimization constraints

## What to Change

Priority order for the next work session:

1. **Builder first** — Improve tree quality before adding traversal complexity.
   A mediocre tree makes even a good kernel look worse.
2. **Direct wide construction** — Replace binary→wide widening with direct
   wide-node build (D1). Reduce collapse bookkeeping (B2).
3. **Leaf dispatch cost** — Profile and minimize `Arc<dyn Intersectable>`
   overhead. Primitive dispatch should become as predictable and data-local as
   possible (T3/E2).
4. **Traversal specialization** — Specialize within the const-generic design,
   not as separate public types (D4, D6).
5. **Split-axis metadata** — Exploit `split_axis` and ray-direction information
   more aggressively for child ordering (D5).
6. **Benchmark binary vs wide** — Understand when wide nodes actually pay for
   themselves (D2).

## Bottom Line

The implementation is already structurally strong. The next step is not to
imitate Embree wholesale, but to remove avoidable construction overhead, reduce
leaf dispatch cost, and tighten the traversal kernel around the existing
const-generic layout. The end state should stay clean, portable, and
reference-worthy, while still being fast enough to earn its place as the best
pure-Rust BVH in practice.

---

## Cross-References

### Related documents

| Document | Relationship |
|----------|-------------|
| `docs/ARCH_REVIEW.md` | Architecture review — BVH is part of the scene intersection pipeline |
| `docs/renderer_arch.md` | Renderer architecture — BVH sits at the intersection level |
| `docs/mesh-design.md` | Mesh design — triangle storage and vertex layout |

### External references

| Reference | Relevance |
|-----------|-----------|
| Embree BVH node layout: `kernels/bvh/bvh_node_aabb.h`, `bvh_node_qaabb.h` | Quantized AABB format, node size computation |
| Embree traversal: `kernels/bvh/node_intersector1.h`, `bvh_traverser1.h` | FMA slab test, stack management, ISA dispatch |
| Embree triangle: `kernels/geometry/trianglei.h`, `triangle_intersector_moeller.h` | SoA triangle format, precomputed edges, SIMD MT |
| tinybvh: `tiny_bvh.h` (single-header) | BVH4_CPU/BVH8_CPU struct layouts, stack compress, BVHTri4Leaf |
| svenstaro/bvh: `src/bvh/bvh.rs`, `src/aabb/aabb.rs` | Entry/exit traversal, consistency checks, FlatBvh |
| Embree SIGGRAPH 2014/2016 papers | Build quality, performance benchmarks, Morton vs SAH |
| Architecture review (v2, 2026-07-23) | Wide-node collapse cost, hit_mask lane construction, specialized W=2/W=4 kernels, co-design of traversal kernel with node format |
| Architecture review (v3, 2026-07-23) | Design constraints (unified Bvh<W>, std::simd, const generics), builder as first-class target, internal specialization, what to keep/change, bottom line |
