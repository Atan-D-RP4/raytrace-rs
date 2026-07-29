# Wavefront Rendering Refactor — Specification Document

**Status:** Draft  
**Target:** `raytrace-rs` rendering system refactor  
**Prerequisites:** SampleStream refactor (complete), `docs/renderer_arch.md`  
**Session reference:** `session-ses_0953.md` (contains full design rationale)

______________________________________________________________________

## Table of Contents

1. [Motivation & Goals](#1-motivation--goals)
2. [Current State Analysis](#2-current-state-analysis)
3. [Phase 0: Scene Type Modernization (LightPrimitive / Primitive)](#3-phase-0-scene-type-modernization)
4. [Phase 1: Integrator Decomposition (process_bounce + GAT)](#4-phase-1-integrator-decomposition)
5. [Phase 2: Const-Generic WavefrontRenderer](#5-phase-2-const-generic-wavefrontrenderer)
6. [Phase 3: renderer_arch.md Integration](#6-phase-3-renderer_archmd-integration)
7. [Phase 4: Batch Optimizations](#7-phase-4-batch-optimizations)
8. [Phase 5: GPU Mapping (rust-gpu)](#8-phase-5-gpu-mapping)
9. [Dependency Order & Merging Strategy](#9-dependency-order--merging-strategy)
10. [Risk & Mitigation](#10-risk--mitigation)

______________________________________________________________________

## 1. Motivation & Goals

### 1.1 Why Wavefront?

The current architecture is a **megakernel path tracer**: one monolithic `Integrator::li()` loop over bounces for a single ray. This is simple and correct but:

- **Prevents SIMD/AVX utilization** on CPU — per-pixel iteration is inherently scalar.
- **Prevents GPU mapping** — the loop structure does not decompose into SIMT-friendly compute shaders.
- **No batch-oriented renderer path** — raster + hybrid renderers from `renderer_arch.md` need per-surface shading, not per-ray `li()`.

Wavefront rendering organizes the render loop into **per-stage kernels over batches of work items**, matching GPU SIMT execution and enabling CPU SIMD auto-vectorization.

### 1.2 Key Design Decisions (from session-ses_0953.md)

| Decision | Rationale |
|----------|-----------|
| **Phase A: Decompose Integrator into stage traits** | Stage decomposition is prerequisite for raster, hybrid, wavefront, and GPU pipelines. Each calls the same `process_bounce()` in different iteration patterns. |
| **Const-generic BATCH size** | `WavefrontRenderer<I, const BATCH: usize>` unifies naive (BATCH=1) and wavefront (BATCH>1) — compiler eliminates compaction/active-mask overhead for BATCH=1. |
| **`type PathState<'a>` GAT on Integrator** | Enables future integrators (volumetric, BDPT) to borrow scene data without breaking `BatchState` storage. |
| **`BounceResult` with `delta_ray` field** | Split paths (Mix material) return the delta child as a separate ray instead of recursing — orchestrator decides handling (recursive for BATCH=1, queue-insertion for BATCH>1). |
| **`LightPrimitive` enum replaces `Arc<dyn Sampleable>`** | Static dispatch for light sampling, pbrt-v4-aligned, already designed in `renderer_arch.md` §2. |

### 1.2 What Does NOT Change

- `Material`, `Bsdf`, `BsdfScatter` traits — per-element interfaces unchanged.
- `SampleStream`, `SamplerRng` — two-stream architecture stays.
- `Hit`, `SurfaceInteraction`, `MaterialHit` — surface interaction types unchanged.
- `Camera`, `Film` traits — unchanged.
- BVH traversal — unchanged.
- `Color3`, `Ray`, `Direction3` — core math types unchanged.
- `PdfKind`, `EmitterPDF`, `MisHeuristic` — PDF infrastructure unchanged.

______________________________________________________________________

## 2. Current State Analysis

### 2.1 Integrator Trait (Current)

```rust
// src/integrator/mod.rs
pub trait Integrator: Send + Sync {
    fn background(&self, direction: Direction3) -> Color3;
    fn env_map(&self) -> Option<&Arc<EnvironmentMap>>;
    fn background_color(&self) -> Color3;
    fn li<S: SampleStream, R: SamplerRng>(
        &self,
        initial_ray: &mut Ray,
        world: &dyn Intersectable,
        lights: &[Arc<dyn Sampleable>],
        stream: &mut S,
        rng: &mut R,
    ) -> Color3;
}
```

**Problems:**
- `li()` bundles all bounces — cannot call per-bounce in a wavefront loop.
- `lights: &[Arc<dyn Sampleable>]` uses dynamic dispatch on every light sample.
- No per-path state abstraction — Bounce-level state (`prev_bsdf_pdf`, `prev_was_delta`) is local variables in `li_inner()`.
- Stream + rng passed as mutable refs — cannot store per-path in an array without lifetime gymnastics.

### 2.2 PathTracingIntegrator (Current)

`li_inner()` (~370 lines, lines 180–521 in `path_tracer.rs`) contains:
- Emission with MIS weight
- NEE (Next Event Estimation) with shadow ray
- Russian Roulette termination
- Material scatter + MIS continuation sampling
- Split path recursive handling

All five concerns are interleaved in one loop body. Extraction into `process_bounce()` makes each callable independently.

### 2.3 Renderer Trait (Current)

```rust
pub trait Renderer<W, C, F>: Send + Sync { ... }
pub struct CpuRenderer<I: Integrator> { ... }
```

`CpuRenderer` is per-pixel-per-sample. No batch concept exists.

### 2.4 Scene Type (Current)

```rust
pub struct Scene {
    objects: Vec<Arc<dyn Intersectable>>,
    important_objects: Vec<Arc<dyn Sampleable>>,  // ← dynamic dispatch
}
```

**Problems:**
- `Arc<dyn Sampleable>` forces vtable dispatch per light sample.
- No type-level distinction between emitters and geometry.
- `renderer_arch.md` §2 already specifies `LightPrimitive` as replacement.

______________________________________________________________________

## 3. Phase 0: Scene Type Modernization

**Goal:** Replace `Arc<dyn Sampleable>` with `LightPrimitive` enum for zero-cost static dispatch.

### 3.1 New Types

```rust
// src/primitive.rs (new file)
use crate::shape::ShapeObject;
use crate::material::Material;
use crate::hittable::{Intersectable, Sampleable, Bounded};
use crate::shape::{SphereShape, QuadShape, MovingSphereShape};

pub enum Primitive {
    Sphere(ShapeObject<SphereShape, Arc<Material>>),
    Quad(ShapeObject<QuadShape, Arc<Material>>),
    MovingSphere(ShapeObject<MovingSphereShape, Arc<Material>>),
    Custom(Arc<dyn Intersectable>),
}

impl Intersectable for Primitive { /* match-delegate */ }
impl Bounded for Primitive { /* match-delegate */ }

pub struct LightPrimitive(Primitive);

// Only shapes known to be sampleable get From impls
impl From<ShapeObject<SphereShape, Arc<Material>>> for LightPrimitive { ... }
impl From<ShapeObject<QuadShape, Arc<Material>>> for LightPrimitive { ... }
impl From<ShapeObject<MovingSphereShape, Arc<Material>>> for LightPrimitive { ... }

impl Sampleable for LightPrimitive { /* match-delegate */ }
impl Intersectable for LightPrimitive { /* delegate via inner Primitive */ }
```

### 3.2 Scene Changes

```rust
// src/scene.rs
pub struct Scene {
    objects: Vec<Primitive>,              // replaces Vec<Arc<dyn Intersectable>>
    lights: Vec<LightPrimitive>,           // replaces Vec<Arc<dyn Sampleable>>
    env_map: Option<Arc<EnvironmentMap>>,
}
```

### 3.3 Integrator Signature Change

```rust
// Was: lights: &[Arc<dyn Sampleable>]
// Becomes:
lights: &[LightPrimitive],
```

**Breaking change.** All call sites update. But `LightPrimitive` implements `Sampleable`, so existing `EmitterPDF` etc. work via match-delegation.

### 3.4 EmitterPDF Update

```rust
// src/pdf.rs — EmitterPDF stores &[LightPrimitive] instead of &[Arc<dyn Sampleable>]
impl<'a> EmitterPDF<'a> {
    pub fn new(objects: &'a [LightPrimitive], origin: Point3, time: f32) -> Self { ... }
}
```

**Files affected:**
| File | Change |
|------|--------|
| `src/primitive.rs` | **New** — Primitive + LightPrimitive enums |
| `src/scene.rs` | Replace `Vec<Arc<dyn Sampleable>>` → `Vec<LightPrimitive>` |
| `src/integrator/path_tracer.rs` | `lights` param type change |
| `src/integrator/mod.rs` | Trait signature change |
| `src/pdf.rs` | EmitterPDF generic over `&[LightPrimitive]` |
| `src/renderer/mod.rs` | Scene type in trait |
| `src/renderer/cpu.rs` | Light iteration type change |
| `src/main.rs` | Scene consumption updated |

### 3.5 File Organization

New files:
- `src/primitive.rs` — Primitive + LightPrimitive
- `src/gbuffer.rs` — GBuffer type (from renderer_arch.md)

Updated exports in `src/lib.rs`:
```rust
pub mod primitive;  // new
pub mod gbuffer;    // new (Phase 3)
```

______________________________________________________________________

## 4. Phase 1: Integrator Decomposition

**Goal:** Extract `process_bounce()` as the universal per-bounce stage primitive.  
Add `type PathState<'a>` GAT for per-integrator state.  
Replace `lights: &[Arc<dyn Sampleable>]` with `lights: &[LightPrimitive]`.

### 4.1 New Core Types

```rust
// src/integrator/mod.rs (new types, could live here or a new submodule)

/// Per-path state carried by the renderer across bounces.
pub struct PathState {
    pub throughput: Color3,
    pub accumulated_color: Color3,
    pub prev_bsdf_pdf: f32,
    pub prev_was_delta: bool,
    pub pixel: (u32, u32),
}

/// Result of one bounce operation.
pub struct BounceResult {
    /// Direct contribution: emission + NEE (already MIS-weighted).
    pub contribution: Color3,
    /// Next ray for non-delta continuation, or None if path terminates.
    pub next_ray: Option<Ray>,
    /// Throughput update for the continuation path (f_cos from scatter).
    pub continuation_throughput: Color3,
    /// Extra delta ray from Split variant, or None.
    pub delta_ray: Option<Ray>,
    /// Throughput for the delta child (delta_f_cos from Split).
    pub delta_throughput: Color3,
    /// BSDF mixture PDF for next bounce's emission MIS weight.
    pub prev_bsdf_pdf: f32,
    /// Whether the scatter was a delta.
    pub was_delta: bool,
}
```

### 4.2 New Integrator Trait

```rust
pub trait Integrator: Send + Sync {
    /// Per-path state. GAT enables integrators that borrow scene data.
    type PathState<'a>: Send + Sync where Self: 'a;

    fn max_depth(&self) -> u32;

    fn init_state<'a>(&self, pixel: (u32, u32)) -> Self::PathState<'a>;

    /// One bounce of path tracing — the universal stage primitive.
    /// Does NOT do: intersection (renderer does), background (renderer calls eval_background)
    fn process_bounce<'a>(
        &self,
        ray: &Ray,
        hit: &MaterialHit<'a>,
        world: &dyn Intersectable,
        lights: &[LightPrimitive],
        state: &mut Self::PathState<'a>,
        bounce: u32,
        rng: &mut impl SamplerRng,
    ) -> BounceResult;

    /// Evaluate background radiance for a miss ray, including MIS weight.
    fn eval_background<'a>(
        &self,
        direction: Direction3,
        ray_time: f32,
        state: &Self::PathState<'a>,
    ) -> Color3;
}
```

**Key differences from current trait:**
- `type PathState<'a>` GAT — replaces local variables `prev_bsdf_pdf`, `prev_was_delta`, `accumulated_color`, `accumulated_attenuation`.
- `process_bounce()` — one bounce, not one path. Takes `&Ray` (immutable), returns `BounceResult` with next ray.
- `eval_background()` — separate method so wavefront renderer can call it on miss paths.
- No `li()` -> `li()` is now a free function or default method that wraps `process_bounce` in a per-pixel loop.
- `lights: &[LightPrimitive]` (from Phase 0).
- No `stream` parameter — correlated 2D samples are NOT used per-bounce (the stream is for camera/lens). The material scatter uses only the RNG.

### 4.3 PathTracingIntegrator::process_bounce

Extracted from current `li_inner()` lines 188–412. The body is:

```
1. Emission with MIS weight (moved from li_inner lines 209-224)
2. Russian Roulette (lines 233-311)
3. NEE (lines 242-298)
4. Scatter + MIS continuation (lines 313-416)
```

Return `BounceResult` instead of mutating local state and looping.

### 4.4 PathTracingIntegrator::li (Backward-compat wrapper)

```rust
impl PathTracingIntegrator {
    /// Full-path entry point. Calls process_bounce in a per-pixel loop.
    /// Equivalent to the old li() — for backward compat or naive renderers.
    pub fn li(
        &self,
        ray: &mut Ray,
        world: &dyn Intersectable,
        lights: &[LightPrimitive],
        rng: &mut impl SamplerRng,
    ) -> Color3 {
        let pixel = (0, 0);
        let mut state = self.init_state(pixel);
        let mut current_ray = *ray;

        for bounce in 0..self.max_depth() {
            match world.intersect(&current_ray, Interval::from(0.001, f32::INFINITY)) {
                None => {
                    return state.accumulated_color
                         + self.eval_background(
                             current_ray.direction.normalize(),
                             current_ray.time,
                             &state,
                         );
                }
                Some(hit) => {
                    let result = self.process_bounce(
                        &current_ray, &hit, world, lights, &mut state, bounce, rng,
                    );
                    state.accumulated_color += result.contribution;
                    state.throughput *= result.continuation_throughput;
                    state.prev_bsdf_pdf = result.prev_bsdf_pdf;
                    state.prev_was_delta = result.was_delta;

                    // Handle Split: trace delta child recursively (BATCH=1 mode)
                    if let Some(delta_ray) = result.delta_ray {
                        let delta_li = self.trace_delta(
                            &delta_ray, world, lights, rng,
                            self.max_depth().saturating_sub(bounce + 1).min(SPLIT_MAX_DEPTH),
                        );
                        state.accumulated_color += state.throughput
                            * result.delta_throughput * delta_li;
                    }

                    match result.next_ray {
                        Some(next) => current_ray = next,
                        None => return state.accumulated_color,
                    }
                }
            }
        }
        state.accumulated_color
    }
}
```

**Removed from trait:** The `background()`, `env_map()`, `background_color()` convenience methods are no longer on the trait. `eval_background()` handles all miss-path logic including MIS. The integrator stores its env_map and background internally.

### 4.5 CpuRenderer Update (Phase 1)

`CpuRenderer` continues to call `li()` per-pixel (unchanged iteration pattern).  
The `Integrator` trait change is source-compatible: `CpuRenderer<I: Integrator>` still works, `li()` still exists on `PathTracingIntegrator`.

```rust
// cpu.rs — minimal change:
// The integrator.li() call stays the same, just the types change.
let radiance = self.integrator.li(&mut cam_ray.ray, world, lights, &mut rng);
// 'lights' is now &[LightPrimitive] instead of &[Arc<dyn Sampleable>]
```

### 4.6 Files Affected (Phase 1)

| File | Change |
|------|--------|
| `src/integrator/mod.rs` | Rewrite trait — add GAT, `process_bounce`, `eval_background`, remove `li()` from trait |
| `src/integrator/path_tracer.rs` | Extract `process_bounce` from `li_inner`, add `li()` as wrapper, add `eval_background` |
| `src/renderer/cpu.rs` | Adjust for lights type change, remove stream param |
| `src/main.rs` | Adjust for changed integrator construction |
| `src/lib.rs` | May add new submodule exports |

______________________________________________________________________

## 5. Phase 2: Const-Generic WavefrontRenderer

**Goal:** `WavefrontRenderer<I, const BATCH: usize>` that unifies naive (BATCH=1) and wavefront (BATCH>1) rendering.

### 5.1 BatchState

```rust
/// Per-batch path state, sized by const generic.
/// With BATCH=1, this is just one path — equivalent to the current per-pixel state.
pub struct BatchState<'a, I: Integrator, const BATCH: usize> {
    pub rays: [Option<Ray>; BATCH],
    pub hits: [Option<MaterialHit<'a>>; BATCH],
    pub states: [I::PathState<'a>; BATCH],
    pub rngs: [HashRng; BATCH],
    pub throughputs: [Color3; BATCH],
    pub contributions: [Color3; BATCH],
    pub active: [bool; BATCH],
    pub pixels: [(u32, u32); BATCH],
    pub count: usize,  // number of active paths (≤ BATCH)
}
```

**Lifetime issue with GAT:** `I::PathState<'a>` has lifetime `'a` tied to the scene. When `MaterialHit<'a>` is returned from intersection, `process_bounce` takes `&MaterialHit<'a>` and the state borrows from scene materials. This is compatible because `BatchState` is created fresh per sample pass and the scene outlives each pass.

**Sampling:** Each slot gets its own `HashRng` (per-pixel seeded, independent). No `SampleStream` per slot — correlated 2D samples (camera AA, lens) are handled at ray generation time outside the bounce loop. The `stream` from the current architecture is only needed once per pixel per sample (at camera ray generation), not per-bounce.

### 5.2 WavefrontRenderer

```rust
pub struct WavefrontRenderer<I: Integrator, const BATCH: usize> {
    integrator: I,
    samples_per_pixel: u32,
}

// Renderer trait implementation
impl<I: Integrator, const BATCH: usize> Renderer<W, C, F> for WavefrontRenderer<I, BATCH>
where
    W: Intersectable,
    C: Camera,
    F: Film,
{
    fn render(&self, camera, film, scene, framebuffer) { ... }
}
```

### 5.3 Render Loop

```rust
fn render_pass(
    &self,
    camera: &C,
    film: &mut F,
    world: &W,
    lights: &[LightPrimitive],
    sample_idx: u32,
) {
    let (width, height) = camera.image_resolution();
    let total_pixels = (width * height) as usize;
    let batch_count = total_pixels.div_ceil(BATCH);

    // Parallel over batches
    (0..batch_count).into_par_iter().for_each(|batch_idx| {
        let start = batch_idx * BATCH;
        let end = (start + BATCH).min(total_pixels);
        let count = end - start;

        let mut batch = BatchState::<I, BATCH>::new();

        // Stage 0: Generate primary rays for this batch
        for i in 0..count {
            let px = start + i;
            let x = (px % width as usize) as u32;
            let y = (px / width as usize) as u32;
            batch.pixels[i] = (x, y);
            batch.active[i] = true;
            batch.states[i] = self.integrator.init_state((x, y));
            batch.rngs[i] = HashRng::for_pixel(x as i32, y as i32, sample_idx);
            // Camera ray generation with per-pixel stream
            let (u, v) = camera_sample(x, y, sample_idx);
            batch.rays[i] = camera.generate_ray_from_pixel(x, y, u, v);
        }
        batch.count = count;

        // Bounce loop
        for bounce in 0..self.integrator.max_depth() {
            self.trace_batch_bounce(&mut batch, world, lights, bounce);
            if batch.count == 0 { break; }
        }

        // Flush to film
        for i in 0..batch.count {
            let (x, y) = batch.pixels[i];
            let color = batch.contributions[i];
            if color.into_inner().is_finite() {
                film.add_sample(x, y, color);
            }
        }
    });
}
```

### 5.4 Batch Bounce

```rust
fn trace_batch_bounce(
    &self,
    batch: &mut BatchState<I, BATCH>,
    world: &W,
    lights: &[LightPrimitive],
    bounce: u32,
) {
    // Stage 1: Intersect all active rays
    for i in 0..batch.count {
        if !batch.active[i] { continue; }
        batch.hits[i] = world.intersect(
            &batch.rays[i], Interval::from(0.001, f32::INFINITY),
        );
    }

    // Stage 2: Shade all hits
    for i in 0..batch.count {
        if !batch.active[i] { continue; }
        match batch.hits[i].take() {
            None => {
                // Miss — accumulate background (MIS-weighted)
                batch.contributions[i] += self.integrator.eval_background(
                    batch.rays[i].direction.normalize(),
                    batch.rays[i].time,
                    &batch.states[i],
                );
                batch.active[i] = false;
            }
            Some(hit) => {
                let result = self.integrator.process_bounce(
                    &batch.rays[i], &hit, world, lights,
                    &mut batch.states[i], bounce, &mut batch.rngs[i],
                );
                batch.contributions[i] += result.contribution;
                batch.throughputs[i] *= result.continuation_throughput;

                // Queue delta child at current depth (not recursive)
                if let Some(delta_ray) = result.delta_ray {
                    batch.queue_extra_ray(i, delta_ray, result.delta_throughput);
                }

                // Update ray, terminate if None
                batch.rays[i] = result.next_ray;
                batch.active[i] = result.next_ray.is_some();
            }
        }
    }

    // Stage 3: Compaction (compiles away for BATCH=1)
    if BATCH > 1 {
        batch.compact();
    }
}
```

### 5.5 BATCH=1 → Old CpuRenderer Behavior

With `const BATCH: usize = 1`:

- `BatchState` is `[Option<Ray>; 1]`, `[bool; 1]` — single slot, stack-allocated.
- `compact()` compiles away (dead code elimination on `if BATCH > 1`).
- The outer batch loop iterates `total_pixels` times (each batch = one pixel).
- The bounce loop iterates `max_depth` times per pixel.
- **Result:** compiler emits a tight per-pixel scalar loop identical to the current `CpuRenderer`.

```rust
// Compile-time type alias for clarity
pub type CpuRenderer<I> = WavefrontRenderer<I, 1>;
```

### 5.6 Files Affected (Phase 2)

| File | Change |
|------|--------|
| `src/renderer/wavefront.rs` | **New** — `WavefrontRenderer`, `BatchState` |
| `src/renderer/mod.rs` | Add `pub mod wavefront`, re-export `WavefrontRenderer` |
| `src/renderer/cpu.rs` | Keep or replace with `type CpuRenderer<I> = WavefrontRenderer<I, 1>` |
| `src/main.rs` | Import new renderer type |
| `src/integrator/mod.rs` | Ensure `PathState: Send + Sync + Clone` (required for array storage) |

______________________________________________________________________

## 6. Phase 3: renderer_arch.md Integration

**Goal:** Implement VisibilityGenerator, GBuffer, RasterRenderer, HybridRenderer from `docs/renderer_arch.md`, now using the decomposed `process_bounce()` primitive.

### 6.1 GBuffer

```rust
// src/gbuffer.rs
pub struct GBuffer<'a> {
    width: u32,
    height: u32,
    pixels: Vec<Option<SurfaceInteraction<'a>>>,
}

impl<'a> GBuffer<'a> {
    pub fn new((width, height): (u32, u32)) -> Self;
    pub fn get(&self, x: u32, y: u32) -> Option<&SurfaceInteraction<'a>>;
    pub fn set(&mut self, x: u32, y: u32, si: SurfaceInteraction<'a>);
    pub fn resolution(&self) -> (u32, u32);
}
```

### 6.2 VisibilityGenerator

```rust
// src/visibility/mod.rs
pub trait VisibilityGenerator<W: Intersectable>: Send + Sync {
    fn generate_visibility(
        &mut self,
        world: &W,
        camera: &impl Camera,
        gbuffer: &mut GBuffer,
    );
}

// Two implementations:
// - RayVisibilityGenerator — uses camera.generate_ray() + world.intersect()
// - TriangleRasterizer — uses view-projection rasterization (future)
```

### 6.3 RasterRenderer

```rust
pub struct RasterRenderer<V> {
    visibility: V,
}

// render():
//   1. visibility.generate_visibility() → GBuffer
//   2. For each GBuffer pixel, call integrator.process_bounce() once (bounce 0 only)
//      Only NEE + emission — no scatter continuation
//   3. Accumulate to film
```

**Key insight:** RasterRenderer calls the same `process_bounce()` as the wavefront renderer, but only once per visible pixel (bounce 0) and with no continuation. The rasterizer handles primary visibility; the integrator handles direct lighting only.

### 6.4 HybridRenderer

```rust
pub struct HybridRenderer<V, I> {
    visibility: V,
    integrator: I,
    samples_per_pixel: u32,
}

// render():
//   1. visibility.generate_visibility() → GBuffer (primary)
//   2. For each pixel needing secondary effects (reflections, GI):
//      a. Generate secondary ray from GBuffer hit
//      b. Call process_bounce() in wavefront mode over all secondary rays
//   3. Composite primary + secondary into film
```

### 6.5 Files Affected (Phase 3)

| File | Change |
|------|--------|
| `src/gbuffer.rs` | **New** — `GBuffer<'a>` |
| `src/visibility/mod.rs` | **New** — `VisibilityGenerator` trait |
| `src/visibility/ray.rs` | **New** — `RayVisibilityGenerator` |
| `src/visibility/triangle.rs` | **New** — `TriangleRasterizer` (deferred, simple stub) |
| `src/renderer/raster.rs` | **New** — `RasterRenderer` |
| `src/renderer/hybrid.rs` | **New** — `HybridRenderer` |
| `src/lib.rs` | Add modules |
| `src/renderer/mod.rs` | Re-export new renderers |

______________________________________________________________________

## 7. Phase 4: Batch Optimizations

**Goal:** Improve wavefront performance for BATCH > 1 with optional optimizations.

### 7.1 Direction Sorting Before Intersection

Group rays by hemisphere octant before the intersect stage to improve BVH cache coherence:

```rust
// In trace_batch_bounce:
// 1. Partition batch.rays into 8 direction octants (in-place, stable)
// 2. Intersect in sorted order
// 3. Un-sort or maintain mapping back to pixel indices
```

**Design decision:** This is an optional stage gated by `const SORT: bool`:

```rust
pub struct WavefrontRenderer<I, const BATCH: usize, const SORT: bool = false> { ... }
```

When `SORT = false` (default), the sort step compiles away. This follows the `BATCH=1` pattern of making optimization choices at compile time.

### 7.2 Material Sorting Before Shading

Group active paths by material type before calling `process_bounce`, improving BSDF cache locality:

- Assign each path a material ID after intersection
- Sort by material ID before the shade stage
- Collect back to pixel order for compaction

Also gated by `const SORT_MATERIAL: bool`. Can share the `SORT` const or be independent.

### 7.3 Compaction Strategies

Current plan: simple prefix-sum compaction (inline, O(BATCH)). For BATCH up to ~256, this is fine. Options for larger batches:

- **Stream compaction** (GPU-style): prefix sum → scatter to new array.
- **Active-mask iteration** (alternative): keep paths in place, iterate only over active mask bits via `iterate_active_indices()` which uses `BATCH`-sized bitmask.

Start with simple compaction. Optimize when profiling shows it as a bottleneck.

### 7.4 Delta Child Queue

In BATCH > 1 mode, Split's delta child cannot be processed recursively (it would break the wavefront). Instead:

```rust
impl<I: Integrator, const BATCH: usize> BatchState<I, BATCH> {
    fn queue_extra_ray(&mut self, idx: usize, ray: Ray, throughput: Color3) {
        // If there's room in the batch, add the delta child as a new active slot
        if self.count < BATCH {
            self.rays[self.count] = Some(ray);
            self.active[self.count] = true;
            self.throughputs[self.count] = self.throughputs[idx] * throughput;
            self.pixels[self.count] = self.pixels[idx];
            self.states[self.count] = self.states[idx].clone_for_split();
            self.count += 1;
        }
        // If batch is full, process delta recursively (fallback)
    }
}
```

For BATCH=1, delta children are handled in `li()` recursively (current behavior). The `queue_extra_ray` method is conditionally compiled via `if BATCH > 1`.

______________________________________________________________________

## 8. Phase 5: GPU Mapping

**Goal:** Document how the decomposed pipeline maps to GPU compute shaders (rust-gpu).

### 8.1 Kernel Mapping

| Wavefront Stage | GPU Compute Shader |
|----------------|-------------------|
| Ray Generation | `gen_rays()` — one thread per pixel, writes `Ray` to SSBO |
| Intersection | `intersect()` — one thread per active ray, traverses BVH, writes `MaterialHit` to SSBO |
| Shading (process_bounce) | `shade()` — one thread per active hit, reads `MaterialHit`, writes next `Ray` + contribution |
| Compaction | `compact()` — prefix sum, scatter active rays |

### 8.2 Data Layout

```rust
// GPU buffer layout (SSBO)
struct GpuWavefrontState {
    rays: [vec4; MAX_PATHS * 2],       // origin.xyz + direction.xyz + time + pad
    hits: [GpuHit; MAX_PATHS],
    states: [GpuPathState; MAX_PATHS],  // throughput, prev_bsdf_pdf, etc.
    active: [u32; MAX_PATHS / 32],      // bitmask
    contributions: [vec4; MAX_PATHS],
}
```

### 8.3 Trait Constraints for GPU

The decomposed `Integrator::process_bounce()` is already GPU-friendly:
- Pure function of inputs → outputs
- No recursive calls (Split returns delta_ray for the orchestrator)
- No `dyn Material` inside the integrator — `Material` is an enum (already is)
- `SamplerRng` is pure hash → no mutable state required per path

**Requirement:** `BounceResult`, `PathState` must be `#[repr(C)]` for GPU layout compatibility.

### 8.4 Requirements for rust-gpu

- `Integrator` implementation must be monomorphizable.
- No `dyn Intersectable` — world type must be concrete (e.g., `Bvh<4>`).
- `Material` enum is already good (match-delegation, no `dyn Bsdf` in hot path).
- `LightPrimitive` enum already good (match-delegation).

______________________________________________________________________

## 9. Dependency Order & Merging Strategy

### 9.1 Implementation Order

```
Phase 0: Scene types (LightPrimitive)
  │
  ▼
Phase 1: Integrator decomposition (process_bounce, GAT)
  │
  ├────────────────┬────────────────┐
  ▼                ▼                ▼
Phase 2a:         Phase 2b:       Phase 3:
Wavefront         CpuRenderer     GBuffer + VisGen
(BATCH≠1)         keeps working    (additive, no
  │               (BATCH=1)        existing code
  ▼                                touched)
Phase 4:                            │
Batch Optimizations                 ▼
                              RasterRenderer +
                              HybridRenderer
```

Each phase is independently mergable:

| PR | Contents | Breaks old? | Test after |
|----|----------|------------|------------|
| PR 0 | `primitive.rs`, LightPrimitive, Scene changes | Yes — scene type change | All scenes render correctly |
| PR 1 | New Integrator trait, process_bounce, PathState GAT | Yes — trait change | Integrator smoke tests pass |
| PR 2 | WavefrontRenderer + BatchState | No — additive | Renders match CpuRenderer pixel-for-pixel |
| PR 3 | GBuffer + VisibilityGenerator + RayVisibilityGenerator | No — additive | GBuffer fills correctly |
| PR 4 | RasterRenderer + HybridRenderer | No — additive | Raster output matches path tracer (simple scenes) |
| PR 5 | Direction/material sorting | No — additive, gated by const bool | Performance benchmarks |

### 9.2 Rollback Strategy

Each PR preserves the old `li()` as a thin wrapper (for Phase 1), so `CpuRenderer` keeps working throughout. `WavefrontRenderer<_, 1>` == `CpuRenderer` after Phase 2.

### 9.3 Testing Strategy

**Per-PR tests:**

| Phase | Test |
|-------|------|
| 0 | `LightPrimitive` match-delegation smoke test (pdf_value, sample_light on known shapes) |
| 0 | Scene construction with new types compiles and produces same object counts |
| 1 | `process_bounce()` produces same `BounceResult` as old `li_inner()` at same inputs (regression test) |
| 1 | `PathTracingIntegrator::li()` wrapper matches old `li()` output pixel-for-pixel on 4×4 test scene |
| 2 | `WavefrontRenderer<_, 1>` produces identical output to `CpuRenderer` (4×4 + small scenes) |
| 2 | `WavefrontRenderer<_, 4>` produces correct output (no off-by-one in batch iteration) |
| 2 | Compaction correctness: paths terminate at right time, contributions accumulate correctly |
| 3 | GBuffer fill + read round-trip |
| 3 | RayVisibilityGenerator produces same hits as direct world.intersect() |
| 4 | RasterRenderer output matches PathTracingIntegrator (diffuse-only scenes, bounce=1) |
| 5 | Direction sorting doesn't change output (only performance) |

**Integration test:**
- Existing `render_4x4_minimal_scene` and `integrator_smoke_test` continue to pass.
- New: `wavefront_vs_cpu_identical` — renders a 4×4 scene with both CpuRenderer and WavefrontRenderer<_, 1>, asserts per-pixel match.

______________________________________________________________________

## 10. Risk & Mitigation

### 10.1 GAT Lifetime Complexity

**Risk:** `type PathState<'a>` with `MaterialHit<'a>` creates complex lifetime relationships in `BatchState`.
**Mitigation:** Start with no borrows (`PathState` owns all data). Future integrators (volumetric) can add GAT lifetimes when needed. The GAT is on the trait but the concrete impl for `PathTracingIntegrator` is `'static` — simplest path.

### 10.2 Split Delta Child in Wavefront Mode

**Risk:** Delta child from Split cannot be processed recursively in wavefront mode. Queueing may overflow BATCH size.
**Mitigation:** 
- BATCH=1: recursive processing (exactly like current code).
- BATCH>1: queue delta child as a new slot. If batch full, trace delta inline (recursive fallback).
- The `SPLIT_MAX_DEPTH = 5` cap limits delta cascade regardless.

### 10.3 Perf Regression for BATCH=1

**Risk:** The generic wavefront loop may not optimize as well as the dedicated `CpuRenderer` loop.
**Mitigation:** 
- Const generics enable aggressive monomorphization.
- `if BATCH > 1` branches are constant-folded to death-code-eliminated.
- `#[inline(always)]` on critical path functions.
- Benchmark comparison: `WavefrontRenderer<_, 1>` vs old `CpuRenderer` head-to-head.

### 10.4 Adaptive Sampling Interaction

**Risk:** Current adaptive sampling logic runs per-sample-pass over the full film. Wavefront batches process pixels out-of-order, which is fine for accumulation but needs coordination with convergence.
**Mitigation:** Convergence check happens at the film level after each sample pass (as it does now). The per-batch processing order doesn't affect the final accumulation — each pixel accumulates independently. The convergence mask just skips converged pixels during ray generation.

### 10.5 Scene Type Breaking Change (Phase 0)

**Risk:** `LightPrimitive` replaces `Arc<dyn Sampleable>` across the entire codebase.
**Mitigation:** 
- `LightPrimitive` implements `Sampleable` — existing code that only calls `Sampleable` methods works unchanged.
- `From` impls for all current shape types make migration ergonomic.
- This change is front-loaded (Phase 0) so Phases 1-4 are built on the correct types.

______________________________________________________________________

## Appendix A: Type Summary

### New Types

| Type | Location | Description |
|------|----------|-------------|
| `Primitive` | `src/primitive.rs` | Enum over geometry types (Sphere, Quad, MovingSphere, Custom) |
| `LightPrimitive` | `src/primitive.rs` | Newtype over Primitive; implements Sampleable |
| `PathState` | `src/integrator/mod.rs` | Per-path bounce state (throughput, accumulated color, PDFs, pixel) |
| `BounceResult` | `src/integrator/mod.rs` | One bounce output (contribution, next_ray, delta_ray, PDFs) |
| `BatchState<'a, I, BATCH>` | `src/renderer/wavefront.rs` | Arrays of per-path state for one batch |
| `WavefrontRenderer<I, BATCH>` | `src/renderer/wavefront.rs` | Const-generic renderer (BATCH=1 → naive, BATCH>1 → wavefront) |
| `GBuffer<'a>` | `src/gbuffer.rs` | 2D array of SurfaceInteraction for primary visibility |
| `RasterRenderer<V>` | `src/renderer/raster.rs` | Pure rasterization renderer |
| `HybridRenderer<V, I>` | `src/renderer/hybrid.rs` | Raster primary + path-traced secondary |

### Modified Types

| Type | Change |
|------|--------|
| `Integrator` trait | Add GAT `PathState<'a>`, add `process_bounce()`, `eval_background()`, remove `li()` from trait |
| `PathTracingIntegrator` | `li()` becomes thin wrapper over `process_bounce()` |
| `CpuRenderer` | Becomes `type CpuRenderer<I> = WavefrontRenderer<I, 1>` or keeps independent |
| `Scene` | `important_objects: Vec<Arc<dyn Sampleable>>` → `lights: Vec<LightPrimitive>` |
| `EmitterPDF` | Stores `&[LightPrimitive]` instead of `&[Arc<dyn Sampleable>]` |
| `Renderer::render()` | Scene type changes to `(&W, &[LightPrimitive])` |

### Removed Types

| Type | Replacement |
|------|-------------|
| `Arc<dyn Sampleable>` (in scene/integrator params) | `LightPrimitive` |

______________________________________________________________________

## Appendix B: Cross-Reference to renderer_arch.md

| renderer_arch.md § | This Spec |
|--------------------|-----------|
| §0 Problem Statement | Addressed by Phases 0-3 |
| §1 Current State | Updated in §2 of this spec |
| §2 Target Architecture: `Primitive`/`LightPrimitive` | Phase 0 |
| §2 Target Architecture: `VisibilityGenerator` | Phase 3 |
| §2 Target Architecture: `GBuffer` | Phase 3 |
| §2 Target Architecture: `RasterRenderer` | Phase 3 |
| §2 Target Architecture: `HybridRenderer` | Phase 3 |
| §2 Target Architecture: `RasterCamera` | Phase 3 (deferred) |
| §6 Evolution Path (steps 1-8) | Phase 3 maps to steps 3-8 |
| §10 Unified Dependency Order | §9 of this spec |
