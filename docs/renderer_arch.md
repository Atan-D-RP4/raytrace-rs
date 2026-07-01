# Hybrid Rendering Architecture

## Decomposition of the Renderer into Visibility + Integration + Composition

______________________________________________________________________

## Changelog

- **v1 (2026-06-27)** — Initial design. Decomposes monolithic `Renderer` into three
  independent concerns: visibility generation, light transport estimation, and film
  composition. Enables rasterizer and hybrid renderer without breaking the existing
  path tracer.
- **v2 (2026-06-28)** — Three-document audit with `docs/denoiser.md` and
  `docs/adaptive-sampling.md`. Updated Film trait to acknowledge `apply_denoiser()`
  extension. Updated CpuRenderer to show denoiser call. Fixed "What Does NOT Change"
  section. Added cross-document integration table (§10) with GBuffer vs
  DenoiserFeatures naming distinction and unified dependency order.
- **v3 (2026-06-28)** — Four-document audit with `docs/samplestream-refactor.md`.
  Fixed `add_sample` signature (removed phantom `weight` param). Added SampleStream
  note to Integrator section. Added `docs/samplestream-refactor.md` cross-reference
  table. Updated unified dependency order to include SampleStream steps 1-4.
- **v4 (2026-06-28)** — Documented `weight: f64` parameter on `add_sample` as a
  deferred upgrade with five use cases: MIS, denoiser confidence, adaptive sampling,
  progressive rendering, and tile merging.
- **v5 (2026-06-28)** — Static dispatch audit. Added `SampleableEnum` with `From` impls
  (replaces `Arc<dyn Sampleable>`). Made `Integrator<W, S>` generic over world type
  (replaces `&dyn Intersectable`). Made `VisibilityGenerator<W>` generic. Updated all
  renderer implementations. Updated "Why generics over dyn T" section with three-strategy
  approach (generics, enums, trait-bound enums). Updated unified dependency order.
- **v6 (2026-06-28)** — Applied Niri/Smithay `render_elements!` pattern to all enums:
  **Descriptor → Concrete → Wrapper**. Added `SampleableKind` descriptor. Updated
  "Why generics over dyn T" section with three-layer pattern description.
- **v7 (2026-06-29)** — Updated §10 cross-reference to reflect `DenoiserFeatures` as
  struct-of-arrays (SoA) following `VarianceEstimator` pattern. Documented SoA vs AoS
  rationale for cache-friendly denoiser passes.

______________________________________________________________________

## 0. Problem Statement

The current `Renderer<W, C, F, S>` trait bundles visibility generation, light
transport, and film writing into a single `render()` call. The `CpuRenderer`
implements this as a per-pixel Monte Carlo loop: `camera.generate_ray()` →
`integrator.li()` → `film.add_sample()`.

This prevents:

1. **Rasterization for primary visibility** — rasterization is not a per-pixel
   ray-generation problem. It transforms triangles to clip space, rasterizes
   fragments, and writes surface interactions. This doesn't fit the
   `generate_ray() → li()` loop.

2. **Hybrid rendering** — rasterize primary visibility, then path-trace secondary
   effects (reflections, GI, shadows). Requires composing two visibility
   mechanisms in one render.

3. **Pluggable visibility backends** — forward rasterizer, deferred rasterizer,
   tile-based rasterizer, ray-traced primary, etc. Each produces the same output
   (`SurfaceInteraction`) through a different mechanism.

The decomposition separates these into independent traits that compose via
generics.

______________________________________________________________________

## 1. Current State

### Traits

```rust
// src/renderer/mod.rs
pub trait Renderer<W, C, F, S>: Send + Sync
where
    W: Intersectable,
    C: Camera,
    F: Film,
    S: Sampler,
{
    fn render(
        &self,
        camera: &C,
        film: &mut F,
        scene: (&W, &[SampleableEnum]),
        framebuffer: Option<SharedFramebuffer>,
    );
    fn resize(&mut self, _width: u32, _height: u32) {}
    fn reset(&mut self) {}
}

// src/integrator/mod.rs
pub trait Integrator<W: Intersectable, S: Sampler>: Send + Sync {
    fn li(
        &self,
        initial_ray: &mut Ray,
        world: &W
        lights: &[SampleableEnum],
        sampler: &mut DimCursor<S>,
    ) -> Color3;
}

// src/camera/mod.rs
pub trait Camera: Send + Sync {
    fn generate_ray(&self, sample: &CameraSampler) -> Option<CameraRay>;
    fn generate_ray_differential(&self, sample: &CameraSampler) -> Option<CameraRay>;
    fn image_resolution(&self) -> (u32, u32);
}

// src/film/mod.rs
pub trait Film: Send + Sync {
    // Might gain a weight: f64 param in the future for MIS, denoiser confidence, adaptive sampling, etc.
    fn add_sample(&mut self, x: u32, y: u32, color: Color3);
    fn merge_tile(&mut self, tile: &FilmTile);
    fn progressive(&self) -> impl Iterator<Item = u8> + '_;
    fn resolution(&self) -> (u32, u32);
    fn reset(&mut self);
    // ... convergence methods
}

// src/hittable.rs
pub trait Intersectable: Send + Sync + Bounded {
    fn intersect<'a>(&'a self, ray: &Ray, ray_t: Interval) -> Option<MaterialHit<'a>>;
}

pub trait Sampleable: Intersectable + Send + Sync {
    fn pdf_value(&self, origin: Vec3, direction: Vec3, time: f64) -> f64;
    fn random_direction(&self, origin: Vec3, u: f64, v: f64, time: f64) -> Vec3;
}
```

#### SampleableEnum (NEW — replaces `Arc<dyn Sampleable>`)

Follows the Niri/Smithay `render_elements!` pattern: **Descriptor → Concrete → Wrapper**.
The wrapper enum delegates the trait via `match`, enabling zero-cost static dispatch.

```rust
// === Descriptor enum (lightweight, Clone+Copy, describes what to build) ===
#[derive(Clone, Copy, Debug)]
pub enum SampleableKind {
    Sphere { center: Point3, radius: f64 },
    Quad { Q: Point3, u: Vec3, v: Vec3 },
    MovingSphere { center: Point3, radius: f64, speed: Vec3 },
}

// === Wrapper enum (delegates Sampleable via match) ===
#[derive(Debug)]
pub enum SampleableEnum {
    Sphere(Sphere),
    Quad(Quad),
    MovingSphere(MovingSphere),
}

impl Sampleable for SampleableEnum {
    fn pdf_value(&self, origin: Vec3, direction: Vec3, time: f64) -> f64 {
        match self {
            Self::Sphere(s) => s.pdf_value(origin, direction, time),
            Self::Quad(q) => q.pdf_value(origin, direction, time),
            Self::MovingSphere(s) => s.pdf_value(origin, direction, time),
        }
    }
    fn random_direction(&self, origin: Vec3, u: f64, v: f64, time: f64) -> Vec3 {
        match self {
            Self::Sphere(s) => s.random_direction(origin, u, v, time),
            Self::Quad(q) => q.random_direction(origin, u, v, time),
            Self::MovingSphere(s) => s.random_direction(origin, u, v, time),
        }
    }
}

// === From impls for ergonomic construction (Niri/Smithay pattern) ===
impl From<Sphere> for SampleableEnum { fn from(s: Sphere) -> Self { Self::Sphere(s) } }
impl From<Quad> for SampleableEnum { fn from(q: Quad) -> Self { Self::Quad(q) } }
impl From<MovingSphere> for SampleableEnum { fn from(s: MovingSphere) -> Self { Self::MovingSphere(s) } }

// === Construction from descriptor ===
impl SampleableEnum {
    pub fn new(kind: &SampleableKind) -> Self {
        match kind {
            SampleableKind::Sphere { center, radius } => Sphere::new(*center, *radius).into(),
            SampleableKind::Quad { Q, u, v } => Quad::new(*Q, *u, *v).into(),
            SampleableKind::MovingSphere { center, radius, speed } =>
                MovingSphere::new(*center, *radius, *speed).into(),
        }
    }
}

// Scene stores Vec<SampleableEnum> instead of Vec<Arc<dyn Sampleable>>
pub struct Scene {
    objects: Vec<IntersectableEnum>,   // also an enum (see CORE_THESIS §2.4)
    lights: Vec<SampleableEnum>,
}
```

**Current state:** `Scene` uses `Vec<Arc<dyn Sampleable>>`.
**Evolution:** Replace with `Vec<SampleableEnum>`. The `From` impls make
construction ergonomic: `scene.add_light(Sphere::new(...))` auto-converts.

### Key Types

```rust
// src/hittable.rs
pub struct SurfaceInteraction<'si> {
    hit: Hit,
    shading_normal: Vec3,
    front_face: bool,
    material: &'si Material,    // lifetime-annotated, not material_id
}

pub struct Hit {
    pub time: f64,
    pub point: Vec3,
    pub mapping_point: Vec3,
    geometric_normal: Vec3,
    pub uv: Option<(f64, f64)>,
}

pub struct MaterialHit<'a> {
    pub hit: Hit,
    pub material: &'a Material,
}

// src/film/tile.rs
pub struct FilmTile {
    pub bounds: [u32; 4],
    pub pixels: Vec<Color3>,
    pub sampled: Vec<bool>,
}

// src/film/rgb.rs
pub struct RgbFilm {
    width: u32,
    height: u32,
    pixels: Vec<Color3>,
    sample_counts: Vec<u32>,
    m_2: Vec<Color3>,           // Welford's online variance
    exposure: f64,
    tone_map: bool,
}
```

### Current CpuRenderer

```rust
// src/renderer/cpu.rs
pub struct CpuRenderer<I, S, Fact>
where
    I: Integrator<S>,
    S: Sampler,
    Fact: SamplerFactory<Sampler = S>,
{
    samples_per_pixel: u32,
    threshold_abs: f64,
    threshold_rel: f64,
    min_samples_before_adapt: u32,
    integrator: I,              // owns the integrator
    sampler_factory: Fact,      // owns the sampler factory
    _phantom: PhantomData<S>,
}
```

The `CpuRenderer` already owns its integrator. The `Renderer` trait's `render()`
method does NOT take an integrator parameter — it's internal to the
implementation.

______________________________________________________________________

## 2. Target Architecture

### The Core Decomposition

Three independent concerns, each a trait:

```
VISIBILITY    — where are the surfaces? (rasterizer OR ray tracer)
INTEGRATION   — what radiance do they produce? (path tracer, direct light, etc.)
COMPOSITION   — how do we combine results? (film writes)
```

Each concern is a trait. Concrete implementations compose them via generics.
No `dyn T`.

### Layer 1: Traits

#### Camera (extended for rasterization)

```rust
/// Ray generation — used by path tracer and hybrid renderer.
pub trait Camera: Send + Sync {
    fn generate_ray(&self, sample: &CameraSampler) -> Option<CameraRay>;
    fn generate_ray_differential(&self, sample: &CameraSampler) -> Option<CameraRay> {
        None
    }
    fn image_resolution(&self) -> (u32, u32);
}

/// Rasterization support — view/projection matrices.
/// Extends Camera, not replaces it.
pub trait RasterCamera: Camera {
    fn view_projection(&self) -> [f64; 16];
    fn view_projection_inverse(&self) -> [f64; 16];
}
```

`PerspectiveCamera` implements both. A pure ray-tracing camera implements only
`Camera`.

**Current state:** `Camera` trait exists. `view_projection()` methods need to be
added to `PerspectiveCamera`.
**Evolution:** Add `RasterCamera` trait. Implement for `PerspectiveCamera`. No
breaking change to existing code.

#### Integrator (stays generic, gains SampleStreamEnum)

```rust
/// Light transport estimation. Given a ray and scene, returns radiance.
pub trait Integrator<W: Intersectable, S: Sampler>: Send + Sync {
    fn li(
        &self,
        initial_ray: &mut Ray,
        world: &W,
        lights: &[SampleableEnum],
        stream: &mut SampleStreamEnum<S>,
    ) -> Color3;
}
```

**Current state:** Generic over `S: Sampler`, takes `&mut DimCursor<S>`.
**Evolution:** Becomes `Integrator<W, S>` (generic over world type). The `DimCursor<S>`
parameter is replaced by `SampleStreamEnum<S>` — a concrete enum that avoids `dyn`
dispatch while enabling future MLT support. See `docs/samplestream-refactor.md`.

#### Film (extended for denoiser)

```rust
pub trait Film: Send + Sync {
    fn add_sample(&mut self, x: u32, y: u32, color: Color3);
    fn merge_tile(&mut self, tile: &FilmTile);
    fn progressive(&self) -> impl Iterator<Item = u8> + '_;
    fn resolution(&self) -> (u32, u32);
    fn reset(&mut self);
    fn apply_denoiser(&mut self);           // NEW — denoiser post-process
    // ... convergence methods
}
```

**Current state:** Trait exists with `add_sample(x, y, color)` (no weight parameter).
`RgbFilm` implements this.
**Evolution:** Add `apply_denoiser()` to the trait (from `docs/denoiser.md` §Phase 1).
Default implementation: no-op. `RgbFilm<D: Denoiser>` overrides with actual denoising.

**Deferred: `weight` parameter on `add_sample`**

The current signature `add_sample(x, y, color)` treats every sample as equally
weighted. A future extension adds an optional `weight: f64` parameter:

```rust
fn add_sample(&mut self, x: u32, y: u32, color: Color3, weight: f64);
```

Use cases for per-sample weights:

| Use Case | How Weight Helps |
|----------|-----------------|
| **MIS (Multiple Importance Sampling)** | When combining direct light + BSDF sampling, each strategy produces samples with different PDFs. The balance/power heuristic assigns weights proportional to 1/PDF. Without weights, MIS samples are treated as equally valid, biasing the result. |
| **Denoiser confidence** | The denoiser could use per-sample weights to know how confident each sample is. Samples near edges or from high-variance strategies get lower weights, improving filter quality. |
| **Adaptive sampling** | Pixels at different convergence states could accumulate with different weights — early samples (high variance) weighted less, later samples (low variance) weighted more. |
| **Progressive rendering** | Temporal accumulation with exponential decay: `weight = alpha^age` gives recent frames more influence than old frames, enabling smooth progressive refinement. |
| **Tile merging** | Samples near tile edges could have different weights for seamless blending across tile boundaries. |

**When to implement:** When any of the above use cases becomes a concrete requirement.
The current unweighted path is correct for uniform-sampling path tracing. The `weight`
parameter is additive — existing `add_sample(x, y, color)` call sites would pass
`weight: 1.0` and behavior is unchanged.

**Cross-reference:** See `docs/denoiser.md` §Architecture Decision and §Phase 1 for
the full `RgbFilm<D>` generic redesign. The denoiser is a generic parameter on Film,
not on Renderer — following the `SamplerFactory` associated-type pattern.

#### VisibilityGenerator (NEW)

```rust
/// Produces primary visibility as SurfaceInteractions in a GBuffer.
/// This is the rasterizer OR the ray tracer's primary visibility pass.
pub trait VisibilityGenerator<W: Intersectable>: Send + Sync {
    fn generate_visibility(
        &mut self,
        world: &W,
        camera: &impl Camera,
        gbuffer: &mut GBuffer,
    );
}
```

**Current state:** Does not exist. Visibility generation is baked into
`CpuRenderer::render()`.
**Evolution:** Extract from `CpuRenderer` into separate implementations.

#### Renderer (simplified)

The `Renderer` trait stays as the top-level orchestration. The `Integrator` is
removed from the trait because it's an implementation detail — each renderer
owns its integrator (if needed).

```rust
pub trait Renderer<W, C, F, S>: Send + Sync
where
    W: Intersectable,
    C: Camera,
    F: Film,
    S: Sampler,
{
    fn render(
        &self,
        camera: &C,
        film: &mut F,
        scene: (&W, &[SampleableEnum]),
        framebuffer: Option<SharedFramebuffer>,
    );

    fn resize(&mut self, _width: u32, _height: u32) {}
    fn reset(&mut self) {}
}
```

**Current state:** `Renderer<W, C, F, S>` — already does NOT take integrator as
parameter. The integrator is internal to `CpuRenderer`.
**Evolution:** No change to the trait signature. The trait is already correct.

### Layer 2: GBuffer (NEW)

```rust
/// Primary visibility buffer. Stores SurfaceInteractions produced by
/// the visibility generator. The shading pass reads this.
pub struct GBuffer<'a> {
    width: u32,
    height: u32,
    pixels: Vec<Option<SurfaceInteraction<'a>>>,
}

impl<'a> GBuffer<'a> {
    pub fn new(dims: (u32, u32)) -> Self { ... }
    pub fn get(&self, x: u32, y: u32) -> Option<&SurfaceInteraction<'a>> { ... }
    pub fn set(&mut self, x: u32, y: u32, si: SurfaceInteraction<'a>) { ... }
    pub fn resolution(&self) -> (u32, u32) { ... }
}
```

The lifetime `'a` is tied to the scene's material storage. On CPU, this is fine —
the GBuffer borrows from the scene. On GPU (Stage 4+), this becomes a buffer of
`material_id` indices, as documented in CORE_THESIS §2.18.

**Current state:** Does not exist.
**Evolution:** Create `GBuffer` type.

### Layer 3: Concrete Implementations

#### PathTracingRenderer (CpuRenderer, unchanged except denoiser call)

```rust
pub struct CpuRenderer<I, S, Fact>
where
    I: Integrator<W, S>,    // now generic over world type
    S: Sampler,
    Fact: SamplerFactory<Sampler = S>,
{
    samples_per_pixel: u32,
    // ... adaptive sampling params ...
    integrator: I,
    sampler_factory: Fact,
    _phantom: PhantomData<S>,
}

impl<W, I, C, F, S, Fact> Renderer<W, C, F, S> for CpuRenderer<I, S, Fact>
where
    W: Intersectable,
    I: Integrator<W, S>,
    C: Camera,
    F: Film,
    S: Sampler,
    Fact: SamplerFactory<Sampler = S>,
{
    fn render(&self, camera, film, scene, framebuffer) {
        let (world, lights) = scene;
        // Current logic: per-pixel Monte Carlo loop
        // For each pixel:
        //   camera.generate_ray() → ray
        //   integrator.li(ray, world, lights) → radiance
        //   film.add_sample(radiance)

        // After sampling loop completes:
        film.apply_denoiser();    // ← NEW (from docs/denoiser.md §6)
    }
}
```

**Current state:** `CpuRenderer` does not call `apply_denoiser()`.
**Evolution:** Add `film.apply_denoiser()` after the sampling loop. This is the only
change to `CpuRenderer` — the rest of the struct and trait impl are unchanged.

**Cross-reference:** See `docs/denoiser.md` §6 for placement details and
`docs/adaptive-sampling.md` §5.3 for the render loop integration.

#### RasterRenderer (NEW)

```rust
pub struct RasterRenderer<V> {
    visibility: V,
}

impl<W, V, C, F, S> Renderer<W, C, F, S> for RasterRenderer<V>
where
    W: Intersectable,
    V: VisibilityGenerator<W>,
    C: RasterCamera,
    F: Film,
    S: Sampler,
{
    fn render(&self, camera, film, scene, framebuffer) {
        let (world, lights) = scene;
        let mut gbuffer = GBuffer::new(camera.image_resolution());
        self.visibility.generate_visibility(world, camera, &mut gbuffer);

        // For each pixel in GBuffer:
        //   si.material().sample(wo, &si) → BsdfSample
        //   Direct lighting: eval BSDF, shadow ray, MIS
        //   film.add_sample(radiance)
    }
}
```

**Current state:** Does not exist.
**Evolution:** New implementation.

#### HybridRenderer (NEW)

```rust
pub struct HybridRenderer<V, I, S> {
    visibility: V,
    integrator: I,
    samples_per_pixel: u32,
    _phantom: PhantomData<S>,
}

impl<W, V, I, C, F, S> Renderer<W, C, F, S> for HybridRenderer<V, I, S>
where
    W: Intersectable,
    V: VisibilityGenerator<W>,
    I: Integrator<W, S>,
    C: RasterCamera,
    F: Film,
    S: Sampler,
{
    fn render(&self, camera, film, scene, framebuffer) {
        let (world, lights) = scene;
        let mut gbuffer = GBuffer::new(camera.image_resolution());

        // 1. Primary visibility via rasterizer
        self.visibility.generate_visibility(world, camera, &mut gbuffer);

        // 2. Secondary effects via path tracer
        //    For each pixel needing reflections/GI/shadows:
        //      Generate secondary ray from GBuffer hit
        //      integrator.li(ray, world, lights) → secondary radiance

        // 3. Composite: GBuffer primary + path-traced secondary
        //    For each pixel:
        //      primary = gbuffer[pixel].material().sample(wo, &si)
        //      secondary = hitbuffer[pixel]  (from integrator)
        //      final = primary + secondary
        //      film.add_sample(final)
    }
}
```

**Current state:** Does not exist.
**Evolution:** New implementation.

#### TriangleRasterizer (NEW, VisibilityGenerator impl)

```rust
pub struct TriangleRasterizer;

impl<W: Intersectable> VisibilityGenerator<W> for TriangleRasterizer {
    fn generate_visibility(
        &mut self,
        world: &W,
        camera: &impl RasterCamera,
        gbuffer: &mut GBuffer,
    ) {
        let vp = camera.view_projection();
        // 1. For each triangle in world:
        //    Transform vertices to clip space via vp
        //    Clip to frustum
        //    Rasterize (scanline or tile-based)
        // 2. For each fragment:
        //    Interpolate vertex attributes (normal, uv)
        //    Write SurfaceInteraction to GBuffer
    }
}
```

**Current state:** Does not exist.
**Evolution:** New implementation.

#### RayVisibilityGenerator (NEW, VisibilityGenerator impl)

```rust
pub struct RayVisibilityGenerator;

impl<W: Intersectable> VisibilityGenerator<W> for RayVisibilityGenerator {
    fn generate_visibility(
        &mut self,
        world: &W,
        camera: &impl Camera,
        gbuffer: &mut GBuffer,
    ) {
        // For each pixel:
        //   camera.generate_ray() → ray
        //   world.intersect(ray) → MaterialHit → SurfaceInteraction
        //   Write to GBuffer
    }
}
```

**Current state:** Does not exist. Primary visibility via ray tracing is baked
into `CpuRenderer`.
**Evolution:** Extract into separate `VisibilityGenerator`. Used by
`HybridRenderer` when rasterizer doesn't cover a pixel.

______________________________________________________________________

## 3. Composition Diagram

```
                    Renderer<W, C, F, S>
                         │
            ┌────────────┼────────────┐
            │            │            │
   CpuRenderer    RasterRenderer  HybridRenderer
   (path trace)   (raster)       (raster + path trace)
            │            │            │
            │     VisibilityGenerator │
            │     ┌─────────┘    ┌────┴────┐
            │     │              │         │
            │  TriangleRasterizer  RayVisibilityGenerator
            │     │              │         │
            └─────┴──────┬───────┘         │
                         │                 │
                   SurfaceInteraction      │
                   (the ABI)          ─────┘
```

All three renderers implement the same `Renderer<W, C, F, S>` trait. The
difference is internal:

- **CpuRenderer**: per-pixel `camera.generate_ray()` → `integrator.li()`
- **RasterRenderer**: `visibility.generate_visibility()` → material eval
- **HybridRenderer**: raster primary → path trace secondary → composite

______________________________________________________________________

## 4. How Production Engines Do This

| Engine | Visibility | Integration | Composition |
|--------|-----------|-------------|-------------|
| **Unreal Lumen** | Nanite rasterizer + SW raster | Lumen ray tracer | Temporal accumulation |
| **Unity HDRP** | Tile/cluster deferred | Per-effect RT toggles | Weighted blend by material |
| **Falcor** | `RenderPass` (raster) | `RenderPass` (RT) | Render graph DAG |
| **Frostbite/Halcyon** | Raster G-buffer | Compute lighting + RT effects | Per-effect pluggable |

The pattern is always: **raster writes primary visibility to G-buffer, RT reads
G-buffer for secondary effects, composition blends by material properties.**

The key insight: `SurfaceInteraction` (or equivalent `ShadingData`) is the ABI
between visibility and shading. Both rasterizer and ray tracer produce it. The
downstream shading system consumes it uniformly.

______________________________________________________________________

## 5. Relationship to CORE_THESIS

| CORE_THESIS Concept | Current | Target |
|---------------------|---------|--------|
| `SurfaceInteraction` (§4) | `SurfaceInteraction<'si>` in `hittable.rs` | **Unchanged** |
| Two execution paths (§2.1) | `CpuRenderer` (ray only) | `CpuRenderer` + `RasterRenderer` + `HybridRenderer` |
| GBuffer (§2.9) | Does not exist | `GBuffer<'a>` with `SurfaceInteraction<'a>` |
| Primary visibility arbitration (§2.9) | Does not exist | `HybridRenderer` composites GBuffer + HitBuffer |
| `SpatialDomain` (§2.4) | `Intersectable` trait | **Deferred** (Stage 1 trigger: 5th leaf type) |
| Render graph (§2.7) | Does not exist | **Deferred** (Stage 7) |
| `LeafNode` enum (§2.4) | `Arc<dyn Intersectable>` | **Deferred** (Stage 1 trigger: 5th leaf type) |

The target architecture enables the CORE_THESIS two-execution-path model without
requiring the full render graph or `LeafNode` enum. Those are later stages.

______________________________________________________________________

## 6. Evolution Path

| Step | What Changes | Breaking? | Files Affected |
|------|-------------|-----------|----------------|
| 1 | Add `RasterCamera` trait | No | `src/camera/mod.rs` |
| 2 | Add `view_projection()` to `PerspectiveCamera` | No | `src/camera/perspective.rs` |
| 3 | Create `GBuffer<'a>` type | No | `src/gbuffer.rs` (new) |
| 4 | Create `VisibilityGenerator` trait | No | `src/visibility/mod.rs` (new) |
| 5 | Create `TriangleRasterizer` impl | No | `src/visibility/triangle.rs` (new) |
| 6 | Create `RayVisibilityGenerator` impl | No | `src/visibility/ray.rs` (new) |
| 7 | Create `RasterRenderer` | No | `src/renderer/raster.rs` (new) |
| 8 | Create `HybridRenderer` | No | `src/renderer/hybrid.rs` (new) |

Steps 1–8 are all additive — no existing code breaks. The `CpuRenderer` and
`Renderer` trait are unchanged.

### What Does NOT Change

- `Renderer<W, C, F, S>` trait — already correct (no integrator parameter)
- `Integrator<S>` trait — already correct (light transport, not visibility). The `S: Sampler` generic will be replaced by `Integrator<S: SampleStream, R: Rng>` in the two-stream refactor (see `docs/samplestream-refactor.md`). This is compatible with the hybrid architecture — the visibility/integration decomposition is independent of how the integrator gets its random numbers.
- `Camera` trait — already correct (ray generation)
- `SurfaceInteraction<'si>` — already correct (the ABI, lifetime-annotated material ref)
- `Hit`, `MaterialHit<'a>` — already correct
- `FilmTile` — already correct

### What DOES Change (from other docs)

- `Film` trait — gains `apply_denoiser()` method (from `docs/denoiser.md` §Phase 1)
- `Film` trait — may gain `weight: f64` parameter on `add_sample` (deferred, see §Film)
- `RgbFilm` — becomes generic `RgbFilm<D: Denoiser = NoDenoiser>` (from `docs/denoiser.md` §Phase 1)
- `CpuRenderer` — gains `film.apply_denoiser()` call after sampling loop (from `docs/denoiser.md` §6)
- `RgbFilm` internals — may extract `VarianceEstimator` (from `docs/adaptive-sampling.md` §2a)
- `Integrator<S>` — becomes `Integrator<W, S>` (generic over world type, from this doc)
- `Integrator<W, S>` — `DimCursor<S>` replaced by `SampleStreamEnum<S>` (from `docs/samplestream-refactor.md`)
- `CpuRenderer` — creates `SampleStreamEnum::indexed()` internally (from `docs/samplestream-refactor.md`)
- `Renderer::render()` — scene parameter changes from `(&W, &[Arc<dyn Sampleable>])` to `(&W, &[SampleableEnum])` (from this doc)

These changes are all additive or backwards-compatible. The `Renderer` trait signature
is unchanged (only the scene parameter type changes). The denoiser, adaptive sampling,
and samplestream changes affect `Film`/`RgbFilm`/`Integrator` internals, not the
renderer abstraction layer.

______________________________________________________________________

## 7. pbrt-v4 Patterns and How They Map

The pbrt-v4 reference implementation uses a three-layer integrator hierarchy
that maps directly to our architecture:

### pbrt-v4 Hierarchy → Our Architecture

```
pbrt-v4                          raytrace-rs
─────────────────────────────    ─────────────────────────────
Integrator                       Renderer<W, C, F, S>
  owns: aggregate, lights          orchestration: render()
  provides: render(), intersect()

ImageTileIntegrator               CpuRenderer<I, S, Fact>
  adds: evaluate_pixel_sample()    tile-based: parallel tile loop
  owns: camera, sampler_prototype

RayIntegrator                     Integrator<S>
  adds: li()                       ray-based: li() per ray
  final: EvaluatePixelSample()
```

The mapping is 1:1. Our `Renderer` is pbrt-v4's `Integrator` (orchestration).
Our `CpuRenderer` is pbrt-v4's `ImageTileIntegrator` (tile-based rendering).
Our `Integrator` is pbrt-v4's `RayIntegrator` (ray-based light transport).

### Key Differences

| Aspect | pbrt-v4 | raytrace-rs | Rationale |
|--------|---------|-------------|-----------|
| Scene ownership | `Integrator` owns aggregate + lights | Passed as parameters to `render()` | Flexibility: same integrator, different scenes |
| Ray type | `RayDifferential` | `Ray` | Simplicity: differentials deferred to Stage 7 |
| Spectral rendering | `SampledWavelengths` | `Color3` (RGB) | Simplicity: spectral deferred |
| Scratch buffer | `ScratchBuffer` passed through | Not needed (no arena allocator) | Simplicity: arena deferred |
| Visible surface | `VisibleSurface` for denoising | Not needed yet | Simplicity: denoiser deferred |

These are production features that we can add later. Our current design is
appropriate for our stage of development.

### Enum-Based Variant Selection (from pbrt-v4)

pbrt-v4 uses an `IntegratorVariant` enum for static dispatch when selecting
integrator types at runtime. This avoids `dyn` dispatch:

```rust
// Static dispatch via enum — no vtable overhead
enum IntegratorVariant<C, S, A>
where
    C: Camera,
    S: Sampler,
    A: Primitive,
{
    Path(PathIntegrator<C, S, A, 5, false>),
    VolPath(VolPathIntegrator<C, S, A, 5, false>),
    BDPT(BDPTIntegrator<C, S, A, 6, false, false>),
    MLT(MLTIntegrator<C, A>),
    SPPM(SPPMIntegrator<C, S, A>),
    AO(AOIntegrator<C, S, A>),
}
```

This pattern is useful for our design. We could add a `RendererVariant` enum
for selecting between path tracer, rasterizer, and hybrid renderer at runtime:

```rust
enum RendererVariant<W, C, F, S, V, I, Fact>
where
    W: Intersectable,
    C: Camera,
    F: Film,
    S: Sampler,
    V: VisibilityGenerator,
    I: Integrator<S>,
    Fact: SamplerFactory<Sampler = S>,
{
    PathTracing(CpuRenderer<I, S, Fact>),
    Raster(RasterRenderer<V>),
    Hybrid(HybridRenderer<V, I, S>),
}
```

### Specialized Integrators (from pbrt-v4)

pbrt-v4's `MLTIntegrator` inherits from `Integrator` directly, NOT from
`RayIntegrator`. This is because MLT has a different execution model — it
doesn't use the tile-based rendering loop. It runs Markov chains that mutate
path samples across the entire image.

This maps directly to our design: the `HybridRenderer` has a different execution
model than `CpuRenderer`. It rasterizes primary visibility, then path-traces
secondary effects. It doesn't fit the per-pixel `generate_ray() → li()` loop.

The lesson: **some renderers don't fit the standard pattern**. The trait
hierarchy should accommodate this by having a common `Renderer` trait at the top,
with different execution models underneath.

### What We Take From pbrt-v4

1. **Three-layer hierarchy** — already mapped to our architecture
2. **Enum-based variant selection** — add `RendererVariant` for runtime selection
3. **Specialized integrators** — `HybridRenderer` doesn't fit `CpuRenderer`'s pattern
4. **Scene ownership** — keep our approach (flexibility) for now; revisit when GPU

______________________________________________________________________

## 8. Future Evolution (not in scope now)

These are CORE_THESIS stages, not part of this refactor:

- **Stage 1:** `LeafNode` enum + `SurfaceInteraction` extraction from `HitRecord`
- **Stage 7:** Render graph with `RenderNode` trait
- **Stage 8:** `Interaction` / `SurfaceInteraction` / `VolumeInteraction` split
- **Stage 13:** Hardware RT migration (`vkCmdTraceRaysKHR`)

The current refactor enables hybrid rendering on CPU without requiring any of
these later stages.

______________________________________________________________________

## 9. Design Decisions

### Why generics over dyn T

The target architecture eliminates `dyn T` from hot paths using three strategies,
following the Niri/Smithay `render_elements!` pattern for enum-based delegation:

1. **Generics for renderer-level composition** — `Renderer<W, C, F, S>`,
   `CpuRenderer<I, S, Fact>`, `RasterRenderer<V>`, `HybridRenderer<V, I, S>`
   are all generic. Zero-cost monomorphization.

2. **Enums for scene objects** — `SampleableEnum` replaces `Arc<dyn Sampleable>`.
   `IntersectableEnum` (from CORE_THESIS §2.4) replaces `Arc<dyn Intersectable>`.
   `From` impls make construction ergonomic. `match` enables type-specific optimization.

3. **Enums for trait-object hot paths** — `SampleStreamEnum<S>` replaces
   `&mut dyn SampleStream`. `ConvergenceCriterionEnum` replaces
   `&dyn ConvergenceCriterion`. `PdfEnum<S>` (already exists) replaces
   `&dyn PDF<S>`.

Each enum follows the **Descriptor → Concrete → Wrapper** pattern:

- **Descriptor** (e.g., `SampleableKind`): lightweight, Clone+Copy, describes what to build
- **Concrete types** (e.g., `Sphere`, `Quad`): implement the trait, hold full state
- **Wrapper enum** (e.g., `SampleableEnum`): delegates via `match`, enables `From` impls

This matches what `niri_render_elements!` and `smithay::render_elements!` do:
generate an enum that wraps concrete types, delegate every trait method via `match`,
and provide `From` impls for ergonomic construction and hierarchical composition.

The remaining `dyn` usage is intentional and justified:

- `Material::Custom(Box<dyn Bsdf>)` — user-extensible, can't enumerate all types
- `Arc<dyn Texture>` — texture trees are recursive, can't flatten to enum
- `BvhNode` internal `Arc<dyn Intersectable>` — stored, not called per-bounce

### Why RasterCamera extends Camera

Rasterization needs view/projection matrices. Ray tracing needs ray generation.
A hybrid renderer needs both. `RasterCamera: Camera` means a rasterization
camera can also generate rays (for secondary effects), but a pure ray-tracing
camera doesn't need to provide matrices.

### Why GBuffer\<'a> borrows from the scene

The `SurfaceInteraction<'si>` has a lifetime-annotated `material: &'si Material`.
The GBuffer stores these, so it borrows from the scene's material storage. This
is correct for CPU. On GPU, `SurfaceInteraction` would use `material_id`
indexing (CORE_THESIS §2.18), and the GBuffer would own its data.

### Why Integrator is generic over W

The `Integrator<W, S>` trait is generic over both the world type and sampler.
This eliminates `&dyn Intersectable` from the integrator's hot path — the
compiler monomorphizes `li()` for each concrete world type (BVH, flat BVH, etc.).
The rasterizer doesn't use an integrator for primary visibility — it evaluates
materials directly. The hybrid renderer uses the integrator for secondary effects
only. The trait boundary is correct.

______________________________________________________________________

## 10. Cross-Document Integration

This document intersects with two other design docs:

### vs `docs/denoiser.md`

| ARCH_HYBRID Section | Denoiser Section | Relationship |
|---------------------|------------------|--------------|
| Film trait (§2) | §Phase 1 — `apply_denoiser()` | Denoiser adds method to Film trait |
| CpuRenderer (§2) | §6 — `film.apply_denoiser()` call | CpuRenderer gains one line after sampling loop |
| RgbFilm (§1, current state) | §Phase 1 — `RgbFilm<D: Denoiser>` | Denoiser makes RgbFilm generic |
| GBuffer\<'a> (§2) | §Phase 2 — DenoiserFeatures | **Different things.** GBuffer stores `SurfaceInteraction` (visibility). DenoiserFeatures stores albedo/normal/depth (per-pixel). See §Naming below. |

**Naming: GBuffer vs DenoiserFeatures**

The denoiser doc's Phase 2 proposes `DenoiserFeatures` as a **struct-of-arrays** (SoA)
on `RgbFilm`: separate `Vec<Color3>` for albedo, `Vec<Vec3>` for normal, `Vec<f64>`
for depth, `Vec<[f64; 3]>` for variance. This follows the `VarianceEstimator` pattern
for cache-friendly sequential access during denoiser passes.

This document's `GBuffer<'a>` stores `SurfaceInteraction<'a>` produced by the
visibility generator. These are different abstractions at different layers:

- **GBuffer** (ARCH_HYBRID): visibility buffer, produced by `VisibilityGenerator`,
  consumed by `RasterRenderer`/`HybridRenderer`. Stores the full surface interaction
  including material reference.
- **DenoiserFeatures** (denoiser.md): per-pixel shading features, produced by the
  integrator at first hit, consumed by the A-Trous wavelet denoiser. Stores only
  the three edge-stopping channels (albedo, normal, depth).

### vs `docs/adaptive-sampling.md`

| ARCH_HYBRID Section | Adaptive Sampling Section | Relationship |
|---------------------|--------------------------|--------------|
| CpuRenderer (§2) | §0.4 — render loop | Adaptive sampling modifies the same loop that gains `apply_denoiser()` |
| RgbFilm (§1) | §2a — VarianceEstimator | VarianceEstimator extraction may change RgbFilm internals |
| Film trait (§2) | §5.3 — `apply_denoiser()` placement | Both agree: after sampling, before final publish |

### vs `docs/samplestream-refactor.md`

| ARCH_HYBRID Section | SampleStream Section | Relationship |
|---------------------|---------------------|--------------|
| Integrator\<W, S> (§2) | §4 — SampleStreamEnum enum | Integrator gains `SampleStreamEnum<S>` parameter (enum, not `dyn`) |
| CpuRenderer\<I, S, Fact> (§2) | §6 — De-generic CpuRenderer | SampleStream may remove `S` and `Fact` generics, creates `IndexedSamplerStream` internally |
| Renderer\<W, C, F, S> (§2) | Not changed by SampleStream | Renderer trait keeps `S: Sampler` — SampleStream only affects Integrator and CpuRenderer internals |
| VisibilityGenerator (§2) | Independent | Visibility decomposition is orthogonal to how integrator gets random numbers |
| SampleableEnum (§2) | Independent | Scene object enums are orthogonal to sampler abstraction |

**Compatibility:** The SampleStream refactor is fully compatible with the hybrid
architecture. The visibility/integration/composition decomposition is independent
of how the integrator gets its random numbers. When SampleStream lands, CpuRenderer
simplifies from `CpuRenderer<I, S, Fact>` to `CpuRenderer<I>` — but the `Renderer`
trait and `VisibilityGenerator` trait are unaffected.

### Unified Dependency Order

| Order | What | Source Doc | Breaking? |
|-------|------|------------|-----------|
| 1 | SampleableEnum + From impls | ARCH_HYBRID.md §2 | No |
| 2 | Integrator\<W, S> generic over world | ARCH_HYBRID.md §2 | No |
| 3 | SampleStreamEnum\<S> enum + IndexedSamplerStream | samplestream-refactor.md Step 1 | No |
| 4 | PdfEnum\<S> (already exists, keep) | pdf.rs | No |
| 5 | §2a VarianceEstimator extraction | adaptive-sampling.md | No |
| 6 | Denoiser trait + NoDenoiser + RgbFilm\<D> | denoiser.md Phase 0-1 | No |
| 7 | BilateralDenoiser | denoiser.md Phase 1 | No |
| 8 | RasterCamera + GBuffer + VisibilityGenerator\<W> | ARCH_HYBRID.md Steps 1-6 | No |
| 9 | RasterRenderer + HybridRenderer | ARCH_HYBRID.md Steps 7-8 | No |
| 10 | A-Trous wavelet + DenoiserFeatures | denoiser.md Phase 2 | No |
| 11 | OIDN integration (optional) | denoiser.md Phase 3 | No |

Steps 1-4 (static dispatch foundation) can proceed in parallel with steps 5-7
(denoiser/adaptive). Steps 8-9 (hybrid rendering) depend on step 1 (SampleableEnum)
and step 2 (Integrator\<W, S>). Step 10 depends on step 6 (needs `RgbFilm<D>` generic).
Step 11 is independent.

______________________________________________________________________
