# SampleStream Refactor — Two-Stream Architecture (SampleStream + RNG) - EXECUTED SUCCESSFULLY

## Changelog

- **v1 (2026-06-28)** — Initial design. Proposed de-genericing `Integrator<S>` and
  `PDF<S>` to `dyn SampleStream`.
- **v2 (2026-06-28)** — Static dispatch audit. Replaced `dyn SampleStream` with
  `SampleStreamKind<S>` enum. Kept `Integrator<W, S>` and `PDF<S>` generic (no
  de-genericing). Updated all code examples, execution order, and summary.
- **v3 (2026-06-28)** — Applied Niri/Smithay `render_elements!` pattern: **Descriptor →
  Concrete → Wrapper**. Renamed `SampleStreamKind<S>` to `SampleStreamEnum<S>`. Added
  `SampleStreamKind` descriptor enum. Added `From` impls for ergonomic construction.
- **v4 (2026-06-30)** — **Two-stream architecture.** Separates correlated 2D sample
  points from independent hash RNG. Removes `S: Sampler` generic from `Integrator`
  and `PDF`. Eliminates fixed 11-dim-per-bounce stride.
- **v4.1 (2026-06-30)** — **No dynamic dispatch.** Replaced `&mut dyn` with concrete
  generics. Zero vtable overhead on hot path. Compiler monomorphizes for each
  stream pair.
- **v4.2 (2026-06-30)** — **Renamed `SobolStream` → `SampleStream`.** The trait
  produces correlated 2D pairs — works for Sobol, stratified grid, and future CMJ.
  Name reflects what it does, not one implementation.
- **v4.3 (2026-06-30)** — **Kept `EmitterPDF`.** Added `LightPDF` as a single-light
  wrapper alongside `EmitterPDF`. Light selection moves to integrator via `SamplerRng::next()`.
  `EmitterPDF` remains for multi-light MIS and non-MIS contexts.
- **v5 (2026-07-13)** — **Doc updated to reflect actual implementation divergences.**
  See changes: `SampleDims` eliminated (replaced by `next_dim` closure), `Rng` → `SamplerRng`,
  `PdfEnum` deleted (PdfKind directly implements PDF), `BsdfSample` → `BsdfScatter` rename,
  `BsdfScatter::Delta` gains `eta: Option<f64>` field.

## Problem

The current architecture uses a **single flat dimension space** (`DimCursor<S>`) shared
across all sampling needs. Every consumer — camera, materials, RR, MIS, PDFs — pulls
from the same linear counter. This forces Sobol dimensions to be consumed for tasks
where Sobol provides zero benefit (discrete decisions, padding).

**Context:** A prior Sobol direction-number bug and a 512-dim cap made this waste
appear catastrophic. With the bug fixed (`v[k] = m[k] << (32 - (k + 1))`) and
`MAX_DIMS = 21200` (full Joe & Kuo dataset), the absolute waste is ~0.1% per path.
The fix is **not** about convergence speed — it's about architectural correctness:

1. **Corruption of Sobol structure** — Discrete decisions (RR, light selection, MIS
   selection) consume Sobol pairs that could stratify the next direction sample.
   Bounce N's direction starts at a dim offset far from bounce N-1's direction,
   diluting the low-discrepancy property across roles that don't benefit from it.

2. **Fixed stride padding** — Early-terminating paths (dielectric, max-attenuation
   cutoff) waste remaining dims in the 11-dim block. A dielectric bounce uses 1
   dim but consumes 11.

3. **MLT incompatibility** — The `sample(n, d)` interface is fundamentally
   incompatible with MLT's mutable state vector with accept/reject.

### Current dimension consumption (per bounce)

| Phase | Dims | Sobol useful? | Source |
|-------|------|---------------|--------|
| RR coin flip | 1 | **No** — discrete | `dim_cursor.next_sample()` |
| Material lobe selection | 1 | **No** — discrete | `dim_cursor.next_sample()` |
| Material direction (2D) | 2 | Yes | `dim_cursor.next_sample()` × 2 |
| Material reserved (x,y,z) | 3 | **No** — unused/padding | `dim_cursor.next_sample()` × 3 |
| MIS strategy selection | 1 | **No** — discrete | `dim_cursor.next_sample()` |
| MIS direction (from selected PDF) | 2 | Yes | `pdfs[i].generate(dim_cursor)` |
| MIS padding to fixed 4 | 1-4 | **No** — waste | `dim_cursor.next_dim()` |
| **Total** | **11** | **4 useful (36%)** | |

### After two-stream architecture

| Phase | Dims | Stream | Source |
|-------|------|--------|--------|
| RR coin flip | 1 | Hash RNG | `rng.next()` |
| Material lobe selection | 1 | Hash RNG | `rng.next()` |
| Material direction (2D) | 2 | SampleStream | `stream.next_2d()` |
| MIS strategy selection | 1 | Hash RNG | `rng.next()` |
| MIS direction (from selected PDF) | 2 | SampleStream | `stream.next_2d()` |
| **Total** | **7** | — | **100% useful** |

With `MAX_DIMS = 21200`, the absolute Sobol waste from discrete decisions is small
(~0.06% of available dims per path). But the two-stream design still wins: discrete
decisions don't consume correlated pairs, there's no fixed stride padding, and MLT
can plug in without fake `sample(n, d)` implementations.

### Per-material breakdown

| Material | RNG dims | Sobol dims | Total | Before |
|----------|----------|------------|-------|--------|
| Lambertian | 1 (MIS sel) | 2 (PDF dir) | **3** | 11 |
| Dielectric | 1 (Fresnel) | 0 (Delta) | **1** | 11 |
| Glossy/Metal | 1 (MIS sel) | 2 (GGX dir) | **3** | 11 |

## Architecture

**Two independent streams, each doing what it's best at:**

```
SampleStream                       HashRng
─────────────                     ───────
Stateful iterator                 Stateful iterator
Produces correlated 2D pairs      Produces independent f64s
next_2d() → (f64, f64)            next() → f64

Used for:                         Used for:
• PDF direction generation        • RR coin flip
  (1×2D per bounce)               • MIS strategy selection
• Material direction              • Light selection
  (when needed)                   • Material lobe selection
```

### Interface Changes

#### 1. New traits (`src/sampler.rs`)

```rust
/// Stateful stream of correlated 2D sample points.
/// Each call advances by one 2D point. No waste.
pub trait SampleStream: Send + Sync {
    /// Returns the next 2D sample point as (u, v) in [0, 1)².
    fn next_2d(&mut self) -> (f64, f64);
}

/// Stateful source of independent random numbers in [0, 1).
pub trait SamplerRng: Send + Sync {
    /// Returns the next independent random value in [0, 1).
    fn next(&mut self) -> f64;
}
```

**These are trait bounds, not concrete types.** Any sampler can implement both.
The integrator is generic over any `(S: SampleStream, R: SamplerRng)` pair — preserving
the same flexibility as today's `Integrator<S: Sampler>`.

#### 2. Existing samplers implement both traits

All three current samplers gain `SampleStream` + `SamplerRng` implementations, so the
integrator works with any combination:

```rust
// === NaiveRandomSampler: pure hash, no QMC structure ===
// Both streams use hash — same as using NaiveRandomSampler for everything today.
impl SampleStream for NaiveRandomSampler {
    #[inline(always)]
    fn next_2d(&mut self) -> (f64, f64) {
        (self.next_hash(), self.next_hash())
    }
}
impl SamplerRng for NaiveRandomSampler {
    #[inline(always)]
    fn next(&mut self) -> f64 { self.next_hash() }
}

// === StratifiedRandomSampler: stratified 2D + hash fallback ===
// SampleStream uses stratified grid for first 2D pair, hash for rest.
// SamplerRng always uses hash.
impl SampleStream for StratifiedRandomSampler {
    #[inline(always)]
    fn next_2d(&mut self) -> (f64, f64) {
        // Stratified 2D for primary samples, hash fallback for rest
        if self.first_pair {
            self.first_pair = false;
            (self.stratified_u(), self.stratified_v())
        } else {
            (self.next_hash(), self.next_hash())
        }
    }
}
impl SamplerRng for StratifiedRandomSampler {
    #[inline(always)]
    fn next(&mut self) -> f64 { self.next_hash() }
}
```

**Usage — same flexibility as today:**

```rust
// Production: Sobol + hash (best quality)
let stream = SampleStreamWriter::for_pixel(x, y, sample_idx);
let rng = HashRng::for_pixel(x, y, sample_idx);
integrator.li(&mut ray, world, lights, &mut stream, &mut rng);

// Debug: pure hash (no QMC structure)
let naive = NaiveRandomSampler::with_seed(seed);
integrator.li(&mut ray, world, lights, &mut naive, &mut naive);

// Comparison: stratified + hash
let strat = StratifiedRandomSampler::new(sqrt_spp, seed);
integrator.li(&mut ray, world, lights, &mut strat, &mut strat);
```

The integrator is **not** tied to Sobol. It's generic over any `(SampleStream, SamplerRng)`
pair. Swap the implementation at the call site — same as swapping `S` today.

#### 2. SampleStream implementation

The current `SobolQmcSampler` is **stateless** — `sample(n, d)` is a pure function of
`(n, d)`. For `SampleStream`, we need a **stateful** wrapper that advances through
dimension pairs:

```rust
/// Stateful Sobol stream — wraps a stateless `SobolQmcSampler` and advances
/// through 2D dimension pairs.
pub struct SampleStreamWriter {
    sampler: SobolQmcSampler,
    sample_idx: u32,
    next_pair: u32,  // increments by 1 per next_2d() call
}

impl SampleStreamWriter {
    pub fn new(sampler: SobolQmcSampler, sample_idx: u32) -> Self {
        Self { sampler, sample_idx, next_pair: 0 }
    }

    pub fn for_pixel(pixel_x: i32, pixel_y: i32, sample_idx: u32) -> Self {
        Self::new(SobolQmcSampler::for_pixel(pixel_x, pixel_y), sample_idx)
    }
}

impl SampleStream for SampleStreamWriter {
    #[inline(always)]
    fn next_2d(&mut self) -> (f64, f64) {
        let d = self.next_pair * 2;
        let u = self.sampler.sample(self.sample_idx, d);
        let v = self.sampler.sample(self.sample_idx, d + 1);
        self.next_pair += 1;
        (u, v)
    }
}
```

**Key insight:** The stateless `SobolQmcSampler::sample(n, d)` is still useful for
other contexts (e.g., pre-generating sample tables). The `SampleStreamWriter` is a
thin stateful adapter for the integrator's sequential consumption pattern.

#### 3. SamplerRng implementation

```rust
/// Hash-based independent random number generator.
/// Each call produces an independent value via SplitMix64.
pub struct HashRng {
    seed: u64,
    counter: u32,
}

impl HashRng {
    pub fn new(seed: u64) -> Self {
        Self { seed, counter: 0 }
    }

    pub fn for_pixel(pixel_x: i32, pixel_y: i32, sample_idx: u32) -> Self {
        let seed = pixel_seed(pixel_x, pixel_y)
            .wrapping_add(sample_idx as u64 * 0x9E3779B97F4A7C15);
        Self { seed, counter: 0 }
    }
}

impl SamplerRng for HashRng {
    #[inline(always)]
    fn next(&mut self) -> f64 {
        let v = hash_sample(self.counter, 0, self.seed);
        self.counter += 1;
        v
    }
}
```

#### 4. PDF trait — remove generic `S`

```rust
// BEFORE (generic, coupled to DimCursor):
pub trait PDF<S: Sampler> {
    fn value(&self, direction: Vec3) -> f64;
    fn generate(&self, dim_offset: &mut DimCursor<S>) -> Vec3;
}

// AFTER (concrete, pure function):
pub trait PDF {
    fn value(&self, direction: Vec3) -> f64;
    fn generate(&self, u: f64, v: f64) -> Vec3;
}
```

Every PDF implementation already reads exactly 2 values from the cursor for
`generate()`. Changing to `(u, v)` is mechanical:

```rust
// CosinePDF — before:
impl<S: Sampler> PDF<S> for CosinePDF {
    fn generate(&self, dim_offset: &mut DimCursor<S>) -> Vec3 {
        let u = dim_offset.next_sample();
        let v = dim_offset.next_sample();
        self.uvw.local_to_world(cosine_hemisphere_direction(u, v))
    }
}

// CosinePDF — after:
impl PDF for CosinePDF {
    fn generate(&self, u: f64, v: f64) -> Vec3 {
        self.uvw.local_to_world(cosine_hemisphere_direction(u, v))
    }
}
```

Same mechanical change for `GgxSamplePDF`, `UniformSpherePDF`, `UniformHemispherePDF`.

#### 5. PdfEnum — deleted (PdfKind now IS a PDF)

> **Note:** The spec proposed de-genericing `PdfEnum<S>` → `PdfEnum`. In the actual implementation, `PdfEnum` was **deleted entirely**. `PdfKind` directly implements the `PDF` trait with `generate()` and `value()` methods — the consolidation suggested in the "Optionally do after" section was already done.

```rust
// BEFORE:
pub struct PdfEnum<S: Sampler> {
    inner: PdfEnumInner,
    _s: std::marker::PhantomData<S>,
}

// AFTER (actual): PdfEnum is gone. PdfKind carries generate() and value() directly.
impl PDF for PdfKind {
    fn value(&self, direction: Vec3) -> f64 {
        PdfKind::value(self, direction)
    }
    fn generate(&self, u: f64, v: f64) -> Vec3 {
        PdfKind::generate(self, u, v)
    }
}
```

#### 6. EmitterPDF and LightPDF

`EmitterPDF` is **kept** — it handles the general multi-light case. A new
`LightPDF` wraps a single light source for when the integrator has already
selected which light to sample:

```rust
/// PDF for a single sampleable light source.
/// Light selection is handled by the integrator — this PDF only generates
/// directions from the selected light.
pub struct LightPDF<'a, T: Sampleable + ?Sized> {
    object: &'a T,
    origin: Point3,
    time: f64,
}

impl<'a, T: Sampleable + ?Sized> LightPDF<'a, T> {
    pub fn new(object: &'a Arc<dyn Sampleable>, origin: Point3, time: f64) -> Self {
        Self { object, origin, time }
    }
}

impl<'a, T: Sampleable + ?Sized> PDF for LightPDF<'a, T> {
    fn value(&self, direction: Vec3) -> f64 {
        self.object.pdf_value(self.origin, direction, self.time)
    }

    fn generate(&self, u: f64, v: f64) -> Vec3 {
        self.object.random_direction(self.origin, u, v, self.time)
    }
}
```

**Usage in the integrator:** Light selection happens via `rng.next()`, then the
selected light is wrapped in `LightPDF` for MIS direction generation:

```rust
// Light selection: hash (discrete)
let light_idx = (rng.next() * lights.len() as f64
    .min(lights.len() as f64 - 1e-15)) as usize;
let light_pdf = LightPDF::new(&lights[light_idx], si.point(), ray.time);
```

`EmitterPDF` remains available for contexts where internal selection is
preferred (e.g., non-MIS light sampling, or when the number of lights is
small enough that selection overhead is negligible).

#### 7. mis_sample — remove generic, accept selection externally

```rust
// BEFORE:
fn mis_sample<S: Sampler>(
    pdfs: &[&dyn PDF<S>],
    eval_fn: impl FnOnce(Vec3) -> Color3,
    dim_cursor: &mut DimCursor<S>,
) -> (Vec3, Color3) {
    let u_select = dim_cursor.next_sample();
    let sel_idx = (u_select * n as f64).min(n as f64 - 1e-15) as usize;
    let direction = pdfs[sel_idx].generate(dim_cursor).unit_vector();
    // ...
}

// AFTER:
fn mis_sample(
    pdfs: &[&dyn PDF],
    eval_fn: impl FnOnce(Vec3) -> Color3,
    sel_idx: usize,
    pdf_u: f64,
    pdf_v: f64,
) -> (Vec3, Color3) {
    let direction = pdfs[sel_idx].generate(pdf_u, pdf_v).unit_vector();
    // ... (rest is identical)
}
```

The selection index comes from `rng.next()` in the integrator. The `(u, v)` for
direction generation come from `sobol.next_2d()`. Each stream provides what it's
best at.

#### 8. Integrator — two streams instead of one cursor

```rust
// BEFORE:
pub trait Integrator<S: Sampler>: Send + Sync {
    fn li(
        &self,
        initial_ray: &mut Ray,
        world: &dyn Intersectable,
        lights: &[Arc<dyn Sampleable>],
        sampler: &mut DimCursor<S>,
    ) -> Color3;
}

// AFTER:
pub trait Integrator<S: SampleStream, R: SamplerRng>: Send + Sync {
    fn li(
        &self,
        initial_ray: &mut Ray,
        world: &dyn Intersectable,
        lights: &[Arc<dyn Sampleable>],
        stream: &mut S,
        rng: &mut R,
    ) -> Color3;
}
```

**No `dyn` dispatch.** The compiler monomorphizes `li()` for each concrete
`(Sobol, R)` pair. Same zero-cost abstraction as today's `Integrator<S: Sampler>`.

`PathTracingIntegrator` becomes non-generic (no `S: Sampler`), but the trait
it implements is generic over the two stream types:

```rust
// Path tracer uses concrete SampleStreamWriter + HashRng
impl Integrator<SampleStreamWriter, HashRng> for PathTracingIntegrator {
    fn li(
        &self,
        ray: &mut Ray,
        world: &dyn Intersectable,
        lights: &[Arc<dyn Sampleable>],
        stream: &mut SampleStreamWriter,
        rng: &mut HashRng,
    ) -> Color3 { ... }
}
```

Inside `li()`, stream calls are direct method calls (no vtable):

```rust
fn li(
    &self,
    ray: &mut Ray,
    world: &dyn Intersectable,
    lights: &[Arc<dyn Sampleable>],
    stream: &mut SampleStreamWriter,
    rng: &mut HashRng,
) -> Color3 {
    for bounce in 0..self.max_depth {
        if let Some(mat_hit) = world.intersect(&ray, Interval::from(0.001, f64::INFINITY)) {
            let si = SurfaceInteraction::from_material_hit(mat_hit, &ray);
            let material = si.material();
            let emission = material.emitted(&si);
            accumulated_color += accumulated_attenuation * emission;

            let max_attenuation = accumulated_attenuation.x
                .max(accumulated_attenuation.y)
                .max(accumulated_attenuation.z);
            if max_attenuation < 1e-6 { return accumulated_color; }

            // RR: hash (discrete decision)
            if bounce >= 5 {
                let survival = max_attenuation.clamp(0.05, 1.0);
                if rng.next() > survival { return accumulated_color; }
                accumulated_attenuation /= survival;
            }

            let wo = -ray.direction.unit_vector();

            // Material uses a next_dim closure — SampleDims was eliminated entirely
            let mut next_mat_dim = || -> f64 { rng.next() };

            if let Some(scatter) = material.scatter(wo, &si, &mut next_mat_dim) {
                let (direction, bias, eta) = match scatter {
                    BsdfScatter::Delta { wi, f_cos, eta } => {
                        (wi, f_cos, eta)
                    }
                    BsdfScatter::NonDelta { pdf_kinds } => {
                        // PdfKind directly implements PDF — no PdfEnum wrapper
                        let is_volume = si.shading_normal().near_zero();

                        // Env PDF: use PdfKind directly (not PdfEnum::new)
                        let env_pdf: PdfKind = if is_volume {
                            PdfKind::UniformSphere
                        } else {
                            PdfKind::Cosine { normal: si.shading_normal() }
                        };

                        // Light selection: hash (discrete)
                        let light_idx = (rng.next() * lights.len() as f64
                            .min(lights.len() as f64 - 1e-15)) as usize;
                        let light_pdf = LightPDF::new(&lights[light_idx], si.point(), ray.time);

                        // MIS selection: hash (discrete)
                        let mis_select = rng.next();

                        // Material PDFs from pdf_kinds array
                        // ... build &dyn PDF refs from PdfKind values ...

                        // Direction: stream
                        let (pdf_u, pdf_v) = stream.next_2d();

                        let eval = |d: Vec3| material.eval(wo, d, &si);
                        let sel_idx = (mis_select * n as f64).min(n as f64 - 1e-15) as usize;
                        let (direction, contribution) = mis_sample(
                            &pdf_refs[..n], eval, sel_idx, pdf_u, pdf_v
                        );
                        (direction, contribution, None)
                    }
                };

                accumulated_attenuation = accumulated_attenuation * bias;
                ray = Ray::new_with_time(si.point(), direction, ray.time);
            } else {
                return accumulated_color;
            }
        } else {
            return accumulated_color + accumulated_attenuation * self.background;
        }
    }
    accumulated_color
}
```

**No fixed 11-dim stride.** Each bounce consumes only what it needs:

- Delta material: 1 RNG (lobe) + 0 Sobol = **1 dim**
- Lambertian: 1 RNG (lobe) + 2 Sobol (direction) + 1 RNG (MIS sel) + 2 Sobol (MIS dir) = **3 RNG + 4 Sobol**
- Glossy: same as Lambertian but with GGX direction

The `debug_assert_eq!(dim_cursor.offset() - bounce_start, 11)` is removed.
Dimension consumption is now variable and minimal.

#### 9. Camera — no change needed

The camera already receives pre-extracted values via `CameraSampler`:

```rust
// CameraSampler already has concrete f64 fields — no generic needed:
pub struct CameraSampler {
    pub x: f64,  // pixel x + AA jitter
    pub y: f64,  // pixel y + AA jitter
    pub lens_u: f64,
    pub lens_v: f64,
    pub time: f64,
}
```

The `CameraSampler::new_sampled()` method currently takes `&mut DimCursor<S>`.
After the refactor, it takes concrete values:

```rust
// BEFORE:
impl CameraSampler {
    pub fn new_sampled<S: Sampler>((x, y): (u32, u32), dim: &mut DimCursor<S>) -> Self {
        let jitter_x = dim.next_sample();
        let jitter_y = dim.next_sample();
        let lens_u = dim.next_sample();
        let lens_v = dim.next_sample();
        let time = dim.next_sample();
        // ...
    }
}

// AFTER — generic over concrete stream types (no dyn):
impl CameraSampler {
    pub fn new_sampled<S: SampleStream, R: SamplerRng>(
        (x, y): (u32, u32),
        stream: &mut S,
        rng: &mut R,
    ) -> Self {
        let (jitter_x, jitter_y) = stream.next_2d();  // AA: correlated 2D
        let (lens_u, lens_v) = stream.next_2d();       // Lens: correlated 2D
        let time = rng.next();                           // Time: hash (independent)
        // ...
    }
}
```

This is a minor change — the camera already receives pre-extracted values.
The only difference is where those values come from.

#### 10. SampleDims — eliminated (replaced by `next_dim` closure)

> **Note:** The spec proposed simplifying `SampleDims` from 6 fields to 3 (u, v, w). In the actual implementation, `SampleDims` was **eliminated entirely**. The `Bsdf::scatter()` signature now takes `next_dim: &mut dyn FnMut() -> f64` instead of a fixed struct, allowing each material to consume exactly as many dimensions as it needs.

```rust
// BEFORE: 6 fields, most unused per material type
pub struct SampleDims {
    pub u: f64,  // lobe selection
    pub v: f64,  // direction u
    pub w: f64,  // direction v
    pub x: f64,  // reserved
    pub y: f64,  // reserved
    pub z: f64,  // reserved
}

// AFTER (actual): no SampleDims struct — the scatter() signature is:
fn scatter(
    &self,
    wo: Vec3,
    si: &SurfaceInteraction,
    next_dim: &mut dyn FnMut() -> f64,
) -> Option<BsdfScatter>;
```

The `next_dim` closure approach gives materials unlimited dimensions (they call
`next_dim()` for each random value they need), eliminating the fixed 3-field cap
and the reserved fields. Materials that need more randomness (e.g., the coated
material's internal Fresnel split) consume dimensions on demand without any
struct changes.

#### 11. Renderer — create both streams per pixel

```rust
// BEFORE:
let sampler = self.sampler_factory.for_pixel(x, y);
let mut dim_cursor = DimCursor::new_at(0, sample_idx, sampler);
let camera_sampler = CameraSampler::new_sampled((x, y), &mut dim_cursor);
let radiance = self.integrator.li(&mut cam_ray.ray, world, lights, &mut dim_cursor);

// AFTER:
let mut stream = SampleStreamWriter::for_pixel(x, y, sample_idx);
let mut rng = HashRng::for_pixel(x, y, sample_idx);
let camera_sampler = CameraSampler::new_sampled((x, y), &mut stream, &mut rng);
let radiance = self.integrator.li(&mut cam_ray.ray, world, lights, &mut stream, &mut rng);
```

#### 12. CpuRenderer — concrete stream generics

```rust
// BEFORE:
pub struct CpuRenderer<I, S, Fact>
where
    I: Integrator<S>,
    S: crate::sampler::Sampler,
    Fact: SamplerFactory<Sampler = S>,
{ ... }

// AFTER — concrete stream types, no dyn:
pub struct CpuRenderer<I, S, R, SFact, RFact>
where
    I: Integrator<S, R>,
    S: SampleStream,
    R: SamplerRng,
    SFact: SampleStreamFactory<SampleStream = S>,
    RFact: RngFactory<Rng = R>,
{
    integrator: I,
    stream_factory: SFact,
    rng_factory: RFact,
}
```

Factory traits for creating per-pixel stream instances:

```rust
pub trait SampleStreamFactory: Send + Sync {
    type SampleStream: crate::sampler::SampleStream;
    fn for_pixel(&self, x: i32, y: i32, sample_idx: u32) -> Self::SampleStream;
}

pub trait RngFactory: Send + Sync {
    type Rng: SamplerRng;
    fn for_pixel(&self, x: i32, y: i32, sample_idx: u32) -> Self::Rng;
}
```

**No `dyn` dispatch anywhere in the render loop.** The compiler monomorphizes
the full path: `CpuRenderer → Integrator::li → SampleStream::next_2d / SamplerRng::next`.

### What Stays the Same

- `Sampler` trait — unchanged, pure indexed source (still useful for other contexts)
- `DimCursor<S>` — unchanged, still useful for sequential access patterns
- `SobolQmcSampler` — unchanged, stateless `(n, d)` lookup
- `NaiveRandomSampler` — unchanged, gains `SampleStream` + `SamplerRng` impls
- `StratifiedRandomSampler` — unchanged, gains `SampleStream` + `SamplerRng` impls
- `SamplerFactory` — unchanged, still useful for creating per-pixel samplers
- `Bsdf::scatter()` — takes `next_dim: &mut dyn FnMut() -> f64` (SampleDims eliminated)
- `ConstantMedium` — uses `hash_sample()` directly, independent of streams
- `Sampleable` — already de-genericed, takes `(u, v)` directly

### MLT Integration (Future)

MLT fundamentally needs a **mutable state vector** with accept/reject. The two-stream
architecture adapts cleanly with **zero `dyn` dispatch**:

```rust
/// MLT sampler — wraps a mutable state vector with accept/reject bookkeeping.
/// Implements both SampleStream and SamplerRng by reading from the same state vector.
/// A single type satisfies both generic bounds.
pub struct MltStream {
    state: Vec<f64>,       // primary sample space [d0, d1, d2, ...]
    cursor: usize,         // current read position
    proposal: Vec<f64>,    // backup for accept/reject
    sigma: f64,
    large_step_prob: f64,
}

impl SampleStream for MltStream {
    fn next_2d(&mut self) -> (f64, f64) {
        let u = self.state[self.cursor];
        let v = self.state[self.cursor + 1];
        self.cursor += 2;
        (u, v)
    }
}

impl SamplerRng for MltStream {
    fn next(&mut self) -> f64 {
        let v = self.state[self.cursor];
        self.cursor += 1;
        v
    }
}
```

The cursor advances linearly regardless of which trait the integrator calls.
`begin_proposal()` snapshots cursor + state, `reject()` restores.

```rust
// MLT integrator — same Integrator trait, concrete MltStream type:
// MltStream implements both SampleStream and SamplerRng, so it fills both generic slots.
impl Integrator<MltStream, MltStream> for PathTracingIntegrator {
    fn li(
        &self,
        ray: &mut Ray,
        world: &dyn Intersectable,
        lights: &[Arc<dyn Sampleable>],
        stream: &mut MltStream,  // same as rng — both point to the same MltStream
        rng: &mut MltStream,     // compiler knows this is MltStream, not dyn
    ) -> Color3 { ... }
}

// Future MLT renderer:
fn render_pixel(&self, pixel: (u32, u32), chain: &mut MltChain) {
    chain.begin_proposal();
    let candidate = self.integrator.li(
        &mut ray, world, lights,
        &mut chain.stream,  // &mut MltStream — passed as both stream and rng
        &mut chain.stream,
    );
    if chain.accept(candidate.luminance()) {
        // keep proposal
    } else {
        // rollback — state vector reverts to pre-proposal state
    }
}
```

**The integrator code stays the same.** The stream implementations change.
No `dyn` anywhere — the compiler monomorphizes for `MltStream`.

## Files to Change

| File | Change | Risk |
|------|--------|------|
| `src/sampler.rs` | Add `SampleStream` + `SamplerRng` traits, `SampleStreamWriter`, `HashRng` | Low |
| `src/pdf.rs` | Remove `S` from `PDF<S>`. Delete `PdfEnum`. `PdfKind` directly implements `PDF`. Add `LightPDF` alongside `EmitterPDF` | Medium |
| `src/integrator/mod.rs` | Change `Integrator<S>` → `Integrator<S, R>`. Update `li()` signature | Medium |
| `src/integrator/path_tracer.rs` | Rewrite `li()` to use two streams. Update `mis_sample` | High |
| `src/camera/mod.rs` | Update `CameraSampler::new_sampled()` to take `(stream, rng)` | Low |
| `src/renderer/cpu.rs` | Update generics to `Integrator<S, R>`. Create `SampleStreamWriter` + `HashRng` | Medium |
| `src/material/mod.rs` | Eliminate `SampleDims` — add `next_dim` param to `Bsdf::scatter()` | Low |
| `src/material/*.rs` | Update all `scatter()` impls to consume dims from `next_dim` closure | Low |

## Implementation Steps

| # | Step | Files | Risk | Breaking? |
|---|------|-------|------|-----------|
| 1 | Add `SampleStream` + `SamplerRng` traits + impls for all samplers | sampler.rs | Low | No |
| 2 | Add `LightPDF` alongside existing `EmitterPDF` | pdf.rs | Low | No |
| 3 | Change `PDF` trait: remove `S`, accept `(u,v)` | pdf.rs, all PDF impls | Medium | Yes |
| 4 | Delete `PdfEnum` — `PdfKind` directly implements `PDF` trait | pdf.rs | Medium | Yes |
| 5 | Update `mis_sample` — remove generic, external selection | path_tracer.rs | Medium | Yes |
| 6 | Eliminate `SampleDims` — use `next_dim` closure in `Bsdf::scatter()` | material/mod.rs, all material impls | Low | Yes |
| 7 | Update `CameraSampler::new_sampled()` | camera/mod.rs | Low | Yes |
| 8 | Change `Integrator<S>` → `Integrator<S, R>` | integrator/mod.rs | Medium | Yes |
| 9 | Rewrite `PathTracingIntegrator::li` to use two streams | path_tracer.rs | High | Yes |
| 10 | Update `CpuRenderer` generics and render loop | renderer/cpu.rs | Medium | Yes |
| 11 | Remove `DimCursor` from hot path (keep for other uses) | sampler.rs | Low | No |

**Note:** The per-bounce dim invariant test added to `src/integrator/mod.rs` (replacing
the `debug_assert!`) should be removed in step 9. The two-stream architecture uses
variable dimension consumption per bounce — the fixed-dim test will be incorrect
after the refactor.

### Migration Strategy

**Phase 1 (non-breaking):** Add `SampleStream` + `SamplerRng` traits alongside existing
`Sampler`. Implement `SampleStreamWriter`, `HashRng`, and add `SampleStream` +
`SamplerRng` impls to `NaiveRandomSampler` and `StratifiedRandomSampler`. Add `LightPDF`
alongside existing `EmitterPDF`. All existing code continues to work.

**Phase 2 (mechanical):** Change `PDF` trait to remove generic. Update all impls.
This is the biggest change but is mostly find-and-replace. Delete `PdfEnum` — `PdfKind` directly implements the `PDF` trait, absorbing all variants.

**Phase 3 (core):** Rewrite `PathTracingIntegrator::li` to use two streams.
Remove fixed 11-dim stride. Update `mis_sample`.

**Phase 4 (cleanup):** Update renderer, camera. Remove `DimCursor` from hot path.
Eliminate `SampleDims` — replace with `next_dim` closure in `Bsdf::scatter()`. Update integrator to use `LightPDF` for MIS.

## What This Buys

| Issue | Before | After |
|-------|--------|-------|
| Sobol structure | Discrete decisions dilute correlation pairs | Each stream has one role, pairs stratify directions |
| Fixed stride | 11 dims/bounce always, padding on early termination | Variable, minimal (1-7) |
| Camera waste | 3/5 dims wasted (time is hash-quality) | Time from hash, AA/lens from stream |
| EmitterPDF waste | 1 selection dim for single-light | `LightPDF` skips selection, `HittablePDF` kept for multi-light |
| Dynamic dispatch | `dyn` on hot path (v3 proposal) | Zero `dyn` — compiler monomorphizes |
| Code complexity | Generic `S` everywhere, 11-dim debug_assert | Concrete stream types, no fixed stride |
| MLT compatibility | Requires fake `sample(n, d)` | Clean `SampleStream + SamplerRng` interface |
| Integrator generic | `Integrator<S: Sampler>` | `Integrator<S, R>` (no `S: Sampler`) |
| PDF generic | `PDF<S: Sampler>` | `PDF` (no generic at all) |

## Cross-reference: renderer_arch.md

This refactor is compatible with the hybrid architecture in `docs/renderer_arch.md`:

- The `Integrator` trait changes from `Integrator<S: Sampler>` to
  `Integrator<S: SampleStream, R: SamplerRng>` — still generic, still zero-cost, but
  with two stream types instead of one sampler type. The visibility/integration
  decomposition is independent of how the integrator gets its random numbers.
- The `Renderer` trait updates to use the new `Integrator` bounds. No `dyn` needed.
- The `Camera` trait is unchanged — it receives pre-extracted values.
- The `Film` trait is unchanged — denoiser operates at the film layer.

## Prerequisites (Completed)

- `Sampleable` de-generic — takes `(u, v)` directly ✓
- Camera/Renderer/Integrator extraction from monolithic camera.rs ✓
- Constant Mediums QMC integration ✓
- renderer_arch.md audit — consistent ✓

## Post-refactor follow-up items

### DONE in the implementation

The following consolidations (proposed as "optionally do after" in the spec) were
**already completed** as part of the refactor:

1. ✅ **PdfKind implements PDF directly** — `PdfKind` has `generate()` and `value()` methods
   (see `src/pdf.rs` and `src/material/mod.rs`). The integrator stores `[PdfKind; MAX_STRATEGIES]`
   on the stack instead of building `PdfEnum` wrappers.
2. ✅ **PdfEnum deleted** — No longer exists. `PdfKind` is used directly everywhere.
3. ✅ **Individual PDF structs (CosinePDF, etc.) removed** — `PdfKind::generate()` dispatches via
   match, eliminating separate structs for each variant.
4. ✅ **MIS hot path uses `&dyn PDF` references to `PdfKind`** — The MIS loop holds
   `[Option<PdfKind>; N]` and takes `&dyn PDF` references to stack copies.

```rust
// Actual implementation — PdfKind directly implements PDF:
impl PDF for PdfKind {
    fn value(&self, direction: Vec3) -> f64 {
        PdfKind::value(self, direction)
    }
    fn generate(&self, u: f64, v: f64) -> Vec3 {
        PdfKind::generate(self, u, v)
    }
}
```

The integrator's MIS step holds `[PdfKind; MAX_STRATEGIES]` on the stack, taking
`&dyn PDF` references to each. One heap-allocation (Vec) gone, the PdfEnum wrapper
tier eliminated.

### Remaining optional items (not yet done)

- Consider renaming the `PDF` trait methods or removing the trait entirely now that
  `PdfKind` carries its own methods (the `impl PDF for PdfKind` delegates are thin
  wrappers).
- `material.pdf()` currently returns `[Option<PdfKind>; 2]` — could delegate to
  `PdfKind::value()` directly, but this is already how the integrator uses it.
