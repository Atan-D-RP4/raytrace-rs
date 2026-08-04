# Architecture Review: `raytrace-rs`

A physically based CPU path tracer in Rust, compared against production renderers.

**Reference renderers:**
**[pbrt-v4](https://github.com/mmp/pbrt-v4)** (C++, ~200K LOC),
**[LuxCoreRender](https://github.com/LuxCoreRender/LuxCore)** (C++/Python, ~350K LOC),
**[appleseed](https://github.com/appleseedhq/appleseed)** (C++, ~680K LOC),
**[OpenMoonRay](https://github.com/OpenMoonRay/openmoonray)** (C++/ISPC, ~2M LOC),
**[NVIDIA Falcor](https://github.com/NVIDIAGameWorks/Falcor)** (C++/Slang, ~300K LOC),
**[Mitsuba 3](https://github.com/mitsuba-renderer/mitsuba3)** (C++/Python, ~180K LOC),
**[Google Filament](https://github.com/google/filament)** (C++/GLSL, ~250K LOC),
**[renderling](https://github.com/schell/renderling)** (Rust/wgpu, ~30K LOC)

> **Last updated:** 2026-07-14 — fresh audit.
> 54 source files, 25 public modules, 21 traits, ~18 enums, ~50 structs.
> 8 core dependencies (image, rand, rayon, winit, softbuffer, tracing, smol, async-tungstenite).

---

## 0. Executive Summary

| Metric | Value |
|--------|-------|
| Source files | 54 `.rs` |
| Public modules | 25 |
| Public traits | 21 (`Intersectable`, `Bounded`, `Sampleable`, `Sampler`, `SamplingSession`, `SampleStream`, `SamplerRng`, `SampleStreamFactory`, `RngFactory`, `Camera`, `Film`, `Integrator`, `Renderer`, `PDF`, `Shape3D`, `Region2D`, `Bsdf`[private], `Texture`, `UVDifferentiable`, `Transform`, `GpuSerializable`) |
| Material types | 8 (6 BSDF + 2 composition: Mix, Coated) |
| Texture types | 5 (SolidColor, Checker, Noise, Image, MappedTexture) |
| Shape types | 2 (Sphere + ShapeObject generic; 9 planar region types via PlanarPatch) |
| Integrators | 1 (PathTracingIntegrator) |
| Renderers | 1 (CpuRenderer — rayon tiled, adaptive) |
| Samplers | 3 (Sobol, NaiveRandom, Stratified) + Factory pattern |
| Scene functions | 10 built-in |
| Unsafe blocks | 2 (GPU serialization only) |
| `unwrap()`/`expect()` | 13 (init code, sampler locks, hardcoded paths) |
| `todo!()` / `unimplemented!()` | 0 |
| TODOs | 20 |
| `#[allow(dead_code)]` | 2 (`perlin.rs`, `environment.rs` spotlight field) |
| `total LOC` | ~7,400 (src/) |

### Key Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Dispatch model** | Enum-based (closed set) | Mirrors pbrt-v4's TaggedPointer; static dispatch, inline-able, no heap alloc |
| **Material polymorphism** | `Material` enum (10 variants) + private `Bsdf` trait | Hot-path dispatch via match; `Custom(Box<dyn Bsdf>)` escape hatch for extensibility |
| **Integrator** | Generic trait `Integrator<S: Sampler>` where `Sampler` uses GAT `Session<'s>: SamplingSession` | Per-pixel session borrow via `begin_pixel`, prevents cross-pixel sample corruption at compile time |
| **Renderer** | Generic trait `Renderer<W,C,F,S>` | Swappable render backends (CPU now, GPU future) |
| **Sampler** | Unified `Sampler` trait with GAT `Session<'s>` + two-stream internals (Sobol for 2D, hash for 1D) | Each thread owns one `Sampler` clone. `begin_pixel()` returns a `SamplingSession`. Compiler monomorphizes per-sampler; two-stream internals protect Sobol correlation structure. |
| **Geometry ownership** | `Vec<Arc<dyn Intersectable>>` | Conservative; arena refactor planned |
| **Film** | `Film` trait + `RgbFilm` (Welford variance) | Adaptive sampling + clean abstraction |
| **Camera** | `Camera` trait + `PerspectiveCamera` | Extensible for new camera models |
| **Acceleration** | Flat BVH (64-byte nodes, iterative) | Cache-line-aligned, no recursion, SAH-built |
| **GPU path** | Material + Texture tree→flat serialization | Flatten at scene-build time, upload as uniform data |

---

## 1. Architecture Overview

### 1.1 Module Hierarchy

```text
src/
├── main.rs                          # Entry point — winit event loop, softbuffer, scene build
├── lib.rs                           # Module declarations (25 pub mod)
│
├── vec3.rs                          # Vec3, Point3/Color3 aliases, ops, concentric_disk
├── ray.rs                           # Ray struct, ParametricCurve
├── interval.rs                      # Interval { min, max }
├── aabb.rs                          # Aabb { x, y, z: Interval }, merge, hit, centroid
├── onb.rs                           # Orthonormal basis (build_from_normal, local↔world)
├── perlin.rs                        # Perlin noise (#[allow(dead_code)])
├── transform.rs                     # Transform trait, Translate, RotateY (+ 5 TODOs)
│
├── sampler/
│   └── mod.rs                       # Sampler(GAT), SamplingSession, SampleStream, SamplerRng, SobolSampler, HashRng
├── pdf.rs                           # PDF trait, PdfKind, power_heuristic, balance_heuristic, MisHeuristic,
│   │                                # SolidAnglePdf/AreaPdf, Distribution1DFixed, sample_discrete
├── distributions.rs                 # Dist1D, Dist2D, Sample1D enum
│
├── hittable.rs                      # Hit, SurfaceInteraction, Intersectable, Bounded, Sampleable, MaterialHit
├── bvh.rs                           # BvhNode enum (Empty/Interior/Leaf), SAH bin build
├── flat_bvh.rs                      # FlatBvh, FlatBvhNode [64-byte aligned], iterative traversal
├── scene.rs                         # Scene builder, 10 scene functions
├── const_medium.rs                  # ConstantMedium<T, SURFACE> volumetric wrapper
│
├── camera/
│   ├── mod.rs                       # Camera trait, CameraSampler, CameraRay
│   └── perspective.rs               # CameraConfig, PerspectiveCamera
│
├── film/
│   ├── mod.rs                       # Film trait, Framebuffer, SharedFb, post_process
│   ├── rgb.rs                       # RgbFilm — Welford variance, PPM/PNG export
│   └── tile.rs                      # FilmTile — per-thread accumulator
│
├── integrator/
│   ├── mod.rs                       # Integrator<S> trait
│   └── path_tracer.rs               # PathTracingIntegrator — MIS, shadow ray, RR
│
├── renderer/
│   ├── mod.rs                       # Renderer trait
│   └── cpu.rs                       # CpuRenderer — rayon tiled, adaptive sampling
│
├── material/
│   ├── mod.rs                       # Material enum (9 variants), Bsdf trait, BsdfScatter, GGX
│   ├── diffuse_reflector.rs         # DiffuseReflector — albedo/π
│   ├── microfacet_reflector.rs      # MicrofacetReflector — GGX conductor/dielectric
│   ├── dielectric.rs                # DielectricMaterial — Fresnel + tint (+ rough)
│   ├── diffuse_light.rs             # DiffuseEmitterMaterial — emissive
│   ├── isotropic.rs                 # IsotropicMaterial — uniform sphere (volumes)
│   ├── mix.rs                       # MixMaterial — weighted stochastic blend
│   ├── coated.rs                    # CoatedMaterial — clear coat over substrate
│   └── gpu.rs                       # GpuMaterialBuffer — tree→flat serialization
│
├── texture/
│   ├── mod.rs                       # Texture trait, TextureCoords/Points/Result
│   ├── impls.rs                     # SolidColor, Checker, Noise, Image, MappedTexture
│   ├── mapping.rs                   # TextureMapping3D/2D, UvGen
│   └── gpu.rs                       # GpuTextureBuffer — tree→flat serialization
│
├── shape/
│   ├── mod.rs                       # Shape3D trait, ShapeObject wrapper
│   └── sphere.rs                    # SphereShape — unit sphere
│
├── planar/
│   ├── mod.rs                       # Region2D trait, PlanarObject, constructors
│   ├── quad.rs                      # QuadRegion
│   ├── tri.rs                       # TriRegion
│   ├── ellipse.rs                   # EllipseRegion
│   ├── annulus.rs                   # AnnulusRegion
│   ├── superellipse.rs              # SuperellipseRegion (Lanczos gamma)
│   ├── rounded_rect.rs              # RoundedRectRegion
│   ├── polygon.rs                   # PolygonRegion
│   ├── function.rs                  # FunctionRegion (predicate + MC area)
│   └── box.rs                       # box3d() — 6 quads → box
│
└── server.rs                        # WebSocket live preview handler
```

### 1.2 Trait Dependency Graph

```text
                             ┌──────────────┐
                             │  Sampler     │←── GAT Session<'s>: SamplingSession
                             │  (unified)   │    begin_pixel(p) → Session
                             └──────┬───────┘
                                    │
                     ┌──────────────┼──────────────┐
                     ▼              ▼              ▼
             ┌────────────┐ ┌──────────┐ ┌──────────┐
             │ Integrator │ │ PDF      │ │ Camera   │
             │<S: Sampler>│ │ (no gen) │ │ :gen_ray │
             │ :li(),     │ │ :sample()│ └──────────┘
             │ :max_bounces│ │ :value()│
             └──────┬─────┘ └──────────┘
                    │
    ┌──────────────┼──────────────┐
    ▼              ▼              ▼
┌─────────┐  ┌──────────┐  ┌──────────┐
│Renderer │  │ Film     │  │ Scene    │
│ :render │  │:add_sample│  │.objects  │
│ :resize │  │:read_image│  │.lights   │
│ :film   │  │:variance │  └──────────┘
└─────────┘  └──────────┘

┌────────────┐  ┌──────────┐  ┌────────────┐  ┌──────────────┐
│Intersectable│  │ Bounded  │  │ Sampleable │  │ MaterialHit  │
│ :intersect │  │:bbox     │  │:sample_pt  │  │ :material    │
└────────────┘  └──────────┘  │:pdf        │  └──────────────┘
                              └────────────┘
       ▲            ▲               ▲              ▲
       │            │               │              │
       └────────────┴───────┬───────┴──────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
       ┌───────────┐ ┌───────────┐ ┌───────────┐
       │ShapeObject│ │PlanarObj  │ │Transform  │
       │<S:Shape3D>│ │<R:Region2D│ │Object     │
       │+Material  │ │ +Surface  │ │<T:Trans,O>│
       └───────────┘ └───────────┘ └───────────┘
                            │
                    ┌───────┴────────┐
                    ▼                ▼
              ┌──────────┐    ┌──────────┐
              │ Shape3D  │    │ Region2D │
              │:intersect│    │:contains │
              │:area     │    │:area     │
              └──────────┘    │:sample   │
                             └──────────┘
```

### 1.3 Dispatch Strategy: Enum Over Trait Objects

raytrace-rs follows pbrt-v4's architectural insight — static dispatch via a closed enum is preferred over dynamic dispatch via `dyn Trait` for hot-path types. The chief exceptions are cross-cutting concerns (scene object storage) and the `Custom` material escape hatch.

| Type | Dispatch | Rationale |
|------|----------|-----------|
| `Material` (10 variants) | Enum match | Hot path (every bounce). Inline-able, no vtable |
| `Bsdf` (private trait, impl'd by materials) | Enum → impl method | Inner material logic — callers match Material enum first |
| `Intersectable` | `dyn` (via `Arc<dyn ...>`) | Heterogeneous scene storage; arena planned |
| `Sampler` | Generic+GAT (`Sampler` trait) | Unified `begin_pixel` → `Session`; internally two-stream (Sobol + hash) |
| `PDF` | Trait object `&dyn PDF` from `[PdfKind; N]` | No longer generic; `PdfEnum` deleted, `PdfKind` directly implements `PDF` |
| `Camera` | Trait w/ concrete `PerspectiveCamera` | Multiple camera models expected (ortho, spherical) |
| `Film` | Trait w/ concrete `RgbFilm` | Multiple film types expected (spectral, AOV) |
| `Integrator` | Generic `<S: Sampler>` | GAT session per pixel; `S::Session<'_>` provides next_2d/next_1d |
| `Renderer` | Generic with 4 type params | CPU now; GPU renderer as separate impl later |
| `Shape3D` | `dyn` (via `ShapeObject`) | Heterogeneous shape storage (sphere + future mesh) |
| `Region2D` | `dyn` (via `PlanarObject`) | Heterogeneous planar regions |

---

## 2. Component Deep Dives

### 2.1 Renderer + Integrator

#### Current State

```text
CpuRenderer<I, S, R, SFact, RFact>   ← trait Renderer<W,C,F,S>
├── owns: I (Integrator), SFact, RFact
├── render() → loops over tiles
│   ├── rayon::for_each on tiles
│   │   └── per pixel: sampler.begin_pixel() → session, I.li(session)
│   ├── FilmTile → merge into RgbFilm
│   └── adaptive: check RgbFilm.variance() → early exit
├── resize() → rebuild Film + tile pool
└── reset() → clear film

PathTracingIntegrator<S: Sampler>   ← trait Integrator<S>
├── li(ray, world, lights, session) → Color3
├── max_bounces() → usize
├── bounce loop:
│   ├── world.intersect(ray) → Option<SurfaceInteraction>
│   ├── add emission
│   ├── Russian roulette (bounce ≥ 5, survive = clamp(throughput, 0.05, 1))
│   ├── NEE: sample light → LightPDF → shadow ray → accumulate direct
│   ├── if material delta → trace reflected ray (no MIS)
│   ├── if material NonDelta → bucket_mis() with env + material PDFs
│   │   └── power heuristic between light-sampling and BSDF-sampling
│   ├── if material Split → li_inner() recursively traces delta branch
│   │   while the non-delta branch proceeds normally
│   └── no fixed dim stride — each bounce consumes what it needs
└── li_inner<S: Sampler>() → recursive helper for Split delta tracing,
    bounded by SPLIT_MAX_DEPTH (5) to prevent exponential cascade
```

**Shadow rays**: Implemented as direct-lighting NEE with explicit occlusion check (commit `24808b5`). The shadow ray tests visibility between the hit point and the sampled light before accumulating direct contribution.

**MIS**: One-sample MIS with power heuristic (α=1). `power_heuristic(n_f * pdf_f, n_g * pdf_g)` from `pdf.rs`.

#### Comparison

| Aspect | raytrace-rs | pbrt-v4 | LuxCore | MoonRay | Mitsuba 3 |
|--------|-------------|---------|---------|---------|-----------|
| Integrator abstraction | `Integrator<S>` trait (GAT session) | `Integrator` base → `RayIntegrator` | `RenderEngine` base | `Engine` class | `Integrator<Float, Spectrum>` template |
| Renderer abstraction | `Renderer<W,C,F,S>` trait | No separate renderer | `RenderEngine` includes it | `Engine` drives | No separate renderer |
| Shadow rays | ✅ Yes | ✅ `Unoccluded()` | ✅ Yes | ✅ Yes | ✅ Yes |
| MIS | ✅ Power heuristic (α=1) | ✅ Power heuristic | ✅ Power heuristic | ✅ Power heuristic | ✅ Power heuristic |
| RR | Throughput-based (clamp 0.05-1.0) | Throughput-based | Throughput-based | Throughput-based | Configurable `rr_depth` |
| Adaptive sampling | ✅ Welford variance + convergence | Yes (variance-guided) | Yes | No | No |
| Tiled rendering | ✅ Pre-allocated pool | `ImageTileIntegrator` | Tile engines | Distributed tiles | `ImageBlock` |
| Bounce controls | Single `max_bounces` | Single + `maxDepth` | Per-type (diffuse/glossy/spec) | Per-type | Single |
| Volumes | ConstantMedium only | Full null-scattering | Full | Decoupled ray march | Full |
| BDPT/MLT | No | Yes (BDPT, MLT, SMLT) | Yes (Hybrid Back-Forward) | No | Yes (BDPT, PSSMLT, Manifold) |

### 2.2 Camera

#### Current State

```text
Camera trait:
  generate_ray(sampler: &mut CameraSampler) → CameraRay

PerspectiveCamera:
  CameraConfig { vfov, focus_distance, defocus_angle, look_from, look_at, up, aspect_ratio }
  → generates rays with thin-lens DOF + jittered pixel AA
```

Minimal but sufficient. Single concrete implementation. The `Camera` trait enables orthographic, spherical, or panoramic cameras as future additions.

#### Comparison

| Aspect | raytrace-rs | pbrt-v4 | LuxCore | Falcor | Filament |
|--------|-------------|---------|---------|--------|----------|
| Trait/interface | `Camera` trait | `Camera` TaggedPointer (4 types) | Property config | `Camera` class | `Camera` class |
| DOF | ✅ Thin-lens | ✅ Thin-lens | ✅ Yes | ✅ Yes | ✅ Physical (focalLength, aperture) |
| Motion blur | ✅ ParametricCurve on Ray | ✅ Yes | ✅ Yes | ✅ Yes | No |
| Camera types | Perspective | Perspective, Ortho, Spherical, Realistic | Perspective, etc. | Perspective + scriptable | Perspective only |

### 2.3 Film

#### Current State

```text
Film trait:
  add_sample(x, y, color, ray_count)
  read_image() → RgbImage
  merge_tile(tile)
  variance() → f64
  is_converged(threshold_rel, threshold_abs) → bool

RgbFilm:
  ┌──────────────────────────────────────────────┐
  │ ScreenBuffer: Vec<[f32; 4]> (RGBA accum)    │
  │ SampleCount: Vec<f32> (per-pixel spp)        │
  │ m_2: Vec<[f32; 3]> (Welford M₂ for variance)│
  │ Resolution: (width, height)                  │
  └──────────────────────────────────────────────┘

FilmTile:
  ┌──────────────────────────────────────────────┐
  │ RgbaAccum: Vec<[f32; 4]>                     │
  │ DirtMask: Vec<bool> (modified since last pub)│
  │ Position: (x, y), Size: (w, h)              │
  └──────────────────────────────────────────────┘

Post-processing:
  post_process(color, exposure, tone_map) →
    Reinhard tone-map (optional) → gamma 2 → [u8; 3]
```

Welford's online algorithm: per-pixel `m_2 += (1 - 1/n) * (x - mean)²`, variance = `m_2 / (n-1)`.

Convergence: pixel is converged when `sqrt(variance) / mean < threshold_rel` OR `sqrt(variance) < threshold_abs`.

#### Comparison

| Aspect | raytrace-rs | pbrt-v4 | LuxCore | MoonRay | Mitsuba 3 |
|--------|-------------|---------|---------|---------|-----------|
| Trait | `Film` trait | `Film` TaggedPointer | `Film` class | `OutputDriver` | `Film` plugin |
| Types | `RgbFilm` | `PixelFilm`, `GBufferFilm` | Single + AOVs | Deep image | `HDRFilm`, `SpecularFilm` |
| Variance tracking | ✅ Welford M₂ | Simple variance | No | No | No |
| Adaptive sampling | ✅ Yes | Yes | Yes | No | No |
| AOVs | No | Alpha, albedo, normal | 20+ | LPEs, deep | Limited |
| Tone-mapping | ✅ Reinhard + gamma | No (raw output) | Yes | Yes | No |
| Denoiser | No | No | Yes (OIDN, OptiX) | Yes | No |
| Tile rendering | ✅ Pre-allocated pool | `ImageTileIntegrator` | Tile engines | Distributed | `ImageBlock` |
| Export | PPM + PNG | EXR | EXR, PNG | Deep EXR | EXR, PNG |

### 2.4 Material System

#### Current State

```text
Material enum (9 variants):
├── Void                           # No material (miss)
├── DiffuseReflector(DiffuseReflector) # albedo/π
├── MicrofacetReflector(MicrofacetReflector) # GGX conductor/dielectric (Fresnel dispatch)
├── Dielectric(DielectricMaterial) # Fresnel/refract/reflect + tint (+ rough)
├── DiffuseEmitter(DiffuseEmitterMaterial) # emissive
├── Isotropic(IsotropicMaterial)   # uniform sphere (volumes)
├── Mix(MixMaterial)               # weighted stochastic blend
├── Coated(CoatedMaterial)         # clear coat over substrate
└── Custom(Arc<Material>)         # Escape hatch

Bsdf trait (private to material module, 7 methods):
  scatter(&self, wo, si, &mut next_dim) → Option<BsdfScatter>
  eval(&self, wo, wi, si) → Color3
  pdf(&self, wo, wi) → f64
  pdf_kind(&self, wo, si) → Option<PdfKind>
  emitted(&self, si) → Color3
  to_gpu(&self, ...) → GpuMaterialType
  gpu_texture_index(&self) → Option<u32>

BsdfScatter enum (3 variants):
  Delta     { wi, f_cos, eta }
  NonDelta  { pdf_kinds: [Option<PdfKind>; MAX_BSDF_STRATS] }
  Split     { delta_wi, delta_f_cos, delta_eta, non_delta_pdf_kinds }

PdfKind enum:
  Cosine { normal } | Ggx { wo, normal, alpha } |
  UniformSphere | UniformHemisphere | Delta { normal }
```

**Key patterns**:

- `Bsdf` is a **private trait** — external code interacts only through the `Material` enum.
- GPU serialization (`GpuMaterialBuffer`) flattens the material tree into a `Vec<GpuMaterialNode>` with child references as indices.
- Composition types (Mix, Coated) use `Box<Material>` (not `Box<dyn Bsdf>`) — the enum dispatch recurses through children.
- Coated material uses an **internal random walk** for rough coating (not a hard Fresnel-split). When the coating scatter produces a Split (one delta + one non-delta branch), the delta branch is handled recursively inside `scatter()` via a random-walk loop (up to `MAX_INTERNAL_BOUNCES = 5`). This correctly handles rough coatings where the coating layer has a glossy lobe.
- The `next_dim` closure approach replaces the old fixed-size `SampleDims` struct — materials consume exactly as many random values as they need, eliminating unused reserved fields.
- `MAX_BSDF_STRATS = 8` gives spare capacity in the PDF kind array for future composition strategies (e.g., environment light PDF).

#### Comparison

| Aspect | raytrace-rs | pbrt-v4 | LuxCore | Mitsuba 3 | Filament |
|--------|-------------|---------|---------|-----------|----------|
| Dispatch | `Material` enum (10 variants) | `Material` TaggedPointer (11) | `Material` virtual base | `BSDF` template plugin | Shader specialization (`.mat` DSL) |
| BSDF per element | Single (match→impl) | Single `BxDF` via TaggedPointer | Single (virtual) | Single (plugin) | Hardcoded shader variants |
| Composition | `Mix` + `Coated` enums | `LayeredBxDF<T,B>` template | `Mix` material type | `blendbsdf`, `mask` plugins | Coating shader variants |
| Layered approach | Fresnel-split + internal random walk for rough coating | MC random walk (Guo et al.) | Hierarchical blend | Plugin composition | Layered shader compilation |
| Rough coating | ✅ Internal random walk (MAX_INTERNAL_BOUNCES=5) | ✅ MC layered walk | ✅ GlossyCoating | ✅ Yes via roughdielectric | N/A |
| Spectral | RGB only | SampledSpectrum (4-point) | Partial (dispersion) | ✅ Full spectral + polarization | RGB only |
| GPU serialization | ✅ Tree→flat node array | `pstd::vector` (arena) | N/A | N/A | Material compiler |
| Delta routing | ✅ `BsdfScatter::Split` variant traces both delta + non-delta | `BxDFFlags` bitmask | Runtime check | Plugin-dependent | N/A |
| Mix efficiency | `eval()` evaluates both children every sample | Evaluates selected lobe only | Same tradeoff | Plugin-dependent | Compiled per variant |

#### Issues

1. **Mix `eval()` evaluates both children every sample.** `scatter()` picks one stochastically via `next_dim()` closure but `eval()` always evaluates both. Acceptable for 2 children; would not scale to N. (The `Mix` material also now handles the one-delta case via `BsdfScatter::Split` — the delta child's contribution is pre-computed, and only the non-delta child is evaluated in `eval()`.)

2. **Coated IOR reads from the coating's DielectricMaterial** (resolved: hardcoded IOR=1.5 was replaced with the actual dielectric IOR during the coated decomposition refactor).

### 2.5 Shapes + Geometry

#### Current State

```text
Shape3D trait:
  intersect_shape(&self, ray, t_range) → Option<Hit>
  bounding_box(&self) → Aabb
  area(&self) → f64
  sample(&self, u, v, t) → Vec3
  sample_direction(&self, origin, u, v, t) → (Vec3, f64)

ShapeObject<S: Shape3D, M: Material>
  └── wraps Shape3D + Material → implements Intersectable + Bounded + Sampleable + MaterialHit

SphereShape  — unit sphere, solid-angle sampling, UV generation

Region2D trait:
  contains(a, b) → bool
  area() → f64
  sample(u, v) → (f64, f64)
  uv(a, b) → (f64, f64)
  bounding_box_area() → f64

PlanarObject<R: Region2D, S: Surface>
  └── wraps Region2D + Surface → implements Intersectable + Bounded + Sampleable + MaterialHit

9 region types: Quad, Tri, Ellipse, Annulus, Superellipse, RoundedRect, Polygon, Function, box3d()
```

**Key insight**: `PlanarObject` + `Region2D` is a parametric surface pattern. A 2D region (defined in `[a,b]×[c,d]` space) is mapped into 3D by a `Surface` (currently a fixed normal + embedding). This gives 9 shape types from one trait + generic wrapper.

#### Comparison

| Aspect | raytrace-rs | pbrt-v4 | LuxCore | Falcor | Filament |
|--------|-------------|---------|---------|--------|----------|
| Primitive count | 2 shapes + 9 planar regions | 7 (Sphere, Cyl, Disk, Tri, Bilinear, Curve, Hair) | ExtMesh (triangle) + primitives | Triangle meshes, curves, SDF | Mesh only (ECS) |
| Triangle mesh | ❌ Not implemented | ✅ `TriangleMesh` | ✅ `ExtTriangleMesh` | ✅ DXR BLAS | ✅ glTF |
| Parametric surfaces | ✅ `Region2D` trait (closed form) | ✅ Bilinear patch | No trait | No | No |
| Implicit primitives | Sphere only | Sphere, Cylinder, Disk | Limited | No | No |
| Curve/hair | No | ✅ `Curve` | No | ✅ Curves | No |
| Texture on primitives | ✅ UV generation per type | ✅ UV per type | ✅ UV | ✅ UV | ✅ UV |

**Biggest gap**: No triangle mesh support. This blocks loading any real-world 3D scene (OBJ, glTF, PLY).

### 2.6 Acceleration Structures

#### Current State

```text
BvhNode enum:
  Empty
  Interior { left, right, bbox }
  Leaf { object, bbox }

Build: SAH binning (32 bins, 3 axes, parallel via rayon::join)

FlatBvh:
  FlatBvhNode [64-byte aligned, 64 bytes each]:
    ┌────┬────┬────┬────┬────┬────┬────┬────┐
    │minx│miny│minz│pad │maxx│maxy│maxz│pad │  16 B
    ├────┴────┴────┴────┴────┴────┴────┴────┤
    │ child[0]_index │ child[1]_index │ flags │  12 B
    ├─────────────────────────────────────────┤
    │ primitive_count │ primitive_offset       │  8 B
    └─────────────────────────────────────────┘   = 36 B payload (padded to 64 B)

  Traversal: iterative with fixed 64-entry stack
  Near-first child ordering for cache efficiency
```

#### Comparison

| Aspect | raytrace-rs | pbrt-v4 | LuxCore | MoonRay | Falcor |
|--------|-------------|---------|---------|---------|--------|
| Build | SAH binning (32 bins) | SAH binning | SAH BVH | Custom BVH | DXR driver |
| Node size | 64 B (cache line) | 32 B | Variable | SIMD-friendly | GPU BLAS |
| Traversal | Iterative + 64-entry stack | Recursive | Recursive | ISPC vectorized | DXR hardware |
| Parallel build | ✅ `rayon::join` | ✅ `ParallelFor` | Yes | Yes | GPU-accelerated |
| Embree support | No | Optional | Optional | ✅ Yes | N/A |
| Flat array | ✅ | ✅ `LinearBVH` in v3 | No | No | GPU buffers |

The flat BVH is production-quality. 64-byte alignment matches cache line size. Iterative traversal avoids recursion depth issues.

### 2.7 Sampling System

#### Current State

```text
Sampler trait (unified, GAT-based):
  Session<'s>: SamplingSession (GAT)
  begin_pixel(p, sample_idx) → Session<'s>
  samples_per_pixel() → u32
  └── per-pixel session cannot outlive the pixel — borrow-checker enforced

SamplingSession trait:
  next_2d() → (u64, u64)     // correlated 2D (Sobol / stratified)
  next_1d() → f64              // independent 1D (hash)
  next_pixel_2d() → (f64, f64) // pixel-filter jitter (may share next_2d)

Two-stream internal implementation:
  Internally, the production SobolSession pairs:
    SampleStreamWriter (stateful Sobol 2D stream)
      [next_pair advances by 1 per call → dims 0-1, 2-3, 4-5...]
    HashRng (SplitMix64 independent 1D)

Concrete types:
  SobolSampler    — production (Sobol + HashRng)
  StreamRngPair   — generic adapter for any (SampleStream, SamplerRng)
  ThreadLocal<T>  — per-thread slot via rayon + Mutex

Legacy traits (still present, used internally):
  QmcSampler      — pure sample(n, d) → f64 (stateless, Sync)
  SampleStream    — next_2d() → (f64, f64)
  SamplerRng      — next() → f64
```

**No fixed dimension stride.** Each bounce consumes only what it needs:

| Use | Dimenions | Stream |
|-----|-----------|--------|
| Material direction (BSDF lobe, GGX, etc.) | 2 | SampleStream `next_2d()` |
| MIS direction (from selected PDF) | 2 | SampleStream `next_2d()` |
| Russian roulette | 1 | SamplerRng `next_1d()` |
| Material lobe selection / MIS strategy selection | 1–2 | SamplerRng `next_1d()` |
| Light selection (NEE) | 1 | SamplerRng `next_1d()` |

Dielectric: 1 RNG (Fresnel split) + 0 Sobol (delta) = **1 dim total**.
Lambertian: 1 RNG (lobe sel) + 2 Sobol (direction) + 1 RNG (light sel) + 2 Sobol (MIS dir) = **3 RNG + 4 Sobol**.

#### Comparison

| Aspect | raytrace-rs | pbrt-v4 | LuxCore | MoonRay | Mitsuba 3 |
|--------|-------------|---------|---------|---------|-----------|
| API | `Sampler` trait with GAT `Session<'s>: SamplingSession` | Stateful `Get1D()`, `Get2D()` | Stateful class | Stateful `SequenceID` | Stateful plugin |
| Thread safety | `ThreadLocal<T>` + `Mutex` per thread | Clone per thread | Per-thread RNG | Per-thread state | Clone per thread |
| Determinism | Same per-pixel seed → same sequences | Same sequence per pixel | Per-pixel + random pass | Pixel+purpose hash | Plugin-dependent |
| Primary sequence | Sobol (21200d, Gray-code) via `SampleStream` + `HashRng` for 1D | Multiple (Sobol, Halton, PMJ, etc.) | Sobol | CMJ/PMJ + hash | Multiple (Independent, Stratified, Sobol, etc.) |
| Scrambling | Digital shift (per-dim hash via splitmix) | Owen, FastOwen, PermuteDigits | None (pixel jitter) | Precomputed tables | None |
| Dim management | Variable per bounce (GAT session consumed as needed) | Implicit (sampler tracks dim) | Per engine | Per purpose | Per plugin |
| Blue noise | No | ✅ BlueNoiseSampler, ZSampler | No | ✅ Precomputed tables | No |
| Max dims | 21200 (full Joe & Kuo dataset) | Unbounded | Unbounded | Precomputed | Unbounded |
| Two-stream | ✅ Sobol for 2D, hash for 1D discrete decisions | Single stream | Single stream | Single stream | Single stream |

**Design note**: The architecture is a hybrid — the `Sampler` trait with GAT session provides a unified interface, while the internal implementation uses two streams (Sobol `SampleStreamWriter` for correlated 2D + `HashRng` for independent 1D). The compiler monomorphizes the integrator for each concrete `Sampler` type, preserving zero runtime overhead. The `QmcSampler` pure `sample(n,d)` API is retained as a building block and is still advantageous for GPU (no per-thread sampler state to maintain).

### 2.8 Texture System

#### Current State

```text
Texture trait:
  value(&self, coords: TextureCoords, points: TexturePoints) → TextureResult

Implementations:
  SolidColor        — constant Color3
  CheckerTexture    — 3D checker with scale, smoothstep edge
  NoiseTexture      — Perlin noise with scale + turbulence
  ImageTexture      — PNG/image via `image` crate
  MappedTexture     — wraps any texture + explicit TextureMapping3D or TextureMapping2D

TextureMapping3D:
  Cubemap, Spherical, Planar  (world-space → uv)

TextureMapping2D:
  Uv, Planar  (per-primitive UV → uv)

GpuTextureBuffer:
  Flattens texture tree → Vec<GpuTextureNode> (GpuTextureType enum + params)
```

### 2.9 Scene Management

#### Current State

```text
Scene:
  config: SceneConfig
  objects: Vec<Arc<dyn Intersectable>>
  important_objects: Vec<Arc<dyn Sampleable>>  (lights and emissive objects)

  builder API: add_object(), add_light(), build() → BVH construction
  10 pre-built scene functions

ConstantMedium<T, SURFACE>:
  Volumetric wrapper — homogeneous scattering medium with density
  Uses IsotropicMaterial for phase function
```

**Ownership**: Currently `Vec<Arc<dyn Intersectable>>` — flexible but imposes ref-counting overhead and prevents flat storage for GPU upload. The arena refactor plan (`docs/arena-refactor-plan.md`) designs a bump-allocated arena with typed storage and automatic light detection.

#### Comparison

| Aspect | raytrace-rs | pbrt-v4 | LuxCore | Filament |
|--------|-------------|---------|---------|----------|
| Ownership | `Vec<Arc<dyn ...>>` | `pstd::vector` (PMR arena) | `boost::shared_ptr` | ECS (SoA) |
| Scene graph | Flat list + BVH | Flat list + BVH | Tree (materials reference) | Entity hierarchy |
| Light detection | Manual `add_light()` | Automatic (areaLight ptr) | Automatic | N/A (punctual only) |
| GPU upload path | Material/texture serialization | `pstd::vector` → GPU buffer | OpenCL buffer image | Buffer upload |
| Scene composability | Builder functions | `ParseFile()` | Python API | gltfio |
| Volumes | `ConstantMedium` wrapper | `Medium` interface | Full volume stack | N/A |

---

## 3. Production Renderer Comparison

### 3.1 Architectural Comparison Matrix

| Feature | raytrace-rs | pbrt-v4 | LuxCore | appleseed | MoonRay | Falcor | Mitsuba 3 | Filament | renderling |
|---------|-------------|---------|---------|-----------|---------|--------|-----------|----------|-----------|
| **Language** | Rust | C++17 (+CUDA) | C++ / Python | C++ / Python | C++ / ISPC | C++ / Slang | C++17 / Python | C++ / GLSL | Rust (wgpu) |
| **LOC (est.)** | ~10K | ~200K | ~350K | ~680K | ~2M+ | ~300K | ~180K | ~250K | ~30K |
| **Dispatch model** | Enum match (static) | TaggedPointer (static) | Virtual (dynamic) | Virtual (dynamic) | ISPC SPMD | Virtual + shader | Template + plugin | ECS + shader gen | Uber-shader + slab |
| **Polymorphism** | Closed enum + `dyn` escape | Closed TaggedPointer | Open virtual hierarchy | Open virtual hierarchy | ISPC + data-driven | Data-driven (shader) | Plugin virtual | Compile-time | Single `MaterialDescriptor` |
| **Rendering** | Offline CPU | Offline CPU + GPU | Offline CPU/GPU | Offline CPU | Offline CPU + GPU | Real-time GPU | Offline CPU/GPU | Real-time GPU | Real-time GPU |
| **MIS** | ✅ Power heuristic | ✅ Power heuristic | ✅ Power heuristic | ✅ Power heuristic | ✅ Power heuristic | Shader-based | ✅ Power heuristic | N/A | N/A (raster) |
| **Shadow rays** | ✅ Explicit NEE | ✅ `Unoccluded()` | ✅ Yes | ✅ Yes | ✅ Yes | Shader-based | ✅ Yes | N/A | N/A (raster) |
| **Adaptive sampling** | ✅ Welford variance | Yes | Yes | Yes | No | No | No | N/A | No |
| **Material composition** | `Mix` + `Coated` enums (Fresnel-split) | `LayeredBxDF<T,B>` (MC walk) | `Mix` + `GlossyCoating` | Layered + OSL | Layered + ISPC | Shader parameters | Plugin `blendbsdf` | Compile-time | Flat PBR descriptor only |
| **BSDF dispatch** | Enum match → impl method | TaggedPointer → template | vtable | vtable | ISPC loops | GPU shader | Template + vtable | Generated GPU code | Uber-shader branch |
| **Triangle meshes** | ❌ Not implemented | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Acceleration** | FlatBVH (64B, SAH) | BVH + Embree | BVH + Embree | BVH + Embree | Embree | DXR BLAS/TLAS | Embree/OptiX/kd-tree | None (raster) | None (raster) |
| **Spectral rendering** | RGB only | ✅ 4-point spectrum | Partial (dispersion) | RGB only | RGB only | RGB only | ✅ Full + polarization | RGB only | RGB only |
| **Differentiable** | No | No | No | No | No | No | ✅ Yes (Dr.Jit AD) | No | No |
| **GPU path** | Serialization only (Vulkan + rust-gpu planned) | CUDA/OptiX | OpenCL/CUDA | No | CPU+GPU hybrid (XPU) | DXR/CUDA/OptiX | CUDA/OptiX/LLVM | Vulkan/Metal/GL | wgpu + rust-gpu SPIR-V |
| **Active development** | Active | Maintenance | Slow | Stalled | Active (DreamWorks) | Active (NVIDIA) | Active (EPFL) | Active (Google) | Active (alpha) |

### 3.2 Where raytrace-rs Excels vs Production Renderers

| Area | Advantage | Why |
|------|-----------|-----|
| **Dispatch performance** | Enum match fast path vs virtual dispatch | pbrt-v4 matches this with TaggedPointer; LuxCore/appleseed use vtable |
| **Sampler API** | Unified GAT-based `Sampler` trait with two-stream internals (Sobol for 2D, hash for 1D) — zero `dyn` dispatch, borrow-checker prevents cross-pixel state corruption | All production renderers use stateful sampler APIs with no compile-time cross-pixel guarantees |
| **QMC dim discipline** | Variable per-bounce consumption, two-stream separates correlated from independent decisions | pbrt-v4 uses implicit cursor — no per-bounce assertion |
| **Adaptive sampling** | Welford's online variance + convergence mask | Only appleseed has comparable adaptive sampling quality |
| **Tile pool** | Pre-allocated, reused across passes | pbrt-v4 allocates per-pass; raytrace-rs eliminates this |
| **GPU material serialization** | Tree→flat at scene-build time | Clean boundary for future GPU integrator |
| **Codebase size** | ~6,500 LOC — comprehensible | Every production renderer is 30-300× larger |
| **Rust safety** | No use-after-free, no data races | C++ renderers all use manual memory management |

### 3.3 Where raytrace-rs Lags

| Gap | Production standard | Effort to close |
|-----|-------------------|-----------------|
| **Triangle meshes** | Universal (OBJ/glTF/PLY) | Medium (new file + parser) |
| **Arena allocation** | pbrt-v4 (PMR), MoonRay (custom) | Medium (~300 lines) |
| **Bidirectional path tracing** | Multiple renderers support it | Large (new integrator) |
| **Volume rendering** | Full null-scattering (pbrt-v4), decoupled (MoonRay) | Large |
| **Spectral rendering** | pbrt-v4, Mitsuba 3 | Large (foundational change) |
| **Advanced sampling** | CMJ/PMJ (MoonRay), Blue noise (pbrt-v4) | Medium (~100 lines) |
| **Complete transform system** | Full 4×4 matrix transforms | Small (~60 lines) |
| **Point3/Color3 newtypes** | Idioimatic in Rust ecosystem | Small (~50 lines) |
| **AOVs / deep images** | Universal in film renderers | Medium |
| **USD/Hydra integration** | MoonRay, appleseed | Large (not applicable) |
| **Denoiser integration** | OIDN (appleseed, LuxCore), OptiX | Medium |

### 3.4 Rust Ecosystem Context

raytrace-rs's architectural patterns mirror the Rust graphics ecosystem's evolution:

| Pattern | In raytrace-rs | Used by |
|---------|---------------|---------|
| **Enum dispatch for hot-path polymorphism** | `Material` (10 variants), `BsdfScatter` (3), `PdfKind` (5) | Performant Rust path tracers; mirrors pbrt-v4's TaggedPointer |
| **Generic type param for strategy pattern** | `Integrator<S: Sampler>`, `Renderer<W,C,F,S>`, `Sampler` (GAT) | `rend3`, `renderling` |
| **Trait object for heterogeneous collections** | `Vec<Arc<dyn Intersectable>>`, `Vec<Arc<dyn Sampleable>>` | All Rust renderers with scene graphs |
| `enum_dispatch`-equivalent pattern | Manual (match + method delegation) | `enum_dispatch` crate auto-generates this |
| **Flat storage + SOA for hot paths** | `FlatBvh` (64B nodes), `RgbFilm` (parallel arrays) | `rend3` ECS, `kajiya` real-time path tracer |
| **Slab allocator for GPU data** | Serialization only (not yet slab-backed) | **`renderling`** (`crabslab` + `craballoc` — dirty-range tracking, bulk commit) |
| **Texture atlas → Image2dArray** | `GpuMaterialNode.texture_index` field | **`renderling`** (`Atlas` packs images into 2D array texture) |
| **Uber-shader + SPIR-V dispatch** | No GPU compute yet | **`renderling`** (rust-gpu `#[spirv]` → SPIR-V → wgpu, single uber-shader) |
| **WebGPU/WGSL target** | GPU serialization groundwork | `wgpu`, `renderling`, `kajiya` |
| `rayon` parallel iteration | `CpuRenderer` tile loop | Universal in Rust parallel compute |
| **repr(C) CPU↔GPU structs** | `GpuMaterialNode`, `GpuTextureNode`, `FlatBvhNode` | `renderling` (`MaterialDescriptor`, `CameraDescriptor`, `PrimitiveDescriptor`) |

---

### 3.5 renderling Deep-Dive — Blueprint for GPU Backend

renderling is the closest Rust+GPU architectural reference for raytrace-rs's planned GPU path tracer. It is **not** a path tracer (it's forward+ rasterization), but its CPU↔GPU dataflow, shader compilation pipeline, and resource management patterns directly inform a WGSL-based GPU backend.

#### 3.5.1 Architecture Comparison

| Dimension | renderling | raytrace-rs (current) | raytrace-rs (GPU target) |
|-----------|-----------|-----------------------|--------------------------|
| **Purpose** | Real-time rasterization (forward+) | Offline CPU path tracer | Offline GPU path tracer |
| **Pattern** | Struct-of-subsystems + slab allocator | Traits + `Arc<dyn Trait>` dispatch | Traits for CPU path, slab/flat for GPU |
| **Renderer** | `Stage::render()` — no trait | `Renderer<W,C,F,S>` trait | Keep trait; GPU impl delegates to compute shaders |
| **Material** | Flat `MaterialDescriptor` (PBR only) | Recursive enum + `Bsdf` trait | ✅ `GpuMaterialNode` already designed (discriminant + children + texture_index) |
| **Scene data** | Slab-backed resources (GPU-first) | `Vec<Arc<dyn Intersectable>>` + BVH | Slab allocator replacing `Vec<Arc<>>`; `FlatBvh` already GPU-ready |
| **Shader lang** | Rust `#[spirv]` → SPIR-V → naga → WGSL | N/A | Rust `#[spirv]` → SPIR-V → Vulkan |
| **Data transfer** | `crabslab` commit (dirty-range tracking) | Serialization only (full rebuild) | Slab commit for incremental scene updates |
| **Texture management** | `Atlas` → `Image2dArray` | `image` crate + CPU | ✅ `GpuMaterialNode.texture_index` aligns with atlas model |

#### 3.5.2 What to Adopt From renderling

**1. Slab allocator pattern (`crabslab` / `craballoc`)**

renderling allocates every GPU resource from a flat u32 slab backed by CPU-accessible memory. `commit()` writes only modified ranges to a `wgpu::Buffer`. This solves:

- **Dirty-range tracking**: No need to re-upload the entire scene when a single material changes
- **Uniform GPU layout**: All `MaterialDescriptor`, `CameraDescriptor`, `PrimitiveDescriptor` live in the same storage buffer — one bind group entry
- **No per-frame allocation**: Resources are allocated once, modified in place

raytrace-rs currently serializes the full material/texture tree each frame. A slab allocator would make updates incremental.

```rust
// renderling pattern:
let slab = Slab::new(device, &BufferDescriptor { /* storage */ });
let mat_id = slab.new_value(MaterialDescriptor::default());
mat_id.modify(|m| m.albedo_factor = Vec4::splat(0.8));
// later:
slab.commit(&mut encoder);  // uploads only dirty slab pages

// raytrace-rs equivalent (current):
let buf = GpuMaterialBuffer::from_material(&material);
buf.flatten();  // full rebuild every time
```

**2. Texture atlas (`Atlas` → `Image2dArray`)**

renderling packs all material textures into a single `wgpu::TextureDimension::D2` array texture. The shader reads `atlas.sample(sampler, uv, texture_layer)` — no unbounded texture binding problem.

raytrace-rs already has `GpuMaterialNode.texture_index: u32` — this maps directly to the atlas layer index. The existing `GpuTextureBuffer` flattens the texture tree into `GpuTextureNode` vecs, which can drive atlas packing.

**3. Single large bind group**

renderling uses one bind group (descriptor set 0) with 13+ entries: geometry slab, material slab, light slab, atlas texture + sampler, irradiance/specular cubemaps, BRDF LUT, shadow map array.

For a GPU path tracer, a single bind group would contain:

- Geometry storage buffer (FlatBvh nodes + primitive data)
- Material storage buffer (GpuMaterialNode array + params)
- Texture atlas (array texture + sampler)
- Light storage buffer
- Ray input / output storage buffers (for wavefront dispatch)

**4. Build-time shader pipeline**

renderling uses `cargo-gpu` to compile Rust `#[spirv]` shaders → `.spv` → `build.rs` auto-generates `linkage/*.rs` with wgpu module/entry-point pairs.

For raytrace-rs, the same pipeline works with Vulkan: write ray intersection and BSDF evaluation in Rust `#[spirv]` shaders, compile at build time, load SPIR-V modules into Vulkan compute pipelines. This matches the project's stated GPU target exactly.

**5. Uber-shader dispatch for material types**

renderling's fragment shader branches on `material.has_lighting` and `debug_channel` at runtime — no shader variant generation.

raytrace-rs's `GpuMaterialType` enum (Void=0 through Coated=7) is designed for exactly this: a SPIR-V switch on `material_type` selects the BSDF evaluation path. Composition nodes (Mix/Coated) recurse through `child_a`/`child_b` indices:

```rust,ignore
// Future rust-gpu compute shader — mirroring raytrace-rs's GpuMaterialType enum
// Path tracer uses compute shaders (not fragment) for explicit wavefront control
#[spirv(compute)]
fn eval_bsdf(mat: &GpuMaterialNode, wo: Vec3, wi: Vec3) -> Vec3 {
    match mat.material_type {
        0 => { /* DiffuseReflector — sample albedo from texture_index */ }
        1 => { /* MicrofacetReflector — GGX conductor/dielectric by fresnel_kind */ }
        2 => { /* Dielectric — Fresnel + refract/reflect (is_rough dispatch) */ }
        5 => { /* Mix — recurse via child_a, child_b, blend by weight */ }
        6 => { /* Coated — Fresnel-split between coating and substrate */ }
        // ...
    }
}
```

#### 3.5.3 What NOT to Adopt

| renderling pattern | Why skip for raytrace-rs |
|--------------------|-------------------------|
| **Struct-of-subsystems** (no traits) | raytrace-rs needs trait boundaries for CPU/GPU interchangeability — `Renderer` trait allows `CpuRenderer` and future `GpuRenderer` to coexist |
| **Uber-shader for all paths** | A path tracer benefits from kernel specialization (wavefront: `ray_gen`, `intersect`, `shade`, `connect`) — a single uber-shader would have divergent warp utilization |
| **Forward+ tile culling** | Not applicable to path tracing (light transport is global, not screen-space tiled) |
| **Flat MaterialDescriptor (no composition)** | raytrace-rs's recursive material tree (Mix, Coated) is a superset — the `GpuMaterialNode` child-index pattern is the right design |
| **Render graph / frame graph** | Path tracing is a single kernel loop (wavefront or mega-kernel), not a DAG of raster passes |

#### 3.5.4 renderling Comparison: Material System

| Aspect | renderling | raytrace-rs (CPU) | raytrace-rs (GPU design) |
|--------|-----------|-------------------|--------------------------|
| **Representation** | Flat `#[repr(C)] MaterialDescriptor` | Recursive `Material` enum (9 variants) | `GpuMaterialNode` array + flat params buffer |
| **PBR model** | Metallic-roughness (albedo, metalness, roughness, AO, normal, emissive) | General BSDFs (Diffuse, GGX conductor/dielectric, Dielectric, Mix, Coated) | Same BSDF set tagged by `GpuMaterialType` discriminant |
| **Composition** | None | Mix (weighted blend), Coated (Fresnel-split) | Child index recursion in shader |
| **Textures** | Atlas: `Image2dArray` sampled by layer | CPU `image` crate → `Texture::value()` | `texture_index` → atlas layer |
| **GPU upload** | Slab commit (dirty-range tracking) | N/A (CPU only) | Slab-backed `GpuMaterialBuffer` |
| **Shader dispatch** | Uber-shader branch on `has_lighting` flag | Enum match → method dispatch | `#[spirv]` match on `material_type` |

#### 3.5.5 Alignment: Existing raytrace-rs GPU Serialization vs renderling Patterns

| raytrace-rs component | Already GPU-ready | Needs renderling-inspired work |
|-----------------------|-------------------|-------------------------------|
| `GpuMaterialNode` (`repr(C)`, discriminant + children + texture_index) | ✅ Yes — maps to rust-gpu `match` dispatch | Replace `Vec<GpuMaterialNode>` with slab allocator (dirty-range tracking) |
| `GpuTextureNode` (`repr(C)`, type + params) | ✅ Yes — maps to atlas layer | Replace with texture atlas (`Image2dArray` packing) |
| `FlatBvh` (64B `repr(C)` nodes, iterative traversal) | ✅ Yes — byte-identical layout for GPU (f64→f32) | No structural changes needed; BVH build remains on CPU |
| `GpuMaterialType` / `GpuTextureType` enums (`repr(u32)`) | ✅ Yes — discriminants match SPIR-V match cases | No changes needed |
| Scene ownership (`Vec<Arc<dyn ...>>`) | ❌ Not GPU-ready | Arena refactor → slab-backed flat arrays |
| Sampler (`Sampler` trait + GAT session, `QmcSampler` pure `sample(n,d)` internally) | ✅ Yes — stateless base, trivially maps to GPU | No changes needed; GPU `n = pixel_x * width + pixel_y; d = bounce * 11 + dim` |
| **→ The GPU serialization boundary is well-designed. The gap is memory management (slab allocator) and shader code (`#[spirv]` kernels), not data structures.** | | |

#### 3.5.6 Staged GPU Migration Path

```text
Phase A — Slab-backed CPU (now → 1 month)
├── Replace Vec<GpuMaterialNode> + Vec<u8> params with crabslab-style slab
├── Replace Vec<GpuTextureNode> with texture atlas
├── Add dirty-range tracking → incremental upload
└── Validate: CpuRenderer still works, serialization is incremental

Phase B — rust-gpu compute shaders (1-3 months)
├── Port FlatBvh traversal to `#[spirv(compute)]` shader
├── Port BSDF eval (Diffuse, GGX, Dielectric) to rust-gpu functions
├── Implement ray-gen kernel (camera + pixel AA)
├── Implement path-trace kernel (bounce loop w/ MIS + shadow ray)
└── Validate: pixel-match CpuRenderer output

Phase C — GpuRenderer + Vulkan backend (3-6 months)
├── Implement Renderer<..., GpuDevice, ...>
├── ash (Vulkan) context + surface integration (or headless via VK_KHR_surface)
├── Texture atlas upload → vk::Image + vk::Sampler
├── Slab commit → vk::Buffer writes
└── cargo-gpu SPIR-V compilation pipeline
└── Validate: GpuRenderer matches CpuRenderer output

Phase D — Wavefront optimization (6-12 months)
├── Split mega-kernel into wavefront stages (ray_gen → intersect → shade → connect)
├── Add tile/wavefront scheduling for GPU occupancy
├── Indirect dispatch for variable-length ray queues
└── Leverage shared Rust code between CPU (enum dispatch) and GPU (rust-gpu match)
```

renderling's contribution to each phase:

- **Phase A**: `crabslab` + `craballoc` pattern, `Atlas` texture packing, dirty-range commit model
- **Phase B**: Build-time shader pipeline (rust-gpu `#[spirv]` → SPIR-V → Vulkan), single-bind-group pattern
- **Phase C**: `Context` + `Frame` abstraction, winit integration, headless mode
- **Phase D**: Indirect dispatch pattern (renderling's compute culling for inspiration); shared Rust code between CPU and GPU via rust-gpu

---

## 4. Code Quality

### 4.1 TODOs (20 total)

| File | Line | Category | TODO |
|------|------|----------|------|
| `main.rs` | 125 | opt-preview | Replace full-frame blit with tile/dirty-rect blits |
| `main.rs` | 135 | viewport | Implement aspect-fit viewport (letterbox/pillarbox) |
| `main.rs` | 408 | gpu | Keep scene-construction boundary for future GPU pipeline |
| `main.rs` | 455 | opt-preview | Propagate cancellation signal for clean exit |
| `main.rs` | 456 | opt-preview | Tile scheduler with periodic publish |
| `main.rs` | 493 | gpu | Keep scene-construction boundary for future GPU pipeline |
| `main.rs` | 538 | gpu | Split accel build from upload/flatten for profiling |
| `hittable.rs` | 13 | type-safety | Point3/Vec3/Color3 are aliases — fields can be mixed up |
| `hittable.rs` | 19 | mapping-2d3d | Move 3D mapping inputs into dedicated payload |
| `hittable.rs` | 75 | type-safety | Point3/Vec3/Color3 are aliases |
| `transform.rs` | 9 | optional | Add rotation/scale variants |
| `transform.rs` | 10 | feat | Macro DSL for ergonomic transform chaining |
| `transform.rs` | 261 | optional | Implement RotateX / RotateZ with cached sin/cos |
| `transform.rs` | 262 | optional | Support transform composition helpers |
| `transform.rs` | 263 | feat | Scene-builder helpers/macros for TransformObject |
| `vec3.rs` | 14 | type-safety | Convert to newtype Point3(Vec3) |
| `vec3.rs` | 21 | type-safety | Convert to newtype Color3(Vec3) |
| `ray.rs` | 7 | refactor | Refactor to Direction(Vec3) newtype |
| `integrator/path_tracer.rs` | 6 | gpu | Mirror boundary in path-trace kernel / WGSL entrypoint |
| `texture/mod.rs` | 12 | renderer-agnostic | Replace direct path-tracer handoff |

**Categories**: GPU pipeline (4), type-safety (4), opt-preview (3), transforms (4), refactor (1), renderer-agnostic (1), mapping (1), gpu-sampler (1). The new `SPLIT_MAX_DEPTH` and `MAX_BSDF_STRATS` constants add to the design documentation but are not TODOs.

**Zero**: `todo!()`, `unimplemented!()`, FIXMEs, HACKs, XXXs.

### 4.2 Unsafe Blocks (2 + 1 dead_code)

| File | Line | Context | Risk |
|------|------|---------|------|
| `material/gpu.rs` | 77 | `std::slice::from_raw_parts` — casting GpuMaterialNode → u32 bytes | Low — well-bounded, tested |
| `texture/gpu.rs` | 92 | `std::slice::from_raw_parts` — casting GpuTextureNode → u32 bytes | Low — well-bounded, tested |
| `environment.rs` | 23 | `#[allow(dead_code)]` on `spotlight` field | Low — constant overhead |

Both serialization unsafe blocks are in GPU serialization code — reinterpreting structs as byte slices for buffer upload. Each has corresponding test coverage. The `environment.rs` dead_code is a constant overhead unused by scene configuration.

### 4.3 `unwrap()`/`expect()` Calls (13 total)

| File | Line | Context | Risk |
|------|------|---------|------|
| `main.rs` | 48 | Softbuffer context creation | Low (init) |
| `main.rs` | 51 | Softbuffer surface creation | Low (init) |
| `main.rs` | 152 | Buffer present | Medium (could panic on lost surface) |
| `main.rs` | 181 | Surface resize | Low (init) |
| `main.rs` | 224 | Event loop | Medium (unreachable) |
| `main.rs` | 581 | Film output size | Low (assertion) |
| `renderer/cpu.rs` | 213 | `current_thread_index()` in Rayon pool | Low (guaranteed in pool) |
| `renderer/cpu.rs` | 214 | `samplers[thread_idx].lock()` | Low (Mutex, uncontended) |
| `sampler/mod.rs` | 394 | `current_thread_index()` in Rayon pool | Low (guaranteed in pool) |
| `sampler/mod.rs` | 395 | `self.items[idx].lock()` | Low (Mutex, uncontended) |
| `scene.rs` | 728 | Image file load (`kiara_1_dawn_4k.hdr`) | Medium (hardcoded path, dev machine) |
| `scene.rs` | 832 | Image file load (`earthmap.png`) | Medium (hardcoded path, dev machine) |
| `material/mod.rs` | 699 | `buf.nodes.last()` in GPU serialization | Low (test, unreachable if nodes non-empty) |

### 4.4 `#[allow(dead_code)]`

- `perlin.rs` — the entire Perlin noise module has `#[allow(dead_code)]`. Used by `NoiseTexture` (confirmed).
- `environment.rs:23` — `spotlight` field on `EnvironmentMap`. A constant angular falloff parameter that should be replaced by a cone-angle texture or scene configuration.

### 4.5 Config & Tooling

| Tool | Status |
|------|--------|
| `rustfmt` | No `rustfmt.toml` — uses default |
| `clippy` | No `clippy.toml`. 1 `#[allow(clippy::...)]` in `pdf.rs` |
| Test modules | 6 `#[cfg(test)]` modules (sampler, flat_bvh, integrator, film/rgb, material, texture/gpu) — same as before, new sampler tests added within `sampler/mod.rs` |
| CI | None (no `.github/`) |
| Workspace | Single crate (no workspace) |
| `Cargo.lock` | Present (deterministic builds) |
| Sobol direction numbers | Full Joe & Kuo 2008 dataset (21200 dims, generated in `build.rs` via include!) — up from 2048 in the old hardcoded file |

---

## 5. Remaining Issues & Roadmap

### 5.1 By Priority

| Pri | Issue | Effort | Impact | Files |
|-----|-------|--------|--------|-------|
| **P0** | Arena refactor (Arc → arena + lifetimes) | ~300 lines, 8 files | Enables GPU storage, auto light detection | `scene.rs`, `hittable.rs`, `bvh.rs`, `flat_bvh.rs`, `const_medium.rs` |
| **P0** | Triangle mesh support + OBJ loader | New file + parser | Unlocks real-world 3D scenes | `shape/mesh.rs`, `shape/obj.rs` |
| **P1** | Complete transform system (RotateX, RotateZ, Scale, matrix, composition) | ~60 lines | Full scene transform support | `transform.rs` |
| **P1** | Point3/Color3/Direction newtypes | ~50 lines | Type safety, prevent coordinate/color confusion | `vec3.rs`, `ray.rs`, `hittable.rs` |
| **P1** | CMJ/PMJ sampler for early bounces | ~100 lines | Better early-preview stratification | `sampler/mod.rs` |
| **P1** | Per-type bounce limits (diffuse/glossy/spec depths) | ~20 lines | Production bounce control | `integrator/path_tracer.rs` |
| **P1** | Light importance sampling (power-based selection) | ~40 lines | Reduce variance in many-light scenes | `pdf.rs`, `integrator/` |
| **P2** | Texture mapping 2D/3D clean split | ~80 lines | Cleaner architecture | `texture/mapping.rs`, `hittable.rs` |
| **P2** | Denoiser integration (OIDN) | ~100 lines | Interactive-quality output at low spp | `film/` |
| **P2** | AOV support (albedo, normal, depth) | ~80 lines | Debug visualization, compositing | `film/` |

### Resolved since last audit

| Issue | Resolution |
|-------|-----------|
| **Rough clearcoat MIS** (was P2) | Coated material now uses internal random walk for rough coating — the `scatter()` method loops up to `MAX_INTERNAL_BOUNCES = 5`, handling the Split (delta + non-delta) component of a rough coating correctly via recursive substrate evaluation. |
| **Sampler refactor** (structural) | Two-stream architecture implemented: `Sampler` trait with GAT `Session<'s>: SamplingSession`, `SobolSampler` (Sobol + HashRng), `StreamRngPair` generic adapter. Eliminated `DimCursor`, fixed 11-dim stride, and `PDF<S>` generic. |
| **PDF System** (structural) | `PdfEnum` deleted, `PdfKind` directly implements `PDF` trait. Added `SolidAnglePdf`/`AreaPdf` domain newtypes, `MisHeuristic` enum, `Distribution1DFixed<const N>`, `balance_heuristic`, `Sample1D` enum, `sample_discrete` helper. |
| **Path splitting** (structural) | `BsdfScatter::Split` variant for Mix one-delta children → `li_inner()` recursive delta trace in integrator. Mix/Coated handle Split in their scatter methods. |
| **Coated IOR hardcode** (was issue #2) | Coated material now reads IOR from the coating's `DielectricMaterial` instead of hardcoded 1.5. |

### 5.2 Development Phases

```text
Phase 1 — Geometry foundation (now → 1 month)
├── Triangle mesh + OBJ parser
├── Full transform system
└── Point3/Color3 newtypes

Phase 2 — Memory model (1-3 months)
├── Arena allocation for scene objects
├── Automatic light detection
└── Flat storage for GPU upload

Phase 3 — Quality (3-6 months)
├── CMJ/PMJ sampler for early bounces
├── Per-type bounce limits
├── Light importance sampling
└── BDPT/MLT exploration

Phase 4 — Production polish (6-12 months)
├── AOV support
├── Denoiser integration
├── Spectral rendering (research)
└── GPU rendering exploration (WGSL/WebGPU)
```

**Completed since last audit**: Sampler refactor (two-stream architecture, GAT `Sampler` trait, `SobolSampler`), PDF system overhaul (`PdfEnum` deleted, `PdfKind` direct impl, `SolidAnglePdf`/`AreaPdf`, `MisHeuristic`, `Distribution1DFixed`), path splitting (`BsdfScatter::Split`, `li_inner`), rough clearcoat MIS (Coated internal random walk), Coated IOR decoupling.

### 5.3 Architectural Positioning

The codebase has **strong architectural fundamentals** that compare favorably with production renderers:

- **Trait boundaries** (Integrator, Renderer, Camera, Film) match the decomposition of pbrt-v4 and exceed it in flexibility (generic type params instead of virtual inheritance).
- **Enum dispatch** for materials matches pbrt-v4's TaggedPointer pattern — all production renderers with virtual dispatch (LuxCore, appleseed) pay a vtable penalty on every BSDF evaluation that raytrace-rs avoids.
- **Pure sampler API** is architecturally distinct from every surveyed renderer and positions the codebase well for GPU. The two-stream internal architecture (Sobol for 2D, hash for 1D) protects Sobol correlation structure while the unified `Sampler` trait with GAT session provides compile-time cross-pixel state safety.
- **GPU serialization** groundwork (material tree → flat node array) is already in place and tested.
- **Adaptive sampling** with Welford's online variance is production-quality.

The gaps are in **scope** (single integrator, no meshes, no bidirectional techniques) and **memory model** (Arc arena). Neither reflects architectural problems — they reflect the codebase's age and focus.

---

## 6. Glossary

| Term | Definition |
|------|-----------|
| **AOV** | Arbitrary Output Variable — per-pixel data channel (albedo, normal, depth) |
| **BDPT** | Bidirectional Path Tracing — paths from both camera and light, connected |
| **BLAS** | Bottom-Level Acceleration Structure (DXR) — per-mesh BVH |
| **BRDF** | Bidirectional Reflectance Distribution Function |
| **BSDF** | Bidirectional Scattering Distribution Function (BRDF + BTDF) |
| **BVH** | Bounding Volume Hierarchy — acceleration structure |
| **CMJ** | Correlated Multi-Jittered — low-discrepancy sequence |
| **DXR** | DirectX Raytracing — Microsoft's GPU raytracing API |
| **ECS** | Entity Component System — data-oriented design pattern |
| **ISPC** | Intel SPMD Program Compiler — C-like language for SIMD |
| **MIS** | Multiple Importance Sampling — combining PDFs from different sampling strategies |
| **NEE** | Next Event Estimation — direct lighting via explicit light sampling |
| **NDF** | Normal Distribution Function — microfacet roughness model |
| **OIDN** | Intel Open Image Denoise |
| **PMJ** | Progressive Multi-Jittered — low-discrepancy sequence |
| **PMR** | Polymorphic Memory Resource — C++17 allocator model |
| **RR** | Russian Roulette — probabilistic path termination |
| **SAH** | Surface Area Heuristic — BVH build cost model |
| **SoA** | Structure of Arrays — data layout optimizing for cache |
| **SPMD** | Single Program Multiple Data — SIMD programming model |
| **TLAS** | Top-Level Acceleration Structure (DXR) — scene-wide BVH |
| **Welford** | Online algorithm for computing variance (single pass) |
