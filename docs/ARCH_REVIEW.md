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

> **Last updated:** 2026-07-01 — fresh audit at `HEAD: 24808b5`.
> 50 source files, 23 public modules, 13 traits, ~15 enums, ~45 structs.
> 8 core dependencies (image, rand, rayon, winit, softbuffer, tracing, smol, async-tungstenite).

---

## 0. Executive Summary

| Metric | Value |
|--------|-------|
| Source files | 50 `.rs` |
| Public modules | 23 |
| Public traits | 13 (`Intersectable`, `Bounded`, `Sampleable`, `MaterialHit`, `Sampler`, `SamplerFactory`, `Camera`, `Film`, `Integrator`, `Renderer`, `PDF`, `Shape3D`, `Region2D`, `Bsdf`[private], `Texture`, `Transform`) |
| Material types | 8 (6 BSDF + 2 composition: Mix, Coated) |
| Texture types | 5 (SolidColor, Checker, Noise, Image, MappedTexture) |
| Shape types | 2 (Sphere + ShapeObject generic; 9 planar region types via PlanarPatch) |
| Integrators | 1 (PathTracingIntegrator) |
| Renderers | 1 (CpuRenderer — rayon tiled, adaptive) |
| Samplers | 3 (Sobol, NaiveRandom, Stratified) + Factory pattern |
| Scene functions | 10 built-in |
| Unsafe blocks | 2 (GPU serialization only) |
| `unwrap()`/`expect()` | 8 (init code, hardcoded paths) |
| `todo!()` / `unimplemented!()` | 0 |
| TODOs | 20 |
| `#[allow(dead_code)]` | 1 (`perlin.rs`) |
| `total LOC` | ~6,500 (src/) |

### Key Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Dispatch model** | Enum-based (closed set) | Mirrors pbrt-v4's TaggedPointer; static dispatch, inline-able, no heap alloc |
| **Material polymorphism** | `Material` enum (10 variants) + private `Bsdf` trait | Hot-path dispatch via match; `Custom(Box<dyn Bsdf>)` escape hatch for extensibility |
| **Integrator** | Generic trait `Integrator<S: Sampler>` | Pluggable sampler strategy, monomorphized per-sampler |
| **Renderer** | Generic trait `Renderer<W,C,F,S>` | Swappable render backends (CPU now, GPU future) |
| **Sampler** | Pure `sample(n, d) -> f64` (deterministic, `Sync`) | Stateless, trivially parallel, cache-friendly, GPU-compatible |
| **Sampler state** | Thread-local GrayCodeCache | No alloc on hot path; pure fn + shared cache |
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
├── lib.rs                           # Module declarations (23 pub mod)
│
├── vec3.rs                          # Vec3, Point3/Color3 aliases, ops, concentric_disk
├── ray.rs                           # Ray struct, ParametricCurve
├── interval.rs                      # Interval { min, max }
├── aabb.rs                          # Aabb { x, y, z: Interval }, merge, hit, centroid
├── onb.rs                           # Orthonormal basis (build_from_normal, local↔world)
├── perlin.rs                        # Perlin noise (#[allow(dead_code)])
├── transform.rs                     # Transform trait, Translate, RotateY (+ 5 TODOs)
│
├── sampler.rs                       # Sampler trait, SobolSampler, NaiveRandom, Stratified
├── pdf.rs                           # PDF<S> trait, PdfEnum, power_heuristic
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
│   ├── mod.rs                       # Material enum (10 variants), Bsdf trait, BsdfSample, GGX
│   ├── lambertian.rs                # LambertianMaterial — albedo/π
│   ├── metal.rs                     # MetalMaterial — GGX conductor
│   ├── dielectric.rs                # DielectricMaterial — Fresnel + tint
│   ├── diffuse_light.rs             # DiffuseLightMaterial — emissive
│   ├── isotropic.rs                 # IsotropicMaterial — uniform sphere (volumes)
│   ├── glossy.rs                    # GlossyMaterial — GGX dielectric
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
                            ┌──────────┐
                            │ Sampler   │←─────────── pure(Sync), sample(n, d)
                            └──────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
            ┌────────────┐ ┌──────────┐ ┌──────────┐
            │ Integrator │ │ PDF<S>   │ │ Camera   │
            │ :li(),     │ │ :sample()│ │ :gen_ray │
            │ :max_bounces│ │ :value() │ └──────────┘
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
| `Sampler` | Generic `<S: Sampler>` | Monomorphized per-sampler; zero-cost |
| `PDF` | Generic `<S: Sampler>` + `PdfEnum` | Mix of known types (Cosine, GGX) + Hittable that needs `dyn` scene ref |
| `Camera` | Trait w/ concrete `PerspectiveCamera` | Multiple camera models expected (ortho, spherical) |
| `Film` | Trait w/ concrete `RgbFilm` | Multiple film types expected (spectral, AOV) |
| `Integrator` | Generic `<S: Sampler>` | Single integrator currently; trait for new types |
| `Renderer` | Generic with 4 type params | CPU now; GPU renderer as separate impl later |
| `Shape3D` | `dyn` (via `ShapeObject`) | Heterogeneous shape storage (sphere + future mesh) |
| `Region2D` | `dyn` (via `PlanarObject`) | Heterogeneous planar regions |

---

## 2. Component Deep Dives

### 2.1 Renderer + Integrator

#### Current State

```text
CpuRenderer<I, S, Fact>           ← trait Renderer<W,C,F,S>
├── owns: I (Integrator), Fact (SamplerFactory)
├── render() → loops over tiles
│   ├── rayon::for_each on tiles
│   │   └── per tile: I.li() for each pixel
│   ├── FilmTile → merge into RgbFilm
│   └── adaptive: check RgbFilm.variance() → early exit
├── resize() → rebuild Film + tile pool
└── reset() → clear film

PathTracingIntegrator<S: Sampler>  ← trait Integrator<S>
├── li(ray, world, lights, sampler) → Color3
├── max_bounces() → usize
└── bounce loop:
    ├── world.intersect(ray) → Option<SurfaceInteraction>
    ├── add emission
    ├── Russian roulette (bounce ≥ 5, survive = clamp(throughput, 0.05, 1))
    ├── if material delta: trace reflected ray, pad 4 dims
    │   └── debug_assert_eq!(11 dims used this bounce)
    └── if material non-delta:
        ├── construct MixturePDF[light, hittable, bsdf]
        ├── sample direction → trace
        ├── weight = 1/pdf_val
        └── debug_assert_eq!(11 dims used this bounce)
```

**Shadow rays**: Implemented as direct-lighting NEE with explicit occlusion check (commit `24808b5`). The shadow ray tests visibility between the hit point and the sampled light before accumulating direct contribution.

**MIS**: One-sample MIS with power heuristic (α=1). `power_heuristic(n_f * pdf_f, n_g * pdf_g)` from `pdf.rs`.

#### Comparison

| Aspect | raytrace-rs | pbrt-v4 | LuxCore | MoonRay | Mitsuba 3 |
|--------|-------------|---------|---------|---------|-----------|
| Integrator abstraction | `Integrator<S>` trait (generic) | `Integrator` base → `RayIntegrator` | `RenderEngine` base | `Engine` class | `Integrator<Float, Spectrum>` template |
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
Material enum (10 variants):
├── Void                           # No material (miss)
├── Lambertian(LambertianMaterial) # albedo/π
├── Metal(MetalMaterial)           # GGX microfacet conductor
├── Dielectric(DielectricMaterial) # Fresnel/refract/reflect + tint
├── DiffuseLight(DiffuseLightMaterial) # emissive
├── Isotropic(IsotropicMaterial)   # uniform sphere (volumes)
├── Glossy(GlossyMaterial)        # GGX dielectric
├── Mix { a: Box<Material>, b: Box<Material>, weight: f64 }
├── Coated { substrate: Box<Material>, coating: Box<Material> }
└── Custom(Box<dyn Bsdf>)         # Escape hatch

Bsdf trait (private to material module, 7 methods):
  sample(&self, wo, u, v) → Option<BsdfSample>
  eval(&self, wo, wi, kind) → Color3
  pdf(&self, wo, wi) → f64
  pdf_kind(&self) -> PdfKind
  emitted(&self, u, v, p) → Color3
  to_gpu(&self, ...) → GpuMaterialType
  gpu_params(&self) → Vec<GpuMaterialNode>
  gpu_texture_index(&self) → Option<u32>

BsdfSample enum (3 variants):
  Delta     { wi, f_cos, pdf_kinds, count }
  NonDelta  { wi, f_cos, pdf_kinds, count }
  Emission  { radiance, pdf_kinds }

PdfKind enum:
  CosineHemisphere | UniformHemisphere | UniformSphere |
  GgxNdf(f32, f32) | LightArea(Vec3, f64) | Delta
```

**Key patterns**:

- `Bsdf` is a **private trait** — external code interacts only through the `Material` enum.
- GPU serialization (`GpuMaterialBuffer`) flattens the material tree into a `Vec<GpuMaterialNode>` with child references as indices.
- Composition types (Mix, Coated) use `Box<Material>` (not `Box<dyn Bsdf>`) — the enum dispatch recurses through children.
- Coated uses a hard Fresnel-split: `f` probability to sample coating, `1-f` probability to sample substrate. `eval()` blends both: `coating * f + substrate * (1-f)`.

#### Comparison

| Aspect | raytrace-rs | pbrt-v4 | LuxCore | Mitsuba 3 | Filament |
|--------|-------------|---------|---------|-----------|----------|
| Dispatch | `Material` enum (10 variants) | `Material` TaggedPointer (11) | `Material` virtual base | `BSDF` template plugin | Shader specialization (`.mat` DSL) |
| BSDF per element | Single (match→impl) | Single `BxDF` via TaggedPointer | Single (virtual) | Single (plugin) | Hardcoded shader variants |
| Composition | `Mix` + `Coated` enums | `LayeredBxDF<T,B>` template | `Mix` material type | `blendbsdf`, `mask` plugins | Coating shader variants |
| Layered approach | Fresnel-split analytic | MC random walk (Guo et al.) | Hierarchical blend | Plugin composition | Layered shader compilation |
| Rough coating | ❌ Hard-split only (correct only for delta) | ✅ MC layered walk | ✅ GlossyCoating | ✅ Yes via roughdielectric | N/A |
| Spectral | RGB only | SampledSpectrum (4-point) | Partial (dispersion) | ✅ Full spectral + polarization | RGB only |
| GPU serialization | ✅ Tree→flat node array | `pstd::vector` (arena) | N/A | N/A | Material compiler |
| Delta routing | ✅ Enum variant (type-level) | `BxDFFlags` bitmask | Runtime check | Plugin-dependent | N/A |
| Mix efficiency | `eval()` evaluates both children every sample | Evaluates selected lobe only | Same tradeoff | Plugin-dependent | Compiled per variant |

#### Issues

1. **Coated eval/pdf Fresnel blend is correct only for delta coating.** For a rough (non-delta) coating, the blend `coating * f + substrate * (1-f)` doesn't match `sample()`'s selection probability — MIS weights become wrong. Not a current problem since coating is always dielectric (smooth).

2. **Coating Fresnel hard-coded to IOR=1.5.** Should read IOR from the coating's `DielectricMaterial`.

3. **Mix `eval()` evaluates both children every sample.** `sample()` picks one stochastically but `eval()` always evaluates both. Acceptable for 2 children; would not scale to N.

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
Sampler trait (pure, deterministic, Sync):
  sample(&self, n: u32, d: u32) → f64
  └── same (n, d) → same f64 everywhere

SamplerFactory trait:
  create() → Self::Sampler

SobolSampler:
  Joe & Kuo 2008 direction numbers (512 dims)
  Gray-code recurrence for O(1) per-sample advance
  Thread-local GrayCodeCache with digital shift scrambling

NaiveRandomSampler:
  SplitMix64 hash of (n, d, seed)

StratifiedSampler:
  Jittered grid for dims 0-1, hash for rest

DimCursor<S: Sampler>:
  ┌──────────────────────────────────────────────┐
  │ sampler: S  (embedded — no separate arg)     │
  │ base: u64  (pixel seed)                      │
  │ offset: u64 (current bounce start dim)       │
  │ sample_idx: u64 (current pixel sample)        │
  └──────────────────────────────────────────────┘

  QMC stride: 11 dims per bounce
  (debug_assert_eq!(bounce_end - bounce_start, 11))
```

**Dimension budget per bounce (11)**:

| Use | Dimenions |
|-----|-----------|
| Pixel AA (jitter) | 2 |
| Lens sample (DOF) | 2 |
| Time sample (motion blur) | 1 |
| BSDF sample | 2 |
| Light sample | 2 |
| RR / padding | 2 |

**Sobol dimension safety**: 512 dims / 11 dims-per-bounce ≈ 46 bounces — sufficient for all practical scenes.

#### Comparison

| Aspect | raytrace-rs | pbrt-v4 | LuxCore | MoonRay | Mitsuba 3 |
|--------|-------------|---------|---------|---------|-----------|
| API | Pure `sample(n,d)` (stateless, Sync) | Stateful `Get1D()`, `Get2D()` | Stateful class | Stateful `SequenceID` | Stateful plugin |
| Thread safety | Thread-local cache + pure fn | Clone per thread | Per-thread RNG | Per-thread state | Clone per thread |
| Determinism | Same `(n,d)` → same value | Same sequence per pixel | Per-pixel + random pass | Pixel+purpose hash | Plugin-dependent |
| Primary sequence | Sobol (512d, Gray-code) | Multiple (Sobol, Halton, PMJ, etc.) | Sobol | CMJ/PMJ + hash | Multiple (Independent, Stratified, Sobol, etc.) |
| Scrambling | Digital shift (per-dim hash) | Owen, FastOwen, PermuteDigits | None (pixel jitter) | Precomputed tables | None |
| Dim management | `DimCursor<S>` auto-advancing + debug_assert | Implicit (sampler tracks dim) | Per engine | Per purpose | Per plugin |
| Blue noise | No | ✅ BlueNoiseSampler, ZSampler | No | ✅ Precomputed tables | No |
| Max dims | 512 | Unbounded | Unbounded | Precomputed | Unbounded |

**Design note**: The pure `sample(n,d)` API is architecturally distinct from every production renderer surveyed — all use stateful sampler APIs. The stateless approach is advantageous for GPU (no per-thread sampler state to maintain) and enables trivial determinism verification (`same n,d → same result`).

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
| **Sampler API** | Pure `sample(n,d)` → deterministic, Sync, trivially GPU-friendly | All production renderers use stateful sampler APIs |
| **QMC dim discipline** | Fixed stride + debug_assert prevents dim aliasing | pbrt-v4 uses implicit cursor — no cross-bounce assertion |
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
| **Enum dispatch for hot-path polymorphism** | `Material` (10 variants), `PdfEnum` (6), `BsdfSample` (3) | Performant Rust path tracers; mirrors pbrt-v4's TaggedPointer |
| **Generic type param for strategy pattern** | `Integrator<S>`, `Renderer<W,C,F,S>`, `PDF<S>`, `DimCursor<S>` | `rend3`, `renderling` |
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
        1 => { /* Lambertian — sample albedo from texture_index */ }
        2 => { /* Metal — GGX microfacet conductor */ }
        3 => { /* Dielectric — Fresnel + refract/reflect */ }
        6 => { /* Mix — recurse via child_a, child_b, blend by weight */ }
        7 => { /* Coated — Fresnel-split between coating and substrate */ }
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
| **Representation** | Flat `#[repr(C)] MaterialDescriptor` | Recursive `Material` enum (10 variants) | `GpuMaterialNode` array + flat params buffer |
| **PBR model** | Metallic-roughness (albedo, metalness, roughness, AO, normal, emissive) | General BSDFs (Lambertian, GGX, Dielectric, Mix, Coated) | Same BSDF set tagged by `GpuMaterialType` discriminant |
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
| Sampler (`pure sample(n,d)`) | ✅ Yes — stateless, trivially maps to GPU | No changes needed; GPU `n = pixel_x * width + pixel_y; d = bounce * 11 + dim` |
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
├── Port BSDF eval (Lambertian, GGX, Dielectric) to rust-gpu functions
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

**Categories**: GPU pipeline (4), type-safety (4), opt-preview (3), transforms (4), refactor (1), renderer-agnostic (1), mapping (1).

**Zero**: `todo!()`, `unimplemented!()`, FIXMEs, HACKs, XXXs.

### 4.2 Unsafe Blocks (2)

| File | Line | Context | Risk |
|------|------|---------|------|
| `material/gpu.rs` | 77 | `std::slice::from_raw_parts` — casting GpuMaterialNode → u32 bytes | Low — well-bounded, tested |
| `texture/gpu.rs` | 92 | `std::slice::from_raw_parts` — casting GpuTextureNode → u32 bytes | Low — well-bounded, tested |

Both are in GPU serialization code — reinterpreting structs as byte slices for buffer upload. Each has corresponding test coverage.

### 4.3 `unwrap()`/`expect()` Calls (8 total)

| File | Line | Context | Risk |
|------|------|---------|------|
| `main.rs` | 50 | Softbuffer context creation | Low (init) |
| `main.rs` | 53 | Softbuffer surface creation | Low (init) |
| `main.rs` | 154 | Buffer present | Medium (could panic on lost surface) |
| `main.rs` | 183 | Surface resize | Low (init) |
| `main.rs` | 226 | Event loop | Medium (unreachable) |
| `main.rs` | 559 | Film output size | Low (assertion) |
| `scene.rs` | 811 | Image file load | Medium (hardcoded path, dev machine) |
| `material/mod.rs` | 789 | `buf.nodes.last()` in test | Low (test) |

### 4.4 `#[allow(dead_code)]`

`perlin.rs` — the entire Perlin noise module has `#[allow(dead_code)]`. It's likely unused by any current scene. Should either be used by `NoiseTexture` (confirm) or removed.

### 4.5 Config & Tooling

| Tool | Status |
|------|--------|
| `rustfmt` | No `rustfmt.toml` — uses default |
| `clippy` | No `clippy.toml`. 1 `#[allow(clippy::...)]` in `pdf.rs` |
| Test modules | 6 `#[cfg(test)]` modules (sampler, flat_bvh, integrator, film/rgb, material, texture/gpu) |
| CI | None (no `.github/`) |
| Workspace | Single crate (no workspace) |
| `Cargo.lock` | Present (deterministic builds) |

---

## 5. Remaining Issues & Roadmap

### 5.1 By Priority

| Pri | Issue | Effort | Impact | Files |
|-----|-------|--------|--------|-------|
| **P0** | Arena refactor (Arc → arena + lifetimes) | ~300 lines, 8 files | Enables GPU storage, auto light detection | `scene.rs`, `hittable.rs`, `bvh.rs`, `flat_bvh.rs`, `const_medium.rs` |
| **P0** | Triangle mesh support + OBJ loader | New file + parser | Unlocks real-world 3D scenes | `shape/mesh.rs`, `shape/obj.rs` |
| **P1** | Complete transform system (RotateX, RotateZ, Scale, matrix, composition) | ~60 lines | Full scene transform support | `transform.rs` |
| **P1** | Point3/Color3/Direction newtypes | ~50 lines | Type safety, prevent coordinate/color confusion | `vec3.rs`, `ray.rs`, `hittable.rs` |
| **P1** | CMJ/PMJ sampler | ~100 lines | Better early-preview stratification | `sampler.rs` |
| **P2** | Rough clearcoat MIS | ~80 lines | Physically correct rough coating | `material/mod.rs` (Coated) |
| **P2** | Per-type bounce limits (diffuse/glossy/spec depths) | ~20 lines | Production bounce control | `integrator/path_tracer.rs` |
| **P2** | Light importance sampling (power-based selection) | ~40 lines | Reduce variance in many-light scenes | `pdf.rs`, `integrator/` |
| **P3** | Texture mapping 2D/3D clean split | ~80 lines | Cleaner architecture | `texture/mapping.rs`, `hittable.rs` |
| **P3** | Denoiser integration (OIDN) | ~100 lines | Interactive-quality output at low spp | `film/` |
| **P3** | AOV support (albedo, normal, depth) | ~80 lines | Debug visualization, compositing | `film/` |

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
├── Rough clearcoat MIS
├── Per-type bounce limits
└── Light importance sampling

Phase 4 — Production polish (6-12 months)
├── AOV support
├── Denoiser integration
├── Spectral rendering (research)
└── GPU rendering exploration (WGSL/WebGPU)
```

### 5.3 Architectural Positioning

The codebase has **strong architectural fundamentals** that compare favorably with production renderers:

- **Trait boundaries** (Integrator, Renderer, Camera, Film) match the decomposition of pbrt-v4 and exceed it in flexibility (generic type params instead of virtual inheritance).
- **Enum dispatch** for materials matches pbrt-v4's TaggedPointer pattern — all production renderers with virtual dispatch (LuxCore, appleseed) pay a vtable penalty on every BSDF evaluation that raytrace-rs avoids.
- **Pure sampler API** is architecturally distinct from every surveyed renderer and positions the codebase well for GPU.
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
