# Architecture Review: `raytrace-rs` vs Production Renderers

Reference renderers:
**[LuxCore](https://github.com/LuxCoreRender/LuxCore)**,
**[OpenMoonRay](https://github.com/Autodesk/moonray)**,
**[renderling](https://github.com/schell/renderling)**
**[pbrt-v4](https://github.com/mmp/pbrt-v4)**

> **Last updated:** 2026-06-20 — comprehensive audit against current `HEAD` (671af5d).
> Previous baseline: commit 85d8021. ~30 commits, ~700 net new lines across 30 files.

---

## 0. Executive Summary: What Changed Since v1

| Item | Status | Detail |
|------|--------|--------|
| **Renderer abstraction** | ✅ Done | `Integrator` trait + `Renderer` trait + `Camera` trait extracted |
| **Camera decomposition** | ✅ Done | Monolithic `camera.rs` → `Camera` trait + `PerspectiveCamera` |
| **Film abstraction** | ✅ Done | `Film` trait + `RgbFilm` + `FilmTile` — post-process separated |
| **Adaptive sampling** | ✅ Done | Welford's online variance, convergence mask, early exit |
| **BsdfSample struct→enum** | ✅ Done | `Delta` / `NonDelta` variants — type-safe per-sample routing |
| **Dielectric tint** | ✅ Done | `DielectricMaterial.tint` — `material::dielectric_tinted(ior, tint)` |
| **QMC dim discipline** | ✅ Done | `debug_assert!(11 dims/bounce)` prevents dimension aliasing |
| **ConstantMedium cleanup** | ✅ Done | Removed `dyn` usage, straight trait objects |
| **Arena refactor** | ❌ Not done | Still `Vec<Arc<dyn ...>>`, `add_light()` manual tracking |
| **Shadow ray** | ❌ Not done | Still ray-probe approach, no explicit `Unoccluded` |
| **Triangle meshes** | ❌ Not done | No mesh/OBJ support |
| **Power heuristic MIS** | ❌ Not done | Fixed `[1/3, 2/3]` weights |
| **Complete transforms** | ❌ Not done | Still only Translate + RotateY |
| **Point3/Color3 newtypes** | ❌ Not done | Still type aliases |
| **Rough clearcoat** | ❌ Not done | Analytic Fresnel split only |

---

## 1. Material System

### Current Design

```
Material enum (9 variants)  ← Box<dyn Bsdf> within composition types
├── Lambertian(LambertianMaterial)
├── Metal(MetalMaterial)          -- GGX conductor
├── Dielectric(DielectricMaterial)  -- with optional tint color
├── DiffuseLight(DiffuseLightMaterial)
├── Isotropic(IsotropicMaterial)
├── Glossy(GlossyMaterial)        -- GGX dielectric
├── Mix { a, b, weight }
├── Coated { substrate, coating }
└── Custom(Box<dyn Bsdf>)
```

Dispatch: match arms on `Material` → delegate to struct methods. `Bsdf` trait has **7 methods** (`sample`, `eval`, `pdf`, `emitted`, `is_emissive`, `clone_box`, `serialize_gpu`). GPU serialization flattens the tree into `Vec<GpuMaterialNode>`.

**Key change since v1**: `BsdfSample` is now an **enum** (`Delta` / `NonDelta`), not a struct. This makes the delta-vs-non-delta distinction a type-level guarantee — the integrator matches on the variant rather than inspecting a `pdf_kind` field:

```rust
pub enum BsdfSample {
    Delta { wi: Vec3, f_cos: Color3 },
    NonDelta { pdf_kind: PdfKind },
}
```

`PdfKind` similarly became an enum (`Cosine`, `Ggx`, `UniformSphere`, `Delta`) instead of the earlier approaches. The integrator constructs stack-allocated PDF objects from the kind on the non-delta path.

### Findings

**✅ Well‑done** (all carried forward from v1)

- **Per‑sample delta‑ness via `BsdfSample::Delta` vs `NonDelta`** — enum variants enforce correct dispatch at the type level. No runtime `is_delta()` needed.
- **`BsdfSample` bundles direction, BSDF×cos, PDF kind** in one variant. Guarantees direction/PDF come from the same sample.
- **GPU tree serialization** — clean and tested (6 tests in `material/mod.rs`). Compositions serialize and reference children by index, forming a valid DAG.
- **Closed‑form Fresnel in Coated::sample()** — direct reflection without delegating to Dielectric (which would double‑Fresnel).
- **Constructor ergonomics** — `.mix()`, `.coated()`, `lambertian()`, `dielectric_tinted()` etc. are intuitive and chainable.

**⚠️ Issues to address** (mostly unchanged — none of these were tackled)

1. **Coated Lacks MIS Between Coating & Substrate** (LuxCore reference)
   Current `Coated::sample()` uses a hard Fresnel-split switch. A rough clearcoat (non‑delta, e.g. satin) would need the coating and substrate lobes combined via MIS, not a hard split. LuxCore's `GlossyCoating` uses:
   ```
   w_coating = 0.5 * (1 + Fresnel(average_value))
   result = coating_sample * w_coating + base_sample * (1-w_coating)
   ```
   **Impact**: Cannot correctly render rough clearcoat (satin, matte clear). Low priority now (coating is always dielectric = perfectly smooth), but will prevent physically correct rough clearcoat later.

2. **Coated `eval()`/`pdf()` Use Fresnel Blend — Correct Only for Delta Coating**
   For a delta coating, `coating.eval()` returns 0 everywhere except the exact specular direction, so this collapses to `(1-f)*substrate.eval()` — correct. For a rough coating, this blend doesn't match `sample()`'s selection probability, making MIS weights wrong and introducing bias.

3. **Coating Fresnel Hard‑coded to IOR = 1.5**
   Should be parametrized on the coating material's IOR from the `DielectricMaterial`.

4. **Mix `eval()` Evaluates Both Children Every Sample**
   The `eval`/`pdf` blend `(1-w)*a + w*b` means every sample evaluates **both** children, wasteful for expensive secondary lobes. Stochastic selection on `sample()` only picks one. This is the same tradeoff as LuxCore.

### Comparison: MoonRay's Component BSDF Architecture

No change from v1 — MoonRay still uses 30+ `BsdfComponent` data classes with `BsdfBuilder` and attenuator chain. Overkill for raytrace-rs's 6 material types. The enum dispatch remains the right call.

---

## 2. 2D/3D Primitives

### Current Design

```
Hittable trait (split into Intersectable + Bounded)
├── Sphere (static + moving)
├── PlanarPatch<R: Region2D>  ← parametric 2D → 3D
│   ├── QuadRegion, TriRegion, EllipseRegion
│   ├── AnnulusRegion, SuperellipseRegion
│   ├── RoundedRectRegion, PolygonRegion
│   └── FunctionRegion (arbitrary predicate)
├── TransformObject<T: Transform, O: Intersectable>
│   ├── Translate { offset }
│   └── RotateY { sin, cos }
├── BvhNode (tree with SAH build)
├── FlatBvh (cache‑friendly flat 64‑byte nodes)
├── ConstantMedium (volumetric scattering)
├── Vec<T: Intersectable> + Arc<T: Intersectable> (blanket impls)
```

Region2D trait:
```
contains(a,b) → bool
area()        → f64
sample(u,v)   → (f64, f64)
uv(a,b)       → (f64, f64)  // default identity
bounding_box_area() → f64   // default = area()
```

**Key change since v1**: `Hittable` trait was **decomposed** into two separate traits — `Intersectable` (intersection queries) and `Bounded` (bounding box queries). The `Sampleable` trait handles light importance sampling with `pdf_value()` and `random()` methods. Hit records were split into `Hit` (raw geometric) and `SurfaceInteraction` (geometric + material + shading normal), which improved `from_material_hit()` construction.

`PlanarPatch` gained type aliases and free constructor functions (`quad()`, `ellipse()`, `tri()`, `annulus()`, `rounded_rect()`, `superellipse()`, `polygon()`, `function_patch()`) for cleaner API.

### Findings

**✅ Well‑done**

- **PlanarPatch + Region2D pattern** maps cleanly to parametric surfaces.
- **8 region types** cover a useful variety. FunctionRegion enables arbitrary shapes.
- **FlatBvh** — 64‑byte nodes, iterative traversal, near‑first child ordering. Production‑quality.
- **SAH BVH build** with 32 bins, all 3 axes evaluated, parallel via `rayon::join`. Solid.

**⚠️ Issues to address** (unchanged from v1)

1. **No Triangle Mesh Support** — **Biggest geometric gap** ❌ Not done
   All three reference renderers use triangle meshes as their primary geometry. raytrace-rs cannot load any standard 3D format (OBJ, glTF, FBX, PLY).

2. **Transform System Incomplete** ❌ Not done
   Only `Translate` and `RotateY` exist. Missing: `RotateX`, `RotateZ`, `Scale`, composition helpers, general 4×4 matrix transform. The same TODOs remain in `transform.rs`.

3. **Arenas & Lifetime: Planned but Un‑implemented** ❌ Not done
   Still using `Vec<Arc<dyn Intersectable>>` and `Vec<Arc<dyn Sampleable<S>>>` in Scene. `add_light()` manual tracking still required. The `docs/arena-refactor-plan.md` design is still valid but un‑implemented.

4. **Sphere Only Implicit** ❌ Not done
   No disk, cylinder, or other implicit primitives beyond sphere.

---

## 3. Path Tracing Integrator

### Current Design

```
PathTracingIntegrator::li()  ← extracted from Camera
├── bounce loop (max_depth)
├── world.intersect() → SurfaceInteraction
├── add emission
├── Russian roulette (bounce ≥ 5, survival = max_attenuation.clamp(0.05, 1.0))
├── material.sample() → BsdfSample::Delta { wi, f_cos }
│   └── Delta path: accumulated_attenuation *= f_cos, trace sample.wi
│       pad 4 dims, debug_assert 11 dims total
└── BsdfSample::NonDelta { pdf_kind }
    └── Construct on-stack PDF from PdfKind variant
        MixturePDF [light, surface, surface] → direction
        f_cos = material.eval(wo, sampled_direction)
        weight = 1 / pdf_val
        accumulated_attenuation *= f_cos * weight
        pad mixture dims to exactly 4, debug_assert 11 dims total
```

**Key changes since v1**:

- **Integrator trait extracted**: `Integrator<S>` trait with `li()` method replaces `Camera::ray_color()`.
- **Renderer abstraction**: `Renderer` trait with `CpuRenderer` implementation handles sample scheduling, tile-based parallel rendering, adaptive sampling, and progressive framebuffer publishing.
- **Camera trait extracted**: `Camera::generate_ray()` separates camera model from the integrator.
- **QMC dimension discipline**: Every bounce must consume exactly 11 fixed dimensions, enforced by `debug_assert_eq!(sampler.offset() - bounce_start, 11)`. Previous `DimCursor` overflow concern is now addressed.
- **Adaptive sampling**: `CpuRenderer` uses Welford's online variance per-pixel. Configurable `threshold_rel` (stddev/mean ratio) and `threshold_abs` (noise floor). Early exit when all pixels converge.
- **Tiled rendering**: Pre-allocated tile pool reused across passes, eliminating repeated heap allocations. Adaptive progressive cadence (1:1 up to 16 passes, 1:4 up to 64, 1:8 thereafter).
- **Film abstraction**: `Film` trait with `RgbFilm` implementation handles accumulation, tone-mapping (Reinhard), gamma correction, and variance tracking.

**Post-processing pipeline**: `post_process(color, exposure, tone_map)` → Reinhard tone-map (optional) → gamma 2 → [u8; 3].

### Findings

**✅ Well‑done** (all carried forward)

- **Clean separation of delta vs non‑delta** via `BsdfSample::Delta` / `BsdfSample::NonDelta` enum variants.
- **Russian roulette** uses throughput‑proportional survival probability with floor clamp — standard and correct.
- **MixturePDF** designs `[light, surface, surface]` gives surface sampling 2/3 weight.
- **Fixed QMC stride** — 11 dims per bounce, enforced by debug assertion. Prevents dimension aliasing bugs.
- **Tiled rendering** with pooled tiles keeps memory allocation off the hot path.
- **Adaptive sampling** with Welford's variance is a significant convergence improvement over fixed-spp rendering.

**⚠️ Issues to address**

1. **No Shadow Ray** — **Most impactful rendering defect** ❌ Not done
   Still using MixturePDF approach without explicit occlusion test. Adding a shadow ray for direct lighting remains the single highest-impact rendering quality improvement available.

2. **Fixed MIS Weights: `[1/3 light, 2/3 surface]`** ❌ Not done
   Still fixed heuristic. Power heuristic (pbrt-v4) or balance heuristic would adapt to scene conditions and reduce variance.

3. **Integrator trait extracted but single implementation** ✅ Done
   `Integrator` trait exists with `PathTracingIntegrator`. The `TODO(renderer-abstraction)` from v1 is resolved. Only one integrator type so far.

4. **Lights: Single Uniform Light Strategy** ❌ Not done
   Same as v1 — always picks a random light object uniformly. Power‑based selection would help for scenes with ~50 area lights (complex_scene).

5. **No Per‑Depth Bounce Limits** ❌ Not done
   Single `max_depth`. LuxCore distinguishes `diffuseDepth`, `glossyDepth`, `specularDepth`.

### Comparison: Rendering Loops

| Aspect | raytrace-rs | LuxCore | MoonRay |
|--------|------------|---------|---------|
| Shadow rays | No | Yes | Yes |
| MIS heuristic | Fixed weights | Power heuristic | Balance heuristic |
| Light strategy | Uniform | Uniform/Power/Log/DLSCache | Per‑layer light sets |
| Integrator container | `Integrator` trait | `PathTracer` engine class | `PathIntegrator` engine |
| Bounce control | Single depth | Per‑type (diffuse/glossy/spec) | Per‑type |
| RR threshold | Bounce ≥ 5, clamp(0.05, 1) | Bounce ≥ 3, cap 0.5 | Throughput threshold |
| Volumes | ConstantMedium (simple) | Full volume integration | Decoupled ray marching |
| Bidirectional | No | Hybrid Back‑Forward mode | No |
| Adaptive sampling | ✅ Welford variance + convergence mask | Yes (variance‑guided) | No |
| Renderer abstraction | ✅ `Renderer` trait + `CpuRenderer` | Engine class hierarchy | Engine class |
| Film abstraction | ✅ `Film` trait + `RgbFilm` + tiles | OutputMgr | `OutputDriver` |

---

## 4. Sampling

### Current Design

```
Sampler trait: sample(&self, n: u32, d: u32) → f64  [pure, deterministic, Sync]

SobolQmcSampler
├── Joe & Kuo 2008 direction numbers (512 dims)
├── Gray‑code recurrence for O(1) per‑sample advance
├── Per‑thread GrayCodeCache with digital shift scrambling
└── for_pixel(pixel_x, pixel_y) → deterministic seed

NaiveRandomSampler   — SplitMix64 hash of (n, d, seed)
StratifiedRandomSampler — jittered grid for dims 0‑1, hash for rest

DimCursor<S: Sampler> { base, offset, sample_idx, sampler }
  — Sampler is now *embedded in the cursor*
  — next_sample() calls sampler.sample(self.sample_idx, d) automatically
```

**Key changes since v1**:

- **Sampler embedded in `DimCursor`**: `DimCursor<S>` now owns a `sampler: S` field alongside `base`, `offset`, and `sample_idx`. This eliminates the separate `sampler` argument at every call site and makes the type generic over the sampler strategy.
- **QMC dimension enforcement**: Every bounce must consume exactly 11 dimensions, enforced by `debug_assert_eq!` — prevents the dimension aliasing that was previously a concern.
- **Digital shift**: Per-dimension scrambling via `splitmix_shift(seed, d)` replaces the previous per-pixel seed scrambling.

### Findings

**✅ Well‑done** (all carried forward)

- **Pure deterministic `sample(n, d)`** — same arguments same result everywhere. Enables `Sync`, deterministic reproduction.
- **Gray‑code cache** — efficient O(1) advance per sample.
- **512 dimensions** — adequate for ~60 bounces × ~8 dims = ~480 dims. Tight but not exceeded.
- **DimCursor with embedded Sampler** — cleaner API than passing sampler separately, trivially correct parallel iteration.
- **Fixed QMC dimension stride** — the debug assertion eliminates the previous class of dim-aliasing bugs.

**⚠️ Issues to address** (unchanged from v1)

1. **No Correlated Multi‑Jittered (CMJ) or Progressive Multi‑Jittered (PMJ) Sequences**
   MoonRay uses PMJ/CMJ for bounces 0‑2. CMJ provides better stratification than Sobol at low sample counts — important for progressive rendering where 1‑4 samples per pixel dominate the early preview.

2. **StratifiedRandomSampler Only Stratifies 2 Dimensions** ❌ Not done
   Only dims 0‑1 (pixel AA) are stratified. True N‑dimensional stratification would improve convergence for the first few bounces, but the Sobol sampler is already the default.

3. **No Blue Noise or Error‑Diffusion Sampling** ❌ Not done
   MoonRay ships precomputed blue‑noise sample tables. Important for real‑time / interactive previews.

### Comparison: Sampling Systems

| Aspect | raytrace-rs | LuxCore | MoonRay |
|--------|------------|---------|---------|
| Sequence type | Sobol (Gray‑code) | Sobol (direction vectors) | CMJ/PMJ + hash fallback |
| Dims | 512 | Unbounded | Precomputed tables |
| Perf model | O(1) via cache | Per‑dim XOR of direction vectors | Table lookup + hash |
| State | Stateless (thread‑local cache) | Per‑thread RNG + pass counter | `SequenceID` hash construction |
| Seed | Per‑pixel deterministic | Per‑pixel + random pass shift | Pixel + sample + purpose hash |
| Stratification | Full Sobol (all dims) | Full Sobol | CMJ for early, hash for late |
| Adaptive sampling | ✅ Yes (Welford variance) | Yes (variance‑guided) | No |

---

## 5. TODOs & Future Direction

### Current TODOs (19 total — down from 37 in v1)

| Category | Count | Files |
|----------|-------|-------|
| GPU pipeline preparation | 3 | main.rs(3) |
| Texture mapping 2D/3D split | 7 | texture/mod.rs(4), texture/mapping.rs(3) |
| Preview optimization | 3 | main.rs(3) |
| Type safety | 4 | hittable.rs(2), vec3.rs(2) |
| Transform system | 5 | transform.rs(5) (optional features + feat) |
| Renderer-agnostic | 1 | texture/mod.rs(1) |
| **Total** | **19** | 0 FIXMEs, 0 HACKs, 0 XXXs |

**TODO reduction**: 37 → 19 (18 resolved). The resolved TODOs included:
- `TODO(renderer-abstraction)` → ✅ Integrator + Renderer + Camera traits
- All `TODO(gpu)` markers in the old camera.rs scene construction boundaries → reduced to 3 remaining in main.rs
- Arena refactor plan is still documented in `docs/arena-refactor-plan.md` but the code-level TODOs were not added to individual hittable files

### What Was Done (Resolved Items from v1)

| Priority | Issue | Status |
|----------|-------|--------|
| P0 | Renderer trait extraction | ✅ Done — `Integrator` + `Renderer` + `Camera` traits |
| P1 | — | |
| P2 | Adaptive sampling | ✅ Done — Welford's variance + convergence mask |
| P3 | — | |
| — | BsdfSample struct→enum | ✅ Done — type-safe delta/non-delta routing |
| — | Dielectric tint | ✅ Done — `dielectric_tinted()` |
| — | QMC dimension enforcement | ✅ Done — fixed 11-dim stride + debug_assert |
| — | Camera decomposition | ✅ Done — `Camera` trait + `PerspectiveCamera` |
| — | Film abstraction | ✅ Done — `Film` trait + `RgbFilm` + `FilmTile` |
| — | Hittable decomposition | ✅ Done — `Intersectable` + `Bounded` + `Sampleable` + `Hit`/`SurfaceInteraction` |

### What Remains (Unresolved from v1)

**P0 — Immediate impact, low effort**

1. **Arena refactor** (~361 lines, 8 files)
   - Eliminates Arc overhead in Scene and primitives
   - Automatic light detection (no `add_light()` ceremony)
   - Flat storage for GPU upload
   - Design documented in `docs/arena-refactor-plan.md` — still valid

2. **Shadow ray** (~50 lines, path_tracer.rs + PDF trait)
   - Direct lighting with explicit visibility test
   - Dramatically reduces noise where occluders sit between hit points and lights
   - The single most impactful rendering quality improvement available

**P1 — High value for medium effort**

3. **Triangle mesh support** (new file + OBJ parser)
   - Unlocks all real‑world scenes
   - Start with a simple indexed mesh + OBJ parser

**P2 — Quality improvements**

4. **Power heuristic MIS** — replace fixed `[1/3, 2/3]` weights with adaptive power heuristic
5. **Rough clearcoat** — MIS between coating and substrate lobes (once the coating can be rough)
6. **Per‑type bounce limits** — add `max_diffuse_depth`, `max_glossy_depth`, `max_specular_depth`

**P3 — Polish**

7. **Complete transform system** — RotateX, RotateZ, Scale, composition macros
8. **Point3/Color3 newtypes** — type‑safety, prevent coordinate/color confusion
9. **Texture mapping 2D/3D split** — clean separation of UV vs world‑space mapping
10. **CMJ/PMJ sampler** — correlated multi‑jittered for early bounces

### Gap vs Production Renderers (Not Yet Addressed)

| Gap | Production status | When to tackle |
|-----|------------------|----------------|
| Triangle meshes | All three use as primary geometry | P1 |
| Spectral rendering | LuxCore + MoonRay support spectral | Not until basics are solid |
| Subsurface scattering | MoonRay: 3 models, LuxCore: volumes | Not planned |
| Normal/bump mapping | All three have per‑layer normal mapping | Not planned |
| Bidirectional path tracing | LuxCore: Hybrid Back‑Forward | Not planned |
| Volume rendering | LuxCore: full integration, MoonRay: decoupled marching | ConstantMedium is adequate for learning |
| Displacement mapping | MoonRay: supported | Planned in old hittable.rs, not currently |

---

## 6. pbrt-v4 Comparison

pbrt-v4 remains the closest architectural cousin to raytrace-rs. The comparison below is updated to reflect current state.

### Material / BxDF System

**pbrt-v4:** `Material` is a `TaggedPointer<11 material types>`. Each material has a typedef `using BxDF = SomeBxDF`. `BSDF` wraps **one** `BxDF` via `TaggedPointer` with a `Frame` for local/world space conversion.

| Aspect | raytrace-rs | pbrt-v4 |
|--------|------------|---------|
| **Material count** | 9 (6 concrete + 3 composition) | 11 (all concrete) |
| **Dispatch** | `Material` enum match → struct methods | `TaggedPointer` dispatch → template `GetBxDF()` |
| **Composition** | `Box<dyn Bsdf>` in Mix/Coated | No composition — `LayeredBxDF<Dielectric, Diffuse>` is a single BxDF class |
| **BSDF per vertex** | Single `Material` reference → match → child | Single `BxDF` via `TaggedPointer` |
| **Flags** | `PdfKind` (what PDF to use) | `BxDFFlags` (scattering type: refl/trans, specular/glossy/diffuse) |
| **Spectral** | RGB only | `SampledSpectrum` (4 wavelength samples, point-sampled) |
| **Sample struct** | `BsdfSample` enum (`Delta`/`NonDelta`) | `BSDFSample` struct (`f`, `pdf`, `flags`, `eta`) |
| **Delta routing** | Enum variant — type-level | `flags` bitmask — runtime check |

**Key difference — Layered materials**: pbrt-v4's `LayeredBxDF<Top, Bottom>` uses a Monte Carlo random walk through the layers (Guo et al. 2018). raytrace-rs uses the analytic Fresnel-split (correct only for smooth dielectric coating). This assessment is unchanged from v1.

**Recent improvement**: `BsdfSample` as an enum with explicit `Delta`/`NonDelta` variants is actually **cleaner** than pbrt-v4's `BxDFFlags` bitmask approach — the Rust compiler enforces exhaustive matching and no runtime flag check is needed.

### Primitives / Shapes

| raytrace-rs | pbrt-v4 |
|------------|---------|
| `Sphere`, `PlanarPatch<R>` (8 regions), `TransformObject<T,O>`, `ConstantMedium` | `Sphere`, `Cylinder`, `Disk`, `Triangle`, `BilinearPatch`, `Curve` |
| Manual transform chaining (Translate, RotateY) | `Transform *renderFromObject` — full 4×4 transforms |
| `Arc<dyn Intersectable>` / `Vec<Arc<>>` | Build-time `pstd::vector<Shape>`, run-time `Primitive` wraps Shape + Material + trans |
| BVH: `BvhNode` + `FlatBvh` | BVH: `BVHAggregate` (LinearBVH in v3) |
| `Region2D` trait for parametric surfaces | No parametric surface trait — quadrics and triangles |

No significant changes from v1 in the primitives area.

### Path Tracing Integrator

The **Integrator** comparison has shifted dramatically since v1:

| Aspect | raytrace-rs | pbrt-v4 |
|--------|------------|---------|
| **Architecture** | `Integrator` trait (generic `S`) + `PathTracingIntegrator` | `Integrator` → `ImageTileIntegrator` → `RayIntegrator` → `PathIntegrator` |
| **Renderer** | `Renderer` trait + `CpuRenderer` (adaptive, tiled) | No separate `Renderer` — integrator drives tiles |
| **Camera** | `Camera` trait + `PerspectiveCamera` | `CameraBase` → `PerspectiveCamera`, `OrthographicCamera` etc. |
| **Film** | `Film` trait + `RgbFilm` (variance tracking) | `Film` base class + `RGBFilm`, `GBufferFilm` |
| **Shadow rays** | No | Yes (`Unoccluded`) |
| **MIS** | Fixed `[1/3, 2/3]` weights | Power heuristic |
| **Adaptive sampling** | ✅ Welford variance | Yes (variance‑guided) |
| **Tile rendering** | ✅ Pre-allocated pool, reused across passes | `ImageTileIntegrator` base |
| **QMC stride** | ✅ 11 dims/bounce, debug-asserted | Per-type bounce dimensions |

**Key convergence**: The architecture now closely mirrors pbrt-v4's integrator hierarchy (though with trait-based generics instead of virtual inheritance). raytrace-rs actually goes beyond pbrt-v4 in a few areas:
- **Adaptive sampling** with Welford variance (pbrt-v4 has a simpler variance tracker)
- **QMC dimension discipline** (pbrt-v4 uses a stateful sampler, no cross-bounce assertion)
- **Pre-allocated tile pool** eliminates per-pass allocation entirely

**Remaining gap**: The shadow ray and power heuristic remain the two biggest quality gaps vs pbrt-v4.

### Sampling

| Aspect | raytrace-rs | pbrt-v4 |
|--------|------------|---------|
| **API** | Pure: `sample(n, d)` | Stateful: `StartPixelSample()`, `Get1D()`, `Get2D()` |
| **State** | Stateless (`Sync`) | Mutable (per-pixel, per-dimension cursor) |
| **Thread safety** | Thread-local cache + pure fn | Clone per thread |
| **Determinism** | Same `(n,d)` → same value | Same sequence for same pixel + sampleIndex |
| **Dim management** | `DimCursor<S>` — auto-advancing + debug assert | Implicit (sampler tracks dimension internally) |
| **Primary sequence** | Sobol (512 dims, Gray-code) | Multiple: Sobol, Halton, PMJ02BN, ZSobol, PaddedSobol, Stratified, Independent, MLT |
| **Scrambling** | Digital shift (per-dimension hash) | Per-dimension hash → Owen, FastOwen, PermuteDigits, None |

No significant architectural change since v1. The sampler is now embedded in `DimCursor`, making the API even cleaner. The debug assertion on fixed stride per bounce is an improvement over pbrt-v4's implicit dimension management.

### Summary: Updated Key Lessons from pbrt-v4

| Lesson | What to adopt | When |
|--------|--------------|------|
| **Shadow ray** | `Unoccluded()` check in direct lighting | P0 — major variance reduction |
| **Power heuristic** | Replace fixed `[1/3,2/3]` weights with `p²/(p²₁+p²₂)` | P1 — adaptive MIS |
| **Light sampler** | Pluggable light selection strategy | P2 — needed for many-light scenes |
| **Mesh support** | `Triangle` shape + indexed mesh storage | P1 — unlocks real scenes |
| **MC layered material** | `LayeredBxDF` random walk for rough coating | P2 — generalization of current Coated |
| **Integrator hierarchy** | ✅ Already done — `Integrator` + `Renderer` + `Camera` traits | Resolved |
| **Spectral rendering** | `SampledSpectrum` | Not a priority |
| **Sampler stateful API** | Keep stateless — better for GPU | Confirmed: current design is correct |
| **BsdfSample as enum** | ✅ Type-level delta routing better than pbrt-v4's flags | Confirmed: current design is superior |
| **QMC dim discipline** | ✅ Fixed stride + debug assert better than implicit cursor | Confirmed: current design is superior |

---

## 7. Summary

### Strengths (Keep & Maintain)

- **BsdfSample enum** — `Delta`/`NonDelta` variants structurally prevent a class of bugs
- **Per‑sample delta routing** via enum match — correct for compositions, type-level
- **Pure Sampler trait** — deterministic, Sync, clean dimension management
- **DimCursor with embedded Sampler** — auto-advancing, prevents dimension aliasing
- **Fixed QMC dimension stride** — 11 dims/bounce enforced by debug assertion
- **Integrator + Renderer + Camera traits** — clean abstraction boundaries
- **Film trait + RgbFilm** — with Welford variance tracking for adaptive sampling
- **PlanarPatch + Region2D** — clean parametric surface pattern
- **FlatBvh** — cache‑friendly, iterative traversal, production‑quality
- **GPU material tree serialization** — recursive flatten with tests
- **Constructor ergonomics** — chaining `.mix()`, `.coated()` is intuitive
- **Adaptive sampling** — Welford's online variance, convergence mask, early exit
- **Tiled rendering** — pre-allocated pool, optimal progressive cadence

### Issues Remaining

| Priority | Issue | Effort | Impact |
|----------|-------|--------|--------|
| P0 | Arena refactor (Arc → Box + lifetimes) | ~361 lines, 8 files | Enables GPU storage, auto lights |
| P0 | Shadow ray for direct lighting | ~50 lines | Major noise reduction |
| P1 | Triangle mesh support | New file + OBJ parser | Unlocks real scenes |
| P2 | Power heuristic MIS | ~30 lines | Reduced variance |
| P2 | Rough clearcoat (MIS between layers) | ~80 lines | Physical rough coating |
| P2 | Per‑type bounce limits | ~20 lines | Production bounce control |
| P3 | Complete transforms (RotateX/Z, Scale) | ~60 lines | Full transform support |
| P3 | Point3/Color3 newtypes | ~50 lines | Type safety |
| P3 | CMJ/PMJ sampler | ~100 lines | Better early-preview quality |
| P3 | Texture mapping 2D/3D split | ~80 lines | Cleaner architecture |

### Progress Since v1 (85d8021 → HEAD)

**Resolved: 10 items** — including the largest architectural items (renderer abstraction, camera decomposition, film abstraction, hittable decomposition).

**Remaining: 10 items** — the P0 items (arena refactor, shadow ray) and P1 (meshes) are the same as v1. The P2/P3 items (power heuristic, transforms, newtypes) remain untouched.

### Development Direction

The codebase has **strong fundamentals** that have only improved since v1. The architecture is now much closer to production renderers (pbrt-v4 especially) with clean trait boundaries for integrator, renderer, camera, and film.

The natural progression remains:

1. **Scaffold for complexity** (now – 3 months): Arena refactor → meshes → shadow ray
2. **Quality** (3 – 6 months): MIS improvements, type safety, per‑type bounce limits, adaptive sampling (✅)
3. **GPU exploration** (6 – 12 months): The existing GPU serialization, pure sampler trait, flat BVH, and clean integrator/renderer boundaries all position the codebase well for a WGSL/GPU pipeline.

**Key insight since v1**: The integrator/renderer/camera/film decomposition means GPU exploration can now proceed as an independent `GpuRenderer` implementing the same `Renderer` trait, rather than requiring a separate pipeline entry point.
