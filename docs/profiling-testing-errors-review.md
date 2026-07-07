# Profiling, Testing & Error Handling Review

> Generated: 2026-06-30
> Last updated: 2026-06-30 (reconciled against fresh codebase scan)
>
> Review of the raytrace-rs project's profiling harness, test suite & coverage, and error handling compared to production renderers (pbrt-v4, LuxCore, MoonRay).

______________________________________________________________________

## Verdict: NEEDS_WORK

Three areas at distinct maturity levels. Profiling is architecturally sound but incomplete. Tests are narrowly scoped. Error handling is development-grade but **several previously flagged issues have already been fixed**.

______________________________________________________________________

## 1. Profiling Harness

### What's Good

- The `profiling` crate with 4 backend options (`tracing`, `puffin`, `tracy`, `optick`) via Cargo features is the right abstraction — matches how production renderers ship profiling backends behind a feature flag.
- `default = ["profile-with-tracing"]` is a sane default: zero-cost when disabled, useful traces when enabled.
- The `#[profiling::function]` annotation on `live_render` and `headless_render` plus manual `profiling::scope!("...")` calls show awareness of the API.
- `tracing-subscriber` is initialized with env-filter, thread IDs, and line numbers — the tracing pipeline is production-ready.

### Current Instrumentation (17 total)

| Type | Count |
|------|-------|
| `profiling::scope!(...)` | 15 |
| `#[profiling::function]` | 2 |

**11 unique scope names across the codebase:**

| Scope | File | Context |
|-------|------|---------|
| `"ui_surface_resize"` | `main.rs:101` | UI resize handler |
| `"ui_blit"` | `main.rs:127` | UI blit path |
| `"ui_present"` | `main.rs:153` | UI present path |
| `"image_save"` | `main.rs:299,548` | Image save (×2) |
| `"scene_build"` | `main.rs:392,442,476` | Scene construction (×3) |
| `"root_bvh_build"` | `main.rs:451,521` | BVH build (×2) |
| `"render"` | `main.rs:457` | Live render dispatch |
| `"render_cpu"` | `main.rs:531` | Headless CPU render dispatch |
| `"camera_render_loop"` | `cpu.rs:101` | Camera render loop |
| `"sample_pass"` | `cpu.rs:137` | Per-sample pass |
| `"complex_scene_build"` | `scene.rs:201` | Complex scene construction |

### Where It's Insufficient

| Scope | Current | Production (LuxCore/MoonRay) |
|-------|---------|------|
| BVH build | `"root_bvh_build"` — single scope | Per-level build timing, SAH evaluation cost |
| BVH traversal | None | Per-ray traversal cost, leaf intersection stats |
| Intersection tests | None | Sphere/quad/transform hit rates |
| Per-sample breakdown | `"sample_pass"` — single scope for the whole pass | Ray-gen / shade / scatter / direct-light per bounce |
| Memory/alloc | None | Allocation counters, pool pressure |
| GPU serialization | None | Buffer upload time |
| Film/tonemap | `"image_save"` | Write cost breakdown |

### Specific Gaps vs Production Renderers

1. **No per-phase breakdown within `sample_pass`.** The entire sample loop is wrapped by one `profiling::scope!`. When the profiler shows a bottleneck, you can't tell whether it's BVH traversal cost, material evaluation, or PDF sampling. pbrt-v4 instruments each bounce phase.

2. **No BVH traversal profiling.** `FlatBvh::intersect` has zero profiling scopes. Production renderers track traversed-node count, leaf-intersection count, early-exit rate. Critical for BVH optimization.

3. **No per-object-type intersection counters.** The `Intersectable` trait has no instrumentation wrapper. You can't see whether ray vs sphere costs dominate vs ray vs planar-patch costs.

4. **No allocator or memory profiling.** The arena refactor is identified as P0 in `ARCH_REVIEW.md`, but there's no profiling to track malloc pressure or `Arc` refcount traffic.

5. **No GPU-side profiling infrastructure.** GPU modules (`material/gpu.rs`, `texture/gpu.rs`) serialize GPU buffers but have zero profiling scopes.

6. **No explicit puffin/tracy/optick initialization.** The `profiling` crate handles backend selection at compile time, but there's no `puffin::GlobalProfiler::new(...)`, `tracy_client::Client::start()`, or Optick init code. The default `profile-with-tracing` path works, but non-tracing backends may not be wired up correctly.

**Recommendation:** Add per-integrator-phase profiling scopes (`ray_gen`, `bvh_traverse`, `material_sample`, `mis_pdf`, `shadow_ray`) and instrument `FlatBvh::intersect` and `ConstantMedium::intersect` at minimum.

______________________________________________________________________

## 2. Test Suite & Coverage

### Current State: 36 `#[test]` functions across 7 modules

**No integration tests. No benchmarks. No dev-dependencies. No property or fuzz testing.**

All tests are standard Rust unit tests embedded in `#[cfg(test)] mod tests` blocks.

### Test Inventory (corrected from original review)

| Module | Tests | What they cover | Gap |
|--------|-------|-----------------|-----|
| `sampler.rs` | 11 | Direction number validity, unit interval ranges, van der Corput exact values, determinism, seed diversity, Sync/Send traits | No edge-case behavior (extreme dimensions, overflow), no QMC stratification quality metrics |
| `film/rgb.rs` | 5 | Variance convergence, convergence mask, RGB8 dimension conversion | No pixel-level correctness tests, no tone mapping tests |
| `material/mod.rs` | 7 | GPU buffer serialization for each material type, Mix/Coated tree flattening, nested compositions, custom material passthrough | No BSDF correctness tests, no energy conservation checks, no reciprocity tests, no textured material sampling tests |
| `flat_bvh.rs` | 5 | Node size, empty BVH, single/two-sphere intersection correctness, multi-object BVH matches BVH | No stress tests (many objects), no SAH cost verification, no transform integration, no correctness vs flat-list baseline |
| `const_medium.rs` | 3 | Volume scattering, sparse medium pass-through, normal at volume hit | No absorption/scattering coefficient sweeps, no heterogeneous medium tests |
| `texture/gpu.rs` | 4 | Sentinel value, node layout size, empty/push buffer | No texture value correctness tests, no mapping tests, no image texture tests |
| `integrator/mod.rs` | 1 | Render a 4×4 minimal scene (basic integration smoke test) | Single tiny test — no bounce correctness, no light sampling, no background handling |

### What Production Renderers Do That This Doesn't

1. **Integration/regression tests (critical gap).** No test renders a scene, saves output, and compares pixel-by-pixel against a known reference. Every production renderer has a scene corpus with golden images. Even a single golden-image test (Cornell box, fixed seed, ±1/255 tolerance) would catch material or integrator regressions that unit tests miss entirely.

2. **No benchmark suite.** The only timing output is the console log. No `#[bench]` functions, no Criterion benchmarks, no performance regression tracking. A commit that doubles BVH traversal time goes undetected. This is especially important for a renderer where performance is a core feature.

3. **No property-based or randomized testing.** Materials have fuzzable sampling paths (compositions with random weights, extreme roughness values, NaN inputs) that aren't tested. The sampler QMC code has no stratification quality metrics.

4. **No error-path tests.** No test for what happens when `ImageTexture::new` fails (it panics), when image resolution is 0, or when the BVH gets NaN positions.

5. **No multi-threaded correctness tests.** No test verifies the film output is identical between single-threaded and multi-threaded render runs. Welford's online algorithm in `rgb.rs` uses shared `m_2` state that could race under concurrency.

### File-by-File Test Gaps (untested modules)

| File | Lines | Missing tests |
|------|-------|--------------|
| `sphere.rs` | 171 | Zero tests. Ray-sphere intersection, moving sphere, sphere sampling/PDF. |
| `planar/mod.rs` + regions (8 files) | 313+ | Zero tests. Quad/Tri/Ellipse intersection, PDF values, UV mapping. |
| `transform.rs` | 215 | Zero tests. Translate/RotateY correctness, AABB after transform. |
| `pdf.rs` | 237 | Zero tests. CosinePDF, GGXSamplePDF, EmitterPDF, MixturePDF value/generate. |
| `integrator/path_tracer.rs` | 162 | Zero tests. Li() correctness, variance, bounce behavior, background handling. |
| `camera/perspective.rs` | 183 | Zero tests. Ray generation, jitter bounds, defocus disk, resolution edge cases. |
| `main.rs` | 572 | Zero tests. CLI args parsing, scene selection, headless vs live mode. |

**Recommendation:** Add a Criterion benchmark suite (3–4 benches: BVH build, ray intersection, sample pass, full Cornell box) and a `tests/` integration directory with at least one golden-image test that renders a fixed scene with fixed seed and compares against a committed reference.

______________________________________________________________________

## 3. Error Handling

### Current State

**12 panic points in production code + 1 in test code.** No error handling crates (`anyhow`, `thiserror`, `eyre`, `snafu`). No custom error types. Minimal `Result<>` propagation.

### Important: Several Original Issues Already Fixed

The three most critical issues from the initial review have been resolved:

| Original Concern | Status | Current Code |
|-----------------|--------|-------------|
| `main.rs:107` — `framebuffer.read().unwrap()` crashes on poisoned lock | ✅ **FIXED** | `let Ok(fb) = self.framebuffer.read() else { error!(); return; }` — graceful degradation |
| `main.rs:125` — `surface.buffer_mut().unwrap()` crashes on resize race | ✅ **FIXED** | `let Ok(mut buffer) = self.surface.buffer_mut() else { error!(); return; }` |
| `main.rs:553-556` — `panic!` when `satty` isn't installed | ✅ **FIXED** | `Err(_) => { info!("satty not found, skipping...") }` — graceful log |

### Current Issues by Category

#### `.unwrap()` — 3 production + 1 test

| File | Line | Code | Severity |
|------|------|------|----------|
| `src/main.rs` | 154 | `buffer.present().unwrap()` | **Medium** — presentation failure (Wayland/X11 context loss) crashes. Mitigation: this is the last UI operation per frame, so a frame drop is acceptable but a crash is not. |
| `src/main.rs` | 226 | `.create_window(...).unwrap()` | **Acceptable** — window creation failure is fatal in a GUI app |
| `src/scene.rs` | 811 | `ImageTexture::new("./earthmap.png").unwrap()` | **High** — missing asset crashes the process. Production renderers substitute a magenta fallback. |
| `src/material/mod.rs` | 849 | `buf.nodes.last().unwrap()` | **Low** — test code only |

#### `.expect()` — 3 calls

| File | Line | Code | Severity |
|------|------|------|----------|
| `src/main.rs` | 51 | `Context::new(...).expect("failed to create softbuffer context")` | **Acceptable** — init failure is fatal |
| `src/main.rs` | 54 | `Surface::new(...).expect("failed to create softbuffer surface")` | **Acceptable** — init failure is fatal |
| `src/main.rs` | 183 | `surface.resize(...).expect("failed to resize surface")` | **Medium** — resize failure on minimized windows crashes |

#### `panic!()` — 5 calls (3 acceptable, 2 problematic)

| File | Line | Code | Severity |
|------|------|------|----------|
| `src/vec3.rs` | 233,245 | `panic!("Vec3 index out of bounds: {}", i)` | **Acceptable** — invariant violation in trait impl |
| `src/aabb.rs` | 71 | `panic!("Invalid axis index: {}", axis)` | **Acceptable** — defensive guard |
| `src/scene.rs` | 290 | `panic!("Failed to load earthmap.png: {e:?}")` | **High** — missing asset crashes. Scene constructors could return `Result<Scene, ...>`. |
| `src/scene.rs` | 658 | `panic!("Failed to load image as Texture: {:?}", e)` | **High** — same pattern in `earth_sphere()` scene builder |

#### `unreachable!()` — 1 call

| File | Line | Code | Severity |
|------|------|------|----------|
| `src/pdf.rs` | 106 | `PdfKind::Delta => unreachable!()` | **Low** — genuine algebraic invariant |

#### Missing infrastructure

- `catch_unwind` — 0 calls
- `std::panic::set_hook` — 0 calls
- Error handling crates (`anyhow`, `thiserror`, `eyre`, `snafu`) — none in `Cargo.toml`
- Custom error types — 0 defined

### What Production Renderers Do

- **pbrt-v4** returns `optional` or zero values for failures — missing textures substitute a checker, malformed scenes print warnings and continue.
- **LuxCore** has a full `boost::optional`/error-return pattern for asset loading. Missing textures trigger a fallback. GPU errors are caught per frame.
- **MoonRay** uses error collectors — errors are aggregated, logged, and the render continues.

### Remaining Issues

| # | Severity | Location | Issue | Fix |
|---|----------|----------|-------|-----|
| 1 | **High** | `scene.rs:290,658,811` | Three locations panic on missing texture/image assets | Return a magenta fallback texture, or change scene constructors to return `Result<Scene>` |
| 2 | **Medium** | `main.rs:154` | `buffer.present().unwrap()` — presentation failure crashes | Change to `if let Err(e) = buffer.present() { error!(...); }` — skip frame, don't crash |
| 3 | **Medium** | `main.rs:183` | `surface.resize().expect()` — crashes on resize failure | Use `if let Err(e) = surface.resize(...) { error!(...); return; }` — skip frame |

### The Broader Question: "Do we need more error handling in a renderer project?"

For a learning/educational path tracer, the current approach is defensible — fast iteration matters more than robustness. Notably, **3 of the 5 original high-severity concerns have already been fixed by the project maintainers**. The remaining gap is concentrated in `scene.rs` scene constructors panicking on missing image assets (lines 290, 658, 811).

Below is a concrete error handling plan informed by the 11-principle discussion in `docs/renderer-error-handling-discussion.md`.

---

## Error Handling Plan: Applying the 11 Principles

### The Quick Fixes (no new crate dependencies)

These should be done immediately regardless of the broader strategy:

1. **`scene.rs:291,660`** — Replace `panic!` on missing texture with a magenta fallback `ImageTexture`. No `Result` needed — consistent with pbrt-v4 behavior.
2. **`main.rs:154`** — Replace `buffer.present().unwrap()` with graceful frame skip.
3. **`main.rs:183`** — Replace `surface.resize().expect()` with `if let Err(e) = ...`.

---

### Phase 1: Classify existing panic points

| Principle | Current code | Verdict | Action |
|-----------|-------------|---------|--------|
| #1 — programmer bugs | `vec3.rs:233,245` — out-of-bounds index | Keep `panic!` | Correct |
| #1 — programmer bugs | `aabb.rs:71` — invalid axis | Keep `panic!` | Correct |
| #1 — programmer bugs | `pdf.rs:106` — Delta unreachable | Keep `unreachable!` | Correct |
| #1 — recoverable error | `scene.rs:290` — missing `earthmap.png` | **Convert to `Result`** | See Phase 2 |
| #1 — recoverable error | `scene.rs:660` — missing `earthmap.png` | **Convert to `Result`** | See Phase 2 |
| #1 — recoverable error | `scene.rs:811` — missing `earthmap.png` (bare `.unwrap()`) | **Convert to `Result`** | See Phase 2 |
| #1 — recoverable error | `main.rs:154` — `buffer.present()` failure | **Graceful error** | Wrap in `if let Err` |
| #1 — recoverable error | `main.rs:183` — `surface.resize()` failure | **Graceful error** | Wrap in `if let Err` |

**Principle #2** (rendering code should almost never return `Result`) is already respected — `sample()`, `scatter()`, `intersect()`, `eval_bsdf()` all return values, not `Result`. The panics that need conversion are all at **system boundaries**: scene construction and UI events.

---

### Phase 2: Introduce typed errors with `thiserror` at library boundaries

Add `thiserror` to `[dependencies]` (not dev — library users need the error types).

#### 2a. Error types needed

```
src/
  error.rs                          # re-exports & common types
  scene/
    error.rs                        SceneError
  texture/
    error.rs                        TextureError
```

```rust
// src/scene/error.rs
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SceneError {
    #[error("failed to load texture '{path}'")]
    TextureLoad {
        path: String,
        source: TextureError,    // or image::ImageError
    },

    #[error("scene construction error: {0}")]
    Construction(String),

    // Future: mesh parse errors
}
```

```rust
// src/texture/error.rs
#[derive(Error, Debug)]
pub enum TextureError {
    #[error("image file not found: {0}")]
    NotFound(String),

    #[error("unsupported image format")]
    UnsupportedFormat,

    #[error(transparent)]
    ImageDecode(#[from] image::ImageError),
}
```

#### 2b. Change scene constructors to return `Result`

```rust
// Before
pub fn complex_scene() -> Self {       // panics on missing texture
    ...
    let emat = match ImageTexture::new("./earthmap.png") {
        Ok(tex) => ...,
        Err(e) => panic!("..."),
    };
    ...
}

// After
pub fn complex_scene() -> Result<Self, SceneError> {
    ...
    let emat = ImageTexture::new("./earthmap.png")
        .map_err(|e| SceneError::TextureLoad {
            path: "./earthmap.png".to_string(),
            source: TextureError::ImageDecode(e),
        })?;
    ...
    Ok(scene)
}
```

Only the two scene constructors that load textures change:
- `complex_scene()` (line 201, uses `earthmap.png` — panic on 290)
- `earth_sphere()` (line 655, uses `earthmap.png` — panic on 660, unwrap on 811)

The 4 texture-free constructors (`cornell_box`, `cornell_box_const_meds`, `checkered_spheres`, `simple_world`) stay as `fn() -> Self`.

#### 2c. Scene `new()` stays infallible

```rust
pub fn new() -> Self { ... }   // no change — never loads files
```

---

### Phase 3: Use `anyhow` at application boundaries

Add `anyhow` to `[dependencies]`. This follows **Principle #4** (anyhow only at application boundaries) and **Principle #5** (composition with thiserror).

```rust
// src/main.rs — the match that selects scenes
fn build_scene(name: &str) -> anyhow::Result<Scene> {
    let scene = match name {
        "cornell_box" => Scene::cornell_box(),
        "cornell_box_const_meds" => Scene::cornell_box_const_meds(),
        "complex_scene" => Scene::complex_scene()
            .context("building complex_scene")?,   // <-- new
        "earth_sphere" => Scene::earth_sphere()
            .context("building earth_sphere")?,     // <-- new
        "checkered_spheres" => Scene::checkered_spheres(),
        "simple_world" => Scene::simple_world(),
        _ => anyhow::bail!("unknown scene: {name}"),
    };
    Ok(scene)
}
```

**This gives us Principle #6** (context answers "what were we trying to do?"):

```
Error: building complex_scene

Caused by:
    failed to load texture 'earthmap.png'

Caused by:
    No such file or directory
```

The `main` function becomes:
```rust
fn main() -> anyhow::Result<()>
```
instead of the current `Result<(), winit::error::EventLoopError>`.

---

### Phase 4: Origin tracking with `#[track_caller]` (future/optional)

Following **Principle #8**, apply the `Located<E>` wrapper to error constructors:

```rust
// src/error.rs
use std::panic::Location;

#[derive(Debug)]
pub struct Located<E> {
    pub error: E,
    pub location: &'static Location<'static>,
}

impl<E> Located<E> {
    #[track_caller]
    pub fn new(error: E) -> Self {
        Self { error, location: Location::caller() }
    }
}

#[macro_export]
macro_rules! err {
    ($e:expr) => { Located::new($e) };
}
```

Usage:
```rust
// Instead of:
Err(SceneError::TextureLoad { ... })

// Write:
Err(err!(SceneError::TextureLoad { ... }))
```

The error now carries **Principle #7** — separate construction site (the `err!` line) from **Principle #6** propagation context (the `.context()` call chain).

When `?` propagates it, both pieces survive:
```
src/scene.rs:291 — failed to load texture 'earthmap.png'

  (constructed at scene.rs:291)

  Caused by:
      building complex_scene  (main.rs:348)
```

**Verdict on when to implement this:** Not needed now. Add it when:
- Asset loading grows complex (OBJ/glTF parsers, multi-file scenes)
- Users start reporting errors that need debugging across subsystem boundaries
- The project starts accepting external contributions

---

### Phase 5: Defensive panics in core rendering

Following **Principle #1**, these stay as panics/asserts:

| Location | Form | Why |
|----------|------|-----|
| `vec3.rs:233,245` — out-of-bounds index | `panic!` | Core invariant |
| `aabb.rs:71` — invalid axis | `panic!` | Core invariant |
| `pdf.rs:106` — Delta unreachable | `unreachable!` | Algebraic invariant |
| Future: BSDF normalization | `debug_assert!` | Numerical invariant, debug-only |
| Future: BVH validity | `debug_assert!` | Data structure invariant, debug-only |
| Future: PDF sums near 1.0 | `debug_assert!` | Numerical correctness, debug-only |

---

### Summary: The 4 Information Kinds Mapped to Our Codebase

| Info Kind | Our Mechanism | Example |
|-----------|---------------|---------|
| **1. Kind (typed)** — *What* failed? | `thiserror` enum variants | `SceneError::TextureLoad`, `TextureError::NotFound` |
| **2. Source location** — *Where* was it constructed? | `Located<E>` + `#[track_caller]` (Phase 4) | `scene.rs:291` |
| **3. Cause chain** — *Why* did it fail? | `#[from]` / `#[source]` on thiserror variants | `ImageError → TextureError → SceneError` |
| **4. Propagation context** — *What were you trying to do?* | `anyhow::Context` | `"building complex_scene"` |

### Implementation Order

| Phase | What | Effort | Dependencies |
|-------|------|--------|-------------|
| **Quick fixes** | Fix 3 unwraps/panics, magenta fallback | 30 min | None |
| **Phase 1** | Add `thiserror` + `anyhow` to Cargo.toml | 5 min | Cargo |
| **Phase 2** | Define `SceneError`, `TextureError`, convert 2 scene constructors | 1–2 hr | Phase 1 |
| **Phase 3** | Wire `anyhow` in `main.rs`, add scene-build context | 30 min | Phase 2 |
| **Phase 4** | `#[track_caller]` + `Located<E>` pattern | Optional | Phase 3 |
| **Phase 5** | Add `debug_assert!` guards to core rendering | Ongoing | None |

### What Goes Unchanged

- `sample()`, `scatter()`, `intersect()`, `eval_bsdf()` — stay pure, no `Result` (**Principle #2**)
- `vec3.rs` / `aabb.rs` / `pdf.rs` panics — correct programmer bug guards (**Principle #1**)
- `main.rs` window init `.expect()` — acceptable, init failure is fatal (**Principle #1** — not a recoverable error)
- `main.rs` `.create_window().unwrap()` — acceptable for the same reason

______________________________________________________________________

## Summary

| Area | Original Verdict | Updated Verdict | Most Critical Fix |
|------|-----------------|-----------------|-------------------|
| **Profiling** | Needs more granular scopes | Still valid — no code changes detected | Add per-phase scopes: `ray_gen`, `bvh_traverse`, `material_sample`, `mis_pdf` |
| **Tests** | Narrow — 29 tests, 0 integration | **Corrected: 36 tests**, still 0 integration, 0 benches | Add 1 golden-image integration test + 3 Criterion benchmarks |
| **Error handling** | 3+ critical weaknesses | **Improved: 3/5 critical issues already fixed** + enriched plan from rendering error handling discussion | See 5-phase plan above: thiserror at boundaries, anyhow at main, debug_assert in core |

### Key Changes from Rendering Error Handling Discussion

| Principle | How It Changes Our Plan |
|-----------|------------------------|
| **#1** — Not everything is an error | Classified all 12 panic points into "keep as panic" (7) vs "convert to Result" (5) |
| **#2** — Rendering code should almost never return Result | Confirmed our core functions (`sample`, `scatter`, `intersect`, `eval_bsdf`) already satisfy this |
| **#3** — Typed errors with `thiserror` | New: `SceneError`, `TextureError` enums in `scene/error.rs` and `texture/error.rs` |
| **#4** — `anyhow` at application boundaries | New: `main.rs` returns `anyhow::Result<()>`, scene match uses `.context()` |
| **#5** — Composition | `SceneError` sources `TextureError`, which `#[from]`s `image::ImageError` |
| **#6** — Context answers "what were we trying to do?" | `"building complex_scene"` breadcrumb propagates with error |
| **#7** — Context ≠ construction site | `.context()` in `main.rs` is *propagation*; `err!()` in scene.rs is *construction* |
| **#8** — `#[track_caller]` | `Located<E>` wrapper planned for Phase 4 |
| **#9** — Wrapping errors in location | `err!()` macro expands to `Located::new(...)` |
| **#10** — `?` still works | All changes preserve `?` compatibility |
| **#11** — Origin and propagation separate | 4-info-kind mental model maps directly to design |

### Key Corrections vs Original Review

| Original Claim | Corrected Value |
|----------------|-----------------|
| 29 tests across 4 modules | **36 tests across 7 modules** (const_medium, film/rgb, and integrator/mod were missed) |
| 13 `unwrap()` calls | **3 production + 1 test** (many were in already-fixed code paths) |
| `framebuffer.read().unwrap()` at main.rs:107 | **Already fixed** — uses `let Ok(fb) = ... else { error!(); return; }` |
| `display_image()` panic at main.rs:553 | **Already fixed** — graceful `info!()` on satty not found |
| `surface.buffer_mut().unwrap()` at main.rs:125 | **Already fixed** — uses `let Ok(mut buffer) = ... else { error!(); return; }` |
