# Adaptive Sampling — Design & Refactor Plan

______________________________________________________________________

## Changelog

- **v1 (2026-06-27)** — Initial implementation. Welford's online variance, hybrid
  convergence threshold, render loop integration.
- **v2 (2026-06-27)** — This document. Codebase-correct abstraction catalog, deferred
  feature criteria, and comparison against prod renderers.
- **v3 (2026-06-28)** — Denoiser integration audit. Added §6 (cross-document
  integration points), updated §2a/§4a to reflect denoiser's per-channel variance
  requirements, added `raw_data()` contract, updated dependency graph.
- **v4 (2026-06-28)** — Bi-directional audit with `docs/denoiser.md`. Resolved
  6 conflicts: raw_data() variance format, overlay_radiance() behavior, VarianceEstimator
  dependency, per-channel variance, double-denoising guard, progressive preview.
  Added cross-references in §5.1/§5.3/§5.4/§5.5/§6.
- **v5 (2026-06-28)** — Three-document audit with `docs/ARCH_HYBRID.md` and
  `docs/denoiser.md`. Added `pixels` dual semantics note (sum vs mean after
  denoising) in §0.1. Added ARCH_HYBRID cross-reference for unified dependency order.
- **v6 (2026-06-28)** — Static dispatch audit. Replaced `&dyn ConvergenceCriterion`
  with `ConvergenceCriterionKind` enum + `From` impls. Replaced `&dyn Film` with
  generic `F: Film` on `SamplingPolicy` trait.
- **v7 (2026-06-28)** — Applied Niri/Smithay `render_elements!` pattern: **Descriptor →
  Concrete → Wrapper**. Renamed `ConvergenceCriterionKind` to `ConvergenceCriterionEnum`.
  Added `ConvergenceCriterionKind` descriptor enum. Added `From` impls for ergonomic
  construction.
- **v8 (2026-06-29)** — Updated §5.6 to reference `DenoiserFeatures` SoA pattern
  (denoiser.md §Phase 2). Variance stored as `Vec<[f64; 3]>` on `DenoiserFeatures`,
  populated from `VarianceEstimator::variance()` during `add_sample()`.

______________________________________________________________________

## 0. Current Implementation

### 0.1 Data Model

Three parallel per-pixel vectors in `RgbFilm` (src/film/rgb.rs):

```
pixels: Vec<Color3>       — accumulated radiance (sum, not mean)
sample_counts: Vec<u32>   — sample count per pixel
m_2: Vec<Color3>          — Welford's M2 accumulator (per-channel)
```

**Note on `pixels` semantics:** After denoising (`docs/denoiser.md` §Phase 1),
`pixels` holds per-pixel **mean** (not sum) — the denoised filtered output. The
`denoised: bool` flag on `RgbFilm` distinguishes the two states. `to_rgb8()` checks
this flag: if `denoised`, uses `pixels` directly; if not, divides by `sample_counts`.

`FilmTile` (src/film/tile.rs) tracks per-tile pixels and a `sampled: Vec<bool>`
mask to distinguish "unsampled" from "sampled with zero contribution".

### 0.2 Welford Update

`RgbFilm::add_sample(x, y, color)` performs a single-sample Welford update per pixel:

```
n_prev = sample_counts[index]
if n_prev == 0:
    pixels = color, counts = 1, m_2 = 0        (variance undefined for n=1)
else:
    mean_prev = pixels / n_prev
    delta = color - mean_prev
    pixels += color
    m_2 += delta^2 * n_prev / (n_prev + 1)     (algebraically equivalent to classic Welford)
    counts += 1
```

`merge_tile` delegates to `add_sample(tx, ty, color)` for each sampled tile pixel
(weight=1.0 since camera-ray premultiplication already applied).

### 0.3 Convergence Criterion

`pixel_variance(idx)` returns `m_2 / (n-1)` (sample variance, Bessel's correction),
taking the max across RGB channels. Returns `INFINITY` for n < 2.

`reset_convergence_mask` computes per-pixel:

```
var_rms = variance / n                           (variance of the mean)
luminance = 0.2126*R + 0.7152*G + 0.0722*B      (sRGB luminance, via LUMINANCE const)
converged = n >= min_samples
            && (var_rms < threshold_abs || var_rms / luminance.max(1e-6) < threshold_rel)
```

### 0.4 Render Loop (src/renderer/cpu.rs)

`CpuRenderer<I, S, Fact>` is generic over integrator, sampler, and sampler factory:

```
pre-allocate converged mask: vec![false; width × height]
pre-allocate tile pool

for each pass:
    if pass ≥ min_samples_before_adapt:
        film.reset_convergence_mask(...)      (refills existing buffer, no alloc)
        if all(converged): publish + break
    else:
        converged.fill(false)

    tile_pool → par_iter_mut:
        for pixel in tile:
            if !converged[global_idx]: ray trace → tile.add_sample()

    sequential merge: film.merge_tile(tile)

    adaptive-cadence progressive publish
    pass timing log every 8 passes
```

### 0.5 Defaults

| Field | Default | Meaning |
|-------|---------|---------|
| `threshold_abs` | `1e-4` | Absolute variance-of-mean floor |
| `threshold_rel` | `0.02` | Relative noise tolerance (stddev ≈ 14% of luminance) |
| `min_samples_before_adapt` | `64` | Minimum passes before convergence check activates |

### 0.6 Tests

- `variance_converges_for_constant_samples` — 1000 identical samples → variance ≈ 0
- `variance_infinity_for_single_sample` — n=1 → variance = INFINITY
- `variance_positive_for_varying_samples` — alternating 0/1 → variance > 0.1
- `convergence_mask_basic` — constant-sampled pixel converges, unsampled doesn't
- `to_rgb8_dimensions` — output buffer size correct

______________________________________________________________________

## 1. Comparison Against Prod Renderers

### 1.1 vs pbrt-v4

pbrt-v4's `VarianceEstimator<Float>` (src/pbrt/util/sampling.h) implements the same
Welford algorithm with identical semantics (`Add`, `Variance`, `Mean`, `Count`).
Key difference: pbrt-v4 also provides `Merge()` for parallel tile aggregation
using the parallel Welford formula:

$$
\begin{aligned}
S &= S_1 + S_2 + \frac{(\mu_1 - \mu_2)^2 \cdot n_1 \cdot n_2}{n_1 + n_2} \\
\bar{x} &= \frac{n_1 \mu_1 + n_2 \mu_2}{n_1 + n_2} \\
n &= n_1 + n_2
\end{aligned}
$$

pbrt-v4 uses variance only for EXR metadata output (diagnostic channels), never
for adaptive sampling decisions. Its render loop runs a fixed SPP for all pixels.

Our `merge_tile` avoids needing `Merge()` by design: each tile contributes exactly
one sample per pixel per pass, so the merge is equivalent to a single `Add()` call.
If tiles were ever split for intra-tile parallelism, `Merge()` would be required.

pbrt-v4 also uses a *running* `mean` field (not recomputing `sum/n` each call),
which is slightly more numerically stable at high sample counts where `sum/n`
loses mantissa bits. Our approach recomputes `mean = pixels / n_prev_f` each
`add_sample` call. For typical render sample counts (< 10k) and radiance ranges
(< ~100), this is not a practical concern.

### 1.2 vs LuxCore

LuxCore implements per-pixel variance-driven adaptive sampling with a **neighborhood
coherence check**: a pixel is not marked converged if its 3×3 or 5×5 neighbor means
differ significantly. This prevents premature convergence on sharp edges where a
pixel's samples are locally consistent (low variance) but the pixel sits on a
boundary with different content.

We lack this neighborhood check. A pixel on a horizontal edge may have low
per-pixel variance but be incorrectly converged. This is the highest-value
quality improvement available at current complexity.

### 1.3 vs OpenMoonRay

OpenMoonRay uses a **global sample budget** model: per-pixel importance weights
are computed each pass, and the next batch of samples is allocated to maximize
global MSE reduction per unit compute cost. This is fundamentally different from
our binary converged/unconverged decision — it distributes samples continuously
across all pixels.

This requires: per-pixel importance weights, a budget allocator, and interaction
with DOF/motion blur/AA variance from multiple dimensions. The complexity is
~2000+ lines vs our ~130 lines. It is justified only when targeting fixed-time
renders (not fixed-SPP) with wide variance disparity across the image.

______________________________________________________________________

## 2. Abstraction Catalog

### 2a. `VarianceEstimator` — composable Welford engine

**Currently:** Welford logic is inlined in `RgbFilm::add_sample` (rgb.rs:82-101).
Three parallel vectors (`pixels`, `sample_counts`, `m_2`) carry the same per-pixel
state separately.

**Proposed:** Extract a standalone type:

```rust
pub struct VarianceEstimator {
    mean: f64,
    m_2: f64,
    n: u64,
}

impl VarianceEstimator {
    pub fn add(&mut self, x: f64);
    pub fn merge(&mut self, other: &Self);  // pbrt-v4 parallel merge
    pub fn variance(&self) -> f64;
    pub fn mean(&self) -> f64;
    pub fn count(&self) -> u64;
}
```

**How it threads through:** `RgbFilm` would hold `Vec<VarianceEstimator>` (or
`Vec<[VarianceEstimator; 3]>` for per-channel) instead of three separate arrays.
`pixels` and `sample_counts` collapse into `VarianceEstimator`'s internal state.
`pixel_variance(idx)` becomes `self.estimators[idx].variance()`.

**Denoiser integration (§6):** The denoiser needs per-channel variance as input.
With `VarianceEstimator` extracted, `raw_data()` returns
`(&[Color3], &[u32], &[[VarianceEstimator; 3]])` — the denoiser calls
`.variance()` on each estimator to get actual `m2/(n-1)` values. Without this
extraction, `raw_data()` must either pre-compute a temporary `Vec<Color3>` of
variance values (wasteful) or expose raw M2 (leaking implementation details).

**When to do this:**

- **Now** if we want per-channel variance (see §4a), parallel tile merge (§2c),
  or clean denoiser integration (§6)
- **Deferred** until then — the inline code is correct and tested

**Effort:** Medium. Requires updating `RgbFilm` fields, `add_sample`, `merge_tile`,
`pixel_variance`, `convergence_mask`, `reset_convergence_mask`, `reset`, and
constructor. ~80 lines changed, net -30 lines (eliminates parallel array bookkeeping).

### 2b. `ConvergenceCriterionKind` enum — pluggable convergence rules

**Currently:** The convergence test is hardcoded in `reset_convergence_mask`
(rgb.rs:200-213): `var_rms < threshold_abs || var_rms / luminance < threshold_rel`.

**Proposed:** Extract into a wrapper enum following the Niri/Smithay `render_elements!`
pattern: **Descriptor → Concrete → Wrapper**. The wrapper delegates via `match`.

```rust
// === Descriptor enum (lightweight, Clone+Copy) ===
#[derive(Clone, Copy, Debug)]
pub enum ConvergenceCriterionKind {
    Hybrid { abs: f64, rel: f64, min_samples: u32 },
    Absolute { threshold: f64, min_samples: u32 },
    Relative { threshold: f64, min_samples: u32 },
    Neighborhood { radius: u32 },
    Never,
}

// === Concrete types (implement ConvergenceCriterion) ===
pub struct HybridThreshold { pub abs: f64, pub rel: f64, pub min_samples: u32 }
pub struct AbsoluteThreshold(pub f64, pub u32);
pub struct RelativeThreshold(pub f64, pub u32);
pub struct NeighborhoodVariance { pub inner: Box<ConvergenceCriterionEnum>, pub radius: u32 }
pub struct NeverConverge;

impl ConvergenceCriterion for HybridThreshold {
    fn is_converged(&self, variance: f64, luminance: f64, n: u32) -> bool {
        n >= self.min_samples
            && (variance < self.abs || variance / luminance.max(1e-6) < self.rel)
    }
}

// === Wrapper enum (delegates via match) ===
pub enum ConvergenceCriterionEnum {
    Hybrid(HybridThreshold),
    Absolute(AbsoluteThreshold),
    Relative(RelativeThreshold),
    Neighborhood(NeighborhoodVariance),
    Never(NeverConverge),
}

impl ConvergenceCriterion for ConvergenceCriterionEnum {
    fn is_converged(&self, variance: f64, luminance: f64, n: u32) -> bool {
        match self {
            Self::Hybrid(h) => h.is_converged(variance, luminance, n),
            Self::Absolute(a) => a.is_converged(variance, luminance, n),
            Self::Relative(r) => r.is_converged(variance, luminance, n),
            Self::Neighborhood(n) => n.is_converged(variance, luminance, n),
            Self::Never(_) => false,
        }
    }
}

// === From impls for ergonomic construction ===
impl From<HybridThreshold> for ConvergenceCriterionEnum {
    fn from(h: HybridThreshold) -> Self { Self::Hybrid(h) }
}
impl From<AbsoluteThreshold> for ConvergenceCriterionEnum {
    fn from(a: AbsoluteThreshold) -> Self { Self::Absolute(a) }
}
impl From<RelativeThreshold> for ConvergenceCriterionEnum {
    fn from(r: RelativeThreshold) -> Self { Self::Relative(r) }
}
impl From<NeverConverge> for ConvergenceCriterionEnum {
    fn from(n: NeverConverge) -> Self { Self::Never(n) }
}

// === Construction from descriptor ===
impl ConvergenceCriterionEnum {
    pub fn new(kind: &ConvergenceCriterionKind) -> Self {
        match kind {
            ConvergenceCriterionKind::Hybrid { abs, rel, min_samples } =>
                HybridThreshold { abs: *abs, rel: *rel, min_samples: *min_samples }.into(),
            ConvergenceCriterionKind::Absolute { threshold, min_samples } =>
                AbsoluteThreshold(*threshold, *min_samples).into(),
            ConvergenceCriterionKind::Relative { threshold, min_samples } =>
                RelativeThreshold(*threshold, *min_samples).into(),
            ConvergenceCriterionKind::Never => NeverConverge.into(),
            // Neighborhood requires recursion — handle separately
            ConvergenceCriterionKind::Neighborhood { radius } =>
                NeighborhoodVariance { inner: Box::new(NeverConverge.into()), radius: *radius }.into(),
        }
    }
}
```

**Note:** The `ConvergenceCriterionKind` descriptor is separate from `ConvergenceCriterionEnum`.
The descriptor is lightweight config; the enum wraps concrete criterion state.

**Variants:**

| Variant | Behavior |
|---------|----------|
| `Hybrid { abs, rel, min_samples }` | Our current logic |
| `Absolute { threshold, min_samples }` | Pure absolute |
| `Relative { threshold, min_samples }` | Pure relative |
| `Neighborhood { radius }` | LuxCore-style edge-aware check |
| `Never` | Disables adaptive sampling (debug) |

The `convergence_mask` / `reset_convergence_mask` methods on `Film` would take
`&ConvergenceCriterionEnum` instead of individual threshold parameters.

**When to do this:** When we need >1 convergence strategy, or when the LuxCore
neighborhood check (§3a) is implemented.

**Effort:** Small (~60 lines for the enum + variants + `From` impls + descriptor).

### 2c. `SamplingPolicy` trait — separates "what to sample" from "how to render"

**Currently:** The convergence mask + `if converged[idx] { continue }` pattern is
hardcoded in the render loop (cpu.rs:128-167, 187-189).

**Proposed:** Extract the decision into a generic trait (no `dyn Film`):

```rust
pub trait SamplingPolicy<F: Film>: Send + Sync {
    fn on_pass_complete(&mut self, film: &F);
    fn should_sample(&self, pixel_idx: usize, pass: u32) -> bool;
    fn is_finished(&self) -> bool;
}
```

The render loop becomes:

```rust
policy.on_pass_complete(&film);
// ...
for each pixel:
    if policy.should_sample(global_idx, sample_idx):
        trace_ray()
// ...
if policy.is_finished(): break
```

**Implementations:**

| Variant | Behavior |
|---------|----------|
| `UniformPolicy` | Always sample (non-adaptive baseline) |
| `VariancePolicy(criterion)` | Our current adaptive |
| `AdaptiveCadencePolicy { inner, cadence }` | Wraps another policy, only recomputes every N passes |
| `BudgetPolicy(budget_mgr)` | OpenMoonRay-style global allocation |

**When to do this:** When we have >1 sampling strategy, or when we need the
global budget model (§4c).

**Effort:** Medium (~30 lines for the trait + `VariancePolicy` impl + render loop
refactor).

______________________________________________________________________

## 3. Immediate Optimizations (No Abstraction Changes)

### 3a. Neighborhood convergence check

**What:** Add a LuxCore-style edge-awareness test. Before marking a pixel
converged, check if its 3×3 neighborhood has consistent means. If neighbor means
differ significantly, the pixel is on an edge and should keep sampling.

**Where:** Inside `reset_convergence_mask`, after the variance check passes, add:

```
let neighbor_max_diff = max |mean(i) - mean(center)| for i in 3×3 window
if neighbor_max_diff > edge_threshold: converged = false
```

**Cost:** ~9 pixel reads per converged pixel per pass. The film is contiguous in
memory, so this is cache-friendly. Only runs on pixels that already pass the
variance check.

**Complexity:** ~30 lines. No trait changes. Can be added directly to the
existing `reset_convergence_mask`.

**When:** Add now if edge artifacts appear in practice.

### 3b. Adaptive convergence mask cadence

**What:** Don't recompute the full-resolution mask every pass. After the first
~64 samples, variance estimates change slowly. Recomputing every 4-8 passes is
sufficient.

**Where:** In the render loop (cpu.rs), gate the mask recomputation:

```
let mask_cadence = if sample_idx < 128 { 1 } else if sample_idx < 512 { 4 } else { 8 };
if sample_idx % mask_cadence == 0 { reset_convergence_mask(...) }
```

**Cost:** Saves ~(7/8) × width × height iterations per skipped pass at steady state.

**Complexity:** ~5 lines in cpu.rs.

### 3c. Sparse active-pixel list

**What:** Instead of scanning the full mask to find non-converged pixels,
maintain a `Vec<u32>` of active pixel indices. When a pixel converges, remove it
from the list (swap-remove). The parallel tile traversal only iterates active pixels.

**Where:** In the render loop, replace `converged[idx]` check with a set lookup
or maintain a per-tile active list.

**Cost:** Eliminates the full-width convergence mask scan. Active list shrinks
geometrically as pixels converge. Trade-off: O(1) lookup per pixel vs O(1) scan
of the bool array.

**Complexity:** ~25 lines. Requires per-tile active index bookkeeping.

### 3d. `u64` sample counts

**What:** Upgrade `sample_counts: Vec<u32>` to `Vec<u64>` to match
`VarianceEstimator`'s natural type and avoid overflow at 4B samples.

**When:** If we extract `VarianceEstimator` (§2a), this comes for free.
Otherwise low priority — u32 overflow is not a practical concern.

______________________________________________________________________

## 4. Deferred Features (Require Abstractions from §2)

### 4a. Per-channel variance convergence

**What:** Instead of one convergence decision per pixel (max over RGB), track
per-channel variance and converge each channel independently. Enables correct
convergence for AOVs (normals, depth, albedo) that have independent noise profiles.

**Requires:** `VarianceEstimator` extracted (§2a) and threaded through `Pixel`
as a composable member, like pbrt-v4's `VarianceEstimator rgbVariance[3]`.

**Denoiser dependency:** The denoiser's `Denoiser::denoise()` signature takes
`variance: &[Color3]` — per-channel variance. Without `VarianceEstimator` (§2a),
the denoiser must either receive raw M2 (and divide by n-1 itself) or we must
allocate a temporary variance buffer each call. With `VarianceEstimator`, the
denoiser receives `&[[VarianceEstimator; 3]]` and calls `.variance()` directly.
This is the cleanest integration path.

**Deferred until:** We add AOV support or spectral rendering. The per-channel
max approach is sufficient for RGB-only rendering. However, if the denoiser is
implemented first (see `docs/denoiser.md`), this becomes a near-term need.

### 4b. Variance heatmap diagnostic output

**What:** Output variance and relative variance as EXR channels (like pbrt-v4)
or as a tone-mapped preview overlay for debugging convergence.

**Requires:** `pixel_variance` already exists — just expose it through progressive
output or a new diagnostic write path.

**When:** Useful for debugging convergence issues. No urgency.

### 4c. OpenMoonRay global sample budget

**What:** Replace binary converged/unconverged with a continuous importance
weight per pixel. Allocate the next batch of samples to maximize global MSE
reduction per unit compute cost.

**Requires:**

- `SamplingPolicy` trait extracted (§2c)
- Per-pixel importance weight tracking (error gradient computation)
- Budget allocator (discrete optimization over pixel × sample allocation)
- Time-budgeted render mode (render for T seconds, not fixed SPP)

**Deferred until ALL of:**

1. The renderer supports a time-budget parameter (not just fixed SPP)
2. Scenes have wide variance disparity (e.g., glossy interiors with small
   bright windows where per-pixel variance varies by 4+ orders of magnitude)
3. The LuxCore neighborhood check (§3a) is implemented and insufficient
4. We have >1 integrator variant (BDPT, MLT) with different convergence profiles

**Complexity:** ~2000+ lines across film, scheduler, and importance modules.
This is a replacement for the render loop itself, not a bolt-on.

______________________________________________________________________

## 5. Denoiser Integration (Cross-Reference to `docs/denoiser.md`)

The denoiser design doc proposes `Denoiser::denoise()` taking per-channel
variance as input. This section maps the integration points.

### 5.1 `raw_data()` contract

The denoiser doc proposes:

```rust
fn raw_data(&self) -> (&[Color3], &[u32], Vec<[f64; 3]>);
//                         pixels    counts   variance (per-channel, R/G/B)
```

**Current reality:** `RgbFilm` stores `m_2: Vec<Color3>` (raw Welford M2),
not actual variance. The denoiser would need to compute `m2 / (n-1)` itself,
or `raw_data()` must return pre-computed variance.

**Two paths:**

| Approach | Pros | Cons |
|----------|------|------|
| `raw_data()` returns raw M2, denoiser divides by (n-1) | No allocation, no stale data | Leaks Welford detail, denoiser must know n |
| `raw_data()` returns pre-computed variance | Clean interface, denoiser gets what it needs | Requires temporary `Vec<[f64; 3]>` or `VarianceEstimator` extraction |

**Recommended:** Extract `VarianceEstimator` (§2a). Then `raw_data()` returns
`(&[Color3], &[u32], &[[VarianceEstimator; 3]])` — the denoiser calls
`.variance()` on each estimator. No temporary allocation, no leaked internals.

**Cross-reference:** See `docs/denoiser.md` §Architecture Decision for the
denoiser's `raw_data()` signature and §Phase 0 for the implementation dependency.

### 5.2 Per-channel vs max-over-RGB

The denoiser's `variance: &[Color3]` input is **per-channel** — R, G, B
independently. This is correct: the denoiser needs to know that the red channel
is noisy while green is clean (common in foliage scenes with green-heavy
luminance weighting).

Our current `pixel_variance(idx)` returns **max-over-RGB** — a single scalar
per pixel. This is sufficient for convergence decisions (a noisy channel should
prevent convergence) but insufficient for the denoiser.

**Integration path:** With `VarianceEstimator` extracted (§2a), the denoiser
receives per-channel variance naturally. Without it, we'd need a separate
`pixel_variance_rgb(idx) -> Color3` method that returns `(var_r, var_g, var_b)`.

### 5.3 `apply_denoiser()` placement in render loop

The denoiser doc says: "renderer calls `film.apply_denoiser()` after the
sampling loop". In the adaptive sampling render loop (cpu.rs), this means:

```
for each pass:
    // ... adaptive sampling logic ...

// After all passes complete (or early exit):
film.apply_denoiser();     // ← HERE
```

**Important:** The denoiser runs **after** the final frame is published for
live preview. For the progressive path (`live_render`), the denoiser should
run on the final framebuffer snapshot, not on intermediate progressive frames.
For the headless path (`headless_render`), it runs before `write_image()`.

**Do NOT denoise intermediate progressive frames** — the denoiser expects
the final sample counts, not partial ones. Denoising a 64-spp preview would
produce different results than denoising the final 2048-spp frame.

### 5.4 `overlay_radiance()` — denoiser output replacement

The denoiser doc proposes `overlay_radiance(radiance: Vec<Color3>)` to replace
pixel data with filtered results. This is a simple `memcpy` of the denoised
buffer into `self.pixels`. However:

- It must NOT reset `sample_counts` or `m_2` — the film's statistical state
  should remain intact for potential re-denoising or diagnostic output.
- It should reset `pixels` to the denoised values AND set `denoised = true`.
- `to_rgb8()` must check `denoised`: if true, use `pixels` directly (already
  per-pixel mean); if false, divide by `sample_counts` (current behavior).

**Cross-reference:** See `docs/denoiser.md` §5 for the corrected
`overlay_radiance()` implementation with the `denoised: bool` flag.

### 5.5 Convergence mask interaction

**Question:** Should the convergence mask be computed before or after denoising?

**Answer:** Before. The convergence mask drives adaptive sampling — it decides
which pixels need more samples. Denoising is a post-process that runs after
all sampling is complete. The convergence mask is never computed after the
final frame.

**Cross-reference:** See `docs/denoiser.md` §6 for the `apply_denoiser()`
placement in the render loop. The denoiser runs after the sampling loop
completes, before the final frame is published to the framebuffer.

**Edge case:** If we implement iterative denoising (denoise → sample more →
denoise again), the convergence mask would need to operate on denoised data.
This is a future consideration, not a current requirement.

### 5.6 Shared variance data

Both adaptive sampling and the denoiser need the same variance data:

| Consumer | Needs | Currently |
|----------|-------|-----------|
| Adaptive sampling | `pixel_variance(idx) -> f64` (max-over-RGB) | ✅ Exists |
| Denoiser | Per-channel variance `[f64; 3]` per pixel | ❌ Missing |
| Both | Sample counts per pixel | ✅ Exists (`sample_counts: Vec<u32>`) |

The `VarianceEstimator` extraction (§2a) serves both consumers. Without it,
we'd need two separate variance access paths. With it, one extraction serves
both.

**DenoiserFeatures integration:** The denoiser's `DenoiserFeatures` struct
(denoiser.md §Phase 2) stores per-pixel variance as `Vec<[f64; 3]>` (SoA).
This is populated from `VarianceEstimator::variance()` during `add_sample()`.
The SoA layout matches how `VarianceEstimator` is stored: cache-friendly
sequential access for both the convergence check and the denoiser pass.

______________________________________________________________________

## 6. Dependency Graph

```
§3a  Neighborhood check ──────────────────────────────── (do now)
§3b  Mask cadence ─────────────────────────────────────── (do now)
§3c  Sparse active list ──────────────────────────────── (do now)

§2a  VarianceEstimator ──┬── §3d  u64 counts (comes free)
                         ├── §4a  Per-channel convergence
                         ├── §5.1  Denoiser raw_data() (clean interface)
                         ├── §5.2  Per-channel variance for denoiser
                         └── §2c  SamplingPolicy ─── §4c  Global budget
                                (needs VarianceEstimator)
§2b  ConvergenceCriterion ── §3a  Neighborhood check (as a variant)
                              §4c  Global budget

§5.3  apply_denoiser() placement ─── render loop (after sampling)
§5.4  overlay_radiance() + denoised flag ─── RgbFilm fields
```

**Denoiser critical path:** §2a (VarianceEstimator) → §5.1 (raw_data) → §5.3
(apply_denoiser placement). The denoiser cannot be cleanly integrated without
`VarianceEstimator` unless we accept a temporary variance allocation or leak
M2 internals.

**Cross-reference:** See `docs/denoiser.md` §Phase 0 for the denoiser's
implementation dependency on §2a, and §File Summary for the integration
points between the two documents.

**Parallel workstreams:**

| Stream | Items | Dependencies |
|--------|-------|-------------|
| Adaptive sampling (do now) | §3a, §3b, §3c | None — independent |
| Denoiser foundation | §2a → §5.1, §5.2, §5.4 | §2a first, then parallel |
| Denoiser integration | §5.3, §5.5, §5.6 | Needs §2a + denoiser module |
| Abstraction (when needed) | §2b, §2c | §2a optional, §3a optional |

The two highest-value items without abstraction work: **§3a** (neighborhood
convergence, fixes edge artifacts, ~30 lines) and **§3c** (sparse active-pixel
list, eliminates full-mask scan, ~25 lines). Both are independent and can be
implemented in parallel.

The `VarianceEstimator` extraction (§2a) is the highest-value abstraction
refactor — it collapses three parallel arrays, enables parallel merge, serves
the denoiser's per-channel variance need, and opens the path to the budget
model. **Do this before the denoiser** if both are planned.
