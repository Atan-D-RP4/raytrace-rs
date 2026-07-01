# Denoiser Implementation Plan

## Changelog

- **v1 (2026-06-28)** — Initial plan. Architecture decision, algorithms, GPU solutions, implementation phases.
- **v2 (2026-06-28)** — Bi-directional audit with `docs/adaptive-sampling.md`. Resolved
  6 conflicts: raw_data() variance format (now `[f64; 3]` per-channel), overlay_radiance()
  preserves counts/m2 (added `denoised: bool`), VarianceEstimator prerequisite (Phase 0),
  per-channel variance contract, double-denoising guard, progressive preview note.
  Added cross-document integration table in §File Summary.
- **v3 (2026-06-28)** — Three-document audit with `docs/ARCH_HYBRID.md` and
  `docs/adaptive-sampling.md`. Renamed "G-buffer fields" to "DenoiserFeatures" to
  avoid collision with ARCH_HYBRID's `GBuffer<'a>` (visibility buffer). Added
  ARCH_HYBRID cross-reference table. Clarified that DenoiserFeatures (albedo/normal/depth
  for edge-stopping) are different from GBuffer (SurfaceInteraction for visibility).
- **v4 (2026-06-28)** — Static dispatch audit. Reviewed all `dyn` usage — denoiser doc
  already uses generics (`RgbFilm<D: Denoiser>`). No changes needed.
- **v5 (2026-06-29)** — Designed `DenoiserFeatures` as struct-of-arrays (SoA) following
  `VarianceEstimator` pattern. Added `set()`, `set_variance()`, `get()`, `reset()` methods.
  Added `Option<DenoiserFeatures>` on `RgbFilm` for zero-cost when not using A-Trous.
  Documented SoA vs AoS rationale for cache-friendly denoiser passes.

## Architecture Decision

**Follow the `SamplerFactory` pattern — denoiser is a generic on `Film`, not on `Renderer`.**

The film already owns the raw data (pixels, variance, sample counts). It's the natural place for post-processing. The renderer calls `film.apply_denoiser()` after the sampling loop — it doesn't need to know the denoiser type. Zero-cost monomorphization: `RgbFilm<NoDenoiser>` compiles to the same code as today's `RgbFilm`.

**Cross-document dependency:** This design depends on `VarianceEstimator` extraction (§2a of `docs/adaptive-sampling.md`). The denoiser needs per-channel variance (`Color3` — R, G, B independently), not the current max-over-RGB scalar from `pixel_variance(idx)`. `VarianceEstimator` provides this cleanly: `raw_data()` returns `&[VarianceEstimator]` and the denoiser calls `.variance()` on each channel. **Implement §2a before Phase 1 of this plan.**

```rust
// Existing pattern (sampler.rs:247):
pub trait SamplerFactory: Send + Sync {
    type Sampler: Sampler;
    fn for_pixel(&self, x: i32, y: i32) -> Self::Sampler;
}

// New denoiser follows the same pattern:
pub trait Denoiser: Send + Sync {
    fn denoise(
        &self,
        pixels: &[Color3],              // accumulated radiance sum per pixel
        sample_counts: &[u32],          // samples per pixel
        variance: &[[f64; 3]],          // per-channel variance (R, G, B independently)
        width: u32,
        height: u32,
    ) -> Vec<Color3>;                   // filtered radiance (per-pixel mean, not sum)
}
```

Note: `variance` is `&[[f64; 3]]` (one `[f64; 3]` per pixel, containing `(var_r, var_g, var_b)`), not `&[Color3]`. This avoids coupling the denoiser to the `Color3` type and makes the per-channel contract explicit. The denoiser calls `VarianceEstimator::variance()` on each channel to populate this buffer.

`RgbFilm<D: Denoiser = NoDenoiser>` becomes generic. The renderer calls `film.apply_denoiser()` after sampling — it doesn't need to know `D`. Backwards-compatible: `RgbFilm::new()` defaults to `NoDenoiser`.

______________________________________________________________________

## What Is Denoising?

### Intuition

Your path tracer is a painter in a dark room. Each sample is a quick glance — one tiny spot, one light path. After thousands of glances per pixel, you know the color roughly. With too few, the image is speckled — that's **noise**. Denoising says: "If this pixel's neighbors have more information and look similar, maybe this pixel should look more like them." It exploits what we know about images (edges are sparse, smooth gradients are common) to fill gaps left by undersampling.

### Rigor

Monte Carlo path tracing estimates the rendering equation via:

$$
L_o(x, \omega_o) = \int_\Omega f_r(x, \omega_i, \omega_o) \, L_i(x, \omega_i) \, |\cos \theta| \, d\omega_i
$$

The Monte Carlo estimator:

$$
\langle L_o \rangle = \frac{1}{N} \sum_{i=1}^{N} \frac{f_r \cdot L_i \cdot |\cos \theta|}{p(\omega_i)}
$$

is **unbiased** — its expectation is the true integral. But with finite samples, each pixel is a random variable with variance:

$$
\sigma^2(p) = E[\langle L_o \rangle^2] - E[\langle L_o \rangle]^2
$$

This variance manifests as noise. Its structure depends on the integrand:

- **Fireflies**: rare paths with tiny p(w) and large f_r — extreme outliers from heavy-tailed distributions.
- **Low-frequency noise**: indirect illumination in shadowed regions — broad spatial correlation.
- **Caustics/specular chains**: hard-to-sample light paths with high variance in small regions.

**Denoising is a bias-variance tradeoff.** It introduces systematic error (bias) in exchange for dramatically reduced variance. A well-tuned denoiser at 256 spp often looks better than raw 4096 spp for a fraction of the cost.

______________________________________________________________________

## Production Renderer Approaches

### Comparison Table

| Renderer | Denoiser(s) | Algorithm | Hand-rolled? | Feature Buffers | Integration |
|---|---|---|---|---|---|
| **LuxCore** | BCD + OIDN | Bayesian collaborative + U-Net CNN | BCD = yes, OIDN = no | BCD: histogram + covariance; OIDN: albedo + normal | Image pipeline plugin (post-process) |
| **OpenMoonRay** | OptiX + OIDN | CNN (both) | **No** — both are wrappers | beauty + albedo + normal | Post-process (CLI / GUI) |
| **PBRT v4** | OptiX only | CNN (NVIDIA) | **No** | beauty + albedo + normal (+ position, variance via GBufferFilm) | Post-process (imgtool CLI) |
| **Unity HDRP** | OIDN (replaced OptiX in 6.5) | CNN | **No** | color + albedo + normal | Post-process |
| **Blender Cycles** | OIDN | CNN | **No** | color + albedo + normal | Post-process |

### LuxCore (most relevant for hand-rolled approach)

LuxCore is the only production renderer with a hand-rolled denoiser: **Bayesian Collaborative Denoiser (BCD)**. It's a non-local Bayesian filter that operates on per-pixel **sample statistics** (histogram of color samples + covariance matrix) rather than the standard albedo/normal AOVs. Based on the EGSR 2017 paper by Boughida & Boubekeur. It includes an anti-fireflies filter. Since v2.4, LuxCore also integrates Intel OIDN as an alternative. Both are post-process plugins in the image pipeline (`film.imagepipelines`).

### OpenMoonRay (DreamWorks)

No hand-rolled denoiser. Wraps NVIDIA OptiX and Intel OIDN via the `mcrt_denoise` module. Clean pimpl abstraction (`Denoiser` -> `DenoiserImpl`). Runtime-selectable (`-mode optix` or `-mode oidn`). Uses beauty + albedo + normal AOVs tagged via `RenderOutput` scene objects. Since OIDN 2.0, supports GPU mode (CUDA). No temporal denoising yet — spatial only per-frame.

### PBRT v4

No built-in denoiser. Uses NVIDIA OptiX via `imgtool denoise-optix` CLI command. The `GBufferFilm` was explicitly designed to support denoising: outputs albedo, shading normal, geometric normal, position, variance, relative variance, depth derivatives, and UV coordinates. The default "zsobol" sampler produces blue noise error distribution, which is more friendly to denoising algorithms.

### Key Insight

Only LuxCore has a hand-rolled denoiser (BCD). Everyone else wraps Intel OIDN or NVIDIA OptiX, both CNN-based. For this codebase: **hand-rolled A-Trous wavelet first** (no external deps, full control), with optional OIDN integration later.

______________________________________________________________________

## Denoiser Algorithms

### 1. Cross-Bilateral Filter (Phase 1 — minimal viable)

Weighted average of neighbors. Weights depend on spatial distance and color similarity. Edge-preserving Gaussian blur.

$$
\hat{c}(p) = \frac{1}{W_p} \sum_q c(q) \cdot g_s(\|p - q\|) \cdot g_r(\|c(p) - c(q)\|)

W_p = \sum_q g_s(\|p - q\|) \cdot g_r(\|c(p) - c(q)\|)

g_s(x) = \exp\!\left(\frac{-x^2}{2 \sigma_s^2}\right) \quad \text{-- spatial kernel}

g_r(x) = \exp\!\left(\frac{-x^2}{2 \sigma_r^2}\right) \quad \text{-- range kernel}
$$

**Variance-adaptive**: modulate sigma_r per-pixel using Welford variance from the film:

$\sigma_r(p) = \sigma_{r,\text{base}} \cdot \max\!\left(1, \frac{\sqrt{\text{Var}(p)}}{\text{luminance}(p)}\right)$

Noisy pixels get wider range kernels -> more aggressive filtering. Converged pixels get tighter kernels -> detail preserved.

| Pro | Con |
|---|---|
| ~60 lines, trivially simple | O(radius^2) per pixel — quadratic in radius |
| Uses existing Welford variance directly | Misses edges in normal/depth space |
| No feature buffers needed | Fireflies pollute neighbors |
| Deterministic, no training | Fixed-radius misses long-range correlation |

**Sweet spot**: Minimal viable denoiser. Will immediately clean up diffuse indirect noise.

### 2. A-Trous Wavelet Filter (Phase 2 — production quality)

Multi-scale edge-aware filter. Cascaded 5x5 kernel with dilation doubling each level. Edge-stopping functions from feature buffers prevent blurring across discontinuities.

**Algorithm:**

Step 1 — Build feature buffers during rendering:

- a(p): first-hit **albedo** (base color, no lighting)
- n(p): first-hit world-space **normal**
- z(p): first-hit linear **depth**
- sigma^2(p): per-pixel variance from Welford (already tracked)

Step 2 — Separate illumination: `i(p) = c(p) / a(p)`. Filter illumination, preserve albedo.

Step 3 — Cascaded A-Trous at level l with dilation 2^l:

$$
i^{(l+1)}(p) = \frac{\sum_k w_k \cdot w_e(p,\, p + 2^l k) \cdot i^{(l)}(p + 2^l k)}{\sum_k w_k \cdot w_e(p,\, p + 2^l k)}
$$

where:

- K is a 5x5 kernel with binomial weights [1, 4, 6, 4, 1] / 16 per row (separable)
- w_e(p, q) is the **edge-stopping function** — product of per-feature Gaussians:

$$
w_{\text{color}}(p, q)    = \exp\!\left(\frac{-\|i(p) - i(q)\|^2}{2 \sigma_c^2}\right)

w_{\text{normal}}(p, q)   = \exp\!\left(\frac{-\|n(p) - n(q)\|^2}{2 \sigma_n^2}\right)

w_{\text{depth}}(p, q)    = \exp\!\left(\frac{-|z(p) - z(q)|}{\sigma_z (\nabla z(p) + \epsilon)}\right)

w_{\text{variance}}(p, q) = \exp\!\left(\frac{-\|\text{var}(p) - \text{var}(q)\|^2}{2 \sigma_v^2}\right)

w_e(p, q) = w_{\text{color}} \cdot w_{\text{normal}} \cdot w_{\text{depth}} \cdot w_{\text{variance}}
$$

The depth weight is normalized by local depth gradient so distant objects with large absolute depth differences aren't erroneously separated.

Step 4 — Recombine: `c_out(p) = a(p) * i^(L)(p)`

**Why it's efficient**: A 5x5 kernel at level 4 (dilation 16) covers 80x80 pixels with only 25 fetches per level x 5 levels = 125 total. A bilateral for the same reach needs 6400 fetches per pixel.

| Pro | Con |
|---|---|
| $O(n)$ — linear in pixels | Needs feature buffers (albedo, normal, depth) |
| Separable -> SIMD-friendly | Coarse-level ringing near sharp edges |
| Industry standard (PBRT, SVGF, LuxCore) | Feature buffers must be filtered or they corrupt edges |
| Multi-scale handles all frequency bands | Parameter tuning per scene |
| No ML, no external deps | Not temporally stable without frame history |

### 3. NL-Means (Phase 3 — optional, for texture-heavy scenes)

Compares **patches** instead of single pixels. Non-local: can find structure across the entire image.

$$
\hat{c}(p) = \frac{1}{W_p} \sum_{q \in \Omega} c(q) \cdot \exp\!\left(\frac{-\|P_p - P_q\|^2}{2\sigma_{\text{pat}}^2 + \text{var}(p) + \text{var}(q)}\right)

\|P_p - P_q\|^2 = \sum_{k \in \text{patch}} g_a(k) \cdot (c(p+k) - c(q+k))^2
$$

- Omega: search window, typically 21x21
- Patch size: 5x5 or 7x7
- g_a(k): Gaussian analysis window, center-weighted
- Variance terms in denominator: when Var(p) is large, the denominator grows, more patches appear "similar", more neighbors contribute -> aggressive filtering where needed.

**Required optimizations**: integral images for patch SSD (O(1) per patch distance), block-based acceleration, PCA pre-filtering.

| Pro | Con |
|---|---|
| Best texture preservation — recovers periodic structure | O(n * S * P) — very expensive |
| No feature buffers needed | Noisy patches don't match clean ones |
| Isolated fireflies get zero weight | Blocky artifacts from insufficient search |

______________________________________________________________________

## GPU Denoising Solutions

Your path tracer is CPU-only today, but the `serialize_gpu()` methods on materials and textures signal future GPU ambitions. Here are the GPU denoising options, ranked by practicality:

### Option A: Intel Open Image Denoise (OIDN) — GPU mode

**Best fit for this codebase.** OIDN 2.x supports GPU backends: CUDA (NVIDIA), HIP (AMD), SYCL (Intel), Metal (Apple). Same API as CPU — just create a GPU device instead of CPU.

**Rust crate**: `oidn` v2.4.1 on crates.io (maintained by Twinklebear). Has `bundled` feature that downloads OIDN binaries automatically. Crate version tracks OIDN version.

```rust
// Cargo.toml
oidn = { version = "2.4", features = ["bundled"] }
```

```rust
use oidn;

// Create GPU device (auto-detects best available)
let device = oidn::Device::new();

// Or explicitly request CUDA device:
// let device = oidn::Device::new_cuda(0);

let mut filter = oidn::RayTracing::new(&device)
    .srgb(false)  // linear HDR input
    .image_dimensions(width as usize, height as usize);

// Set auxiliary buffers for better quality
filter.set_albedo_image(&mut albedo_buf);
filter.set_normal_image(&mut normal_buf);

filter.filter(&input_img, &mut output).unwrap();
```

**Key facts:**

- Apache 2.0 license
- CPU: SSE4.1 through AVX-512, AMX. GPU: NVIDIA (CUDA), AMD (HIP), Intel (SYCL/XMX), Apple (Metal)
- OIDN 3 (shipping H2 2026) adds temporal denoising using motion vectors
- Unity 6.5 deprecated OptiX in favor of OIDN, citing better cross-vendor support and performance

### Option B: NVIDIA OptiX Denoiser

CNN-based, ships with NVIDIA drivers. Best quality at 16+ spp with NVIDIA GPUs. Used by PBRT, Arnold, V-Ray, Blender Cycles.

- Requires NVIDIA GPU (Maxwell or later)
- Uses tensor cores for acceleration
- Supports multi-AOV denoising (beauty + albedo + normal)
- Temporally coherent mode available
- Improvements come via driver updates, not SDK rebuilds
- **Limitation**: NVIDIA-only. Unity deprecated it in favor of OIDN for cross-vendor support.

**Rust bindings**: No maintained Rust crate. Would need FFI via `optix-sys` or raw bindings. More friction than OIDN.

### Option C: NVIDIA NRD (Real-Time Denoisers)

**Different use case**: NRD is designed for **real-time** denoising in games (1 spp per frame, temporal accumulation). Three specialized denoisers:

- **REBLUR**: Recurrent blur. Fast (2.3ms at 1440p on RTX 4080). Good default.
- **RELAX**: A-Trous wavelet based. Higher quality (3.0ms). Designed for RTXDI.
- **SIGMA**: Per-light shadow-only denoiser (0.4ms).

**Inputs**: diffuse + specular radiance (separated), normals, roughness, viewZ, motion vectors, hit distance. More complex integration than OIDN.

**When to use**: If you move to real-time GPU path tracing with temporal accumulation. Overkill for offline batch rendering.

### Option D: AMD FidelityFX Denoiser

MIT license, open source. Specialized denoisers for specific workloads:

- **Shadow Denoiser**: Spatio-temporal, for raytraced soft shadows
- **Reflection Denoiser**: For stochastic reflections (SSSR, raytraced reflections)

**Limitation**: Not a general-purpose beauty denoiser. Designed for shadow masks and reflection signals, not full path-traced images. Use alongside OIDN/OptiX, not instead of them.

### Option E: Hand-Rolled Compute Shaders

Implement bilateral, A-Trous, or NL-Means as compute shaders in whatever graphics API you choose (Vulkan, wgpu, Metal, D3D12). The algorithms are the same — just parallelized across GPU threads.

**When to use**: Full control, no external deps, consistent behavior across vendors. The A-Trous wavelet is particularly GPU-friendly (separable, constant work per pixel per level).

**Trade-off**: You're reimplementing what OIDN already does better (trained CNNs beat hand-tuned filters for most scenes). But for a learning project or when you need deterministic vendor-neutral behavior, hand-rolled is valid.

### GPU Recommendation

| Scenario | Use |
|---|---|
| CPU path tracer, want best quality now | OIDN CPU (via `oidn` crate with `bundled` feature) |
| GPU path tracer, any vendor | OIDN GPU (CUDA/HIP/SYCL/Metal via same `oidn` crate) |
| GPU path tracer, NVIDIA only, need maximum quality | OptiX denoiser |
| Real-time GPU path tracing in games | NRD (REBLUR or RELAX) |
| Just shadows/reflections, not full beauty | AMD FidelityFX Denoiser |
| Learning project, full control | Hand-rolled A-Trous compute shader |

______________________________________________________________________

## Implementation Plan

### Phase 0: VarianceEstimator Extraction (prerequisite — from adaptive-sampling.md §2a)

**Goal:** Extract a composable Welford engine that provides per-channel variance. This is a prerequisite for clean denoiser integration — the denoiser needs per-channel variance (`[f64; 3]` per pixel), not the current max-over-RGB scalar.

**What changes:** `RgbFilm` holds `Vec<VarianceEstimator>` (or `Vec<[[f64; 3]; 3]>` for per-channel) instead of three separate arrays (`pixels`, `sample_counts`, `m_2`). `pixel_variance(idx)` becomes `self.estimators[idx].variance()`. `raw_data()` returns estimators directly — no temporary allocation.

**Effort:** ~80 lines changed, net -30 lines (eliminates parallel array bookkeeping).

**Cross-reference:** See `docs/adaptive-sampling.md` §2a for full design and §5.1-§5.6 for denoiser integration points.

### Phase 1: Core Infrastructure + Bilateral Denoiser

**Goal**: Working denoiser with minimal code changes. Bilateral filter using existing Welford variance.

**Files to change:**

#### 1. `src/denoiser/mod.rs` (NEW — ~40 lines)

Denoiser trait, NoDenoiser, re-exports.

#### 2. `src/denoiser/bilateral.rs` (NEW — ~80 lines)

BilateralDenoiser with variance-adaptive range sigma.

```rust
pub struct BilateralDenoiser {
    spatial_sigma: f64,
    range_sigma_base: f64,
    variance_scale: f64,
    radius: u32,
}
```

#### 3. `src/lib.rs` — add `pub mod denoiser;`

#### 4. `src/film/mod.rs` — add to Film trait

```rust
fn apply_denoiser(&mut self);
```

#### 5. `src/film/rgb.rs` — make RgbFilm generic

```rust
pub struct RgbFilm<D: Denoiser = NoDenoiser> {
    width: u32,
    height: u32,
    pixels: Vec<Color3>,           // accumulated radiance sum (or denoised mean if denoised=true)
    sample_counts: Vec<u32>,       // sample count per pixel (preserved after denoising)
    m_2: Vec<Color3>,             // Welford M2 (preserved after denoising)
    exposure: f64,
    tone_map: bool,
    denoiser: D,
    denoised: bool,                // prevents double-denoising
}

impl RgbFilm<NoDenoiser> {
    pub fn new(dimensions: (u32, u32), exposure: f64, tone_map: bool) -> Self {
        Self::with_denoiser(dimensions, exposure, tone_map, NoDenoiser)
    }
}

impl<D: Denoiser> RgbFilm<D> {
    pub fn with_denoiser(dimensions: (u32, u32), exposure: f64, tone_map: bool, denoiser: D) -> Self {
        let (width, height) = dimensions;
        Self {
            width, height,
            pixels: vec![Color3::ZERO; (width * height) as usize],
            sample_counts: vec![0; (width * height) as usize],
            m_2: vec![Color3::ZERO; (width * height) as usize],
            exposure, tone_map, denoiser,
            denoised: false,
        }
    }

    /// Replace pixel radiance with denoised values.
    /// Does NOT reset sample_counts or m_2 — statistical state is preserved
    /// for potential re-denoising, diagnostic output, or variance heatmaps.
    fn overlay_radiance(&mut self, radiance: Vec<Color3>) {
        self.pixels = radiance;
        self.denoised = true;
    }

    /// Extract raw data for denoiser consumption.
    /// Returns (pixels, sample_counts, variance_per_channel).
    /// Variance is computed from m_2 / (n-1) — requires VarianceEstimator (§2a).
    fn raw_data(&self) -> (&[Color3], &[u32], Vec<[f64; 3]>) {
        let variance: Vec<[f64; 3]> = self.pixels.iter()
            .zip(self.sample_counts.iter())
            .zip(self.m_2.iter())
            .map(|((_, &n), m2)| {
                if n > 1 {
                    let nf = n as f64;
                    [m2.x / (nf - 1.0), m2.y / (nf - 1.0), m2.z / (nf - 1.0)]
                } else {
                    [f64::INFINITY; 3]
                }
            })
            .collect();
        (&self.pixels, &self.sample_counts, variance)
    }
}

impl<D: Denoiser> Film for RgbFilm<D> {
    fn apply_denoiser(&mut self) {
        if self.denoised {
            return;  // already denoised — no-op
        }
        let (pixels, counts, variance) = self.raw_data();
        let filtered = self.denoiser.denoise(pixels, counts, &variance, self.width, self.height);
        self.overlay_radiance(filtered);
    }

    fn read_image(&self) -> Vec<u8> {
        // After denoising, pixels hold per-pixel mean (not sum).
        // to_rgb8() must handle both states.
        self.to_rgb8()
    }

    // ... existing methods unchanged ...
}
```

**Key design decisions:**

- `overlay_radiance()` replaces `pixels` but does NOT reset `sample_counts` or `m_2`. Statistical state is preserved for diagnostic output, variance heatmaps, or potential re-denoising.
- `denoised: bool` prevents double-denoising. `apply_denoiser()` is a no-op if already denoised.
- `to_rgb8()` must check `denoised`: if true, use `pixels` directly (already per-pixel mean); if false, divide by `sample_counts` (current behavior).
- `raw_data()` returns a temporary `Vec<[f64; 3]>` for variance. With `VarianceEstimator` extracted (§2a of adaptive-sampling.md), this becomes `&[[VarianceEstimator; 3]]` — no allocation.

#### 6. `src/renderer/cpu.rs` — call `film.apply_denoiser()` after sampling loop

```rust
// After the main sampling loop completes, BEFORE publishing the final frame:
film.apply_denoiser();

// Then publish final frame to framebuffer (if live preview):
if let Some(ref framebuffer) = framebuffer {
    let rgb = film.progressive();
    // ... publish to framebuffer ...
}
```

**Important:** The denoiser runs **after** the final frame is published for live preview. For the progressive path (`live_render`), the denoiser should run on the final framebuffer snapshot, not on intermediate progressive frames. The denoiser expects complete sample counts — denoising a 64-spp preview would produce different results than denoising the final 2048-spp frame. Progressive frames are NOT denoised.

#### 7. `src/main.rs` — wire up (optional, backwards-compatible)

```rust
// With denoiser:
let film = RgbFilm::with_denoiser(
    camera.image_resolution(), config.exposure, config.tone_map,
    BilateralDenoiser::new(4, 3.0, 0.15, 2.0),
);

// Without denoiser (default, same as today):
let film = RgbFilm::new(camera.image_resolution(), config.exposure, config.tone_map);
```

**Total**: ~170 new lines, ~50 lines changed. No breaking changes.

### Phase 2: A-Trous Wavelet + DenoiserFeatures

**Goal**: Production-quality denoiser with edge-stopping from albedo/normal/depth.

Additional changes:

- `DenoiserFeatures` struct (SoA) on `RgbFilm` — stores per-pixel albedo, normal, depth as separate contiguous arrays for cache-friendly access (see §DenoiserFeatures below)
- Integrator outputs features at first hit via `set_features(pixel_idx, albedo, normal, depth)`
- `src/denoiser/atrous.rs` (NEW — ~250 lines): A-Trous wavelet with 5 levels, separable 5x5 binomial kernel, 4-channel edge-stopping

### DenoiserFeatures — Struct-of-Arrays for Edge-Stopping

**NOT the same as `GBuffer<'a>` from `docs/ARCH_HYBRID.md`** (which stores `SurfaceInteraction` for visibility). `DenoiserFeatures` stores per-pixel shading features consumed by the A-Trous wavelet denoiser.

Follows the `VarianceEstimator` pattern: a struct that encapsulates per-pixel state, stored as `Vec<DenoiserFeatures>` (or inline SoA fields) on `RgbFilm`. The SoA layout ensures cache-friendly sequential access during the denoiser's horizontal/vertical passes.

```rust
/// Per-pixel features for edge-stopping in A-Trous wavelet denoiser.
/// Stored as struct-of-arrays on RgbFilm for cache-friendly access.
///
/// The integrator writes features at first hit. The denoiser reads them
/// during edge-stopping function evaluation.
///
/// Naming: "DenoiserFeatures" — NOT "GBuffer" (which is visibility).
#[derive(Debug, Clone)]
pub struct DenoiserFeatures {
    /// First-hit albedo (base color, no lighting). Used for color edge-stopping.
    pub albedo: Vec<Color3>,
    /// First-hit world-space normal. Used for normal edge-stopping.
    pub normal: Vec<Vec3>,
    /// First-hit linear depth. Used for depth edge-stopping with gradient normalization.
    pub depth: Vec<f64>,
    /// Per-pixel variance from Welford (R, G, B channels). Used for variance-aware filtering.
    /// Populated from VarianceEstimator extraction (adaptive-sampling.md §2a).
    pub variance: Vec<[f64; 3]>,
    /// Width × height for bounds checking.
    width: u32,
    height: u32,
}

impl DenoiserFeatures {
    pub fn new(width: u32, height: u32) -> Self {
        let n = (width * height) as usize;
        Self {
            albedo: vec![Color3::ZERO; n],
            normal: vec![Vec3::ZERO; n],
            depth: vec![f64::INFINITY; n],
            variance: vec![[f64::INFINITY; 3]; n],
            width,
            height,
        }
    }

    /// Write features at a pixel (called by integrator at first hit).
    #[inline(always)]
    pub fn set(&mut self, x: u32, y: u32, albedo: Color3, normal: Vec3, depth: f64) {
        let idx = (y * self.width + x) as usize;
        self.albedo[idx] = albedo;
        self.normal[idx] = normal;
        self.depth[idx] = depth;
    }

    /// Update variance for a pixel (called by Film after each sample).
    #[inline(always)]
    pub fn set_variance(&mut self, x: u32, y: u32, variance: [f64; 3]) {
        let idx = (y * self.width + x) as usize;
        self.variance[idx] = variance;
    }

    /// Read all features for a pixel (called by denoiser).
    #[inline(always)]
    pub fn get(&self, x: u32, y: u32) -> (Color3, Vec3, f64, [f64; 3]) {
        let idx = (y * self.width + x) as usize;
        (self.albedo[idx], self.normal[idx], self.depth[idx], self.variance[idx])
    }

    /// Reset all features (for re-rendering).
    pub fn reset(&mut self) {
        self.albedo.fill(Color3::ZERO);
        self.normal.fill(Vec3::ZERO);
        self.depth.fill(f64::INFINITY);
        self.variance.fill([f64::INFINITY; 3]);
    }
}
```

**Why SoA (not AoS)?**

The A-Trous wavelet filter processes each feature channel independently in its horizontal/vertical passes. With SoA, the filter reads `albedo[idx]` sequentially across pixels — a single contiguous stride. With AoS (e.g., `Vec<PixelFeatures>` where each `PixelFeatures` contains all features), the filter would stride across unrelated data, causing cache misses.

This matches how `VarianceEstimator` is stored: `Vec<[VarianceEstimator; 3]>` (per-channel) rather than a single `VarianceEstimator` per pixel. Both patterns optimize for the access pattern of the consumer.

**Integration with RgbFilm:**

```rust
pub struct RgbFilm<D: Denoiser = NoDenoiser> {
    width: u32,
    height: u32,
    pixels: Vec<Color3>,           // accumulated radiance sum
    sample_counts: Vec<u32>,       // sample count per pixel
    m_2: Vec<Color3>,             // Welford M2
    exposure: f64,
    tone_map: bool,
    denoiser: D,
    denoised: bool,
    features: Option<DenoiserFeatures>,  // None when NoDenoiser (zero-cost)
}

impl<D: Denoiser> RgbFilm<D> {
    /// Enable feature collection (called when A-Trous denoiser is configured).
    pub fn enable_features(&mut self) {
        self.features = Some(DenoiserFeatures::new(self.width, self.height));
    }

    /// Write features at first hit (called by integrator).
    #[inline(always)]
    pub fn set_features(&mut self, x: u32, y: u32, albedo: Color3, normal: Vec3, depth: f64) {
        if let Some(ref mut f) = self.features {
            f.set(x, y, albedo, normal, depth);
        }
    }
}
```

**Note:** `features` is `Option<DenoiserFeatures>` — `None` when using `NoDenoiser` or `BilateralDenoiser` (Phase 1, doesn't need features). Only allocated when A-Trous (Phase 2) or OIDN (Phase 3) is configured. This preserves the zero-cost property: `RgbFilm<NoDenoiser>` has no feature overhead.

### Phase 3: OIDN Integration (optional)

**Goal**: Wrap Intel Open Image Denoise for ML-based denoising.

- Add `oidn = { version = "2.4", features = ["bundled"] }` to Cargo.toml
- `src/denoiser/oidn.rs` (NEW — ~100 lines): OidnDenoiser wrapping the `oidn` crate
- Works on CPU and GPU (CUDA/HIP/SYCL/Metal) with same API

### Phase 4: GPU Compute Shader Denoiser (future)

**Goal**: Hand-rolled A-Trous or bilateral as a compute shader.

- Implement in wgpu/Vulkan/Metal compute pipeline
- Same algorithm as Phase 2 A-Trous, but parallelized across GPU threads
- Each workgroup processes one tile (e.g., 16x16 pixels)
- Shared memory for tile data, global reads for edge-stopping features
- Useful when moving to GPU path tracing

______________________________________________________________________

## Testing Strategy

1. **Unit tests**: Construct synthetic noisy image (known variance), denoise, verify error reduction
2. **Regression test**: Render cornell_box at 64 spp with/without denoiser, verify denoised output has lower PSNR against 4096 spp reference
3. **Visual inspection**: Render each preset scene at low spp with bilateral and A-Trous
4. **Performance benchmark**: Time render + denoise vs. render-only at equivalent quality

______________________________________________________________________

## File Summary

### Phase 0 (from adaptive-sampling.md)

| File | Change | Lines |
|---|---|---|
| `src/film/rgb.rs` | Extract VarianceEstimator, replace 3 parallel arrays | ~80 changed, net -30 |

### Phase 1

| File | Change | Lines |
|---|---|---|
| `src/denoiser/mod.rs` | **NEW** — Denoiser trait + NoDenoiser | ~40 |
| `src/denoiser/bilateral.rs` | **NEW** — BilateralDenoiser | ~80 |
| `src/denoiser/atrous.rs` | **NEW** — AtrousDenoiser (Phase 2) | ~250 |
| `src/denoiser/oidn.rs` | **NEW** — OidnDenoiser (Phase 3) | ~100 |
| `src/denoiser/features.rs` | **NEW** — DenoiserFeatures (SoA) | ~80 |
| `src/lib.rs` | Add `pub mod denoiser;` | 1 |
| `src/film/mod.rs` | Add `apply_denoiser()` to Film trait | ~5 |
| `src/film/rgb.rs` | Make `RgbFilm<D>` generic, add `denoised: bool`, fix `overlay_radiance`, `raw_data`, add `features: Option<DenoiserFeatures>` | ~60 |
| `src/renderer/cpu.rs` | Call `film.apply_denoiser()` after sampling loop | ~5 |
| `src/main.rs` | Optional: wire up BilateralDenoiser | ~5 |

**Phase 1 total**: ~190 new lines, ~70 lines changed. No breaking changes.

### Cross-Document Integration Points

| Denoiser Doc Section | Adaptive Sampling Doc Section | Dependency |
|---|---|---|
| `raw_data()` variance format | §5.1 `raw_data()` contract | Denoiser needs §2a (VarianceEstimator) for clean interface |
| Per-channel variance | §5.2 Per-channel vs max-over-RGB | Denoiser needs `[f64; 3]`, not max-over-RGB scalar |
| `apply_denoiser()` placement | §5.3 render loop placement | Both agree: after sampling, before final publish |
| `overlay_radiance()` behavior | §5.4 overlay + denoised flag | Aligned: preserve counts/m2, use `denoised: bool` |
| Progressive preview | §5.3 progressive frames | Denoiser does NOT run on intermediate frames |
| Double-denoising guard | §5.4 denoised flag | Aligned: `denoised: bool` prevents re-denoising |
| `DenoiserFeatures` variance | §5.6 shared variance | VarianceEstimator → `DenoiserFeatures.variance` SoA |

| Denoiser Doc Section | ARCH_HYBRID Doc Section | Relationship |
|---|---|---|
| §Phase 1 — `apply_denoiser()` on Film | §2 Film trait | ARCH_HYBRID acknowledges the extension |
| §Phase 1 — `RgbFilm<D: Denoiser>` | §1 RgbFilm (current state) | ARCH_HYBRID shows pre-generic snapshot |
| §6 — CpuRenderer denoiser call | §2 CpuRenderer | ARCH_HYBRID acknowledges the one-line addition |
| §Phase 2 — DenoiserFeatures (SoA) | §2 GBuffer\<'a> | **Different things.** GBuffer = visibility (SurfaceInteraction). DenoiserFeatures = per-pixel shading features for edge-stopping. SoA layout matches VarianceEstimator pattern. |
