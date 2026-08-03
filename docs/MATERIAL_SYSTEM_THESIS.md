# Material System — Design & Evolution (v3)

## Status

**Foundation implemented (v3, 2026-08-01). The three §0 gaps remain deferred.**

The v1 hard prerequisite is met: `samplestream-refactor.md` is fully
implemented, and the lobe-primitive material-tree restructure that v2's
cross-renderer analysis (§2.4, §4.10) pointed at has landed on top of it —
`DiffuseReflector`, `MicrofacetReflector` (Fresnel: Conductor{η,κ} /
Dielectric{ior}; absorbs former Metal + Glossy), `Dielectric` (smooth/rough
unified; absorbs former Dielectric + RoughDielectric), plus the unchanged
`DiffuseEmitter`/`Isotropic`/`Mix`/`Coated`. Verified behavior-preserving
(byte-identical fixed-seed render). See changelog and §4.10's status note.

Still deferred, deliberately: the three §0 gaps (measured reflectance data,
physically-correct multi-bounce layering, performance/fidelity dial) and
everything in §7's non-goals. §4.4's stochastic-evaluation signature
question (Phase B) is now unblocked but still undecided — it should be
decided against the landed `SampleStream`/`SamplerRng` shapes, not on
paper.

Soft prerequisites (should be stable, need not be finished): `renderer_arch.md`,
`adaptive-sampling.md`, `denoiser.md`, `mesh-design.md`. They gate the
remaining backlog, not the landed foundation.

One structural note carried over from the CORE_THESIS review: this doc
deliberately separates an aspirational section (§3) from the actionable one
(§4), and marks the aspirational section as non-implementable by construction.
That split is intentional, for the reason CORE_THESIS itself became a source of
confusion — a vision doc that reads like a live roadmap gets implemented
piecemeal by accident. §3 should never gain a checkbox.

______________________________________________________________________

## Changelog

- **v3 (2026-08-01)** — Lobe-primitive foundation implemented (Status moved from
  "deferred" to "foundation implemented"). Material tree restructured to
  intrinsic-composition primitives: Metal + Glossy → `MicrofacetReflector` with
  a `Fresnel` enum (Conductor{η,κ} / Dielectric{ior}) and optional albedo;
  Dielectric + RoughDielectric → one `DielectricMaterial` (roughness option +
  mirror-threshold delta fallback, coupled single-H sampler preserved);
  Lambertian → `DiffuseReflector`. Deliberately no standalone transmittor leaf —
  transmission stays a coupled lobe inside the dielectric composite (§4.10
  status note). `GpuMaterialType` renumbered (DiffuseReflector=0 … Coated=6)
  with `fresnel_kind`/`is_rough` dispatch flags; no GPU shader consumes the
  buffer yet. Verified behavior-preserving: byte-identical fixed-seed render,
  83/83 tests. Side effects: fixed a pre-existing `Material::is_delta()`
  dispatch gap (near-mirror rough dielectrics now correctly skip NEE);
  smooth-dielectric `reflectance_estimate` returns the Fresnel value instead of
  the default 1.0; `const_medium` phase tags merge (volume seeds shift for
  glossy/rough-dielectric-in-media scenes). The §0 gaps and the §4.2–§4.13
  leanings are untouched and remain the backlog.
- **v2 (2026-07-09)** — Cross-renderer expansion. Added detailed comparisons
  against Mitsuba 3, LuxCoreRender, appleseed, OpenMoonRay, NVIDIA Falcor,
  Google Filament, and renderling (Rust/wgpu). Identified 7+ stealable
  implementation patterns from these renderers — precomputed energy compensation
  tables (appleseed), BSDF component classification with type-mask filtering
  (Mitsuba 3), measured BRDF with best-fit parametric importance sampling
  (Falcor), quality-level compile-time dial (Filament), slab-allocator GPU
  material serialization (renderling), BsdfBuilder pattern (MoonRay), and
  dual-use CPU/GPU code paths (renderling). Updated §4 architectural leanings
  with new patterns. Expanded bibliography with renderer source references. No
  code changes proposed.
- **v1 (2026-07-07)** — Initial draft. Synthesized from a three-way design
  discussion (two independent external material-architecture proposals, cross-
  checked against pbrt-v4's actual source/documentation, and against this
  project's existing `Bsdf`/`PdfKind`/`GpuMaterialNode` catalog). Establishes
  current-state audit, production comparison, the non-actionable North Star
  section, near-term architectural leanings, and a first open-questions list. No
  code changes proposed for immediate execution.

______________________________________________________________________

## 0. Purpose & Scope

The current material system (`DiffuseReflector`, `MicrofacetReflector`,
`Dielectric`, `DiffuseEmitter`, `Isotropic`, `Mix`, `Coated`, all behind the
`Bsdf` trait) is
correctly interface-unified but has no answer to three gaps:

1. **Measured/tabulated reflectance data** — no leaf type exists for it.
2. **Physically correct multi-bounce layering** — `Coated` is a documented
single-bounce approximation (see `Coated::emitted()`'s own comment admitting it
can't account for the coating's effect on emission).
3. **A deliberate performance/fidelity dial** — today, trading detail for speed
means hand-picking a different material; there's no mechanism for a scene, an
LOD system, or a user to make that trade-off systematically.

This doc is about closing those three gaps **without disturbing what already
works** — the `Bsdf` trait, the enum-of-leaves-plus-`Arc<dyn Bsdf>`-for-
composition shape, and the `GpuMaterialNode` flattening scheme all stay.

______________________________________________________________________

## 1. Current State (Audit)

| Piece | Shape | Notes |
|---|---|---|
| `Bsdf` trait | `scatter`, `eval`, `pdf`, `pdf_kind`, `emitted`, `is_emissive`, `reflectance_estimate`, `is_delta`, `ggx_alpha` | Deterministic given `(wo, wi)` — no RNG in `eval`/`pdf` today; `scatter` draws from a `next_dim` closure over the two-stream `SampleStream`/`SamplerRng`. |
| Leaf variants | `DiffuseReflector`, `MicrofacetReflector` (Fresnel: Conductor{η,κ} / Dielectric{ior}), `Dielectric` (smooth/rough unified), `DiffuseEmitter`, `Isotropic` | Inline structs, no heap allocation. `MicrofacetReflector` absorbs former Metal + Glossy; `Dielectric` absorbs former Dielectric + RoughDielectric (v3). |
| Composite variants | `Mix` (stochastic lobe selection), `Coated` (single-bounce Fresnel blend) | Both `Arc<dyn Bsdf>` children — the only place this codebase pays vtable cost on the material side. |
| `PdfKind` | Enum, avoids `Box<dyn PDF>` in the hot path | Already the right shape for what follows. |
| `GpuMaterialNode` | `material_type` / `param_offset` / `child_a` / `child_b` / `texture_index` | Flat, index-based — textures and (per this doc) measured/precomputed tables both reduce to "buffer + index" under this scheme with no new GPU-side concept required. v3: `GpuMaterialType` renumbered (DiffuseReflector=0 … Coated=6); `MicrofacetReflector` carries a `fresnel_kind` flag plus η/κ params, `Dielectric` an `is_rough` flag. No GPU shader consumes the buffer yet. |
| Known open issues (already tracked, relevant here) | GGX is single-scatter (no Kulla-Conty/Turquin compensation); GGX samples the NDF, not VNDF (Heitz 2018); `Coated::emitted()` sums coat+substrate emission because it has no `wo` to compute a real Fresnel split | Carried forward from prior review — not re-litigated here, but §4.8 explains why one future tier resolves the first and third for free. |


______________________________________________________________________

## 2. Comparison Against Production Renderers

### 2.1 pbrt-v4

- Virtual dispatch is deliberately avoided for almost every polymorphic type in
  pbrt via `TaggedPointer`; the one named exception is Integrator selection,
  because that's a once-per-render decision, not a hot-path call. Materials and
  BxDFs are squarely in the "hot path" category — this validates keeping `Bsdf`
  enum/monomorphized rather than boxed wherever the call volume is high.
- Layered materials are handled by `LayeredBxDF`: exactly two interfaces plus
  one homogeneous medium between them, evaluated via **stochastic Monte Carlo
  random walk** (Guo, Hašan & Zhao 2018 — see §11), not precomputation. More
  than two layers is handled by nesting a `LayeredBxDF` inside another.
  `f()`/`PDF()` on this type are themselves noisy estimators — explicitly
  acceptable in pbrt's own documentation, since it's just one more error source
  that shrinks with sample count.
- The two interface types are template parameters — dynamic dispatch is
  supported for arbitrary user-supplied pairs, but the common cases
  (`CoatedDiffuseBxDF`, `CoatedConductorBxDF`) get dedicated monomorphized types
  for speed. This is a direct, explicit, documented precedent for the
    generic-for-common/dyn-for-rare split this project already uses elsewhere
    (`Sampleable`, `SampleStreamEnum`, etc.).
- pbrt-v3's kitchen-sink `UberMaterial` was **removed** in v4 in favor of small,
  individually composable, physically-grounded materials. Worth internalizing:
  the "principled uber-shader" pattern is an *authoring convention* for DCC
  interchange (Blender, Arnold, RenderMan all expose one), not necessarily the
  right internal renderer representation. pbrt-v4 itself moved away from it
  internally.
- Measured BRDF representation moved off raw MERL-style tabulation to Dupuy &
  Jakob's adaptive parameterization (§11) — worth reading before committing to a
  table format, since raw MERL tables have real practical downsides (memory,
  importance-sampling difficulty) this newer representation targets directly.

### 2.2 OpenPBR / Disney Principled BSDF

An authoring convention: diffuse + specular/metal + clearcoat + sheen +
subsurface + transmission lobes, combined via a **fixed, heuristic** energy
weighting (not full radiative-transfer simulation — this is explicit in the
spec, since a real-time-friendly artist tool can't afford per-shading-point
adding-doubling). Covers most everyday appearances with a small, intuitive
parameter set; does not aim for exact multilayer interference, glints, or
discrete flake sparkle. This is a **preset that could be built on top of** the
granular primitives in §1, not a parallel system to construct alongside them.

### 2.3 The Layered-Materials Literature — Three Families, Not One

| Family | Representative work | Precompute | Per-sample cost | Texture-compatible |
| --- | --- | --- | --- | --- |
| **Naive linear combination** | Weidlich & Wilkie 2007 (arbitrarily layered microfacet surfaces via a transmission-factor blend) | none | cheap | yes |
| **Discretized scattering-matrix / adding-equations** | Jakob, d'Eon, Jakob & Marschner 2014; Zeltner & Jakob 2018; Belcour 2018 (statistical-operator variant) | **high, per unique parameter combination** | cheap (table/Fourier lookup) | poor — combinatorial blowup under spatially-varying params |
| **Stochastic random walk** | Guo, Hašan & Zhao 2018 (pbrt‑v4's `LayeredBxDF`); improved by Xia et al. 2020, Gamboa et al. 2020 | none | expensive, variable‑length, noisy | yes, fully — re‑simulates locally with whatever local params apply |

This table *is* the three-tier structure proposed in §4.2. It isn't a novel
invention for this project — it's the literature's own natural clustering,
confirmed independently by two different external reviewers of this exact
question landing on the middle-and-right columns without knowing about each
other.

### 2.4 Mitsuba 3 — Plugin-Based BSDF Composition (C++/Python, ~180K LOC)

- **Plugin/registration system** with `MI_EXPORT_PLUGIN` macro
  auto-instantiating all variant combinations (scalar/JIT ×
  RGB/spectral/mono/polarized = 13 variants). Rust analogue: derive macro +
  feature flags.
- **`BSDFFlags` bitfield** classifying lobes (Diffuse/Glossy/Delta/Null) plus
  `BSDFContext` type-mask for selective component evaluation. Enables MIS with
  per-lobe PDFs without requiring separate BSDF objects.
- **`eval_pdf_sample()` combined dispatch** — single method returning spectrum,
  pdf, and sample together, amortizing virtual dispatch (~20% perf win on CUDA
  per docs).
- **Adapter-compositor layering:** `coating` BSDF wraps any nested BSDF, adding
  a Fresnel interface + Beer's law absorption, forwarding/transforming child
  flags. Single-bounce (Weidlich & Wilkie 2007), known energy loss for
  diffuse/rough bases. Validates §4.2 Tier 2's shape (a BSDF that wraps another
  BSDF).
- **`principled`** (Disney-style) and **`measured`** (Dupuy-Jakob adaptive
  parameterization) coexist as peer variants — neither is second-class.
  Validates §3's "no lobe is a second-class citizen."
- **GPU path:** Dr.Jit JIT-compiles arrays-of-pointers for virtual dispatch on
  CUDA; material params accessed as struct fields, no UBO indirection.
  `DRJIT_CALL_TEMPLATE_BEGIN/END` enables vectorized dispatch over arrays of
  BSDF pointers.

### 2.5 LuxCoreRender — CPU+GPU VM-Interpreter Material System (C++/Python,
~350K LOC)

- Single abstract `Material` base class with virtual `Evaluate`/`Sample`/`Pdf` —
  no monomorphization or tagged union on CPU. `MaterialType` enum (~20 entries)
  for RTTI dispatch. All parameters stored as `TextureConstPtr` forming a
    delegated parameter DAG — re-evaluated per-hitpoint.
- **GPU path:** materials compiled to flat `MaterialEvalOp` bytecode at scene
  load; OpenCL kernel interprets via stack machine (`material_funcs.cl`).
  Recursive Mix/GlossyCoating handled via VM call/return, not recursion.
  Accepted tradeoff: "slowest but requires no kernel recompilation."
- **`GlossyCoatingMaterial`**: SchlickBSDF coating over arbitrary base, with
  `multibounce` flag for energy compensation.
- **`MixMaterial`**: stochastic lobe selection — trivially energy-conserving via
  mutual exclusion.
- Per-material `isVisibleIndirectDiffuse/Glossy/Specular` flags — coarse
  visibility filter for indirect paths, the simplest form of a performance dial
  found in any production renderer.
- **Key limitation:** no measured BRDF support, no material LOD system.

### 2.6 appleseed — Virtual + CRTP Wrapper + Precomputed Energy Tables (C++,
~680K LOC)

- Clean virtual base class with typed `void*` input values (pre-evaluated into
  arena-allocated flat structs per shading point — avoids per-call texture
  re-evaluation).
- **`BSDFWrapper<BSDFImpl>` CRTP template:** wraps every BSDF to inject
  direction culling, cosine adjoint correction, shadow terminator fix. Separates
  cross-cutting concerns from scattering physics — each BSDF impl is pure math.
- **`attenuate_substrate`/`attenuate_emission` virtual protocol:** any BSDF
  under a coating defines how light passes through itself. Enables generic
  layering without the coating knowing the substrate type.
- **Precomputed 3D albedo tables** (eta × roughness × cosθ, 90×90×90, ~171KB
  baked C++ array) for energy-conserving dielectric layer attenuation in
  `GlossyLayerBSDF` and `PlasticBRDF`. **This is the key finding:** it provides
  a cheaper path to energy conservation than Tier 2's random walk — a single
  table lookup instead of a variable-length bounce loop. Relevant to §4.8.
- Scattering mode bitfields (`modes` parameter on all BSDF methods) enable
  caller-side lobe filtering. `m_min_roughness` per ray avoids expensive
  near-mirror GGX eval.
- CPU-only — no GPU path.

### 2.7 OpenMoonRay — DSO Plugin System + BsdfBuilder + ISPC (C++/ISPC, ~2M LOC)

- Materials compiled as dynamically-loaded DSO plugins with RDL2 typed
  attributes. Each plugin registers `createLobes()` which calls a
  **`BsdfBuilder`**.
- **`BsdfBuilder` pattern:** materials don't return a BSDF struct; they call
  builder methods (`addMicrofacetIsotropicBRDF()`, `addLambertianBRDF()`, etc.)
  which internally arena-allocate `BsdfLobe` objects into a fixed-size array
  (max 16 lobes). Separates parameter resolution from lobe construction. In Rust
  this maps to a builder taking `&mut Bsdf` with typed add methods.
- Layering via **`DwaLayerMaterial`** (two submaterials blended via per-pixel
  mask) and **`DwaMixMaterial`** (weighted blend). `DwaBase` has explicit
  `startAdjacentComponents()`/`endAdjacentComponents()` for energy partitioning
  across lobes.
- Toksvig normal-map anti-aliasing: `computeMicrofacetRoughness()` boosts
  roughness proportional to normal-map variance. `NormalAAStrategy` enum.
  Per-material `mGlitterApproximateForSecondaryRays` flag for quality degrade on
  indirect paths — closest thing to a material LOD found in a CPU renderer.
- **ISPC vectorized hot path:** `Bsdfv.ispc` exports `Bsdfv_init`,
  `BsdfLobev_eval`, `BsdfLobev_sample` — SIMD over shading points.
- CPU/ISPC only — no GPU material serialization.

### 2.8 NVIDIA Falcor — Slang Material Instances + MERL Measured BRDF
(C++/Slang, ~300K LOC)

- **`IMaterialInstance` Slang interface** with `eval<S>`, `sample<S>`,
  `evalPdf`, `evalAlbedo`. Each material type produces a concrete instance
  struct (e.g., `StandardMaterialInstance`). `[anyValueSize(N)]` attribute
  enables stack allocation and monomorphization — maps to Rust enum dispatch.
- **`LayeredBSDF`**: Guo et al. Monte Carlo random walk between two `IBSDF`
  interfaces with optional participating medium (sigmaT, g, albedo) between
  them. Top BSDF positive hemisphere exposed for wi.z>0, bottom's negative for
  wi.z<0. **Direct production precedent for Tier 2.**
- **MERL measured BRDF:** `MERLMaterialData` stores tabulated data in a GPU
  buffer + **`DiffuseSpecularData` best-fit parametric approximation for
  importance sampling** — since tabulated BRDFs are expensive to sample
  directly. `MERLMix` variant supports per-texel BRDF selection via an index
  map. This addresses a key open question (§6.5) — the parametric-fit-for-IS
  approach is stealable regardless of whether the final table format is
  Dupuy-Jakob or MERL.
- **Flat GPU material buffer:** All material params packed into a single
  `StructuredBuffer<MaterialDataBlob>`. `MaterialParamLayout` defined via
  X-macro (`MATERIAL_PARAMS(PARAM)`). Maps to Rust `macro_rules!` for CPU/GPU
  layout sync.
- `MaterialSystem::optimizeMaterials()` replaces constant textures with uniform
  parameters. `computeMinRoughness()` with clamping factor for singular
  detection.

### 2.9 Google Filament — Uber-Shader + Quality-Level Performance Dial
(C++/GLSL, ~250K LOC)

- Single PBR model (Cook-Torrance GGX) with compile-time variant selection.
  `#define SHADING_MODEL_LIT` gates which evaluation code compiles. `matc` tool
  generates shader variants for feature combinations (DIR, DYN, SRE, SKN, FOG,
  SSR, STE).
- **Quality levels: `FILAMENT_QUALITY` (LOW/NORMAL/HIGH) selects fast vs
  accurate BRDF math** — e.g., `SPECULAR_V_SMITH_GGX_FAST` vs the full
  correlated Smith. Auto-picks LOW for mobile, HIGH for desktop with explicit
  override. **Most direct production precedent for the North Star's fidelity
  dial.** The mechanism is simple: a compile-time constant gates which math
  function is called, with no per-branch overhead for the fast path.
- Layering via extra lobes within a single evaluation: clearcoat (isotropic GGX,
  always isotropic), sheen (Charlie distribution, Estevez 2017), subsurface
  (thickness-based transmission). **Multi-bounce Kulla-Conty** energy
  compensation for high-roughness GGX energy loss — another precedent for §4.8
  without Tier 2's cost.
- GPU material serialization: three descriptor sets (PER_VIEW, PER_RENDERABLE,
  PER_MATERIAL) with structured `#[repr(C)]`-style UBOs matching std140 layout.
  `MaterialInstance` with dynamic UBO offsets for per-object parameter
  variations.
- BRDF LUT pre-computed at startup (IBL integration term).

### 2.10 renderling — Slab-Allocator GPU Material System (Rust/wgpu, ~30K LOC)

- **Most directly relevant comparison** because it's Rust → wgpu, the same
  language and GPU API target as this project's eventual GPU path.
- Single `#[repr(C)]` `MaterialDescriptor` struct with `Id<T>` slab pointers for
  texture references — no trait hierarchy, no enum dispatch. Flat enough to
  memcpy into a `StorageBuffer<[u32]>`.
- **Slab allocator** (`craballoc`/`crabslab`): three separate slabs (geometry,
  materials, lighting) bound as `&[u32]` storage buffers in a single bind group.
  `Hybrid<T>` CPU/GPU bridge with dirty tracking. Single `commit()` flushes all
  changes via `queue.write_buffer()`. **Direct Rust precedent for §4.7's GPU
  flattening** — and materially simpler than the VM-interpreter approach
  (LuxCore) or X-macro layout (Falcor).
- **Dual-use code pattern:** `pub fn fragment_impl(...)` in shared `shader.rs`
  called from both CPU unit tests and `#[spirv(fragment)]` GPU entry point. No
  duplication. `#[cfg(cpu)]` / `#[cfg(gpu)]` split on modules.
- **Texture atlas:** all material textures packed into a single `Image2dArray`
  via `crunch` rectangle packing. References become `(layer, offset, size)`, not
  `TextureView`. Avoids binding limits — one bind group covers all texture slots
  for all materials.
- **Clone-counted GPU lifecycle:** `WeakHybrid` detects when CPU handles are
  dropped; `upkeep()` reclaims slab slots. No explicit GPU free() — resource
  lifetime matches Rust's ownership semantics.
- **Gaps confirmed** (matching doc's §0 assessment): no layered/coated
  materials, no measured BRDF, no performance/fidelity dial. Single hard- coded
  Cook-Torrance model with no BSDF dispatch.
- ⚠ Dependency: renderling uses `rust-gpu` (SPIR-V from Rust), but the
  renderling devlog (2026) flags an intended move toward WGSL. Worth watching
  before committing to a GPU compilation target.

### 2.11 Common Threads Across 9 Renderers

| Renderer | Architecture style | Layering model | Measured BRDF | Perf dial | GPU materials |
|---|---|---|---|---|---|
| **pbrt-v4** | TaggedPointer enum, monomorphized | Stochastic RW (Guo 2018), two-interface | Dupuy-Jakob adaptive | — | CPU-only |
| **Mitsuba 3** | Plugin registration, virtual dispatch | Adapter coating (W&W 2007), single-bounce | Dupuy-Jakob adaptive (TensorFile) | Variant system, per-lobe srate | Dr.Jit JIT arrays-of-pointers |
| **LuxCore** | Virtual hierarchy (CPU) + VM bytecode (GPU) | GlossyCoating (Schlick + multibounce), Mix | Not found | Indirect-visibility flags | `Material` flat union + VM opcodes |
| **appleseed** | Virtual + CRTP wrapper, factory plugins | GlossyLayer/Plastic (precomputed albedo tables), Blend/Mix | Not found | Scattering mode bitfields, `m_min_roughness` | CPU-only |
| **MoonRay** | DSO plugins + BsdfBuilder + ISPC | DwaLayer (mask blend), DwaMix (weight blend) | Not found | Normal AA, glitter LOD, per-material flags | CPU/ISPC |
| **Falcor** | Slang interface + monomorphized instances | LayeredBSDF (Guo 2018 random walk), two-interface + medium | MERL + MERLMix + **parametric best-fit IS** | Roughness clamping, `optimizeMaterials()` | `MaterialDataBlob` flat buffer + X-macro |
| **Filament** | Uber-shader + compile-time variants | Extra lobes (clearcoat/sheen) in single eval | Not found | **`FILAMENT_QUALITY` (LOW/NORMAL/HIGH)** | 3-set UBOs (view/renderable/material) |
| **renderling** | Single `#[repr(C)]` struct, no dispatch | Not found | Not found | Not found | **Slab allocator + `Id<T>` + commit cycle** |
| **OpenPBR** | Authoring convention, not engine-internal | Fixed heuristic weighting (not RT simulation) | Not found | — | Authoring-format only |

**Five patterns that cross-cut multiple renderers:**

1. **Flat GPU material buffers win** — Every renderer with a GPU path (LuxCore
   VM, Falcor `MaterialDataBlob`, renderling slab) converges on a flat buffer of
   material structs with type-headers, indexed by pointer or slab `Id`. No
   renderer uses per-material GPU shader variants. This strongly validates §4.7
   and gives us concrete implementation precedents in both C++ and Rust.

2. **Stochastic random walk for layering is production-validated** — Both
   pbrt-v4 and Falcor ship production implementations of Guo et al.'s
   position-free random walk for layered BSDF evaluation. This is no longer a
   research-only technique. The "lean: build Tier 2 before Tier 1" in §4.2 is
   well-supported.

3. **Nobody has a good material LOD system** — The closest production dials are
   Filament's compile-time quality levels (which affect *all* materials
   globally), MoonRay's per-material flags for secondary-ray approximations, and
   LuxCore's indirect-visibility toggles. No renderer has the granular
   per-object fidelity dial the North Star describes. This is a genuine
   innovation opportunity.

4. **Measured BRDF support is rare and uses parametric IS** — Only two renderers
   support measured BRDFs (pbrt-v4, Falcor), and both use a parametric
   approximation for importance sampling rather than sampling the tabulated data
   directly. The measured data stores the "truth" for eval; the parametric fit
   handles sampling. This is worth codifying as a design pattern in §4,
   regardless of which table format we adopt.

5. **Rust-specific: slab allocators outperform VM interpreters** — Comparing
   LuxCore's VM bytecode (whose own docs admit it's "slowest") with renderling's
   slab-allocator approach for GPU material dispatch, the slab pattern is
   simpler, faster, and more Rust-idiomatic. This should be the default pattern
   for any GPU material system in this project.

______________________________________________________________________

## 3. The North Star

> **This section is not a task list. It is never "done." Nothing in this section
> should ever appear as a line item in an implementation-phase table. Its only
> job is to give future revisions of this doc — and any other spec that touches
> the material system — a consistent standard to justify decisions against. If a
> future addition can't point at a sentence here as its motivation, that's a
> signal to question the addition, not to add a checkbox here.**

The ideal material system for this engine:

- Treats **completeness as a spectrum with a runtime dial**, not a fixed choice
  baked in at compile time or asset-authoring time. Analytic, measured, and
  multi-bounce-simulated representations of the same conceptual material
  coexist, and the renderer — or the user — decides which one gets paid for, per
  object, per scene, per render.
- Makes **no lobe or representation a second-class citizen**: a measured BRDF, a
  stochastically-layered stack, and a three-line analytic Lambertian should all
  be equally first-class values behind the same interface, callable from the
  same integrator code, with no special-casing anywhere in the shading path.
- Sacrifices **as little raw performance as possible** for that flexibility —
  the cost of choosing a cheap representation should be exactly the cost of that
  representation, never a tax paid by every material for the mere possibility
  that some other material is expensive.
- Gives users **granular, legible control** over the fidelity/performance
  trade-off, rather than a single global quality slider — hero assets at full
  fidelity, background geometry cheap, by deliberate choice rather than by the
  renderer's inability to do otherwise.

This is a compass, not a milestone. It exists so that when a future version of
this doc proposes something concrete, there's a shared standard to check it
against.

______________________________________________________________________

## 4. Patterns Identified / Architectural Leanings

These are the actionable conclusions. Everything here is provisional (v1), but
this is the section future implementation work should actually read.

### 4.1 One interface, tiered leaves and composites

No new top-level system. `Bsdf` stays the unification point. A measured leaf and
a stochastically-layered composite are new **variants**, not new architecture —
this is mechanically identical to how `DiffuseReflector` and `MicrofacetReflector`
already coexist.

### 4.2 Layering is three tiers, not one axis

Per §2.3's table:

- **Tier 0 (have it today):** `Coated`, naive single-bounce blend. Cheapest,
  biased at grazing angles / dark substrates, zero new cost.
- **Tier 1 (precomputed table):** cheap eval, expensive and
  parameter-combination-limited bake. Best suited to a small number of fixed,
  non-textured hero materials (a specific car paint recipe, not "any rough
  clearcoat with a spatially-varying roughness map").
- **Tier 2 (stochastic random walk):** no bake step, fully texture-compatible,
  noisy per-sample estimate. Follow pbrt's shape exactly — two interfaces plus
  an optional homogeneous medium, `Box<dyn Bsdf>`-recursable for depth beyond
  two layers, meaning this is naturally an **extension of `Coated`'s existing
  shape**, not a new N-ary data structure.

Current lean: build Tier 2 before Tier 1. It reuses more of what exists
(`Coated`'s shape, the existing MIS/integrator structure once §4.4 is settled),
doesn't need an offline bake pipeline, and is the tier that also resolves §4.8's
energy-conservation payoff. Tier 1 is deferred until a concrete hero-asset use
case actually needs it — see §7.

### 4.3 Measured/tabulated data is an orthogonal leaf, not a fourth layering
tier

Slots in next to `DiffuseReflector`/`MicrofacetReflector` as `Bsdf::Measured(...)`. It composes into
any layer stack (any tier) as a sub-layer, since layering machinery only needs
its children to satisfy the `Bsdf` trait.

Sharp edge to keep in view: a measured BRDF **cannot be spatially textured at
the reflectance-shape level** — there's no parameter left to swap a texture
value into, since the whole shape *is* the measurement. The common production
workaround is a spatial tint/scale multiplier applied on top of the lookup,
which is not physically part of the measurement but is the accepted escape hatch
if any per-texel variation is wanted at all.

Target representation to actually study before committing to a table format:
Dupuy & Jakob 2018 (§11), not raw MERL. Confirmed storage cost from that paper:
~16 KiB per spectral channel for isotropic materials, ~544 KiB for anisotropic —
worth using as the real memory-budget reference point rather than MERL's
original (much larger) raw-grid footprint.

### 4.4 Stochastic evaluation is a trait-signature decision, not an
implementation detail

A Tier 2 leaf/composite needs RNG (or `SampleStream`) access inside `eval()` and
`pdf()`, not just `sample()` — this is the one piece of "surgery" this doc
requires on the existing `Bsdf` trait. Two consequences:

1. This is a **hard dependency on `samplestream-refactor.md`** landing first —
deciding this signature before that refactor's `SampleStream`/`Rng` traits
stabilize means deciding it twice.
2. MIS weighting code that currently treats `material.pdf()` as a deterministic
value needs to tolerate a noisy estimate for any material using Tier 2. pbrt's
own documentation states this is fine for Monte Carlo estimators in general —
but it's a real assumption shift worth deciding deliberately rather than
discovering while debugging unexplained variance later.

### 4.5 Generic-for-common, dyn-for-rare

Direct precedent: pbrt-v4's `LayeredBxDF` takes its two interface types as
template parameters, with dedicated monomorphized types (`CoatedDiffuseBxDF`,
`CoatedConductorBxDF`) for the common pairs and ordinary dynamic dispatch as the
fallback for arbitrary ones. Same shape this project already applies to
`Sampleable`/`SampleStreamEnum`/`ConvergenceCriterionEnum`. Lean: monomorphize
`Coated<Dielectric, DiffuseReflector>` / `Coated<Dielectric, MicrofacetReflector>` as fast paths
once real scene data says they're common enough to be worth it (this can't be
decided on paper — see §6.7); `Box<dyn Bsdf>` handles everything else.

### 4.6 Detail-level resolution is coarse, not per-sample

One of the two external proposals suggested a `Material::createBSDF(point,
DetailLevel)` factory called per shading sample. **Reject this shape.** Read
literally, it heap-allocates or re-resolves a trait object on every call —
exactly the anti-pattern already flagged elsewhere in this codebase (68
`Arc::new` calls for solid-color textures), now recurring at the material layer
where it would be paid far more often.

The underlying idea (a material can resolve to different concrete
representations depending on desired fidelity) is sound; only the *frequency* of
the decision needs to change. Lean: resolve detail level **coarsely** — once per
object per LOD-change event (a camera-distance threshold crossing, an explicit
scene-author override, or once at scene-load if dynamic LOD isn't built yet) —
via the same `Descriptor → Concrete → Wrapper` pattern already used for
`SampleStreamKind`/`ConvergenceCriterionKind`. The result is a committed,
already-resolved `Bsdf` variant; every shading sample after that just calls into
whatever's already there, with zero extra indirection versus today.

### 4.7 GPU tiering

Measured (§4.3) and Tier 1 precomputed-table data both reduce to "a big buffer
of floats, referenced by index" — structurally identical to how image textures
already flow through `GpuMaterialNode`'s `texture_index`. No new GPU-side
concept needed for either.

Tier 2 (stochastic random walk) is a genuinely harder GPU problem, and not just
because it's more expensive per sample: it's a **variable-iteration-count
workload** (bounce count inside the walk differs per invocation), which causes
real wavefront/warp divergence in a compute shader — worse than "different
material enum variant per pixel," which is a one-time branch rather than a
variable-length loop per lane. Lean: keep Tier 2 CPU-only (or a preview/hero-
render-only path) until deliberately revisited; let the GPU path lean on Tier 0
and Tier 1, both boring, static-cost buffer lookups.

### 4.8 Energy conservation as a side effect, not a separate task

A correctly-implemented Tier 2 random walk conserves energy by construction — it
actually simulates the inter-reflection, so there's nothing left to separately
compensate for. This is a free, non-urgent resolution path for two issues
already on record for this codebase: the undocumented GGX energy loss at high
roughness (no Kulla-Conty/Turquin compensation today), and the
`Coated::emitted()` workaround that exists only because `Coated` can't currently
reason about what happens between coat and substrate. Not a reason to prioritize
Tier 2 on its own — but worth remembering when weighing whether it's worth the
render-time cost later, since building it also retires two existing known issues
for any material that opts into it.

### 4.9 Precomputed energy compensation tables (appleseed, Filament)

A correctly-implemented Tier 2 random walk conserves energy by construction (see
§4.8), but it's expensive. **appleseed's `GlossyLayerBSDF` provides a cheaper
alternative:** a 3D table (eta × roughness × cosθ, 90×90×90, ~171KB) stores the
directional albedo of a dielectric microfacet layer, computed offline from the
analytic GGX model. At runtime, `attenuate_substrate()` does a single table
lookup instead of a bounce loop.

**This changes the cost calculus of §4.8.** The energy-conservation payoff needs
*not* require Tier 2 — a precomputed table can correct both the GGX energy loss
and the `Coated::emitted()` approximation at the cost of a trilinear lookup,
one-time parameter-combination limitation (non-textured layer parameters), and a
build-time bake step.

The lean from §4.8 remains valid (Tier 2 resolves energy conservation for *any*
layer parameter combination, including textured ones), but we should recognize
the precomputed-table approach as a viable intermediate step rather than
deferring it entirely. A plausible staging:

1. **Build the precomputed table** for the common case (dielectric coating over
diffuse/conductor, uniform parameters).
2. **Use it as the default** in `Coated` when parameters are uniform (no spatial
variation).
3. **Fall back to Tier 2** (or the current single-bounce approximation) when
layer parameters vary spatially.

**Reference:** `AlbedoTable2D`/`AlbedoTable3D` in appleseed's
`energycompensationtables.cpp` (171KB baked C++ data), used in
`glossylayerbsdf.cpp` and `plasticbrdf.cpp`.

### 4.10 BSDF component classification with type-mask filtering (Mitsuba 3,
appleseed)

Both Mitsuba 3 and appleseed classify BSDF lobes using a bitfield of component
types (Diffuse/Glossy/Specular/Delta, each OR'd with Reflect/Transmit), and pass
a type-mask through the evaluation context to tell the BSDF which components to
consider.

**Mitsuba 3's `BSDFFlags`** is the cleaner design: a bitfield enum with compound
flags (e.g., `Reflection = DiffuseReflection | GlossyReflection |
DeltaReflection`), stored per-component in `m_components` vector and OR'd into
`m_flags`. A `BSDFContext` struct carries the `type_mask` and `component` index,
allowing integrators to selectively query sub-lobes for MIS.

**appleseed's scattering mode bitfields** achieve the same goal via a `modes`
parameter on every BSDF method. If only `Diffuse` is requested, a glossy-only
BSDF returns zero immediately.

**Lean:** Adopt a `BSDFFlags` bitfield for lobe classification and a
`BSDFContext`-like struct for per-component evaluation requests. This enables:
- MIS with per-lobe PDFs without requiring separate BSDF objects.
- Selective evaluation (integrator says "only glossy for this call").
- Clean composition: a layered BSDF ORs its children's flags.
- The flags type can also serve as the interface for `is_delta()`,
  `is_emissive()`, etc., replacing the current per-material methods.

The `BSDFFlags::Delta` discriminant is particularly important — it replaces
individual `is_delta()` checks and makes MIS dispatch self-documenting. This is
compatible with pbrt-v4's approach and would fold into the existing `PdfKind`
enum naturally.

**Status note (v3):** the lobe-primitive restructure of 2026-08-01 partially
realizes this lean. The tree now classifies lobes at the type level
(`DiffuseReflector` / `MicrofacetReflector` / the coupled dielectric composite)
and parameterizes the reflector's Fresnel behavior via the `Fresnel` enum
(`Conductor{η,κ}` | `Dielectric{ior}`) — the Mitsuba-style "component
classification" move. Not yet adopted: the `BSDFContext`-style type-mask for
per-component evaluation, and `BSDFFlags::Delta` as a replacement for the
per-material `is_delta()` methods. The deliberate non-standalone transmittor
(the coupled dielectric sampler must not be split into independent lobes) is the
one place this project's shape intentionally diverges from a pure
component-composition model; pbrt-v4's fused `DielectricBxDF` is the precedent.

### 4.11 Measured BRDF with parametric importance sampling (Falcor)

Falcor's `MERLMaterialData` stores the measured BRDF as a GPU buffer of raw
tabulated values, but uses a **`DiffuseSpecularData` best-fit parametric
approximation for importance sampling**, not the table itself. The table is only
evaluated; sampling is done via the parametric fit's analytic sampling.

This pattern addresses an open question (§6.5) that the v1 doc had identified
but not answered: "measured data representation — commit to studying Dupuy &
Jakob's parameterization before choosing a table format." The parametric-fit-
for-IS pattern is **orthogonal to the table format choice** — it works with both
  raw MERL grids and Dupuy-Jakob adaptive parameterizations. This means:

- We can adopt the Dupuy-Jakob parameterization as our storage format (compact,
  good eval behavior, following pbrt-v4's lead).
- And separately build a parametric importance-sampling fit for each measured
  dataset at load time (compute best-fit Cook-Torrance + Lambertian params that
  approximate the measured data, store alongside the true data).

**MERLMix** extends this further: multiple measured BRDFs in a single buffer
with a per-texel index map, enabling texture-driven BRDF variation across a
surface without per-pixel material ID changes.

### 4.12 Quality-level compile-time dial (Filament)

Filament's `FILAMENT_QUALITY` (LOW/NORMAL/HIGH) is the closest production
implementation of the North Star's fidelity dial. It works as a compile-time
constant that selects between fast and accurate math for specific BRDF terms:

```rust
const QUALITY: QualityLevel = match target() {
    Target::Desktop => QualityLevel::High,
    Target::Mobile  => QualityLevel::Low,
};
// Compile-time: only the chosen branch survives monomorphization
if QUALITY >= QualityLevel::High {
    // full correlated Smith-GGX visibility
} else {
    // fast Schlick approximation
}
```

**Lean:** Introduce a `MaterialQuality` enum (not a global — per-object per the
North Star) that selects:

- **Full:** Full Tier 2 random walk for layering, analytic GGX with multi-bounce
  compensation, measured BRDF with full table eval.
- **Medium:** Precomputed energy compensation tables (§4.9) instead of random
  walk, single-bounce coating approximation (§4.2 Tier 0), parametric-fit BRDF
  sampling.
- **Low:** Single-bounce coating, cheap BRDF math (Schlick approximations, fast
  visibility term), uniform lighting response, no measured data.

The granularity need not be three levels — the key insight from Filament is that
the quality decision is a *compile-time or resolve-time* constant in the
relevant monomorphized code path, not a runtime branch. For a Rust renderer this
maps to either a const generic or an enum discriminant that controls which
functions are called, with zero overhead for the chosen path.

This can coexist with §4.6's coarse-resolution mechanism: the `Descriptor`
already encodes the quality level, and the `Concrete` resolved type is
monomorphized for that quality at resolve time.

### 4.13 Slab-allocator GPU material serialization (renderling)

renderling's slab-allocation system
([`craballoc`](https://github.com/schell/craballoc) /
[`crabslab`](https://github.com/schell/crabslab)) is a concrete Rust
implementation of the GPU flattening described in §4.7, and should be the
default pattern when that path is built.

Key design elements:
- A `MaterialDescriptor` is a flat `#[repr(C)]` struct — no nested allocations,
  no trait objects, no enums. Texture and sub-material references are `Id<T>`
  (slab-relative u32 indices with `Id::NONE` sentinel for optional slots).
- The slab allocator manages a single `wgpu::Buffer` as a typed arena. CPU
  writes go into a staging copy; `commit()` detects dirty regions and flushes
  them to GPU in one `write_buffer()` call per slab.
- Three slabs (geometry, materials, lighting) are bound as `&[u32]` storage
  buffers in one bind group. Every shader reads every slab — no per-material
  binding.
- `Hybrid<T>` is a CPU-side handle that mirrors a slab-stored value. `modify()`
  queues an update. `upkeep()` checks `WeakHybrid` counts to reclaim unused
  slots (reference-counted GPU resource lifecycle).

**Comparison against alternatives:** This is simpler than LuxCore's VM bytecode
(which the LuxCore team themselves describe as the "slowest" approach), more
idiomatic Rust than Falcor's X-macro C preprocessor pattern, and directly
compatible with wgpu. For the GPU material path, adopt the
`#[repr(C)]`-flat-struct + `Id<T>` slab pattern.

**Open question:** Whether to merge the material slab with the existing
`GpuMaterialNode`+`texture_index` scheme (putting material params and texture
indices in the same slab buffer) or keep them separate (one slab for flat
material params, another for texture atlas). renderling keeps separate slabs per
concern; this has better cache behavior (material lookups don't compete with
texture data) at the cost of more bindings. Lean toward separate slabs following
renderling's proven model, but revisit when the actual GPU pipeline shape is
known.

### 4.14 BsdfBuilder / ephemeral material instance pattern (MoonRay, Falcor)

Two renderers independently converged on the insight that the material
evaluation object at a shading point is **ephemeral** — created at the hit
point, used for one scattering event, discarded — and should be constructed
differently from the persistent scene material.

- **MoonRay's `BsdfBuilder`:** A material's `createLobes()` receives a
  `BsdfBuilder` and calls typed add methods (`addMicrofacetIsotropicBRDF()`)
  that arena-allocate `BsdfLobe` objects into a fixed-size array. The builder
  handles lobe labeling, energy partitioning labels, and culling.
- **Falcor's `IMaterialInstance`:** Each material type produces a small
  monomorphized instance struct. The interface is polymorphic but the concrete
  struct fits in a fixed-size buffer (`[anyValueSize(N)]`), enabling stack
  allocation.

**Lean:** Adopt MoonRay's builder pattern for BSDF construction in the ephemeral
`sample()`/`eval()` path. The scene material stores parameter descriptors
(texture handles, scalar values, blend weights). At each shading point, a
`BsdfBuilder` resolves these parameters (sampling textures, applying normal
maps, computing Fresnel terms) and constructs the concrete lobe array for that
specific point. This separates concerns cleanly:
- **Scene material:** parameter storage, serialization, scene-graph concerns.
- **BsdfBuilder:** parameter resolution, lobe construction, energy partitioning.
- **Lobe array (`Bsdf` container):** compact, hot-path evaluation, sampling,
  MIS.

In Rust, the builder pattern avoids `Box<dyn Bsdf>` per-hit allocation by
writing into a fixed-size array or small-vector on the stack. The current `Bsdf`
trait (`eval`/`sample`/`pdf` on the *material* directly) already encodes this
distinction — the material's `sample()` internally constructs the ephemeral BSDF
state. The builder pattern makes this construction explicit and composable.

______________________________________________________________________

## 5. Sketch: Proposed Type Catalog Additions (illustrative only, not final)

```rust
// New leaf variant — orthogonal to layering, see §4.3
Bsdf::Measured(MeasuredData)   // table/parameterization TBD, see §6.5

// Layering stays inside Coated's existing shape (§4.2) — this is a sketch of
// *how* Coated's internals might branch, not a proposal to rename or split it
enum LayerModel {
    Naive,                      // today's single-bounce blend — Tier 0
    Stochastic { max_bounces: u32 },  // Guo/Hašan/Zhao-style random walk — Tier 2
    // Precomputed(TableHandle) — Tier 1, deferred, see §7
}
```

The exact shape of `Coated`'s internal branch (a field switch vs. a new
sibling variant `StochasticCoated` vs. something else) is genuinely open —
see §6.1. This sketch exists to make the discussion concrete, not to
pre-decide it.

______________________________________________________________________

## 6. Open Questions

1. **Does Tier 2 live as a mode switch inside `Coated`, or a new sibling
   variant?** Affects whether existing `Coated` call sites need to change at
   all when Tier 2 ships, versus needing a migration.
2. **Exact `Rng`/`SampleStream` threading signature for `eval`/`pdf`.** A full
   extra parameter on `Bsdf::eval`/`pdf` (affects every existing leaf, even
   deterministic ones, who'd ignore it), or a separate `StochasticBsdf: Bsdf`
   trait that only Tier 2 variants implement, with the integrator branching on
   which trait is present? The latter avoids touching every existing leaf's
   signature but adds a dispatch branch to the integrator. **v3 note:** the
   `samplestream-refactor.md` hard prerequisite is met (see Status) — this
   question is now decidable against the landed `SampleStream`/`SamplerRng`
   shapes and remains undecided.
3. **Does the existing MIS power-heuristic implementation need a variant for
   a noisy `pdf()`, or is treating the noisy estimate as-is actually fine**, as
   pbrt's own documentation claims for its own estimators? Needs to be checked
   against this project's specific MIS code, not assumed by analogy.
4. **Tier 1 table format and bake trigger** — offline authoring tool, lazy
   bake on first use, or scene-load bake? And: do we just accept "Tier 1
   requires non-textured, uniform layer parameters" as a hard constraint
   rather than trying to solve the combinatorial-explosion problem, given
   Tier 2 exists specifically to avoid it?
5. **Measured data representation** — commit to studying Dupuy & Jakob's
   parameterization (§11) in detail before choosing a table format; raw MERL
   grids are the historically-cited option but pbrt-v4 itself has moved past
   them for stated reasons (memory, importance sampling). **Partial answer
   from v2:** (§4.11) the parametric-fit-for-importance-sampling pattern
   (Falcor MERL) is orthogonal to table format choice and should be adopted
   regardless of which format is picked.
6. **Detail-level resolution granularity and trigger** — per-object? per
   material-instance? What triggers re-resolution — is automatic
   distance-based LOD in scope for v1 at all, or is "explicit scene-author
   override only, no automatic switching" the right starting constraint?
   (Current lean: the latter — automatic LOD is listed as a non-goal, §7.)
7. **Which `Coated<Top, Bottom>` pairs are common enough to monomorphize?**
   Not decidable on paper — needs real scene/profiling data once something
   exists to profile.
8. **GPU buffer format for measured/Tier-1 data** — reuse the existing
   `texture_index`-style indirection directly, or give it a dedicated buffer
   class? Leaning toward reuse (§4.7) but not confirmed against the actual
   `GpuMaterialNode`/`GpuTextureNode` code. **Partial answer from v2:**
   (§4.13) renderling's slab allocator pattern (`Id<T>` slab-relative
   indices, separate slabs per concern) is the recommended Rust implementation
   pattern, with separate material and texture slabs following renderling's
   proven model.
9. **Neural BRDF** — explicitly out of scope for v1 (§7). If ever revisited:
   does it reuse the `Measured` leaf's interface (swap table lookup for
   inference), or does it need its own variant given very different
   cost/precompute characteristics?
10. **Interaction with `mesh-design.md`'s deferred per-face material plan** —
    can different faces of one mesh resolve to different detail levels, or is
    detail level strictly per material-instance for v1? (Current lean: strictly
    per material-instance — per-face LOD is a compounding of two deferred
    features and should stay deferred until both are independently justified.)

______________________________________________________________________

## 7. Non-Goals (This Iteration)

- **Neural BRDF** — inference cost and training/acquisition pipeline are both
  well beyond current priorities. Revisit only if Tier 2 and measured-data
  support both feel insufficient for a concrete material that's actually
  blocking something.
- **Tier 1 (precomputed layered tables) as a from-scratch build** — Tier 2 is
  more aligned with the existing MIS/integrator shape, needs no bake pipeline,
  and resolves more open questions for less new infrastructure. Only build
  Tier 1 if a specific hero-asset need appears that Tier 2's per-sample cost
  can't afford.
- **Automatic screen-coverage-driven LOD** — a real feature, but it's a
  renderer/scheduling concern layered *on top of* the detail-level resolution
  mechanism in §4.6, not part of this doc. This doc only needs to make manual/
  explicit detail-level resolution possible; automatic triggers are a separate
  future doc.
- **An OSL-style arbitrary shader graph** — already explicitly deferred
  (elsewhere) until glTF/OBJ prerequisites are satisfied. Not revisited here.

______________________________________________________________________

## 8. Prerequisites & Dependency Order

| Doc | Relationship | Hard/Soft |
|---|---|---|
| `samplestream-refactor.md` | `SampleStream`/`Rng` traits must exist and be stable before §4.4's signature question can be closed | **Met (v3)** — implementation complete; §4.4 must be decided against the landed two-stream shapes |
| `renderer_arch.md` | Any new `Bsdf` variant needs an `albedo()`-style extraction path for GBuffer/`DenoiserFeatures` bridging | Soft |
| `denoiser.md` | Same reason as above — new variants must be accounted for wherever albedo/normal AOVs are derived from a material | Soft |
| `mesh-design.md` | Per-face material assignment is deferred there; §6.10 notes the interaction but doesn't require it resolved first | Soft |
| `adaptive-sampling.md` | No direct interaction identified | None |

The v1 gate is lifted: `samplestream-refactor.md`'s implementation steps (its
Phase 3–4, per that doc) are complete, and the v3 foundation builds directly on
them. The remaining backlog's own dependencies still apply — §4.4 against the
landed `SampleStream`/`SamplerRng` shapes, and any new `Bsdf` variant against
`renderer_arch.md`/`denoiser.md`'s extraction paths.

______________________________________________________________________

## 9. Illustrative Future Phases (will be rewritten once this doc stabilizes)

These exist to make the scope concrete, not as a committed plan:

- **Phase 0 (done, v3)** — Lobe-primitive material-tree restructure
  (`DiffuseReflector` / `MicrofacetReflector` + `Fresnel` / unified
  `Dielectric`), implementing §4.10's type-level lobe classification without
  touching any of the §4.2/§4.4 machinery. Its behavior-preservation proof
  (byte-identical fixed-seed render) is the model for how later phases should
  be verified.

- **Phase A** — `Measured` leaf. No `Rng` dependency; could theoretically
  start anytime but held per Status.
- **Phase B** — Close §4.4/§6.2: the `Rng`-in-`eval`/`pdf` signature decision,
  riding directly on `samplestream-refactor.md` landing.
- **Phase C** — Tier 2 stochastic layering inside/beside `Coated`.
- **Phase D** — Detail-level resolution mechanism (§4.6), reusing the
  `Descriptor → Concrete → Wrapper` pattern.
- **Phase E** — Tier 1 investigation, lowest priority, only if triggered by a
  concrete need (§7).

______________________________________________________________________

## 10. Cross-Document References

| This Doc | Related Doc | Relationship |
|---|---|---|
| §4.4 | `samplestream-refactor.md` | Hard prerequisite — met (v3); see §8 |
| §4.7 | `renderer_arch.md` §GPU material buffer | Measured/Tier-1 data should reuse the same indirection as `texture_index` |
| §4.3, §4.8 | Prior architecture review (this project, undated in `docs/`) | GGX energy loss and `Coated::emitted()` gaps referenced here are tracked there in more detail |
| §6.10 | `mesh-design.md` §Open Questions | Per-face material deferral noted on both sides |
| §3 | `CORE_THESIS.md` | Structural precedent for separating aspiration from roadmap — see Status section above |

______________________________________________________________________

## 11. Bibliography

**Layered materials — stochastic family (current lean, §4.2 Tier 2):**

- Guo, Y., Hašan, M., & Zhao, S. (2018). *Position-Free Monte Carlo Simulation
  for Arbitrary Layered BSDFs.* ACM Transactions on Graphics 37(6), Article
  279 (SIGGRAPH Asia 2018). https://shuangz.com/projects/layered-sa18/
- Xia, M. et al. (2020). Improved importance sampling for the position-free
  layered BSDF approach (referenced via PBR Book 4th ed., Ch. 14 Further
  Reading — confirm full citation before external use).
- Gamboa, L. E., Gruson, A., & Nowrouzezahrai, D. (2020). Efficient
  multi-layer approach without bidirectional sampling (referenced via PBR
  Book 4th ed., Ch. 14 Further Reading — confirm full citation before
  external use).

**Layered materials — precomputed/discretized-matrix family (§4.2 Tier 1, deferred):**

- Jakob, W., d'Eon, E., Jakob, O., & Marschner, S. (2014). *A Comprehensive
  Framework for Rendering Layered Materials.* ACM Transactions on Graphics
  33(4), Article 118.
- Zeltner, T., & Jakob, W. (2018). Layered-material composition work applying
  the adding equations to discretized scattering matrices (referenced via PBR
  Book 4th ed., Ch. 14 Further Reading — exact title/venue not independently
  verified in this pass; confirm before citing externally).
- Belcour, L. (2018). *Efficient Rendering of Layered Materials Using an
  Atomic Decomposition with Statistical Operators.* ACM Transactions on
  Graphics 37(4).

**Layered materials — naive/real-time family (§4.2 Tier 0 precedent):**

- Weidlich, A., & Wilkie, A. (2007). Arbitrarily layered micro-facet surfaces
  via linear transmission-factor combination (GRAPHITE 2007 — exact title
  recalled with moderate confidence; confirm before citing externally).

**Measured BRDF data (§4.3):**

- Matusik, W., Pfister, H., Brand, M., & McMillan, L. (2003). *A Data-Driven
  Reflectance Model.* ACM Transactions on Graphics 22(3). [MERL database —
  historical/foundational reference, superseded as a target representation
  by the entry below.]
- Dupuy, J., & Jakob, W. (2018). *An Adaptive Parameterization for Efficient
  Material Acquisition and Rendering* ("Powitacq"). ACM Transactions on
  Graphics 37(6), Article 274 (SIGGRAPH Asia 2018).
  https://rgl.epfl.ch/publications/Dupuy2018Adaptive — storage cost confirmed
  at ~16 KiB/spectral-channel (isotropic), ~544 KiB (anisotropic); this is the
  representation to study before committing to a table format (§6.5).

**Principled / authoring-convention BSDFs (§2.2):**

- Burley, B. (2012). *Physically Based Shading at Disney.* SIGGRAPH 2012
  Course Notes.
- Academy Software Foundation (2023). *OpenPBR — A New Open Standard for
  Physically Based Shading.* SIGGRAPH 2023.
  https://github.com/AcademySoftwareFoundation/OpenPBR

**Subsurface scattering (out of scope this iteration, §7 by omission — retained for future reference):**

- Christensen, P. (2015). *Physically-Based Subsurface Scattering in
  Production.* SIGGRAPH 2015 Course.

**Neural representations (explicit non-goal, §7):**

- Müller, T. et al. (2022). *Neural BRDFs: Representation, Acquisition, and
  Rendering of Real-World Materials.* ACM Transactions on Graphics 41(6).

**Cross-renderer comparisons (§2.4–§2.10) — source repositories:**
- [Mitsuba 3](https://github.com/mitsuba-renderer/mitsuba3):
  - `include/mitsuba/render/bsdf.h` (BSDF base class, BSDFFlags, BSDFContext)
  - `src/bsdfs/measured.cpp` (Dupuy-Jakob measured BRDF)
  - `src/bsdfs/principled.cpp` (principled BSDF)
  - `src/bsdfs/coating.cpp` (coating adapter).
  - plugin system docs: https://mitsuba.readthedocs.io/

- [LuxCoreRender](https://github.com/LuxCoreRender/LuxCore):
  - `include/slg/materials/material.h` (Material base class),
  - `include/slg/materials/material_types.cl` (GPU opcodes),
  - `include/slg/materials/material_funcs.cl` (GPU VM interpreter),
  - `include/slg/materials/glossycoating.h` (GlossyCoating),
  - `include/slg/materials/mix.h` (Mix).

- [appleseed](https://github.com/appleseedhq/appleseed):
  - `src/appleseed/renderer/modeling/bsdf/bsdf.h` (BSDF base +
    `attenuate_substrate` protocol)
  - `src/appleseed/renderer/modeling/bsdf/bsdfwrapper.h` (CRTP wrapper)
  - `glossylayerbsdf.cpp` (precomputed albedo tables)
  - `energycompensationtables.cpp` (171KB baked table data)
  - `plasticbrdf.cpp` (internal-scattering correction)

- [OpenMoonRay](https://github.com/OpenMoonRay/openmoonray):
  - `lib/rendering/shading/BsdfBuilder.h` (BsdfBuilder pattern)
  - `moonshine/lib/material/dwabase/DwaBase.h` (attribute keys, normal AA)
  - `dso/material/DwaLayer/` (layering)
  - `lib/rendering/shading/Bsdfv.ispc` (ISPC vectorized hot path)

- [NVIDIA Falcor](https://github.com/NVIDIAGameWorks/Falcor):
 - `Source/Falcor/Rendering/Materials/IMaterialInstance.slang` (material
   instance interface),
  - `LayeredBSDF.slang` (Guo et al. random walk)
  - `Scene/Material/MERLMaterialData.slang` (MERL + parametric IS fit)
  - `MERLMixMaterialData.slang` (per-texel BRDF selection)
  - `MaterialSystem.slang` (flat GPU buffer)
  - `StandardMaterialParamLayout.slang` (X-macro layout)

- [Google Filament](https://github.com/google/filament):
  - `shaders/src/surface_brdf.fs` (BRDF evaluation, `FILAMENT_QUALITY` gates),
  - `surface_material.fs` (uber-shader),
  - `libs/filamat/src/shaders/CodeGenerator.cpp` (quality-level define
    generation),
  - `libs/filabridge/include/private/filament/UibStructs.h` (UBO layout),
  - `libs/filamat/include/filamat/MaterialBuilder.h` (material parameter API).
  - PBR theory docs: https://google.github.io/filament/

- [renderling](https://github.com/schell/renderling):
  - `crates/renderling/src/material.rs` (MaterialDescriptor, `#[repr(C)]`
    layout)
  - `material/cpu.rs` (slab allocator, `Hybrid<T>`)
  - `pbr/shader.rs` (dual-use `fragment_impl`),
  - `atlas/` (texture atlas with `crunch` packing)
  - `Cargo.toml` (rust-gpu dependency).
  - crates:
    - https://github.com/schell/craballoc
    - https://github.com/schell/crabslab

**Foundational / cross-cutting:**

- Pharr, M., Jakob, W., & Humphreys, G. *Physically Based Rendering: From Theory
  to Implementation*, 4th ed. Chapter 14 (§14.3 "Scattering from Layered
  Materials") is the direct reference for §2.1/§2.3/§4.2/§4.4.
  https://pbr-book.org/
- pbrt-v4 project (GitHub README / release notes) — source for the
  `UberMaterial` removal and Dupuy-Jakob measured-BRDF adoption facts cited in
  §2.1. https://github.com/mmp/pbrt-v4
- Veach, E. (1997). *Robust Monte Carlo Methods for Light Transport Simulation.*
  Stanford PhD thesis. [MIS foundation — already a running reference elsewhere
  in this project's docs; relevant here for §6.3.]
- Open Shading Language.
  https://github.com/AcademySoftwareFoundation/OpenShadingLanguage [Referenced
  for completeness per §2.2's "custom shader" escape hatch; not otherwise
    discussed in this doc — see the separate, already-deferred shading-graph
    discussion.]

______________________________________________________________________

## 12. Confidence Notes

Per this project's own documentation discipline (e.g., `profiling-testing-
errors-review.md`'s self-graded verdict), an honest accounting of source
confidence for this doc specifically:

- **High confidence, independently verified this pass:** Guo/Hašan/Zhao 2018,
  Dupuy & Jakob 2018 (exact titles, authors, venues, and — for Dupuy/Jakob —
  storage figures confirmed via direct search rather than recalled). All
  cross-renderer architecture facts in §2.4–§2.10 researched via direct GitHub
  code inspection of each renderer's material/BSDF source files,
  cross-referenced against their documentation where available. Renderling's
  slab allocator and `Id<T>` design verified from crate source.
- **High confidence, well-established/widely cited:** Matusik et al. 2003
  (MERL), Jakob et al. 2014, pbrt-v4's `TaggedPointer`/Integrator-exception
  fact, pbrt-v4's `LayeredBxDF` two-interface/random-walk/noisy-estimator facts,
  pbrt-v3 UberMaterial removal. Filament's `FILAMENT_QUALITY` design is
  well-documented in the official materials guide and PBR theory documentation.
- **Medium confidence, recalled rather than independently verified this pass —
  confirm before external citation:** Zeltner & Jakob 2018's exact title/venue,
  Weidlich & Wilkie 2007's exact title/venue, Christensen 2015 and Müller et al.
  2022's exact details (carried forward from the original source material this
  doc synthesizes, not independently re-checked here). MoonRay's `BsdfBuilder`
  and `Bsdfv` ISPC vectorization patterns verified from GitHub source but not
  from the official documentation (print/book references for MoonRay
  architecture are sparse).
- **Medium confidence, verified via GitHub code search but not from official
  documentation:** LuxCore's GPU VM-interpreter design (`MaterialEvalOp`,
  `material_funcs.cl`) — confirmed the opcodes and interpreter loop exist in the
  source but the design is not externally documented in detail. appleseed's
  `AlbedoTable2D`/`energycompensationtables.cpp` — table data confirmed present
  but the exact generation method and accuracy figures not independently
  validated. Falcor's `MERLMaterialData` + `DiffuseSpecularData` parametric fit
  — the data structures are confirmed present in the Slang source but the
  fitting procedure and sampling quality not independently evaluated.
- **Architectural recommendations (§4, §6, §7):** original synthesis for this
  project, not sourced from any single external document — cross-checked for
  internal consistency against this project's actual `Bsdf`/`GpuMaterialNode`/
  `samplestream-refactor.md` shapes where those were available in context.
