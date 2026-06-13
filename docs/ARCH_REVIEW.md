
# Architecture Review: `raytrace-rs` vs Production Renderers

Reference renderers: **[LuxCore](https://github.com/LuxCoreRender/LuxCore)**, **[OpenMoonRay](https://github.com/Autodesk/moonray)**, **[renderling](https://github.com/schell/renderling)**

---

## 1. Material System

### Current Design

```
Material enum (9 variants)  ← pure-mat..
├── Lambertian(Material)
├── Metal(MetalMaterial)          -- GGX conductor
├── Dielectric(DielectricMaterial)
├── DiffuseLight(DiffuseLightMaterial)
├── Isotropic(IsotropicMaterial)
├── Glossy(GlossyMaterial)        -- GGX dielectric
├── Mix { a: Box<dyn Bsdf>, b: Box<dyn Bsdf>, weight }
├── Coated { substrate: Box<dyn Bsdf>, coating: Box<dyn Bsdf> }
└── Custom(Box<dyn Bsdf>)
```

Dispatch: match arms on `Material` → delegate to struct methods. `Bsdf` trait has 10 methods (`sample`, `eval`, `pdf`, `emitted`, `is_emissive`, `is_delta`, `gpu_node`, `clone_box`, `serialize_gpu`). GPU serialization flattens the tree into `Vec<GpuMaterialNode>`. Texture uniform via `Option<Arc<dyn Texture>>`.

### Findings

**✅ Well‑done**

- **Per‑sample delta‑ness via `pdf_kind`**, not `material.is_delta()` — matches how Coated/Mix must work. Fixed last session.
- **`BsdfSample` bundles direction, BSDF×cos, PDF, and pdf_kind** in one struct. Guarantees direction/PDF come from the same sample, structurally preventing the Glossy direction‑PDF mismatch bug.
- **GPU tree serialization** is clean and tested (6 tests in  material/mod.rs). Compositions serialize and reference children by index, forming a valid DAG.
- **Closed‑form Fresnel in Coated::sample()** — direct reflection without delegating to Dielectric (which would double‑Fresnel). Fixed last session.
- **Constructor ergonomics** — `.mix()`, `.coated()`, `lambertian()`, `dielectric()` etc. are intuitive and chainable.

**⚠️ Issues to address**

1. **Coated Lacks MIS Between Coating & Substrate** (LuxCore reference)

   Current `Coated::sample()`:
   ```
   if u < f  → coating reflection (delta, pdf=1, f_cos=1)
   else      → substrate sample, f_cos /= (1-f)
   ```

   This is correct for a perfectly smooth clearcoat (delta coating). But a **rough** clearcoat (non‑delta, e.g. satin finish) needs the coating and substrate lobes combined via MIS, not a hard Fresnel‑split switch. LuxCore's `GlossyCoating` uses:
   ```
   w_coating = 0.5 * (1 + Fresnel(average_value))
   result = coating_sample * w_coating + base_sample * (1-w_coating)
   pdf    = coating_pdf * w_coating + base_pdf * (1-w_coating)
   ```
   Where `w_coating` ensures the coating is never sampled less than 50% of the time regardless of Fresnel — reducing variance.

   **Impact**: Cannot correctly render rough clearcoat (satin, matte clear). Low priority now (coating is always dielectric = perfectly smooth), but will prevent physically correct rough clearcoat later.

2. **Coated `eval()`/`pdf()` Use Fresnel Blend — Correct Only for Delta Coating**

   ```
   f * coating.eval() + (1-f) * substrate.eval()
   ```
   For a delta coating, `coating.eval()` returns 0 everywhere except the exact specular direction, so this collapses to `(1-f)*substrate.eval()` — correct because MIS only runs on non‑delta paths (substrate).

   **But**: for a rough coating, this blend doesn't match `sample()`'s selection probability, making MIS weights wrong and introducing bias. Must fix when rough coating is added.

3. **Coating Fresnel Hard‑coded to IOR = 1.5**

   LuxCore's GlossyCoating takes configurable `index`, `Ks` (Fresnel weight at normal incidence), `Ka` (absorption), `depth`. Hard‑coded IOR is a simplification — adequate for now but limits glass types.

4. **Dielectric Implements Dead `eval()`/`pdf()`**

   Dielectric is always delta. The trait requires `eval`/`pdf`, so they exist but return 0/0. Not harmful, but the trait could document that delta materials must implement these trivially.

5. **Mix Uses Simple Stochastic Selection (Same as LuxCore)**

   Both raytrace-rs and LuxCore use `u < weight` to pick one child per sample. MoonRay uses lobe CDF with one‑sample MIS. The simple approach is adequate — stochastic mix with `eval`/`pdf` blending handles the MIS. But the `eval`/`pdf` blend `(1-w)*a + w*b` means every sample evaluates **both** children, which is wasteful for expensive secondary lobes.

### Comparison: MoonRay's Component BSDF Architecture

MoonRay doesn't use a material enum. Instead, it uses **30+ `BsdfComponent` data classes** (parameter bundles) assembled by `BsdfBuilder`, which manages an **attenuator chain** for automatic energy‑conserving layering:

```
addComponent → OVER flag → stageAttenuator (e.g. OneMinusFresnel)
             → UNDER flag → subsequent lobes scaled by previous transmittance
```

**Worth adopting?** No — overkill for raytrace-rs. The enum dispatch is simpler, equally expressive for the material count (<10 types), and serializes cleanly for GPU. MoonRay's component model is production‑scale for a team of 50+ engineers maintaining hundreds of material configurations. For 6 material types, the enum is right.

---

## 2. 2D/3D Primitives

### Current Design

```
Hittable trait
├── Sphere (static + moving)
├── PlanarPatch<R: Region2D>  ← parametric 2D → 3D
│   ├── QuadRegion, TriRegion, EllipseRegion
│   ├── AnnulusRegion, SuperellipseRegion
│   ├── RoundedRectRegion, PolygonRegion
│   └── FunctionRegion (arbitrary predicate)
├── TransformObject<T: Transform, O: Hittable>
│   ├── Translate { offset }
│   └── RotateY { sin, cos }
├── BvhNode (tree with SAH build)
├── FlatBvh (cache‑friendly flat 64‑byte nodes)
├── ConstantMedium (volumetric scattering)
├── Vec<T: Hittable> + Arc<T: Hittable> (blanket impls)
```

Region2D trait:
```
contains(a,b) → bool
area()        → f64
sample(u,v)   → (f64, f64)
uv(a,b)       → (f64, f64)  // default identity
bounding_box_area() → f64
```

### Findings

**✅ Well‑done**

- **PlanarPatch + Region2D pattern** maps cleanly to parametric surfaces. Unit‑square region (QuadRegion) scales to world via `corner + a·side_a + b·side_b`. Simple, composable, mathematically sound.
- **8 region types** cover a useful variety. FunctionRegion enables arbitrary shapes (text, logos, procedural masks). Good for a learning renderer.
- **FlatBvh** — 64‑byte nodes, iterative traversal, near‑first child ordering. Production‑quality.
- **SAH BVH build** with 32 bins, all 3 axes evaluated, parallel via `rayon::join`. Solid.

**⚠️ Issues to address**

1. **No Triangle Mesh Support** — **Biggest geometric gap**

   All three reference renderers use triangle meshes as their primary geometry:
   - LuxCore: `TriangleMesh` with MBVH for instancing
   - MoonRay: `PolygonMesh`, optional tessellation of subdivision surfaces
   - renderling: indexed vertex buffers + indices on GPU slab

   raytrace-rs cannot load any standard 3D format (OBJ, glTF, FBX, PLY). This dramatically limits scene complexity. Any interesting scene beyond spheres and quads requires a mesh loader.

2. **Transform System Incomplete**

   Only `Translate` and `RotateY` exist. Missing:
   - `RotateX`, `RotateZ` (marked TODO)
   - `Scale` (non‑uniform scaling)
   - Composition helpers (marked TODO)
   - General 4×4 matrix transform

   The `Transform` trait requires implementing `hit`, `bbox`, `ray`, `object_to_world_direction` — heavy per‑transform. For the 3 current transforms this is manageable, but every new transform duplicates the same pattern. A macro (marked TODO) would help.

3. **Arenas & Lifetime: Planned but Un‑implemented**

   The  `docs/arena-refactor-plan.md` is well‑designed:

   | Current | Planned |
   |---------|---------|
   | `Arc<dyn Hittable>` everywhere | `Vec<Box<dyn Hittable>>` in Scene, lifetime‑borrowed BVH |
   | Manual `add_light()` duplication | Auto‑detect emissives by `is_emissive()` |
   | Separate `light_objects` list | Light BVH built from emissive indices in sorted objects |
   | Arc overhead + scattered allocations | GPU‑ready flat storage |

   **Benefit vs effort**: ~361 lines across 8 files. Reduces Arc overhead, eliminates manual light tracking, moves storage toward GPU readiness. Worth doing before GPU pipeline.

4. **Sphere Only Implicit**

   Production renderers support: sphere, box, cylinder, disk, torus, cone, hyperboloid, paraboloid. Adding a `Disk` or `Cylinder` would be useful for scene variety without full mesh support.

---

## 3. Path Tracing Integrator

### Current Design

```
ray_color()  ← per‑bounce loop
├── hit test
├── add emission
├── Russian roulette (bounce ≥ 5, survival = max_attenuation.clamp(0.05, 1.0))
├── material.sample() → BsdfSample
│   ├── Delta path:  accumulated_attenuation *= f_cos, trace sample.wi
│   └── Non‑delta:   MixturePDF [light, surface, surface] → direction
│                     f_cos = material.eval(wo, sampled_direction)
│                     weight = 1 / pdf_val
│                     accumulated_attenuation *= f_cos * weight
└── miss → background
```

MIS uses Veach's one‑sample model: pick a direction from the mixture PDF, evaluate both the material and the PDF at that direction, weight by `1/p_mixture`.

### Findings

**✅ Well‑done**

- **Clean separation of delta vs non‑delta** via `pdf_kind`. Composition materials (Coated/Mix) route correctly per‑sample.
- **Russian roulette** uses throughput‑proportional survival probability with floor clamp — standard and correct.
- **MixturePDF** designs `[light, surface, surface]` gives surface sampling 2/3 weight (surface PDF is usually a better match for the integrand). Simple fixed heuristic — adequate for learning.

**⚠️ Issues to address**

1. **No Shadow Ray** — **Most impactful rendering defect**

   Current path for direct lighting:
   ```
   MixturePDF.generate() → direction
   trace ray in direction
   if hits light → contribution = f_cos * 1/pdf_val
   if hits occluder → continue path (lost light contribution)
   ```

   This is **standard next‑event estimation without a shadow ray** — sampling a direction toward a light and tracing a ray to see if it gets there. If an occluder is in front of the light, the ray hits the occluder and the light contribution is lost.

   **The issue**: The light contribution is blocked by whatever geometry is in front of the light. This is *functionally correct* but *extremely noisy*. Production renderers use a **shadow ray** (short‑circuit occlusion test):
   ```
   sample point on light → if visible (shadow ray hits nothing):
       pdf = light_selection_pdf * light_solid_angle_pdf
       f_cos = material.eval(wo, light_direction)
       contribution = f_cos * light_radiance / pdf
   ```

   Adding a shadow ray reduces variance significantly because:
   - The light PDF evaluates to high values (narrow cone from hit point to light)
   - An occluder within that cone blocks it completely, but the weight `1/pdf_val` explodes if the PDF was high
   - With a shadow ray, you explicitly check visibility before evaluating the BRDF — no variance explosion from occluded lights

   **However**, the current ray‑probe approach is not wrong — pbrt-v4 uses a similar approach (trace ray, check what it hits). The distinction is that LuxCore and others separate "direct light sampling" (shadow ray) from "indirect scattering" (BSDF ray) for better variance control.

   **Recommendation**: Add explicit shadow rays for direct lighting. This is standard practice and dramatically improves convergence for scenes with occluders near lights.

2. **Fixed MIS Weights: `[1/3 light, 2/3 surface]`**

   The `[light, surface, surface]` mixture means surface sampling is always 2× more likely than light sampling. This is a scene‑independent heuristic. LuxCore uses the **Power Heuristic** which adapts naturally: `weights = pdf_i² / sum(pdf_j²)`. The balance heuristic (pbrt) uses `weights = pdf_i / sum(pdf_j)`.

   Neither is "wrong" — the fixed weights are unbiased. But adaptive heuristics reduce variance tailored to the scene. The current approach is a simplification of Veach's one‑sample model where selection probabilities are fixed rather than optimal.

3. **Integrator Bound to `Camera` Struct (TODO: `renderer-abstraction`)**

   `ray_color()` lives inside `Camera`. It uses `self.background`, `self.max_depth` etc. Extracting a `Renderer` trait would enable:
   - Multiple integrators (path tracer, direct only, albedo, normals)
   - Separate GPU kernel entrypoint
   - Clean boundary for testing

   Marked `TODO(renderer-abstraction)` at lines 8 and 385.

4. **Lights: Single Uniform Light Strategy**

   LuxCore has 4 strategies (`Uniform`, `Power`, `LogPower`, `DLSCache`). raytrace-rs always picks a random light object uniformly — fine for <10 lights, but for the `complex_scene` with ~50 area lights, power‑based selection would reduce variance.

5. **No Per‑Depth Bounce Limits**

   LuxCore distinguishes `maxPathDepth.diffuseDepth`, `.glossyDepth`, `.specularDepth`. raytrace-rs has a single `max_depth`. This is a minor feature — adequate for learning.

### Comparison: Rendering Loops

| Aspect | raytrace-rs | LuxCore | MoonRay |
|--------|------------|---------|---------|
| Shadow rays | No | Yes | Yes |
| MIS heuristic | Fixed weights | Power heuristic | Balance heuristic |
| Light strategy | Uniform | Uniform/Power/Log/DLSCache | Per‑layer light sets |
| Integrator container | Camera method | `PathTracer` engine class | `PathIntegrator` engine |
| Bounce control | Single depth | Per‑type (diffuse/glossy/spec) | Per‑type |
| RR threshold | Bounce ≥ 5, clamp(0.05, 1) | Bounce ≥ 3, cap 0.5 | Throughput threshold |
| Volumes | ConstantMedium (simple) | Full volume integration | Decoupled ray marching |
| Bidirectional | No | Hybrid Back‑Forward mode | No |

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

DimCursor { base, offset } — auto‑advancing dimension wrapper
```

### Findings

**✅ Well‑done**

- **Pure deterministic `sample(n, d)`** — same arguments same result everywhere. Enables `Sync`, deterministic reproduction, no state corruption. Matches the production approach.
- **Gray‑code cache** — efficient O(1) advance per sample (just XOR one direction vector per dim). Rebasing on pixel change is amortized across all samples for that pixel.
- **512 dimensions** — adequate for ~60 bounces × ~8 dims = ~480 dims. Tight but not exceeded.
- **DimCursor** — prevents dimension aliasing (the bug that caused Silent Wrong in earlier versions). Clean design.

**⚠️ Issues to address**

1. **No Correlated Multi‑Jittered (CMJ) or Progressive Multi‑Jittered (PMJ) Sequences**

   MoonRay uses PMJ/CMJ for bounces 0‑2 (where correlation matters most), falling back to hash‑based for deeper bounces. CMJ provides better stratification than Sobol at low sample counts — important for progressive rendering where 1‑4 samples per pixel dominate the early preview.

   **Impact**: Visible structured noise at low sample counts compared to CMJ. The Sobol sequence is already high‑quality, so this is a refinement, not a defect.

2. **StratifiedRandomSampler Only Stratifies 2 Dimensions**

   Only dims 0‑1 (pixel AA) are stratified. Everything else falls back to hash. True N‑dimensional stratification would improve convergence for the first few bounces, but the Sobol sampler is already the default and handles this better.

3. **No Blue Noise or Error‑Diffusion Sampling**

   MoonRay ships precomputed blue‑noise sample tables. Blue noise trades low‑frequency noise for high‑frequency (perceptually less visible) noise. Important for real‑time / interactive previews. Could be interesting for raytrace-rs's live preview mode.

4. **DimCursor `base + offset` Overflow After ~4B `next()` Calls**

   Not a practical concern (a ray uses ~480 dims), but worth noting: the `u32` addition can wrap. A `debug_assert!` or `NonZeroU32` for `(max_dim - base)` would catch accidental overflow.

### Comparison: Sampling Systems

| Aspect | raytrace-rs | LuxCore | MoonRay |
|--------|------------|---------|---------|
| Sequence type | Sobol (Gray‑code) | Sobol (direction vectors) | CMJ/PMJ + hash fallback |
| Dims | 512 | Unbounded (direction vector gen) | Precomputed tables |
| Perf model | O(1) via cache | Per‑dim XOR of direction vectors | Table lookup + hash |
| State | Stateless (thread‑local cache) | Per‑thread RNG + pass counter | `SequenceID` hash construction |
| Seed | Per‑pixel deterministic | Per‑pixel + random pass shift | Pixel + sample + purpose hash |
| Stratification | Full Sobol (all dims) | Full Sobol | CMJ for early, hash for late |
| Adaptive sampling | No | Yes (variance‑guided) | No |

---

## 5. TODOs & Future Direction

### All 37 TODOs by Category

| Category | Count | Files |
|----------|-------|-------|
| GPU pipeline preparation | 8 | main.rs(3), camera.rs(1), scene.rs(3) |
| Texture mapping 2D/3D split | 7 |  texture/mod.rs(4),  texture/mapping.rs(3) |
| Preview optimization | 6 | camera.rs(2), main.rs(4) |
| Renderer abstraction | 5 | camera.rs(2), hittable.rs(2) |
| Optional features | 3 | transform.rs(3) |
| Type safety | 3 | vec3.rs(2), hittable.rs(1) |
| Displacement mapping | 1 | hittable.rs(1) |
| **Total** | **37** | 0 FIXMEs, 0 HACKs, 0 XXXs |

**Plus**: Arena refactor (361 lines across 8 files — documented separately in `docs/`)

### Priority‑Ordered Recommendations

**P0 — Immediate impact, low effort**

1. **Arena refactor** (~361 lines, 8 files)
   - Eliminates Arc overhead
   - Automatic light detection (no `add_light()` ceremony)
   - Flat storage for GPU upload
   - Unblocks: cleaner scene construction, material → emissive auto‑detection

2. **Shadow ray** (~50 lines, camera.rs + PDF trait)
   - Direct lighting with explicit visibility test
   - Dramatically reduces noise where occluders sit between hit points and lights
   - The single most impactful rendering quality improvement available

**P1 — High value for medium effort**

3. **Triangle mesh support** (new file + OBJ loader)
   - Unlocks all real‑world scenes
   - Required for any serious scene beyond spheres and quads
   - Start with a simple indexed mesh + OBJ parser

4. **Renderer trait extraction** (~100 lines, new  `renderer.rs`)
   - `TODO(renderer-abstraction)` — move `ray_color` into separate trait
   - Enables: multiple integrators, GPU mirror, cleaner testing
   - Also enables the `TODO(gpu)` at line 434

**P2 — Quality improvements**

5. **Per‑type bounce limits** — add `max_diffuse_depth`, `max_glossy_depth`, `max_specular_depth`
6. **Rough clearcoat** — MIS between coating and substrate lobes (once the coating can be rough)
7. **Power heuristic MIS** — replace fixed `[1/3, 2/3]` weights with adaptive power heuristic

**P3 — Polish**

8. **Complete transform system** — RotateX, RotateZ, Scale, composition macros
9. **Point3/Color3 newtypes** — type‑safety, prevent coordinate/color confusion
10. **Adaptive sampling** — variance‑guided sample allocation per pixel
11. **Texture mapping 2D/3D split** — clean separation of UV vs world‑space mapping
12. **CMJ/PMJ sampler** — correlated multi‑jittered for early bounces

### Gap vs Production Renderers (Not Yet Addressed)

| Gap | Production status | When to tackle |
|-----|------------------|----------------|
| Triangle meshes | All three use as primary geometry | P1 |
| Spectral rendering | LuxCore + MoonRay support spectral | Not until basics are solid |
| Subsurface scattering | MoonRay: 3 models, LuxCore: volumes | Not planned |
| Normal/bump mapping | All three have per‑layer normal mapping | Not planned |
| Bidirectional path tracing | LuxCore: Hybrid Back‑Forward | Not planned |
| Volume rendering | LuxCore: full integration, MoonRay: decoupled marching | ConstantMedium is adequate for learning |
| Displacement mapping | MoonRay: supported | TODO in hittable.rs, not planned |

---

## 6. pbrt-v4 Comparison

pbrt-v4 is the closest architectural cousin to raytrace-rs — both are CPU Monte Carlo path tracers with explicit scene description, unlike LuxCore (production engine) or MoonRay (studio renderer). The comparison is instructive because pbrt-v4 is the **textbook reference** for the techniques raytrace-rs implements.

### Material / BxDF System

**pbrt-v4:** `Material` is a `TaggedPointer<11 material types>` (same dispatch pattern as a Rust enum). Each material has a `using BxDF = SomeBxDF` typedef. `Material::GetBSDF()` instantiates the **single** correct BxDF type. `BSDF` wraps **one** `BxDF` via `TaggedPointer` with a `Frame` for local/world space conversion.

```
Material::GetBSDF() → BSDF { bxdf, shadingFrame }
BSDF::Sample_f()   → bxdf.Sample_f(wo, u, u2, mode, flags)
```

Each `BxDF` (10 types) implements `f()`, `Sample_f()`, `PDF()`, `Flags()`. `Flags()` returns `BxDFFlags`: a bitmask of `{Reflection, Transmission, Diffuse, Glossy, Specular}`.

| Aspect | raytrace-rs | pbrt-v4 |
|--------|------------|---------|
| **Material count** | 9 (6 concrete + 3 composition) | 11 (all concrete) |
| **Dispatch** | `Material` enum match → struct methods | `TaggedPointer` dispatch → template `GetBxDF()` |
| **Composition** | `Box<dyn Bsdf>` in Mix/Coated | No composition — `LayeredBxDF<Dielectric, Diffuse>` is a single BxDF class |
| **BSDF per vertex** | Single `Material` reference → match → child | Single `BxDF` via `TaggedPointer` |
| **Flags** | `PdfKind` (what PDF to use) | `BxDFFlags` (scattering type: refl/trans, specular/glossy/diffuse) |
| **Spectral** | RGB only | `SampledSpectrum` (4 wavelength samples, point-sampled) |

**Key difference — Layered materials: MC random walk vs analytic Fresnel split**

pbrt-v4's `LayeredBxDF<Top, Bottom, twoSided>` uses a **Monte Carlo random walk** through the layers (Guo et al. 2018). For `CoatedDiffuseBxDF` (= `LayeredBxDF<DielectricBxDF, DiffuseBxDF, true>`):

```
Entrance interface → sample transmission → random walk in medium
→ scatter at bottom interface → random walk back → exit
```

The random walk correctly models:
- Rough interfaces (microfacet coating)
- Multiple scattering within layers
- Volumetric absorption/scattering in the coating
- Thin-film interference

raytrace-rs uses the analytic Fresnel-split:
```
if u < f → reflection (delta), else → substrate sample
```

This is correct only for **smooth dielectric coating**. pbrt-v4's approach is more expensive (N random walk bounces per BSDF evaluation) but physically general. raytrace-rs's approach is cheaper (one sample, closed-form) but limited to smooth clearcoat.

**What raytrace-rs should learn**: The `LayeredBxDF` template pattern — parameterizing the top and bottom layers as type parameters — is elegant. A rough-coating mode could use a similar MC walk, but the analytic approach is fine for the current learning scope.

**BSDFSample differences**:

| raytrace-rs `BsdfSample` | pbrt-v4 `BSDFSample` |
|--------------------------|----------------------|
| `wi: Vec3` | `wi: Vector3f` |
| `f_cos: Color3` (already × cos) | `f: SampledSpectrum` (not × cos) |
| `pdf: f64` | `pdf: Float` |
| `pdf_kind: PdfKind` | `flags: BxDFFlags` + `eta: Float` |
| — | `pdfIsProportional: bool` |

pbrt-v4 returns raw BSDF value `f` (not × cos) and applies `AbsDot(wi, n)` in the integrator. raytrace-rs returns `f_cos` (already × cos). Both are correct — just a convention difference.

**`pdfIsProportional`** is interesting: when `true`, the actual PDF is proportional but not equal to the returned value. The integrator handles this by calling `BSDF::PDF()` explicitly for MIS weights. raytrace-rs doesn't have this — its PDF values are always exact.

### Primitives / Shapes

| raytrace-rs | pbrt-v4 |
|------------|---------|
| `Sphere`, `PlanarPatch<R>` (8 regions), `TransformObject<T,O>`, `ConstantMedium` | `Sphere`, `Cylinder`, `Disk`, `Triangle`, `BilinearPatch`, `Curve` |
| Manual transform chaining (Translate, RotateY) | `Transform *renderFromObject` — full 4×4 transforms |
| `Box<dyn Hittable>` / `Arc<dyn Hittable>` | Build-time `pstd::vector<Shape>`, run-time `Primitive` wraps Shape + Material + trans |
| BVH: `BvhNode` + `FlatBvh` | BVH: `BVHAggregate` (LinearBVH in v3) |
| `Region2D` trait for parametric surfaces | No parametric surface trait — quadrics and triangles |

pbrt-v4's key pattern: **`Shape` is a `TaggedPointer<Sphere, Cylinder, Disk, Triangle, BilinearPatch, Curve>`**. Each shape has `Intersect(ray)`, `Sample(u)`, `PDF(ctx)`, `Area()`. The `Sample(ctx, u)` variant gives solid-angle PDF for direct lighting from a point — all shapes implement this.

**Triangle meshes** are pbrt-v4's primary geometry. `Triangle::CreateTriangles(mesh)` returns `pstd::vector<Shape>`, one per face. Meshes are stored in a global `TriangleMesh` list with shared vertex/index buffers. This is the same approach raytrace-rs needs for mesh support — indexed mesh + per-triangle Shapes.

**Transform system**: pbrt-v4 uses full 4×4 `Transform` objects with `renderFromObject` and `objectFromRender` stored as pointers. All shapes store these pointers. This is simpler than raytrace-rs's `Transform` trait + `TransformObject<T,O>` generic — no type parameter per transform operation.

### Path Tracing Integrator

pbrt-v4's `PathIntegrator` is the direct analogue of raytrace-rs's `ray_color()`. Here's the point-by-point:

**Direct Lighting (SampleLd)** — the most instructive comparison:

```
raytrace-rs:                              pbrt-v4:
  MixturePDF[light, surface, surface]       LightSampler.Sample(ctx, u) → light
  sampling_pdf.generate() → direction        light.SampleLi(ctx, uLight, λ) → ls
  ray = scattered                            f = bsdf->f(wo, wi) * AbsDot(wi, n)
  pdf_val = sampling_pdf.value(unit)         if !Unoccluded(intr, ls.pLight) → 0
  weight = 1/pdf_val                         w_l = PowerHeuristic(1, p_l, 1, p_b)
  accumulated_attenuation *= f_cos * weight  L += w_l * ls.L * f / p_l
```

**Critical differences**:

1. **Shadow ray (`Unoccluded`)**: pbrt-v4 traces a shadow ray (`IntersectP` — just check occlusion, no scattering). raytrace-rs doesn't, relying on the ray-probe hitting the light or not. pbrt-v4's approach has **much lower variance** because `light.SampleLi` can sample narrow solid-angle cones from the light position, and the shadow ray is cheap (just boolean occlusion).

2. **Power heuristic**: pbrt-v4 uses `PowerHeuristic(1, p_l, 1, p_b)` = `p_l² / (p_l² + p_b²)`. raytrace-rs uses fixed mixture weights `[1/3, 2/3]`. The power heuristic is adaptive — when `p_l >> p_b` (light sampling is much better), weight approaches 1 for the light technique and vice versa.

3. **Light selection**: pbrt-v4's `LightSampler` is pluggable (bvh, uniform, power). raytrace-rs selects lights uniformly. For scenes with many lights of varying power, power-based selection reduces variance.

4. **No `pdfIsProportional` in raytrace-rs**: pbrt-v4 marks `pdfIsProportional` for BSDF samples where the PDF estimate is approximate (e.g., layered BxDFs), falling back to explicit `bsdf.PDF()` for MIS weights.

**Russian roulette** — essentially identical:
```
raytrace-rs:                               pbrt-v4:
  if bounce >= 5:                            if rrBeta.MaxComponentValue() < 1 && depth > 1:
    survival = max_attenuation.clamp(..)       q = 1 - rrBeta.MaxComponentValue()
    if rr > survival: return                   if sampler.Get1D() < q: break
    accumulated_attenuation /= survival          beta /= 1 - q
```

**Integrator hierarchy**:

pbrt-v4:
```
Integrator (base)
  └── ImageTileIntegrator (tile-based rendering)
        └── RayIntegrator (Li per ray)
              ├── PathIntegrator (MIS)
              ├── VolPathIntegrator (volumes + MIS)
              ├── SimplePathIntegrator (educational, no MIS)
              ├── RandomWalkIntegrator (simple, uniform sampling)
              ├── SimpleVolPathIntegrator (delta tracking)
              └── AOIntegrator (ambient occlusion)
```

raytrace-rs: one `ray_color()` method in `Camera`. Marked `TODO(renderer-abstraction)`.

**The cleaned `Camera.ray_color()`** is functionally closer to `SimplePathIntegrator` with `sampleLights=true, sampleBSDF=true` (the textbook-style path tracer). pbrt-v4's `PathIntegrator` is more advanced (power heuristic, light sampler, regularize, etaScale, surface visible albedo for denoising).

### Sampling

This is where raytrace-rs and pbrt-v4 diverge most in design philosophy:

| Aspect | raytrace-rs | pbrt-v4 |
|--------|------------|---------|
| **API** | Pure: `sample(n, d)` | Stateful: `StartPixelSample()`, `Get1D()`, `Get2D()` |
| **State** | Stateless (`Sync`) | Mutable (per-pixel, per-dimension cursor) |
| **Thread safety** | Thread-local cache + pure fn | Clone per thread |
| **Determinism** | Same `(n,d)` → same value | Same sequence for same pixel + sampleIndex |
| **Dim management** | `DimCursor` — explicit at call site | Implicit (sampler tracks dimension internally) |
| **Primary sequence** | Sobol (512 dims, Gray-code) | Multiple: Sobol, Halton, PMJ02BN, ZSobol, PaddedSobol, Stratified, Independent, MLT |
| **Scrambling** | Digital shift (per-pixel seed) | Per-dimension hash → Owen, FastOwen, PermuteDigits, None |

**The stateful vs stateless tradeoff**:

pbrt-v4's `Get1D()/Get2D()` implicitly advances a dimension counter. The sampler handles stratification internally. The user never sees dimension indices.

raytrace-rs's `sample(n, d)` is explicit — the caller passes both sample index and dimension. This is mathematically pure and trivially `Sync`, but requires `DimCursor` discipline to avoid aliasing. The `DimCursor` was added precisely because manual `d+1` was error-prone.

**Neither approach is architecturally superior.** pbrt-v4's stateful approach is simpler for the caller (no cursor management) but requires per-thread sampler cloning and makes the sampler non-Sync. raytrace-rs's stateless approach enables `Sync` and trivially correct parallel iteration, at the cost of explicit dimension management. For a GPU pipeline, the stateless approach maps naturally to `(threadId, dim)` indexing.

**Sampler diversity**: pbrt-v4 ships 9 sampler types. raytrace-rs has 3 (Sobol, Naive, Stratified). Of these, pbrt-v4's `PMJ02BNSampler` (progressive multi-jittered with blue noise) is the most interesting for raytrace-rs — it provides better stratification at low sample counts than Sobol, which matters for interactive preview.

### The `TaggedPointer` Pattern

pbrt-v4 uses `TaggedPointer` extensively — it's their version of a Rust enum. Every major type hierarchy (`BxDF, Material, Shape, Sampler, Light, Camera, Texture, Medium, Filter`) is a `TaggedPointer`. This is functionally identical to Rust's `enum` dispatch via match arms.

Where pbrt-v4's approach differs from raytrace-rs's enum dispatch:
- **Extension**: Adding a new BxDF type in pbrt-v4 means adding it to the `TaggedPointer<...>` template argument list AND implementing the interface. In raytrace-rs, you add a variant to the enum AND add match arms. Same effort.
- **Template dispatch**: pbrt-v4 uses `Dispatch(lambda)` which expands to `switch(type_tag) { case T: return ptr->method(); }`. This is C++'s version of Rust match dispatch. Equivalent performance.
- **GPU compilation**: pbrt-v4 compiles the same `TaggedPointer` dispatch for CUDA/OptiX via `__device__` annotations. raytrace-rs would compile the same enum dispatch for WGSL via `@switch` — same pattern.

### Summary: Key Lessons from pbrt-v4

| Lesson | What to adopt | When |
|--------|--------------|------|
| **Shadow ray** | `Unoccluded()` check in direct lighting | P0 — major variance reduction |
| **Power heuristic** | Replace fixed `[1/3,2/3]` weights with `p²/(p²₁+p²₂)` | P1 — adaptive MIS |
| **Light sampler** | Pluggable light selection strategy | P2 — needed for many-light scenes |
| **BxDFFlags** | Richer per-sample metadata (refl/trans, specular/glossy) for future use | P3 — minor |
| **Mesh support** | `Triangle` shape + indexed mesh storage | P1 — unlocks real scenes |
| **Shape::Sample(ctx)** | Solid-angle shape sampling from a point | Already present in raytrace-rs via `HittablePDF` |
| **MC layered material** | `LayeredBxDF` random walk for rough coating | P2 — generalization of current Coated |
| **Integrator hierarchy** | `Renderer` trait extraction | P1 — enables multiple integrators |
| **Integrator class hierarchy** | `ImageTileIntegrator` → tile-based rendering | P3 — big refactor, optional |
| **Spectral rendering** | `SampledSpectrum` | Not a priority — RGB is adequate for learning |
| **Sampler stateful API** | Keep stateless — better for GPU | Confirmed: current design is correct |
| **`pdfIsProportional`** | Not needed — exact PDF values from all materials | Confirmed: current approach is sufficient |


## 7. Summary

### Strengths (Keep & Maintain)

- **BsdfSample** struct — direction+PDF coupling structurally prevents a class of bugs
- **Per‑sample delta routing** via `pdf_kind` — correct for compositions
- **Pure Sampler trait** — deterministic, Sync, clean dimension management
- **DimCursor** — prevents dimension aliasing
- **PlanarPatch + Region2D** — clean parametric surface pattern
- **FlatBvh** — cache‑friendly, iterative traversal, production‑quality
- **GPU material tree serialization** — recursive flatten with tests
- **Constructor ergonomics** — chaining `.mix()`, `.coated()` is intuitive

### Issues to Address

| Priority | Issue | Effort | Impact |
|----------|-------|--------|--------|
| P0 | Arena refactor (Arc → Box + lifetimes) | ~361 lines, 8 files | Enables GPU storage, auto lights |
| P0 | Shadow ray for direct lighting | ~50 lines | Major noise reduction |
| P1 | Triangle mesh support | New file + OBJ parser | Unlocks real scenes |
| P1 | Renderer trait extraction | ~100 lines | Clean boundary for GPU |
| P2 | Power heuristic MIS | ~30 lines | Reduced variance |
| P2 | Rough clearcoat (MIS between layers) | ~80 lines | Physical rough coating |
| P2 | Per‑type bounce limits | ~20 lines | Production bounce control |
| P3 | Complete transforms (RotateX/Z, Scale) | ~60 lines | Full transform support |
| P3 | Point3/Color3 newtypes | ~50 lines | Type safety |
| P3 | Adaptive sampling | ~200 lines | Faster convergence |

### Development Direction

The codebase has **strong fundamentals**: clean material dispatch, correct physics (since the last round of fixes), performant BVH, deterministic sampling. The natural progression is:

1. **Scaffold for complexity** (now – 3 months): Arena refactor → meshes → shadow ray → renderer trait
2. **Quality** (3 – 6 months): MIS improvements, type safety, per‑type bounce limits, adaptive sampling
3. **GPU exploration** (6 – 12 months): The existing GPU serialization, the `TODO(gpu)` markers, the pure sampler trait, and the flat BVH all position the codebase well for a WGSL/GPU pipeline. The renderling project demonstrates the CPU‑GPU hybrid pattern (slab allocator, shared shader code, texture atlas) that would be the next architectural step.
