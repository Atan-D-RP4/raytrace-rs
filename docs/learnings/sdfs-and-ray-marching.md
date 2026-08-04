# SDFs, Ray Marching, and Path Tracing

First-principles learning session — 2026-07-20/27. 8 nodes + capstone.

---

## 1. sdf-defined

A signed distance function (SDF) maps any 3D point to a scalar: the distance to the nearest surface. Positive outside, zero on the surface, negative inside.

**Sphere:** `sdf(p) = |p - c| - r`
- `p = (7,0,0)`, center `(10,0,0)`, r=5 → `|(-3)| - 5 = -2` (inside)
- `p = (15,0,0)` → `|5| - 5 = 0` (on surface)
- `p = (12,0,0)` → `|2| - 5 = -3` (inside)

**Box:** `sdf(p) = length(max(|p| - half_size, 0)) + min(max(q.x, q.y, q.z), 0)`

**Key insight:** An SDF isn't sphere-specific. It's a general representation for any continuous shape. The sphere formula `|p| - r` is just one example.

---

## 2. no-analytic-intersect

A sphere's SDF plugged into a ray equation gives a quadratic — solvable in one step. A general SDF plugged into a ray gives a transcendental equation — no closed-form solution. This is why iteration is required.

**Key insight:** The SDF can be *any* continuous function. There is no general "solve `sdf(ray(t)) = 0` for `t`" formula.

---

## 3. sphere-tracing

Advance the ray by the SDF value at the current point — the distance to the nearest surface. The SDF guarantees no step overshoots a surface because the value is the minimum distance to any surface in the scene.

**Algorithm:**
```
t = ray_t.min
for _ in 0..MAX_ITER:
    p = ray.at(t)
    d = sdf(p)
    if d.abs() < EPSILON_HIT (1e-4): return Hit
    if t + d > ray_t.max: return Miss
    t += d
return Miss
```

**Why safe:** SDF value at `p` is the distance to the nearest surface. A sphere of radius `d` around `p` contains no surfaces. Stepping forward by `d` stays inside this empty sphere.

**Worst case:** Near-parallel grazing — the ray hugs the surface, each step advances microscopically. This is the "parallel-surface problem."

**Name:** "Sphere tracing" — not about spheres! Named because each step carves out a sphere of empty space around the current point.

---

## 4. sdf-composition (Constructive Solid Geometry / CSG)

SDFs compose via min/max:
- **Union:** `min(a, b)` — surface of either shape
- **Intersection:** `max(a, b)` — surface of both shapes
- **Subtraction:** `max(a, -b)` — shape a minus shape b

**Why `-b` flips inside-out:** An SDF returns positive outside, negative inside. Negating it flips every sign — the "inside" region becomes positive, and `max` picks it up as "outside" the combined shape.

**Positioning:** Translate the subtracted shape before composing: `max(sphere(p), -cylinder(p - offset))`

**Operator overloading pattern:**
```rust
let shape = sphere(1.0) | box(0.5);     // union  → min(a, b)
let shape = sphere(1.0) & box(0.5);     // intersect → max(a, b)
let shape = sphere(1.0) - box(0.5);     // subtract → max(a, -b)
```

Note: `+` does NOT work for union — `a + b` ≠ `min(a, b)`.

**Key insight:** No other geometry representation makes CSG this simple. Triangles need mesh booleans (expensive, fragile). SDFs compose with one `min` or `max` call.

---

## 5. sdf-normals

Surface normals from SDFs via numerical central differences — 6 SDF evaluations:

```
normal(p) = normalize(
    sdf(p + εx) - sdf(p - εx),
    sdf(p + εy) - sdf(p - εy),
    sdf(p + εz) - sdf(p - εz)
)
```

**Why gradient = normal:** The surface is the level set `sdf(p) = 0`. The gradient of any function at a point on its level set is perpendicular to that level set. For a true SDF (satisfying the eikonal equation `|∇sdf| = 1`), the gradient points directly toward the nearest surface.

**Connection to displacement vectors:** If an SDF returned a vector pointing to the nearest surface instead of a scalar distance, that vector would be parallel to the surface normal. The gradient extracts exactly this direction.

**In raytrace-rs:** normals use forward-AD dual numbers (`Dual<f32, 3>`) instead of central differences — one `eval::<Dual<f32,3>>` call returns the value and all three derivatives exactly. 1 eval vs 6, no epsilon tuning, and the same generic SDF code serves both paths (`eval::<f32>` for marching, `eval::<Dual>` for gradients).

---

## 6. sphere-tracing vs. ray tracing

| | BVH + triangle | Sphere tracing |
|---|---|---|
| Intersection | Exact `t` | Approximate (within epsilon) |
| Cost | O(log N) BVH + 1 test | O(steps) SDF evals |
| Predictability | Bounded, deterministic | Geometry-dependent (5-200+ steps) |
| Surface type | Only explicit meshes | Any continuous function |
| Memory | Vertex buffers, BVH nodes | Zero geometry memory (just code) |
| Normal | Analytic from vertices | Numerical gradient (6 SDF evals) |

**Key trade-off:** Predictability vs. freedom. Ray tracing is exact and bounded but only for explicit geometry. Sphere tracing handles any surface but with variable cost.

**Hybrid approach:** Scene BVH contains both `ShapeObject<MeshShape>` and `ShapeObject<SdfShape>`. BVH traversal is the same — it calls `intersect_shape` on whatever shape is at the leaf. Integrator never knows which.

**Analogy to material system:** Simple materials (Lambertian, Metal) are bounded and deterministic like BVH ray-triangle intersection. Coated material's internal Monte Carlo walk is iterative with unpredictable cost like sphere tracing.

---

## 7. sdf-shadows-ao

**Shadows:** March from surface point `p` toward light `L`. If the SDF drops below `ε_shadow` before reaching the light distance, the point is shadowed. Uses larger epsilon (1e-3 vs 1e-4) and fewer iterations (32 vs 256) than primary visibility.

**Ambient occlusion:** March along the normal from `p`. Track how far you go before hitting something. `ao = min(1.0, occlusion_distance / ao_radius)`. Corners occlude fast (dark), open planes occlude slowly (bright).

**Key insight:** Both effects reuse the sphere-tracing loop — no extra infrastructure. In triangle path tracers, shadows need BVH traversal and AO needs precomputed maps. With SDFs, both emerge from the same marching loop.

---

## 8. sdf-path-tracer-integration (capstone)

**What stays the same (entire integrator):**
- PathTracingIntegrator — calls `intersect()` on `Arc<dyn Intersectable>`
- BSDF evaluation — gets SurfaceInteraction with `p`, `n`, `wo`
- NEE / light sampling — samples lights, fires shadow ray
- MIS weighting — uses BSDF PDF + light PDF
- Russian roulette — uses throughput + depth
- Ray differentials — propagated from hit point
- Film / AOVs — receives color values

**What changes:**
- `Shape3D` impl: new `SdfShape` with sphere tracing in `intersect_shape`
- `occluded()`: shadow march with larger epsilon + fewer iterations
- Normal computation: forward-AD dual gradient — 1 eval returns value + derivatives (exact, no epsilon)
- `area()` / `sample()`: approximate instead of exact

**Key insight:** The integrator is geometry-agnostic. `SdfShape` is just another `Sh` in `ShapeObject<Sh, M>`. Once `p` and `n` are known, the SDF surface behaves identically to any other surface.

---

## Full ray trace through SDF scene

```
1. Camera fires ray through pixel
2. BVH traversal → SdfShape leaf          [vs ray tracing: same BVH]
3. Sphere trace: step by sdf(p)           [sphere tracing]
   └─ SDF defined by composition tree     [sdf-defined, sdf-composition]
   └─ No analytic solve → iterative       [no-analytic-intersect]
4. Hit: sdf(p) < ε_hit                    [sphere tracing: termination]
5. Normal: dual-number forward AD          [sdf-normals: 1 eval]
6. SurfaceInteraction (p, n, wo)          [integration: unchanged]
7. BSDF eval + NEE                        [integration: unchanged]
   └─ Shadow ray: sphere trace p→light    [sdf-shadows]
   └─ AO: march along normal              [sdf-ao]
8. Pixel color → film                     [integration: unchanged]
```

---

## Misconceptions corrected

1. **SDF is not sphere-specific** — the sphere formula `|p| - r` is just one example. An SDF can be any continuous function defining any shape.
2. **Box SDF is not `max((p-r), 0)`** — correct: `length(max(|p| - half_size, 0)) + min(max(q.x, q.y, q.z), 0)`
3. **`+` doesn't work for union** — union is `min(a, b)`, not `a + b`.
4. **Sphere tracing is not about spherical geometry** — named for the sphere of empty space carved at each step.
5. **The SDF's gradient (not the scalar itself) points to the nearest surface** — and at the surface, the gradient is the normal.

## Connections to raytrace-rs

- Implemented in `src/shape/sdf/`: `SdfShape<F: SdfFn>` (`mod.rs`), forward-AD dual numbers (`dual.rs`), object-safe dispatch for custom SDFs (`dispatch.rs`), CSG expression tree (`expr.rs`), space repetition (`impls.rs`)
- `ShapeObject<SdfShape<F>, M>` inherits all integrator integration via existing traits
- Scene BVH treats SDF shapes identically to spheres/quads/meshes
- `occluded()` override implements the shadow march with relaxed epsilon/iterations
- `SdfExpr` CSG via operator overloading: `|` union, `&` intersection, `-` subtraction, plus smooth union
- Transforms are deliberately NOT part of the SDF system — translate/rotate/scale go through the existing `TransformObject`, identical to any other shape
- `SdfRepeat<F: SdfFn>` handles space repetition (a non-injective fold that `TransformObject` can't express)
- Hybrid scenes: meshes + SDF shapes in the same scene BVH

## Implementation status

**Landed:**
- `SdfShape<F: SdfFn>` as a `Shape3D` impl — sphere tracing in `intersect_shape`, shadow march in `occluded()`, bounding box, approximate `area()`/`sample()`
- Gradients via forward-AD dual numbers — 1 eval vs 6 central differences; mean curvature via nested duals
- March hardening: SOR over-relaxation, interior traversal (`was_outside`), self-intersection guard, minimum physical step
- `SdfExpr` CSG tree: Union / Intersect / Subtract / SmoothUnion + `Custom(Box<dyn DynSdfFn>)` escape hatch that preserves gradients in all three eval contexts (`f32`, `Dual<f32, 3>`, nested duals) via `DynEval` dispatch
- `SdfRepeat<F: SdfFn>` wrapper for space repetition
- Transforms through existing `TransformObject` — zero transform variants in the SDF tree

**Direction / future:**
- Struct-per-primitive refactor mirroring the material system: `SdfSphere`, `SdfBox`, `SdfCylinder`, ... as standalone structs with `From` impls into a thin `SdfExpr` enum
- Packet/SIMD evaluation — deliberately deferred until wavefront rendering lands (ray batching changes the eval API shape)
- Voxel SDF adapter for mesh-to-SDF conversion (future)
