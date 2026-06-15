# Starlight Rendering Engine

## Design Specification v3 — Spatial Microkernel

______________________________________________________________________

## Changelog from v1

The core vision is unchanged. These specific problems raised in critique are
resolved:

- **Rasterization is not a search problem** — formalized as a separate execution
  path that uses the BVH for culling, not per-pixel dispatch. Two execution
  paths are now explicitly defined and their interaction documented.
- **Simulation granularity** — `SimulationPass` redesigned as data-parallel grid
  kernels, not per-cell command recording.
- **Portal recursion bounds** — fixed stack depth (32 entries, max portal depth
  16), cycle detection via per-sub-BVH frame counter.
- **Primary visibility arbitration** — explicit rule: raster writes GBuffer for
  primary visibility; ray traversal writes secondary-effects buffer for specular
  reflections, AO, and refractions. `ShadingPass` composites from both.
- **Multi-queue deferral** — start with a single `GRAPHICS | COMPUTE` queue;
  multi- queue is an explicit later optimization, not an assumption.
- **rust-gpu replaces build.rs GLSL emission** — shaders written in Rust,
  compiled to SPIR-V. A shared `no_std` types crate makes the `leaf_types.glsl`
  generation approach obsolete: type tag constants are the same Rust enum used
  in both host and shader, by construction.

______________________________________________________________________

## Changelog from v2

The core implementation plan is unchanged. These conceptual and architectural
clarifications are integrated:

- **Thesis updated** — *"The spatial router is the universal. Domains are
  sovereign. SurfaceInteraction is the ABI."* More precise and more durable than
  v2's formulation. The word "BVH" is removed from the thesis because the thesis
  must survive replacing the BVH.
- **New §0.5: The Spatial Microkernel Model** — The OS/microkernel analogy
  formalized as a design lens, not a metaphor. BVH-as-VFS,
  SurfaceInteraction-as-ABI, sovereign-leaves-as-drivers. This section explains
  why the design is shaped the way it is and where it is heading structurally.
- **Invariant 1 revised** — "Spatial router" replaces "BVH" throughout the
  invariant statement. Implementation remains a BVH. The invariant now survives
  replacing it with hardware RT structures, sparse DAGs, or clipmap hierarchies.
  New Invariant 9 added: `SurfaceInteraction` is an ABI, not a convenience type.
- **Evolution notes added throughout** — Each underspecified abstraction now
  carries an explicit note: current state, trigger condition, and target state.
  These are not deferrals. They are commits: when condition X is true, do Y.
- **`SpatialDomain` trait** — Identified as the architectural target for
  sovereign leaves. `LeafNode` enum remains correct now. The trait is introduced
  at the trigger point documented in §2.4.
- **Portal capability path** — `world_id: u32` is correct now. `WorldHandle`
  capability replaces it when streaming or networked worlds are needed. Trigger
  and migration path documented in §2.12.
- **Simulation tension documented** — The philosophical gap between "spatial
  router is universal" and "simulation domains are external" is named. Two
  resolution paths given with explicit triggers. §2.11.
- **`SurfaceInteraction` evolution path** — The current struct is
  surface-centric by design and correctly so. The `Interaction` /
  `SurfaceInteraction` split from PBRT is the target when volumes need non-
  surface hit records. §4.
- **Material centralization tension** — Named and reasoned. Centralized buffer
  is correct now. The conditions under which it becomes a constraint are
  documented. §2.13.
- **New §1.7–1.13** — OS references (seL4, Plan 9, OSTEP), additional renderer
  references (Embree, Mitsuba 3, Falcor, MoonRay), hardware RT references
  (Vulkan RT, DXR, RTX) added with motivation.
- **New §10: Evolution Roadmap** — All evolution notes collected into a single
  reference table: current → trigger → target for every identified abstraction
  threshold.

______________________________________________________________________

## 0. Thesis

> **The spatial router is the universal. Domains are sovereign.
> SurfaceInteraction is the ABI.**

Starlight is a hybrid real-time renderer where heterogeneous scene geometry —
analytic shapes, SDF fields, triangle meshes, foam-based captures, fractals,
participating media, and portals to recursive sub-worlds — coexists in a single
spatial index. Two execution paths operate over this index simultaneously:

**The ray-driven path** (Starlight core): one GPU compute dispatch per frame,
one spatial traversal per pixel. When a traversal reaches a leaf, that leaf runs
its own intersection function — analytic formula, SDF ray march, foam graph
walk, fractal distance estimator, or a portal that recurses the traversal into a
sub-index. The traversal code is ignorant of leaf contents. A uniform
`SurfaceInteraction` is returned regardless of how the hit was found. Secondary
effects (reflections, AO, soft shadows, refractions) are computed here.

**The raster path**: the spatial index is used for view-frustum culling and LOD
selection, not per-pixel traversal. Culled leaf lists feed traditional Vulkan
draw calls into a deferred GBuffer. Primary visibility comes from here. It is
fast, tile-coherent, and occupies a separate graphics pipeline.

Both paths share the same spatial index, the same material buffer, the same
simulation state buffers, and the same `SurfaceInteraction` semantics. The
`ShadingPass` unifies their outputs under a single PBR lighting model.

The current implementation of the spatial router is a BVH. This is correct and
sufficient. The invariant is the *role*, not the *implementation*.

______________________________________________________________________

## 0.5 The Spatial Microkernel Model

This section formalizes the design lens that makes the architecture coherent. It
is not required reading to implement the engine; it is required reading to
understand why the design is shaped the way it is and where it is heading.

### The OS Stack Analogy

The standard OS request stack:

```
User Process
    ↓
Syscall ABI
    ↓
Kernel
    ↓
VFS
    ↓
Filesystem Driver
    ↓
Physical Device
```

maps directly to Starlight's rendering stack:

```
Integrator / ShadingPass
    ↓
SurfaceInteraction          ← the ABI
    ↓
Traversal Engine            ← the kernel
    ↓
Spatial Router (BVH)        ← the VFS
    ↓
Leaf / Spatial Domain       ← the filesystem driver
    ↓
Geometry Representation     ← the physical device
```

This is not metaphor. It is the same delegation structure, applied to a
different resource (space instead of files).

### SurfaceInteraction Is an ABI

A syscall ABI exists because the kernel cannot depend on what is below it:

```
read(fd, buffer, count) → bytes

The kernel doesn't care whether the fd points to:
  EXT4 / NTFS / tmpfs / procfs / a network filesystem
```

Likewise, `ShadingPass` cannot depend on what produced the hit:

```
intersect(ray) → SurfaceInteraction { position, normal, uv, material_id }

The shading pass doesn't care whether the hit came from:
  triangle / sdf / foam / fractal / portal / volume
```

That is an ABI. The shading system depends on that contract and nothing else.
This is why `SurfaceInteraction` is defined once in `starlight-types`, why
changing it is a breaking change, and why Invariant 9 exists.

The analogy is load-bearing: every design decision that touches
`SurfaceInteraction` should ask "would this break the ABI?" If yes, it must be
versioned carefully or kept out of the shared struct.

### The BVH Is a VFS, Not a Filesystem

People conflate "filesystem" with EXT4. Linux's key insight was separating:

```
VFS (routes requests)   ←→   EXT4 (stores data)
```

The VFS does not store files. It routes requests to whatever driver is
registered for that path. EXT4 is one driver. tmpfs is another. procfs is
another. All expose the same interface; all do completely different things
internally.

Starlight's BVH is the VFS. It does not own geometry. It routes ray queries to
the responsible spatial domain. The BVH's job is:

```
find responsible spatial domain
```

not:

```
intersect triangles
```

This distinction matters when considering future acceleration structure options.
Hardware RT cores, sparse voxel DAGs, clipmap hierarchies — all of these can
play the VFS role. The leaf domains don't change.

### Sovereign Leaves Are Device Drivers

Linux device drivers all implement the kernel ABI (`read`, `write`, `ioctl`,
`mmap`...) but internally behave completely differently. A network driver and a
block driver satisfy the same contract; what happens below the contract is
entirely their business. The kernel doesn't know and doesn't need to know.

Starlight's leaf types are device drivers. They all satisfy:

```rust
trait SpatialDomain {
    fn intersect(ray: Ray) -> Option<SurfaceInteraction>;
    fn bounds()            -> Aabb;
    fn sample(rng: Rng)    -> Vec3;
    fn pdf(point: Vec3)    -> f32;
}
```

Then `TriangleDomain`, `SdfDomain`, `FoamDomain`, `FractalDomain`,
`PortalDomain` are all drivers. The BVH routes the request. The domain handles
it. The traversal engine is ignorant of what the domain does internally.

> **See §2.4 for the implementation path from `LeafNode` enum to `SpatialDomain`
> trait.**

### TLAS/BLAS Is the Weak Form of This

DXR and Vulkan RT discovered the same separation in hardware:

```
Ray → TLAS → BLAS → Triangle
```

The TLAS routes to BLAS instances. The BLAS contains geometry. The hardware
traverses the TLAS; the intersection shader runs on normal shader cores.
Hardware RT cores accelerate the routing — not the intersection — because the
routing is what can be accelerated uniformly. The intersection is domain-
specific.

The Shader Binding Table (SBT) in DXR/Vulkan RT is the dispatch table that maps
BLAS types to intersection shaders. This is exactly what
`match leaf.type_tag { GpuLeafType::TriangleMesh => ..., GpuLeafType::SdfVolume => ... }`
implements in the Starlight traversal shader — in software, for now.

Starlight generalizes this:

```
Ray → BVH → Spatial Domain → Domain-specific traversal
```

The BLAS becomes `TriangleDomain` instead of `TriangleAccelerationStructure`.
The SBT becomes the traversal `match` statement. The migration from software
traversal to hardware RT (§6 Stage 13) is a change of routing mechanism, not a
change of domain interface.

### The Strongest Architectural Insight

Traditional renderers:

```
Renderer owns representation.
(PBRT, Unreal, LuxCore: the renderer decides how geometry is stored)
```

Starlight:

```
Domains own representation.
Renderer owns transport.
```

This is why adding a new leaf type requires no changes to the shading system.
The shading system only speaks `SurfaceInteraction`. It doesn't know what
produced that interaction. The domain owns its internal geometry entirely: its
storage format, its intersection algorithm, its simulation coupling, its
differentiability. The renderer owns traversal and light transport.

This is the payoff of leaf sovereignty. It is also the reason the design is
structurally closer to an OS than to a traditional renderer.

### What This Architecture Actually Is

Not "a renderer built around a BVH."

A **spatial microkernel** — a minimal kernel whose only responsibility is
routing spatial queries to the correct domain and collecting the results through
a uniform ABI. The renderer (path tracing, rasterization, shading) is the first
*service* running on top of this kernel. Simulation is another service. Power
Foam reconstruction is another service. The kernel — traversal + routing

- `SurfaceInteraction` — supports all of them without knowing about any of them.

This framing explains why traditional renderer literature only partially helps:
the problems of traversal delegation, ABI stability, domain sovereignty, and
routing-layer abstraction are operating systems problems. The rendering math
(BSDF evaluation, MIS, path tracing) is solved by renderer literature. The
structural problems require a different vocabulary.

______________________________________________________________________

## 1. Influences

### 1.1 Starlight (inigid, 2025) — the runtime architecture

The Zig + GLSL Vulkan compute shader prototype that proves the concept: ~2,500
lines, one dispatch, one BVH, sovereign leaves (analytic box/sphere, SDF
torus/gyroid, Mandelbulb fractal, portal nesting), GPU thermal diffusion with
real geometry displacement, all in one shader, lit by one sky.

Core ideas taken directly:

- One BVH as the universal spatial address space for rendering and simulation.
- Per-pixel BVH traversal in a compute shader; leaf decides its own
  intersection.
- Portal leaf that transforms the ray and recurses the traversal into a
  sub-hierarchy.
- Simulation cells co-located with rendering leaves, writing to shared GPU
  buffers that leaf intersection reads directly.
- Rendering is spatial search. What you find at each leaf is up to you.

Reference: `reddit.com/r/computergraphics/comments/1skkxrm`

### 1.2 Power Foam (Govindarajan et al., arXiv 2604.24994, 2026) — a sovereign

leaf type

Power Foam is a differentiable 3D scene representation simultaneously amenable
to tile-based rasterization and real-time ray tracing from identical data.

**Foundation (Radiant Foam, ICCV 2025)**: space partitioned into Voronoi cells
defined by learnable site positions. Ray traversal hops from cell to cell
through a Delaunay adjacency graph in constant time per step — no BVH needed
inside the foam. The representation is differentiable: cell boundaries move
continuously as site positions change.

**Power Foam's extension**: replaces unbounded Voronoi cells with a bounded
power diagram (weighted Voronoi with per-site radii). Every cell is clipped to a
sphere bound, giving it a finite screen-space projection for tile rasterization.
The adjacency graph becomes the Čech complex (all pairwise- overlapping spheres)
— a superset of the α-complex, slightly over-connected but cheap to maintain and
correct in output. A dipole face (interior/exterior boundary) acts as
macro-scale geometry. Detail sites provide displacement along the dipole axis. A
spherical Voronoi captures directional radiance.

**As a Starlight leaf type**: when the BVH reaches a `PowerFoamCell` AABB, the
leaf's intersection function walks the Čech complex from cell to cell until it
finds the dipole face. The `SurfaceInteraction` returned is identical to every
other leaf. The BVH above is ignorant of the internal adjacency structure.

**Dual-path capability**: `PowerFoamCell` is the only leaf type that
additionally participates in a raster pass. `PowerFoamRasterPass` projects
sphere bounds as screen-space discs and tile-rasterizes dipole faces into the
GBuffer, alongside any triangle mesh leaves processed by `MeshRasterPass`. Both
raster passes read the same cell data as the traversal shader — no second
representation.

**Differentiability**: Power Foam is the first leaf type optimizable from
observations. Given real images, gradient descent through the foam rendering
reconstructs the scene.

Reference: `powerfoam.github.io`

### 1.3 Niri compositor (YaLTeR) — the Rust structural pattern

Two specific patterns are taken from Niri's source code
(`github.com/YaLTeR/niri`):

**The `niri_render_elements!` macro pattern**: given `VariantName = ConcreteType` pairs, generates a concrete enum with delegated trait impls and
`From<T>` for each variant. This replaces `Box<dyn Trait>` with zero-cost static
dispatch. The engine uses this pattern for both the `LeafNode` enum (leaf type
dispatch) and the `FramePassNode` enum (render graph pass dispatch).

**The `NiriRenderer` supertrait pattern**: a trait alias that bundles capability
bounds under one name, plus an `AsGlesRenderer` escape hatch for direct hardware
access. The engine's `GpuRenderer` trait and `AsVkDevice` escape hatch follow
this exactly.

What Niri does NOT contribute: Niri has no render graph. The render graph is
built fresh.

### 1.4 rust-gpu (Embark Studios) — shaders in Rust

`rust-gpu` (`github.com/EmbarkStudios/rust-gpu`) is a Rust compiler backend that
emits SPIR-V. All GPU shaders in the engine are written as Rust functions in a
dedicated shader crate, compiled to SPIR-V at build time.

The architectural consequence: a `no_std`-compatible `starlight-types` crate
defines `SurfaceInteraction`, `GpuBvhNode`, `GpuMaterial`, `GpuLeafType`, and
all other shared types once. Both the CPU host crate and the GPU shader crate
import this crate. The leaf type tags are a Rust `#[repr(u32)]` enum in
`starlight-types`. The traversal shader's `match leaf.type_tag` uses the same
enum variant values as the host code's `LeafNode` serialization — not because a
build script generates a header, but because it is the same Rust code compiled
to two different targets.

This eliminates the entire `leaf_types.glsl` generation approach from v1. Type
synchronization between CPU and GPU is architectural, not procedural.

### 1.5 renderling (schell) — a rust-gpu renderer in practice

`renderling` (`github.com/schell/renderling`) is a GPU-driven Rust renderer that
uses rust-gpu for all of its shaders. It demonstrates a complete production
workflow for:

- Organizing a multi-crate workspace where shader code and host code share
  types.
- Using `spirv-std` for shader builtins (`Image`, `Sampler`,
  `gl_GlobalInvocationID`).
- Structuring shader entry points as Rust `fn` annotated with
  `#[spirv(compute(...))]`.
- GPU-driven rendering with indirect draw calls dispatched from GPU-visible
  data.
- Bindless resource access patterns under rust-gpu.

renderling is the closest existing reference implementation for the "shaders in
Rust" approach at production scale.

### 1.6 raytrace-rs (Atan-D-RP4) — the existing foundations

The engine's CPU path tracer. Already contains the right building blocks:

- `flat_bvh.rs`: GPU-ready flat BVH with `repr(C)` 64-byte nodes, iterative
  traversal with a 64-entry explicit stack, near-child-first ordering for early
  termination.
- `material/gpu.rs`: `GpuMaterialType` (`#[repr(u32)]`), `GpuMaterialNode`
  (`repr(C)`), `GpuMaterialBuffer` serialization.
- `texture/gpu.rs`: same pattern for textures.
- `hittable.rs`: `HitRecord` with existing `TODO(renderer-agnostic)` calling for
  `SurfaceInteraction`. The refactor is already anticipated.
- `bvh.rs`: Tree BVH using `Arc<dyn Hittable>` — the `Arc<dyn Hittable>` is what
  gets replaced by the `LeafNode` enum.
- `planar/`: Quad, triangle, box, annulus, ellipse, superellipse, rounded rect,
  polygon.
- `transform.rs`, `onb.rs`, `pdf.rs`, `sampler.rs`: acceleration infrastructure.

### 1.7 seL4 Microkernel — Capability Systems

`sel4.systems` — The formally-verified L4 microkernel. The key concept:

```
Object + Capability = Permission to interact
```

No global namespace. No giant central manager. Every object interaction is
mediated by a *capability* — an unforgeable token of authority to perform a
specific operation on a specific object. The kernel grants capabilities;
processes can delegate their capabilities to others; without a capability, no
access is possible.

Applied to Starlight: a portal currently holds `target_world: u32`, an index
into a global world table. This works for fixed, pre-loaded worlds. It breaks
for streaming worlds, procedural worlds, and networked worlds. The capability
model replaces this with `capability: WorldHandle` — the portal holds authority
to enter a world without knowing where the world lives.

This is the conceptual target for the portal evolution documented in §2.12. The
seL4 reference is not for immediate implementation; it is the vocabulary and the
proven-correct model for the architecture that capability-based portals will
follow.

Reference: Klein et al., "seL4: Formal Verification of an OS Kernel," SOSP
2009\. `sel4.systems`

### 1.8 Plan 9 from Bell Labs — Everything Is a Namespace

Pike et al., Bell Labs. The architecture that takes "everything is a file"
seriously as a systems principle, not a slogan. Plan 9's 9P protocol routes all
resource access — filesystem, device, process, network — through the same
uniform namespace interface. The VFS is not an implementation detail; it is the
architecture.

The VFS analogy in §0.5 is sharpest when understood through Plan 9: the routing
layer can encompass geometry, simulation state, portals, and neural fields — all
through the same `SurfaceInteraction` protocol — because the protocol is the
architecture, and the implementations behind it are free to do anything.

### 1.9 Embree — Traversal Separated From Intersection

Intel's Embree (`embree.github.io`) is the most direct reference implementation
of the spatial-router idea at production quality. Embree provides high-
performance BVH construction and traversal, and exposes *intersection callbacks*
— the application provides what to do when a ray reaches a leaf. Embree does not
own geometry. It routes.

This is exactly what Starlight's traversal shader does, on GPU instead of CPU.
Embree did it in 2011. The traversal shader does it in SPIR-V.

Study Embree specifically for:

- Filter functions (the CPU equivalent of custom intersection shaders in DXR)
- The `RTCGeometry` abstraction, which is the CPU-side production equivalent of
  `SpatialDomain`
- `rtcSetGeometryIntersectFunction` / `rtcSetGeometryOccludedFunction` — the
  full/shadow-ray split that Starlight implements as `traverse` / `occluded`

### 1.10 Mitsuba 3 — Heterogeneous Differentiable Rendering

Mitsuba 3 (`mitsuba-renderer.org`) is the production system closest to
Starlight's goals: heterogeneous geometry types, differentiable rendering
through geometry representations (Dr.Jit backend), pluggable integrators that
operate on a uniform interaction type.

Key structural parallels:

| Mitsuba 3 | Starlight |
|---|---|
| `Intersection` | `SurfaceInteraction` |
| `Shape` plugin | `SpatialDomain` |
| `Integrator` plugin | `ShadingPass` |
| Dr.Jit differentiable backend | Power Foam differentiable path |
| `BSDFSample` | `BsdfSample` |
| `SamplingIntegrator` | `ShadingPass` + `TraversalPass` |

Study Mitsuba 3 for the `Shape` → `Intersection` → `BSDF` pipeline and for the
`Medium` integration, which is the reference for the `VolumeInteraction`
evolution path documented in §4.

### 1.11 Falcor — Render Graph Architecture

NVIDIA Research's Falcor (`github.com/NVIDIAGameWorks/Falcor`) is a research
renderer built on a render graph. Its `RenderPass` abstraction closely matches
`RenderNode`. Its resource reflection and barrier model are directly comparable
to `ResourcePool`.

Study Falcor for:

- `RenderPass::reflect()` — resource declaration by name and type (the Falcor
  equivalent of `reads()/writes()`)
- Resource lifetime management across passes
- Shader variable binding patterns under DX12/Vulkan

### 1.12 MoonRay — Production Path Tracer Architecture

DreamWorks Animation's MoonRay (`github.com/dreamworksanimation/moonray`,
open-sourced 2023). Notable for:

- Per-component BSDF architecture (`BsdfComponent`) — the reference
  implementation of the layered material composition that Starlight's `Coated`
  material targets
- Light sampling at production scale (power-weighted selection, `DLSCache`)
- Geometry plugin system — another `SpatialDomain` reference implementation at
  production quality

### 1.13 Hardware Ray Tracing — Vulkan RT, DXR, RTX

The hardware expressions of the sovereign-leaf idea. These are not immediately
used by Starlight, but they prove the architecture is correct at the silicon
level and define the migration target.

**Vulkan Ray Tracing** (`VK_KHR_ray_tracing_pipeline`,
`VK_KHR_acceleration_structure`): TLAS/BLAS separation, shader binding tables,
callable shaders. The SBT is the hardware dispatch table mapping BLAS types to
intersection shaders — exactly what the traversal shader's `match leaf.type_tag`
implements in software. The migration from software traversal to
`vkCmdTraceRaysKHR` (§6 Stage 13) uses this extension.

**DirectX Raytracing (DXR)**: Any-hit shaders, closest-hit shaders, miss
shaders. The shader type hierarchy maps to Starlight's intersection function
variants (`intersect_*` for full hits, `occluded` for shadow rays).

**NVIDIA RTX**: RT cores traverse BVH nodes on dedicated hardware; intersection
testing runs on normal shader cores. Hardware-level proof that the
routing/intersection separation is correct: the silicon separates them. RT cores
accelerate the part of traversal that is geometry-agnostic; the geometry-
specific part runs in programmable shaders. Starlight's software architecture
mirrors this decomposition.

______________________________________________________________________

## 2. Architecture

### 2.1 The Two Execution Paths

This is the most important architectural clarification from v1. Rasterization
and ray traversal are not the same operation and must not be conflated.

```
                    ┌─────────────────────────────────────┐
                    │         Spatial Router (BVH)         │
                    │  universal address space for all     │
                    │  rendering AND simulation            │
                    └────────────┬──────────┬─────────────┘
                                 │          │
          ┌──────────────────────▼──┐   ┌──▼──────────────────────────┐
          │      RASTER PATH        │   │         RAY PATH            │
          │  (primary visibility)   │   │   (secondary effects)       │
          ├─────────────────────────┤   ├─────────────────────────────┤
          │ BvhCullPass             │   │ TraversalPass               │
          │  frustum + occlusion    │   │  compute dispatch           │
          │  → draw lists           │   │  one ray per pixel          │
          │                         │   │  BVH traversal              │
          │ MeshRasterPass          │   │  match leaf.type_tag        │
          │  vkCmdDrawIndexed       │   │  → leaf intersection fn     │
          │  → GBuffer              │   │  → SurfaceInteraction       │
          │                         │   │  → HitBuffer (2ndary hits)  │
          │ PowerFoamRasterPass     │   │                             │
          │  tile splat raster      │   │ Reflections, AO, refractions│
          │  → GBuffer              │   │ soft shadows, caustics      │
          └───────────┬─────────────┘   └─────────────┬───────────────┘
                      │                               │
                      └──────────────┬────────────────┘
                                     ▼
                           ┌──────────────────┐
                           │   ShadingPass    │
                           │  reads GBuffer   │
                           │  reads HitBuffer │
                           │  unified PBR     │
                           │  composites both │
                           └──────────────────┘
```

**The invariant**: the spatial router is queried by both paths, but by different
mechanisms. The ray path walks the BVH per pixel in a compute shader. The raster
path uses the BVH to produce a visible-object list (frustum culling AABB nodes),
then issues traditional draw calls. The raster path does not walk the BVH per
pixel.

### 2.2 The Workspace

Three crates. The boundary between them is the CPU/GPU split:

```
starlight/
├── types/          starlight-types       no_std, shared CPU+GPU
├── shaders/        starlight-shaders     rust-gpu SPIR-V target
└── engine/         starlight-engine      host, Vulkan, ash
```

**`starlight-types`** is `no_std`-compatible. It contains every type that must
be visible in both host code and shader code: `SurfaceInteraction`,
`GpuBvhNode`, `GpuLeafType`, `GpuLeafData`, `GpuMaterialType`,
`GpuMaterialNode`, `GpuTextureType`, `GpuTextureNode`, `Ray`, `LightNode`, and
the math primitives `Vec3`, `Vec4`, `Mat4`, `Aabb`. Alignment and sizing are
`repr(C)` throughout. This crate evolves directly from `raytrace-rs`'s existing
`material/gpu.rs`, `texture/gpu.rs`, and `flat_bvh.rs`.

**`starlight-shaders`** targets `spirv-unknown-vulkan1.2`. It imports
`starlight-types` and uses `spirv-std` for shader builtins. Each leaf
intersection function is a Rust function. The traversal entry point is annotated
`#[spirv(compute(threads(8, 8)))]`. No GLSL, no HLSL, no `build.rs` header
emission.

**`starlight-engine`** is the host. It imports `starlight-types` for the shared
types, loads the SPIR-V compiled from `starlight-shaders`, and drives Vulkan via
`ash`. The `LeafNode` host enum, `RenderGraph`, `EngineState`, `ResourcePool`,
and all Vulkan resource management live here.

### 2.3 The Shared Types Crate — Why It Replaces build.rs

In v1, `build.rs` emitted a `leaf_types.glsl` header from Rust enum ordinals so
that the GLSL `switch` statement matched the Rust enum values. This is correct
but fragile — it is a procedural synchronization, not an architectural one.

With rust-gpu, the same `GpuLeafType` enum is used in both contexts:

```rust
// In starlight-types/src/leaf.rs
// Imported by starlight-engine (host) and starlight-shaders (SPIR-V)
#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GpuLeafType {
    AnalyticSphere  = 0,
    AnalyticAabb    = 1,
    TriangleMesh    = 2,
    SdfVolume       = 3,
    HeightField     = 4,
    Fractal         = 5,
    ConstantMedium  = 6,
    Portal          = 7,
    PowerFoam       = 8,
}
```

The host code serializes leaf nodes with these constants. The shader code does:

```rust
// In starlight-shaders/src/traversal.rs
use starlight_types::leaf::GpuLeafType;

match GpuLeafType::from_u32(leaf.type_tag) {
    Some(GpuLeafType::AnalyticSphere) => intersect_sphere(ray, leaf, leaf_data),
    Some(GpuLeafType::SdfVolume)      => intersect_sdf(ray, leaf, leaf_data),
    Some(GpuLeafType::PowerFoam)      => intersect_foam(ray, leaf, leaf_data),
    Some(GpuLeafType::Portal)         => traverse_portal(ray, leaf, bvh, stack),
    _ => LeafHit::miss(),
}
```

There is nothing to synchronize. The ordinal `AnalyticSphere = 0` is the same
value in both compilation units because it is the same source. Adding a new leaf
type means adding one variant to `GpuLeafType` in `starlight-types` and one arm
to the traversal match in `starlight-shaders`. The compiler enforces
exhaustiveness in both.

### 2.4 The `leaf_nodes!` Macro

The `leaf_nodes!` macro (Niri pattern) operates on the host-side `LeafNode` enum
— the rich CPU type owning the full Rust data for each leaf. `GpuLeafType` is
the thin `repr(u32)` tag shared with the shader.

```rust
// In starlight-engine/src/scene/leaf/mod.rs
leaf_nodes! {
    LeafNode => {
        AnalyticSphere  = AnalyticSphere,
        AnalyticAabb    = AnalyticAabb,
        TriangleMesh    = TriangleMesh,
        SdfVolume       = SdfVolume,
        HeightField     = HeightField,
        Fractal         = FractalParams,
        ConstantMedium  = ConstantMedium,
        Portal          = Portal,
        PowerFoam       = PowerFoamCell,
    }
}
```

Generated:

```rust
pub enum LeafNode {
    AnalyticSphere(AnalyticSphere),
    // ...
}

impl RenderLeaf for LeafNode {
    fn intersect(&self, ray: &Ray, t: Interval) -> Option<LeafHit> {
        match self { /* delegates to each inner type */ }
    }
    fn aabb(&self) -> Aabb {
        match self { /* delegates */ }
    }
    fn gpu_type_tag(&self) -> GpuLeafType {
        match self {
            LeafNode::AnalyticSphere(_) => GpuLeafType::AnalyticSphere,
            // ... one arm per variant, enforced exhaustive
        }
    }
}

impl SimLeaf for LeafNode {
    fn simulation_cell(&self) -> Option<&dyn SimulationCell> {
        match self { /* delegates */ }
    }
}

impl GpuSerialize for LeafNode {
    fn write_bytes(&self, dst: &mut [u8]) { match self { /* per-variant */ } }
    fn byte_size(&self) -> usize { match self { /* per-variant */ } }
}

impl From<AnalyticSphere> for LeafNode { ... }
// ...From<T> for each variant
```

______________________________________________________________________

> **Evolution Note — `LeafNode` enum → `SpatialDomain` trait**
>
> **Now:** `LeafNode` enum with macro-generated static dispatch. Zero-cost,
> exhaustive, compile-time enforced. The compiler tells you when a new leaf type
> is not handled everywhere.
>
> **Why this is correct now:** Dynamic dispatch on the CPU hot path costs a
> vtable lookup per leaf during BVH construction and scene traversal setup. At
> ≤5 leaf types, the enum match is faster and the exhaustiveness guarantee is
> more valuable than the flexibility of a trait. The enum is also trivially
> serializable and requires no lifetime management.
>
> **Trigger:** When the fifth concrete leaf type is implemented and working, or
> when a leaf type needs to live in a separately compiled module (plugin
> architecture, community extensions, dynamic loading from a scene file that
> describes its own leaf types).
>
> **Target:**
>
> ```rust
> // In starlight-engine/src/scene/leaf/domain.rs
> pub trait SpatialDomain: Send + Sync {
>     fn intersect(&self, ray: &Ray, interval: Interval)
>         -> Option<SurfaceInteraction>;
>     fn bounds(&self) -> Aabb;
>     fn sample(&self, rng: &mut Rng) -> Vec3;
>     fn pdf(&self, point: Vec3) -> f32;
>     fn gpu_type_tag(&self) -> GpuLeafType;
>     fn gpu_serialize(&self, dst: &mut [u8]);
>     fn gpu_byte_size(&self) -> usize;
> }
> ```
>
> The `LeafNode` enum becomes one `impl SpatialDomain` — the static dispatch
> version. Plugin leaf types provide their own `impl SpatialDomain`. The BVH
> holds `Vec<Box<dyn SpatialDomain>>`. The macro-generated enum remains
> available as the preferred path for the built-in leaf types; the trait adds
> the escape hatch.
>
> **Migration:** Introduce the trait alongside the enum (not replacing it). Add
> a blanket `impl SpatialDomain for LeafNode`. Validate identical output.
> Migrate incrementally — built-in leaves stay in the enum; plugin leaves use
> the trait directly.

______________________________________________________________________

### 2.5 The `render_pass_nodes!` Macro

Same structural pattern applied to the render graph's pass collection:

```rust
render_pass_nodes! {
    FramePassNode => {
        Simulation      = SimulationPass,
        BvhCull         = BvhCullPass,
        MeshRaster      = MeshRasterPass,
        FoamRaster      = PowerFoamRasterPass,
        BvhBuild        = BvhBuildPass,
        Traversal       = TraversalPass,
        Shading         = ShadingPass,
        PostProcess     = PostProcessPass,
    }
}
```

Generated: `FramePassNode` enum + `impl RenderNode` delegating `reads()`,
`writes()`, `record()` + `From<T>` for each variant.

The render graph holds `Vec<FramePassNode>`, performs topological sort, and
inserts `vkCmdPipelineBarrier` calls between passes with conflicting resource
access.

### 2.6 The `GpuRenderer` Trait

Niri's `NiriRenderer` supertrait pattern adapted for Vulkan:

```rust
// In starlight-engine/src/engine/renderer.rs
pub trait GpuRenderer:
    VulkanDevice
    + HasCommandBuffer
    + HasDescriptorPool
    + HasMemoryAllocator
{
    type Error: std::error::Error + Send + Sync + 'static;
}

// Escape hatch — Niri's AsGlesRenderer analog
pub trait AsVkDevice {
    fn vk_device(&self)   -> &ash::Device;
    fn vk_physical(&self) -> vk::PhysicalDevice;
    fn vk_queue(&self)    -> vk::Queue;
}
```

All `unsafe` Vulkan calls are confined to pass `record()` implementations and
this escape hatch. No `unsafe` in the render graph orchestration layer.

### Buffer Layout Convention

**Rule: buffer layout types in `starlight-types` use primitive arrays `[f32;
N]`, not `glam::Vec3`.**

GLSL `vec3` has 16-byte alignment in storage buffers (std140/std430). `[f32; 3]`
compiles to SPIR-V `OpTypeArray float 3` with 4-byte element alignment — no
special `vec3` rule applies. Using `[f32; 3]` for buffer layout types eliminates
the alignment mismatch between CPU `repr(C)` structs and GPU SSBO reads.

The convention:

- **Buffer layout types** (`LightNode`, `SurfaceInteraction`, `GpuBvhNode`, leaf
  data structs): use `[f32; 3]` for 3-component fields.
- **Shader computation code**: convert to `glam::Vec3` at read time
  (`Vec3::from(array)`). Use `Vec3` freely in math.
- **`bytemuck::Pod + Zeroable`**: derive on all buffer layout types.

`scalarBlockLayout` is enabled as a device feature (belt-and-suspenders) because
it enables ergonomic future use and costs nothing. It is not load-bearing for
correctness.

### 2.7 The Render Graph

Built fresh following Frostbite's frame graph pattern (GDC 2017). Falcor
(`github.com/NVIDIAGameWorks/Falcor`) is the production reference for this
pattern — see §1.11.

```rust
// In starlight-engine/src/graph/mod.rs
pub trait RenderNode {
    fn reads(&self)  -> &[ResourceHandle];
    fn writes(&self) -> &[ResourceHandle];
    fn record(&self, cmd: &mut CommandRecorder, pool: &ResourcePool);
}

pub struct RenderGraph {
    nodes: Vec<FramePassNode>,
}

impl RenderGraph {
    pub fn compile(&self) -> ExecutionPlan {
        // 1. Build DAG: edge from A→B if A writes a resource that B reads
        // 2. Topological sort (Kahn's algorithm)
        // 3. For each adjacent pair with write→read on same resource:
        //    emit BarrierDescriptor { src_stage, dst_stage, src_access, dst_access }
        // 4. Single GRAPHICS|COMPUTE queue — no multi-queue for now
    }

    pub fn execute(
        &self,
        plan: &ExecutionPlan,
        cmd: &mut CommandRecorder,
        pool: &ResourcePool,
    ) {
        for step in &plan.steps {
            if let Some(b) = &step.barrier { b.record(cmd); }
            step.node.record(cmd, pool);
        }
    }
}
```

**Single-queue constraint**: all passes record into a single command buffer
submitted to one `VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT` queue.
`FoamRasterPass` uses dynamic rendering (`VK_KHR_dynamic_rendering`, core in
Vulkan 1.3) — no pre-created `VkRenderPass` objects.

Multi-queue (compute queue parallel to graphics queue) is deferred. When added,
the render graph barrier model extends to queue family ownership transfers and
timeline semaphores, but the `reads()`/`writes()` interface on `RenderNode` does
not change.

### 2.8 The Traversal Shader (rust-gpu)

```rust
// In starlight-shaders/src/traversal.rs

use spirv_std::{spirv, Image, glam::UVec3};
use starlight_types::{
    bvh::GpuBvhNode,
    leaf::{GpuLeafType, GpuLeafData},
    material::GpuMaterialNode,
    interaction::SurfaceInteraction,
    ray::Ray,
};

#[spirv(compute(threads(8, 8)))]
pub fn traversal_main(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(descriptor_set = 0, binding = 0, storage_buffer)] bvh:         &[GpuBvhNode],
    #[spirv(descriptor_set = 0, binding = 1, storage_buffer)] leaf_types:  &[u32],
    #[spirv(descriptor_set = 0, binding = 2, storage_buffer)] leaf_data:   &[u8],
    #[spirv(descriptor_set = 0, binding = 3, storage_buffer)] materials:   &[GpuMaterialNode],
    #[spirv(descriptor_set = 0, binding = 4, storage_buffer)] sim_buffers: &[f32],
    #[spirv(descriptor_set = 0, binding = 5)] camera_ubo:                  &CameraParams,
    #[spirv(descriptor_set = 0, binding = 6, storage_image)] output:
        &Image!(2D, format=rgba32f, sampled=false),
) {
    let pixel = id.xy();
    let ray = generate_primary_ray(pixel, camera_ubo);
    let hit = traverse_bvh(ray, bvh, leaf_types, leaf_data, sim_buffers);
    let color = match hit {
        Some(interaction) => shade_secondary(&interaction, materials),
        None              => sample_sky(ray.direction),
    };
    unsafe { output.write(pixel, color.extend(1.0)); }
}
```

**Sky model**: Initially `sample_sky()` returns a constant or gradient
background. For physically based rendering, the sky is promoted to a `SkyLight`
— a large distant sphere with an HDR environment map sampled as a light source,
added to the light buffer at scene load.

```rust
fn traverse_bvh(
    ray: Ray,
    bvh: &[GpuBvhNode],
    leaf_types: &[u32],
    leaf_data: &[u8],
    sim: &[f32],
) -> Option<SurfaceInteraction> {
    let mut stack = [0u32; 32];
    let mut stack_top = 0usize;
    let mut closest: Option<SurfaceInteraction> = None;
    // ... standard BVH descent ...

    // At leaf:
    let leaf_type = GpuLeafType::from_u32(leaf_types[leaf_idx]);
    let interaction = match leaf_type {
        Some(GpuLeafType::AnalyticSphere) => intersect_sphere(ray, leaf_idx, leaf_data),
        Some(GpuLeafType::AnalyticAabb)   => intersect_aabb(ray, leaf_idx, leaf_data),
        Some(GpuLeafType::TriangleMesh)   => intersect_trimesh(ray, leaf_idx, leaf_data),
        Some(GpuLeafType::SdfVolume)      => intersect_sdf(ray, leaf_idx, leaf_data),
        Some(GpuLeafType::HeightField)    => intersect_heightfield(ray, leaf_idx, leaf_data, sim),
        Some(GpuLeafType::Fractal)        => intersect_fractal(ray, leaf_idx, leaf_data),
        Some(GpuLeafType::ConstantMedium) => intersect_medium(ray, leaf_idx, leaf_data),
        Some(GpuLeafType::Portal)         => traverse_portal(
                                                ray, leaf_idx, leaf_data,
                                                bvh, leaf_types, leaf_data, sim,
                                                &mut stack, &mut stack_top),
        Some(GpuLeafType::PowerFoam)      => intersect_foam(ray, leaf_idx, leaf_data),
        None                              => None,
    };
    // update closest...
    closest
}

/// Shadow ray — short-circuit occlusion test.
/// Returns true if any leaf is hit before t_max.
/// No SurfaceInteraction computed. Early termination on first hit.
/// Cheaper than full traversal: no normal, no UV, no material lookup.
fn occluded(
    ray: Ray,
    bvh: &[GpuBvhNode],
    leaf_types: &[u32],
    leaf_data: &[u8],
    t_max: f32,
) -> bool {
    // Same BVH descent, returns true immediately on any leaf hit.
}
```

### 2.9 Primary Visibility Arbitration

The `ShadingPass` reads two buffers:

- **GBuffer** (from `MeshRasterPass` + `PowerFoamRasterPass`): albedo, normal,
  depth, material ID for primary visibility of rasterizable objects.
- **HitBuffer** (from `TraversalPass`): `SurfaceInteraction` for secondary ray
  hits — specular reflections, AO samples, refraction rays, shadow rays.

**The arbitration rule:**

> Primary visibility is always the GBuffer. The raster path writes the surface
> seen by the primary camera ray. The `TraversalPass` does not compete for
> primary visibility; its output is used only for secondary effects. For pixels
> with no raster coverage (SDF-only leaves, fractal leaves, portals), the
> `TraversalPass` also provides primary visibility via the HitBuffer. The
> `ShadingPass` selects: if GBuffer depth for a pixel is valid, use GBuffer as
> primary; otherwise fall back to HitBuffer primary.

For secondary effects, the `ShadingPass` traces a reflection ray or AO ray for
pixels whose GBuffer material has specular response. These hits come from the
HitBuffer and are composited over the GBuffer primary using a weighted blend.

**Known optimization:** `TraversalPass` currently runs for all pixels
unconditionally (see §7). For fully-rasterized frames, this wastes compute. A
later optimization pass reads GBuffer depth and dispatches traversal only for
pixels with no raster coverage and for pixels needing secondary effects. This is
tracked as a Stage 7 optimization, not deferred indefinitely.

### 2.10 Power Foam Dual-Path Integration

`PowerFoamCell` is the only leaf type participating in both execution paths:

```
BvhCullPass:
  AABB cull → PowerFoamCell AABB visible → add to foam draw list

PowerFoamRasterPass:
  for each PowerFoamCell in draw list:
    project sphere bounds → screen-space disc tiles
    rasterize dipole faces as splats
    write → GBuffer (albedo, world-normal, depth, material_id)

TraversalPass:
  BVH descent → LEAF_POWER_FOAM
  intersect_foam(ray, leaf_data):
    walk Čech complex adjacency from entry sphere
    find closest dipole face along ray
    apply detail-site displacement
    return SurfaceInteraction
  write → HitBuffer (for secondary hits only)
```

Both paths read `PowerFoamCell` data from the same GPU buffer. Since both are
reads, the render graph can insert a `NONE` barrier between them. The
`ShadingPass` reads both outputs.

For the foam adjacency walk: start with brute-force iteration over all dipole
faces in the cell's data block. Add graph-walk traversal only after profiling
confirms brute-force is the bottleneck. 128 sites per cell is a reasonable
starting threshold estimate; profile before committing to that number.

### 2.11 Simulation — Data-Parallel Kernels

**The v1 problem**: `SimulationPass` iterating all `SimLeaf`s and calling
`step()` on each would produce one compute dispatch per cell, creating CPU-side
overhead proportional to cell count.

**The v2/v3 design**: simulation cells are typed grid kernels. A
`ThermalDiffusionCell` is not one cell in the BVH — it is a simulation domain
covering a region of space and referenced by multiple BVH leaves. The
`SimulationPass` dispatches one kernel per simulation type across all active
domains:

```rust
pub struct SimulationPass {
    thermal_domains: Vec<ThermalDomainHandle>,
    fluid_domains:   Vec<FluidDomainHandle>,
}

impl RenderNode for SimulationPass {
    fn record(&self, cmd: &mut CommandRecorder, pool: &ResourcePool) {
        // One dispatch for ALL thermal domains
        if !self.thermal_domains.is_empty() {
            cmd.dispatch_thermal_diffusion(
                pool.thermal_params_buffer(),
                pool.thermal_state_buffer(),
                self.thermal_domains.len() as u32,
            );
        }
        // One dispatch for ALL fluid domains
        if !self.fluid_domains.is_empty() {
            cmd.dispatch_fluid(/* ... */);
        }
    }
}
```

This is O(simulation_types) dispatches per frame, not O(simulation_cells).

______________________________________________________________________

> **Architectural Tension: Simulation Outside the Spatial Model**
>
> The thesis states: "The spatial router is the universal." Yet `ThermalDomain`
> and `FluidDomain` live outside the spatial router — referenced by ID from leaf
> data structs and read via SSBO offsets. Simulation domains are not BVH leaves;
> they are external services.
>
> This tension is real and unresolved by design. Two resolution paths:
>
> **Path A — Pragmatic (current):** Simulation domains remain external. They are
> services that geometry leaves consume; they do not need to be queryable by
> rays. A `HeightField` reads temperature to compute displacement, but rays do
> not intersect temperature fields. This is correct for thermal diffusion, fluid
> displacement, and fracture propagation.
>
> **Path B — Principled:** Simulation domains become `SpatialDomain` leaves that
> participate in BVH traversal. They do not return `SurfaceInteraction`
> directly, but they can be spatially queried — "what is the temperature at this
> point?" becomes a ray query into a `ThermalDomain` leaf. This is required for
> physically-based volume rendering where the medium is both a simulation domain
> and an intersectable geometry.
>
> **Trigger for Path B:** When a simulation domain must itself be intersectable
> — participating media (fog, smoke, fire) where the medium is queried along the
> ray path, not just at a leaf surface. At that point, `ConstantMedium`
> integration and volume rendering require the simulation domain to be a
> traversable `SpatialDomain`. The `VolumeInteraction` evolution (§4) and Path B
> here are the same trigger.
>
> Path A is correct through Stage 9. Re-evaluate at Stage 10 when
> `ConstantMedium` is fully implemented and volume rendering requirements are
> concrete.

______________________________________________________________________

### 2.12 Portal Recursion — Bounded Stack and Cycle Detection

The existing `flat_bvh.rs` already uses a 64-entry explicit stack for BVH
traversal. Portal recursion extends this stack model.

**Stack discipline**: the traversal stack holds `TraversalState` entries —
either a BVH node index in the current world, or a portal frame boundary marker.
Total entries: 32 (conservative, sufficient for any practical scene depth).

**Portal frame marker**:

```rust
// In starlight-types/src/bvh.rs
pub struct PortalFrame {
    pub sub_bvh_root:  u32,
    pub ray_transform: Mat4,
    pub scale_factor:  f32,
    pub world_id:      u32,
}
```

**Maximum portal depth**: 16 nested portals. When the stack would exceed 16
portal frame markers, the portal leaf returns `LeafHit::miss()`.

**Cycle detection**: each sub-BVH root node carries a `last_visited_frame`
counter. In the traversal shader, before entering a portal:

```rust
if portal_frame.world_id == current_world_id {
    return LeafHit::miss();  // direct self-portal cycle
}
if bvh_roots[portal_frame.sub_bvh_root].last_frame == current_frame {
    return LeafHit::miss();  // already visited this world this frame
}
```

This prevents infinite loops. Cost: one buffer read per portal traversal.

______________________________________________________________________

> **Evolution Note — `world_id: u32` → `WorldHandle` Capability**
>
> **Now:** `PortalFrame.world_id: u32` is an index into a global table of worlds
> loaded at startup. This is correct for a fixed set of worlds.
>
> **Why this is correct now:** A pre-populated table is the simplest model. All
> worlds are known at scene load. No lazy loading. No network. No procedural
> generation. Cycle detection uses the same `u32` as the identity check.
>
> **Trigger:** When worlds need to be loaded on demand — streamed from disk or
> network, procedurally generated at traversal time, or defined by an external
> system that didn't exist at scene load.
>
> **Target (seL4 capability model, §1.7):**
>
> ```rust
> // WorldHandle is an opaque authority token — unforgeable, delegatable
> pub struct WorldHandle(Arc<dyn WorldLoader>);
>
> pub trait WorldLoader: Send + Sync {
>     // Called when a ray first enters this portal.
>     // May block briefly on first load; subsequent calls return cached.
>     fn load(&self) -> Result<Arc<BvhRoot>, WorldError>;
>
>     // Stable identity for cycle detection, independent of load state.
>     fn world_id(&self) -> u64;
> }
> ```
>
> The portal holds a `WorldHandle`. It does not know where the world lives, how
> it was created, or whether it is cached. It only knows that it has authority
> to enter. Cycle detection uses `WorldLoader::world_id()` — a stable 64-bit
> identity that does not require the world to be loaded.
>
> **On GPU:** The `WorldHandle` is not directly representable in SPIR-V. The
> migration requires the traversal shader to request worlds by `world_id` into a
> GPU-side world table that the host populates asynchronously. The shader marks
> un-loaded worlds as `LeafHit::miss()` and the host queues loading for the next
> frame. This is the standard streaming architecture.
>
> **When to implement:** Stage 12+. Not before worlds need to be dynamic.

______________________________________________________________________

### 2.13 The Material / BSDF System

The material system is geometry-agnostic: every leaf returns
`SurfaceInteraction`, and the `ShadingPass` evaluates the material referenced by
`material_id` without knowing the leaf type.

#### Material Dispatch

The `Material` enum dispatches to 9 concrete material types (6 scattering + 3
composition). Each material implements `sample()`, `eval()`, `pdf()`, and
`gpu_node()`. The GPU serialization flattens the material tree into a flat
`Vec<GpuMaterialNode>` buffer, indexed by `material_id` in `SurfaceInteraction`.

On CPU: `Material` enum match → struct methods. On GPU: `GpuMaterialType`
(`#[repr(u32)]`) tag → `match` in the shading shader → per-type BSDF evaluation.

#### BSDF Evaluation Operations

Every material implements three core operations:

1. `sample(wo, rng) → BsdfSample`: sample a direction, return direction +
   BSDF×cos + PDF + pdf_kind.
2. `eval(wo, wi) → Color3`: evaluate the BSDF at a direction pair. Returns `f ×
   |cos θ_i|`.
3. `pdf(wo, wi) → f64`: evaluate the PDF for a given direction pair.

`BsdfSample` carries `pdf_kind: PdfKind` indicating delta or non-delta, enabling
the integrator to route paths differently without calling `material.is_delta()`.

#### Layered Materials

`Coated { substrate, coating }` uses an analytic Fresnel split for smooth
dielectric clearcoat. For rough clearcoat, a `coating_roughness: f32` field and
GGX NDF evaluation are needed. pbrt-v4's `LayeredBxDF` uses a Monte Carlo random
walk (Guo et al. 2018) — deferred as a future extension.

#### GPU Material Serialization

```
GpuMaterialNode {
    material_type: GpuMaterialType,  // #[repr(u32)] tag
    data:          GpuMaterialData,  // repr(C) per-type data
    child_a:       u32,              // index into buffer (composition)
    child_b:       u32,              // index into buffer (composition)
}
```

Composition materials (Mix, Coated) reference children by index. The GPU shader
reads this buffer and dispatches on `material_type`. Serialization is tested (6
tests in `material/mod.rs`). Migration to `starlight-types` is a move, not a
rewrite.

______________________________________________________________________

> **Material Centralization Tension**
>
> The design asserts "leaf sovereignty" while `material_id → global material
> buffer` is centralized. This is a known and intentional tradeoff, not an
> oversight.
>
> **Why centralized is correct now:**
>
> - Materials are shared across leaf types (two `TriangleMesh` leaves sharing
>   one material is both common and correct)
> - The GPU shading shader reads from a flat SSBO; there is no viable
>   alternative for GPU dispatch
> - The `ShadingPass` reads `material_id` from `SurfaceInteraction` — neither
>   leaf type nor shading pass knows about the other's internals
>
> **The constraint this creates:** A leaf whose material does not fit the shared
> BSDF model — a neural radiance field, a Power Foam cell with spatially-varying
> directional radiance, or a volumetric emission field — must still reduce to a
> `material_id` that indexes the shared buffer. The shared buffer must grow to
> accommodate these cases, or a separate buffer must be added.
>
> **Alternative architecture (not now):** `SurfaceInteraction` grows an inline
> `material_data: [u8; 32]` payload that the leaf writes and the shading pass
> reads. This enables per-leaf-instance material parameters without a shared
> buffer lookup. Tradeoff: larger `SurfaceInteraction` struct; loss of material
> sharing; more complex shading dispatch.
>
> **Trigger for reconsideration:** When a leaf type needs material parameters
> that vary per-instance in ways not capturable by a `material_id` index — for
> example, Power Foam's spherical Voronoi directional radiance (which is
> different per foam cell and too large to live in the shared material buffer).

______________________________________________________________________

### 2.14 Direct Lighting — Shadow Rays and MIS

Direct lighting is the most impactful rendering quality subsystem.

#### The Direct Lighting Algorithm

Following pbrt-v4's `SampleLd` pattern:

1. Sample a light from the light buffer
2. Sample a point on the light surface
3. Evaluate the BSDF toward the light point: `wi = normalize(light_point -
   hit_point)`, `f = material.eval(wo, wi)`
4. Trace a shadow ray: `visible = occluded(shadow_ray)` (boolean, no
   `SurfaceInteraction`)
5. Compute MIS weight (power heuristic): `weight = p_light² / (p_light² +
   p_bsdf²)`
6. Accumulate: `if visible: L += f * light.radiance * weight / p_light`

#### Shadow Ray Execution

Shadow rays are short-circuit occlusion tests — boolean, no normals, no UVs, no
material lookup, early termination on first hit. The `TraversalPass` exposes two
entry points:

- `traverse(ray) → Option<SurfaceInteraction>` — full traversal
- `occluded(ray) → bool` — shadow ray, early termination

Both share the same BVH traversal code. Both are expressed as Rust functions in
`starlight-shaders/src/traversal.rs` sharing the traversal loop.

#### MIS Strategy

The power heuristic:

```
weight_i = pdf_i² / Σ(pdf_j²)
```

When light sampling has a much higher PDF than BSDF sampling, the power
heuristic gives most weight to light sampling — correct because light sampling
is more efficient in this regime. This is the standard Veach MIS approach used
by pbrt-v4, LuxCore, and MoonRay.

### 2.15 Bounce Control and Russian Roulette

#### Per-Type Bounce Limits

```
max_diffuse_depth  = 5
max_glossy_depth   = 8
max_specular_depth = 12
```

The bounce type is determined by `BsdfSample.flags`. Standard practice — LuxCore
and MoonRay both use per-type limits.

#### Russian Roulette

After the minimum bounce threshold (5):

```rust
if bounce >= 5 {
    let survival = clamp(max(throughput.r, throughput.g, throughput.b), 0.05, 1.0);
    if random > survival { terminate path }
    throughput /= survival;  // unbias
}
```

The floor clamp (0.05) prevents over-eager termination in dark regions.

### 2.16 Light Buffer — Storage and Sampling

#### Light Discovery

Lights are not declared separately. At scene load, the host walks `LeafNode`
list and collects emissive leaves:

```rust
Scene::build():
    for (i, leaf) in leaf_nodes.iter().enumerate():
        if leaf.material().is_emissive():
            let r = leaf.material().emitted();
            light_buffer.push(LightNode {
                leaf_index: i as u32,
                area:       leaf.aabb().surface_area(),
                radiance:   [r.x, r.y, r.z],
                luminance:  0.2126 * r.x + 0.7152 * r.y + 0.0722 * r.z,
            })
```

#### LightNode Layout

```rust
// In starlight-types/src/light.rs
// [f32; 3] convention — see Buffer Layout Convention in §2.6
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightNode {
    pub leaf_index: u32,       // offset 0
    pub area:       f32,       // offset 4
    pub radiance:   [f32; 3],  // offset 8
    pub luminance:  f32,       // offset 20
}
// 24 bytes, bytemuck::Pod, zero-copy upload via bytemuck::cast_slice
```

#### Light Sampling

Initial: uniform selection (adequate for \<10 lights).

```rust
fn sample_light(hit_point: Vec3, rng: Rng) -> (Vec3, f32, f32) {
    let idx = rng.u32(0, light_count);
    let light = light_buffer[idx];
    let light_point = sample_point_on_light(light, rng);
    let to_light = light_point - hit_point;
    let dist = to_light.length();
    let wi = to_light / dist;
    let p_light = (1.0 / light_count as f32) * (dist * dist / light.area);
    (light_point, light.radiance, p_light)
}
```

**Future — Power-Based Selection (50+ lights):**

```
weight[i] = light[i].area * luminance(light[i].radiance)
cdf[i]    = sum(weight[0..=i]) / sum(weight[0..N])
sample:   binary search on CDF
```

LuxCore's `Power` strategy. `DLSCache` (direct light sampling cache) is more
sophisticated but further deferred.

### 2.17 Transform System

raytrace-rs currently has only `Translate` and `RotateY`. The `Transform` trait
pattern (one type per operation) duplicates boilerplate at every new transform.

#### The Solution: Mat4 Everywhere

pbrt-v4 stores a single `Transform` (4×4 matrix pair) on every shape. Starlight
follows this pattern.

On the **host side**: transforms are `Mat4` stored in leaf data structs. The
inverse is computed once at scene load and stored alongside the forward matrix.

On the **GPU side**: leaf data structs store a `TransformPair`:

```rust
// In starlight-types/src/leaf.rs
#[repr(C)]
pub struct TransformPair {
    pub world_from_object: Mat4,
    pub object_from_world: Mat4,
}
```

The shader reads the pre-computed inverse. No runtime matrix inversion in the
hot path.

```rust
fn intersect_sphere(ray: Ray, leaf: &SphereLeaf) -> Option<SurfaceInteraction> {
    let local_origin = leaf.transform.object_from_world * ray.origin.extend(1.0);
    let local_dir    = leaf.transform.object_from_world * ray.direction.extend(0.0);
    // ... sphere intersection in object space ...
    let world_point  = leaf.transform.world_from_object * hit_point.extend(1.0);
    let world_normal = leaf.transform.world_from_object.transpose() * normal.extend(0.0);
}
```

#### What This Eliminates

- `Transform` trait with 5 methods
- `TransformObject<T, O>` generic
- `RotateX`, `RotateZ`, `Scale` types
- Composition helpers (matrix multiplication replaces them)

#### Migration Path

`Transform` trait and `TransformObject` wrapper removed in Stage 0. Existing
scenes rewritten to store composed `Mat4` directly in leaf data. One-time
migration, no behavioral change. `onb.rs` stays — used by BSDF evaluation, not
by transforms.

### 2.18 Unified CPU/GPU Abstraction Layer

#### The Problem

Without a shared abstraction, CPU and GPU implementations diverge silently.
`Material::eval()` on CPU and `match GpuMaterialType` on GPU can drift apart
invisibly until a visual regression appears.

#### The Solution

```rust
// In starlight-engine/src/traits.rs

pub trait Intersectable {
    fn intersect(&self, ray: &Ray) -> Option<SurfaceInteraction>;
    fn occluded(&self, ray: &Ray, t_max: f32) -> bool;
}

pub trait Shading {
    fn eval(&self, wo: Vec3, wi: Vec3) -> Color3;
    fn sample(&self, wo: Vec3, rng: &mut Rng) -> BsdfSample;
    fn pdf(&self, wo: Vec3, wi: Vec3) -> f64;
    fn emitted(&self) -> Color3;
    fn is_emissive(&self) -> bool;
}

pub trait LightSampling {
    fn sample(&self, point: Vec3, rng: &mut Rng) -> (Vec3, Color3, f32);
    fn pdf(&self, point: Vec3, wi: Vec3) -> f32;
    fn area(&self) -> f32;
    fn radiance(&self) -> Color3;
}

pub trait Texturing {
    fn sample(&self, uv: [f32; 2]) -> Color3;
    fn sample_normal(&self, uv: [f32; 2]) -> Vec3;
    fn sample_roughness(&self, uv: [f32; 2]) -> f32;
}
```

GPU functions follow the same signatures by convention, operating on the same
data types from `starlight-types`. The shared types crate enforces data
compatibility at compile time. The CPU path is the reference; GPU divergence is
caught by pixel-level comparison.

#### Data Format Conversion

```
CPU Type             GPU Type              Conversion
──────────────────   ─────────────────     ──────────────────
Scene                GpuSceneBuffers       Scene::flatten()
Material (enum)      GpuMaterialNode       Material::gpu_node()
Texture (enum)       GpuTextureNode        Texture::gpu_node()
LeafNode (enum)      GpuLeafType+LeafData  LeafNode::serialize()
BvhNode<LeafNode>    GpuBvhNode[]          BvhNode::flatten()
LightNode            LightNode (repr(C))   direct copy
HitRecord            SurfaceInteraction    HitRecord::to_interaction()
```

**Hardware texture sampling**: textures are `VkImage` objects with bindless
descriptors via `VK_EXT_descriptor_indexing`. Each texture is uploaded with
mipmaps, backed by `VkImageView` and `VkSampler`. `GpuTextureNode.image_index`
indexes into a bindless texture array. The traversal shader calls hardware
bilinear/anisotropic sampling. The atlas approach from v1 is obsolete.

### 2.19 Scene Format Support — glTF and OBJ

#### glTF 2.0 — The Primary Format

| Feature | glTF Element | Starlight Mapping |
|---|---|---|
| Triangle meshes | `mesh.primitives` | `TriangleMesh` leaf |
| PBR materials | `material.pbrMetallicRoughness` | `Material::Glossy` + textures |
| Texture maps | `image` + `sampler` + `texture` | `Texture::Image` + bindless |
| Transforms | `node.translation/rotation/scale` | `Mat4` in leaf data |
| Cameras | `camera.perspective` | `Camera` params |
| Node hierarchy | `node.children` | Parent-child transform composition |

**PBR metallic-roughness mapping:**

```
baseColorFactor = [r,g,b,1]                 → Lambertian { albedo }
baseColorFactor + metallicFactor > 0.5      → Glossy (GGX dielectric)
metallicFactor = 1, roughness = 0           → Metal (GGX conductor)
emissiveFactor = [r,g,b]                    → DiffuseLight { color }
alphaMode = "BLEND"                         → deferred (OIT required)
alphaMode = "MASK"                          → raster path only (discard)
```

**glTF extensions (priority order):**

| Extension | Priority |
|---|---|
| `KHR_lights_punctual` | P0 — needed for lit scenes |
| `KHR_materials_unlit` | P1 — UI/emissive constant |
| `KHR_materials_clearcoat` | P2 — maps to `Coated` |
| `KHR_materials_transmission` | P2 — maps to `Dielectric` |
| `KHR_materials_ior` | P2 — configurable IOR |
| `KHR_materials_specular` | P3 |
| `KHR_materials_emissive_strength` | P3 — HDR emissives |

**Deferred extensions**: `KHR_animation_pointer`, `KHR_materials_iridescence`,
`KHR_materials_volume`, `KHR_materials_sheen`, `KHR_materials_dispersion`.

#### OBJ — Legacy Compatibility

OBJ is simpler and useful for quick testing and legacy scenes. Supports vertex
positions/normals/UVs, face triangulation, group boundaries, MTL material
mapping. OBJ support is implemented in Stage 2 alongside triangle mesh support.

#### Unified Loading Pipeline

```
glTF file → GltfLoader ──┐
                          ├→ Scene { LeafNode[], Material[], Texture[], Light[], Camera }
OBJ file  →  ObjLoader ──┘
                          ↓
                   Scene::flatten()
                          ↓
                   GpuSceneBuffers { bvh_nodes, leaf_types, leaf_data,
                                     materials, textures, lights, camera }
```

Loaders produce a `Scene` struct. `flatten()` converts to GPU buffers. Loaders
don't know about GPU specifics. The GPU path doesn't know about file formats.
`Scene` is the single source of truth.

______________________________________________________________________

## 3. Module Layout

```
starlight/
│
├── types/                          starlight-types  [no_std]
│   └── src/
│       ├── lib.rs
│       ├── interaction.rs          SurfaceInteraction       ← from HitRecord TODO
│       ├── ray.rs                  Ray                      ← raytrace-rs ray.rs
│       ├── bvh.rs                  GpuBvhNode, PortalFrame  ← raytrace-rs flat_bvh.rs
│       ├── leaf.rs                 GpuLeafType, GpuLeafData, TransformPair
│       ├── light.rs                LightNode
│       ├── material.rs             GpuMaterialType/Node     ← raytrace-rs material/gpu.rs
│       ├── texture.rs              GpuTextureType/Node      ← raytrace-rs texture/gpu.rs
│       └── math/
│           ├── vec3.rs             Vec3, Vec4               ← raytrace-rs vec3.rs
│           ├── mat4.rs             Mat4
│           └── aabb.rs             Aabb                     ← raytrace-rs aabb.rs
│
├── shaders/                        starlight-shaders  [spirv target]
│   └── src/
│       ├── lib.rs
│       ├── traversal.rs            BVH traversal + leaf dispatch + occluded()
│       ├── shading.rs              PBR shading entry point
│       ├── leaves/
│       │   ├── analytic.rs         intersect_sphere, intersect_aabb
│       │   ├── trimesh.rs          intersect_trimesh (Möller–Trumbore)
│       │   ├── sdf.rs              intersect_sdf (ray march)
│       │   ├── heightfield.rs      intersect_heightfield (reads sim buffer)
│       │   ├── fractal.rs          intersect_fractal (distance estimator)
│       │   ├── medium.rs           intersect_medium (participating media)
│       │   ├── portal.rs           traverse_portal (recursive BVH)
│       │   └── power_foam.rs       intersect_foam (Čech complex walk)
│       └── simulation/
│           ├── thermal.rs          thermal diffusion kernel
│           └── fluid.rs            fluid sim kernel
│
└── engine/                         starlight-engine  [host]
    └── src/
        ├── lib.rs
        ├── traits.rs               Intersectable, Shading, LightSampling, Texturing
        │
        ├── engine/
        │   ├── mod.rs              EngineState
        │   ├── renderer.rs         GpuRenderer trait + AsVkDevice
        │   └── resource_pool.rs    ResourceHandle, BufferHandle, GpuBuffer<T>
        │
        ├── graph/
        │   ├── mod.rs              RenderGraph: DAG, topological sort, barriers
        │   ├── node.rs             RenderNode trait
        │   └── passes/
        │       ├── mod.rs          render_pass_nodes! + FramePassNode
        │       ├── simulation.rs   SimulationPass (data-parallel kernels)
        │       ├── bvh_cull.rs     BvhCullPass → draw lists
        │       ├── bvh_build.rs    BvhBuildPass
        │       ├── mesh_raster.rs  MeshRasterPass (dynamic rendering)
        │       ├── foam_raster.rs  PowerFoamRasterPass (tile splat)
        │       ├── traversal.rs    TraversalPass (Starlight compute dispatch)
        │       ├── shading.rs      ShadingPass (unified PBR)
        │       └── post.rs         PostProcessPass (TAA, tonemap)
        │
        ├── scene/
        │   ├── mod.rs              Scene: BvhNode<LeafNode>, material buffer, light buffer
        │   ├── bvh.rs              BvhNode<LeafNode> tree    ← raytrace-rs bvh.rs
        │   ├── light.rs            light discovery + LightNode construction
        │   ├── flatten.rs          Scene::flatten() → GpuSceneBuffers
        │   ├── gltf.rs             glTF 2.0 loader
        │   ├── obj.rs              OBJ + MTL loader
        │   └── leaf/
        │       ├── mod.rs          leaf_nodes! macro + LeafNode enum
        │       ├── domain.rs       SpatialDomain trait  [introduced at trigger]
        │       ├── analytic.rs     AnalyticSphere, AnalyticAabb
        │       ├── trimesh.rs      TriangleMesh
        │       ├── sdf.rs          SdfVolume
        │       ├── heightfield.rs  HeightField
        │       ├── fractal.rs      FractalParams
        │       ├── medium.rs       ConstantMedium
        │       ├── portal.rs       Portal
        │       └── power_foam/
        │           ├── mod.rs      PowerFoamCell
        │           ├── diagram.rs  bounded power diagram, Čech complex
        │           ├── surface.rs  dipole face, detail sites
        │           └── radiance.rs spherical Voronoi directional radiance
        │
        ├── material/
        │   ├── mod.rs              Material enum
        │   ├── interaction.rs      SurfaceInteraction (wraps types/ version)
        │   ├── scatter.rs          BSDF, importance sampling, MIS
        │   └── texture.rs          Texture enum
        │
        └── simulation/
            ├── mod.rs              SimulationDomain trait
            ├── thermal.rs          ThermalDomain: grid state, GPU buffer
            ├── fluid.rs            FluidDomain
            └── fracture.rs         FractureDomain
```

______________________________________________________________________

## 4. The `SurfaceInteraction` Contract

Every leaf intersection function returns the same type, defined once in
`starlight-types`:

```rust
// In starlight-types/src/interaction.rs
// [f32; 3] convention — see §2.6
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SurfaceInteraction {
    pub point:       [f32; 3],  // world-space hit position
    pub normal:      [f32; 3],  // shading normal
    pub geo_normal:  [f32; 3],  // geometric normal (for self-intersection offset)
    pub uv:          [f32; 2],
    pub t:           f32,        // ray parameter at hit
    pub material_id: u32,        // index into flat material buffer
    pub object_id:   u32,        // per-instance data, simulation coupling
    pub world_id:    u32,        // which sub-world (0 = root)
}
// 48 bytes. Convert to Vec3 at read time: Vec3::from(interaction.point).
```

**Why no `flags` or `eta` in `SurfaceInteraction`**: these describe the result
of a scattering event, not the geometry. The leaf doesn't know whether the
surface it hit is specular or diffuse — that depends on the material, which is
evaluated by the `ShadingPass`. Flags and eta live in `BsdfSample`, not in the
hit record. This matches pbrt-v4's design.

The `ShadingPass` operates entirely on `SurfaceInteraction`. It does not know
whether the hit came from an analytic sphere, SDF ray march, foam cell walk, or
fractal distance estimator. This is the payoff of leaf sovereignty.

```
BxDFFlags (stored in BsdfSample, not here):
    REFLECTION   = 0x01
    TRANSMISSION = 0x02
    DIFFUSE      = 0x04
    GLOSSY       = 0x08
    SPECULAR     = 0x10
```

______________________________________________________________________

> **Evolution Note — `SurfaceInteraction` → `Interaction` Hierarchy**
>
> The current `SurfaceInteraction` is surface-centric by design. Every field
> (`normal`, `geo_normal`, `uv`) assumes a surface was hit. This is correct for
> all current and near-term leaf types: analytic shapes, triangle meshes, SDFs,
> foam cells, fractals, and portals all return well-defined surface hits.
>
> **Trigger:** When `ConstantMedium` needs to return a scattering event inside a
> participating medium — a volumetric hit that has no surface normal, no UV, and
> no geometric normal. Or when a neural field returns a density sample rather
> than a surface point.
>
> **Target (PBRT's solution):**
>
> ```rust
> // In starlight-types/src/interaction.rs
>
> /// Common fields for any ray-domain interaction.
> #[repr(C)]
> #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
> pub struct Interaction {
>     pub point:       [f32; 3],
>     pub t:           f32,
>     pub world_id:    u32,
>     pub object_id:   u32,
>     pub material_id: u32,
>     _pad:            u32,
> }
>
> /// Surface hit: Interaction + geometric surface fields.
> #[repr(C)]
> #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
> pub struct SurfaceInteraction {
>     pub base:       Interaction,
>     pub normal:     [f32; 3],
>     pub geo_normal: [f32; 3],
>     pub uv:         [f32; 2],
>     _pad:           [f32; 2],
> }
>
> /// Volume scattering event: Interaction + phase function parameters.
> #[repr(C)]
> #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
> pub struct VolumeInteraction {
>     pub base:    Interaction,
>     pub phase_g: f32,   // Henyey-Greenstein asymmetry parameter
>     _pad:        [f32; 3],
> }
> ```
>
> The traversal shader returns `Option<Interaction>` tagged with interaction
> kind (surface vs. volume via a flag bit, or via separate output buffers). The
> `ShadingPass` dispatches on kind.
>
> **ABI note:** This is a breaking change to `SurfaceInteraction`. Bump the
> `starlight-types` major version. Both CPU and GPU implementations must update
> simultaneously. The CPU path tracer is the reference for validating the
> migration. Place a `// TODO(volume-interaction): split when ConstantMedium is
> implemented in Stage 8` comment in `interaction.rs` now.
>
> **Why not now:** All leaf types through Stage 7 return surface hits. Adding
> the hierarchy before needing it adds struct indirection with no benefit and
> makes the ABI migration harder to sequence.

______________________________________________________________________

## 5. Design Invariants

Nine invariants. Breaking any one collapses a load-bearing assumption.

1. **The spatial router is the universal address space** for both rendering and
   simulation. The current implementation is a BVH. The architecture must
   survive replacing it with a hardware RT structure, sparse DAG, or any
   equivalent spatial index. The invariant is the *routing role*, not the *BVH
   implementation*.

2. **Every leaf returns `SurfaceInteraction`** (or, after the §4 evolution, a
   compatible `Interaction` subtype). The shading pass is completely agnostic to
   leaf type. No leaf has a special shading path.

3. **Rasterization is not in the leaf-dispatch loop.** Raster passes consume
   visible leaf lists produced by `BvhCullPass`. They issue draw calls. They do
   not walk the BVH per pixel.

4. **The render graph owns all barriers.** No pass manually calls
   `vkCmdPipelineBarrier`. Barriers are derived from `reads()`/`writes()`
   declarations and inserted by the render graph.

5. **Type synchronization between CPU and GPU is architectural, not
   procedural.** The shared `starlight-types` crate defines `GpuLeafType` and
   all GPU-facing structs once. Both host code and rust-gpu shader code import
   the same crate.

6. **The `unsafe` boundary is narrow.** All `unsafe` Vulkan calls are confined
   to pass `record()` implementations and the `AsVkDevice` escape hatch.

7. **Simulation is data-parallel, not per-object.** Simulation domains expose
   grid-style GPU kernels. `SimulationPass` dispatches O(simulation_types)
   kernels per frame, not O(simulation_cells).

8. **Primary visibility arbitration is explicit.** GBuffer is primary; HitBuffer
   is secondary effects. `ShadingPass` documents the selection rule for pixels
   with and without GBuffer coverage.

9. **`SurfaceInteraction` is an ABI.** It is defined once in `starlight-types`.
   Every subsystem that consumes it depends only on this contract. Changing
   `SurfaceInteraction` is a breaking change. Extend additively. Never remove or
   reorder fields without a major version bump of `starlight-types` and
   coordinated updates to every consumer.

______________________________________________________________________

## 6. Implementation Sequence

Each stage produces a runnable, visually verifiable output. The existing CPU
path tracer is never broken — it runs in parallel as the reference.

The sequence follows Review 1's recommendation: CPU-only stages first, then
incremental GPU stages, raster added *after* the full GPU path tracer is
working. This ensures the compute path tracer is the reference for every pixel
before raster is introduced.

**Evolution integration points are marked inline. These are not optional
deferrals — they are scheduled architectural transitions with defined triggers.**

______________________________________________________________________

### Stage 0 — Arena refactor + workspace setup + shared types crate

**Arena refactor**: Replace `Arc<dyn Hittable>` with `Vec<Box<dyn Hittable>>`
in `Scene`. The BVH lifetime-borrows from the scene's object list. Automatic
light detection via `is_emissive()`. Light BVH built from emissive indices.
This eliminates Arc overhead, removes scattered allocations, moves storage
toward GPU readiness.

**Transform system cleanup**: Remove `Transform` trait and `TransformObject<T,O>`.
Store `Mat4` directly in leaf data structs (§2.17). Rewrite existing `Translate`
and `RotateY` scenes to composed `Mat4`.

**Workspace setup**: Create the three-crate workspace. Move `material/gpu.rs`,
`texture/gpu.rs`, and `flat_bvh.rs` GPU-facing types from `raytrace-rs` into
`starlight-types`. Verify that `starlight-engine` re-imports them and the
existing CPU renderer still compiles and produces identical output.

Add `// TODO(volume-interaction): split when ConstantMedium is implemented in Stage 8` to `interaction.rs`. Add `// TODO(spatial-domain): introduce trait when fifth leaf type is working` to `leaf/mod.rs`. These comments are the
formal record of the evolution commitments.

______________________________________________________________________

### Stage 1 — `LeafNode` enum + `SurfaceInteraction` (CPU only)

Introduce the `leaf_nodes!` macro and `LeafNode` enum in `starlight-engine`.
Replace `Arc<dyn Hittable>` in `BvhNode` leaves with `LeafNode`. Extract
`SurfaceInteraction` from `HitRecord` (the `TODO(renderer-agnostic)` in
`hittable.rs`). Verify identical render output from the CPU path tracer. This
eliminates vtable dispatch from the BVH hot path.

______________________________________________________________________

### Stage 2 — Triangle mesh support + OBJ loader (CPU only)

Add `TriangleMesh` leaf type with indexed vertex buffers and per-triangle index
triples. Implement Möller–Trumbore ray-triangle intersection on CPU. Add a
minimal OBJ parser: positions (`v`), normals (`vn`), UVs (`vt`), faces (`f`),
material groups (`usemtl`). Verify by loading an OBJ file and rendering on the
CPU path tracer.

The mesh data is serialized into the flat leaf data buffer as a
`GpuTriangleMesh` struct. This is the third concrete leaf type (AnalyticSphere,
AnalyticAabb, TriangleMesh).

______________________________________________________________________

### Stage 3 — rust-gpu shader crate baseline (first GPU leap)

Add `starlight-shaders` to the workspace. Configure the rust-gpu toolchain.
Write a trivial compute shader (one thread writes a gradient to a storage image)
using `spirv-std` builtins. Load the compiled SPIR-V in a minimal `ash` Vulkan
context and display the result. This validates the rust-gpu build pipeline
before any shader logic is added.

**Development discipline**: test each rust-gpu feature incrementally. Avoid
heavy closures, iterators, or dynamic dispatch inside shaders early on.
rust-gpu's subset of Rust has sharp edges; find them on trivial shaders, not on
the traversal loop.

______________________________________________________________________

### Stage 4 — Triangle mesh intersection on GPU

Port Möller–Trumbore to the traversal shader. Upload the flat BVH, leaf data,
and material buffers using `starlight-engine`'s `ResourcePool`. Output must
match the CPU reference render for analytic spheres and triangle meshes.

This is the first `TraversalPass` implementation. All GPU types come from
`starlight-types`. The `match leaf.type_tag` statement is live for the first
time.

______________________________________________________________________

### Stage 5 — Direct lighting + shadow rays on GPU

Add the `ShadingPass` as a GPU compute shader. Implement the direct lighting
algorithm from §2.14: read HitBuffer, sample light, eval BSDF, trace shadow ray
via `occluded()`, MIS weight, accumulate.

Extend the CPU path tracer to use explicit shadow rays with the same algorithm.
Compare output pixel-by-pixel with the GPU. This is the first validation of the
shading pipeline.

**`occluded()` implementation note**: this must not be a modified
`traverse_bvh()` that returns a `SurfaceInteraction`. It is a separate shader
function sharing the traversal loop that returns `bool` on first hit. Verify at
the SPIR-V level (inspect the compiled output) that the compiler is not
generating dead normals/UV code in the `occluded()` path.

______________________________________________________________________

### Stage 6 — Indirect lighting, Russian roulette, bounce limits

Complete the GPU path tracer with indirect lighting, Russian roulette (survival
∝ `max(throughput)`, starting at bounce 5), and per-type bounce limits
(`max_diffuse_depth`, `max_glossy_depth`, `max_specular_depth`). Use a shared
RNG implementation in `starlight-types` with `#[cfg]` guards for CPU vs. GPU.

This is the Starlight equivalent of pbrt-v4's `PathIntegrator`. The CPU path
tracer in raytrace-rs is the reference implementation — identical algorithm,
different execution target.

> **Evolution trigger for Stage 6:** With the GPU path tracer complete, this is
> also the point to add `SdfVolume` to the traversal shader (fourth leaf type).
> At the fifth leaf type (e.g., `Fractal` or `HeightField`), create
> `leaf/domain.rs` with the `SpatialDomain` trait as documented in §2.4.
> Validate that both the enum dispatch and trait dispatch produce identical
> output before proceeding to Stage 7.

______________________________________________________________________

### Stage 7 — Render graph + raster path

Now that the full GPU path tracer works for every pixel, add raster as an
optimization for primary visibility. Introduce `RenderGraph` with
`FramePassNode`. Add `BvhCullPass` (compute, outputs draw lists) and
`MeshRasterPass` (dynamic rendering, writes GBuffer). Add `ShadingPass` reading
GBuffer + HitBuffer, applying the primary visibility arbitration rule.

Since the compute path tracer already produces correct output for every pixel,
the GBuffer can be validated against ray-traced primary within tolerance. This
reduces the risk of debugging two execution paths simultaneously.

**TraversalPass optimization (tracked here, not deferred):** Add a pre-pass that
reads GBuffer depth and marks pixels as "raster-primary" or "ray-primary".
Dispatch the `TraversalPass` only for ray-primary pixels and pixels needing
secondary effects. This eliminates the full-screen unconditional dispatch noted
in §2.9.

______________________________________________________________________

### Stage 8 — Additional leaf types in shader + `Interaction` split

Add `GpuLeafType::SdfVolume` (ray march a hardcoded sphere SDF). Add
`GpuLeafType::Portal` with the bounded stack and cycle detection (§2.12).

> **`Interaction` evolution trigger for Stage 8:** Add
> `GpuLeafType::ConstantMedium`. At this point, the
> `SurfaceInteraction`-is-surface-only limitation becomes concrete. Implement
> the `Interaction` / `SurfaceInteraction` / `VolumeInteraction` split as
> documented in §4. Bump the major version of `starlight-types`. Update both CPU
> and GPU paths simultaneously. The CPU path tracer validates the migration —
> identical output for surface leaves; `VolumeInteraction` code validated
> against pbrt-v4's `ConstantMedium` output.

At this point: one dispatch, multiple sovereign leaf types, portals working.
This is the Starlight microkernel running.

______________________________________________________________________

### Stage 9 — Simulation coupling

Add `SimulationPass` with `ThermalDomain` as the first domain type. Implement
the thermal diffusion kernel in `starlight-shaders/src/simulation/thermal.rs`.
Wire `HeightField` leaf to read the domain's output buffer. The render graph
schedules `SimulationPass` before `TraversalPass` and inserts the barrier.
Surface displaces from simulation in real time.

> **Simulation tension review for Stage 9:** After implementing `ThermalDomain`
> and `ConstantMedium`, evaluate whether simulation domains need to be spatially
> queryable (Path B in §2.11). If participating media require temperature-driven
> density queries during traversal, begin the `SpatialDomain` migration for
> simulation domains. If thermal data is still only read by geometry leaves at
> their surface, Path A remains correct.

______________________________________________________________________

### Stage 10 — Power Foam leaf

Implement `RadiantFoamCell` first (unbounded Voronoi, ray tracing only).
Validate the cell-walk traversal against the Power Foam paper's algorithm.
Extend to `PowerFoamCell` with sphere bounds. Add `PowerFoamRasterPass`. The
same leaf works in both execution paths from identical data. Implement brute-
force adjacency first; add graph-walk only after profiling confirms it is the
bottleneck.

Add the differentiable training loop: loss from image comparisons, gradients
through the foam rendering, Adam optimizer updating site positions, radii,
directional radiance. Engine reconstructs Power Foam captures from real-world
images.

______________________________________________________________________

### Stage 11 — glTF loader

Implement `gltf.rs`: JSON scene graph parsing, binary buffer accessors, image
decoding. Map glTF PBR metallic-roughness to Starlight materials (§2.19).
Support `KHR_lights_punctual`.

Implement `Scene::flatten()` — the unified conversion from CPU `Scene` to
`GpuSceneBuffers`. Define the trait interfaces in `traits.rs`
(`Intersectable`, `Shading`, `LightSampling`, `Texturing`) and implement them
for CPU types. GPU shader functions follow the same interface by convention.

Verify by loading the glTF spec sample models (Box, Duck, FlightHelmet).

______________________________________________________________________

### Stage 12 — `WorldHandle` capability (portal streaming)

> **Trigger:** A scene requires worlds that are not pre-loaded at startup.
>
> Introduce the `WorldLoader` trait and `WorldHandle` type as documented in
> §2.12. Migrate `PortalFrame.world_id: u32` to `Portal.capability:
> WorldHandle`. Add a GPU-side world table populated asynchronously by the host.
> The traversal shader marks un-loaded worlds as `LeafHit::miss()` for one frame
> while loading is queued. Validate: pre-loaded worlds behave identically to
> Stage 8 portals.

______________________________________________________________________

### Stage 13 — Hardware RT migration

> **Trigger:** Hardware RT is available on the target device and profiling
> confirms the compute `TraversalPass` is the bottleneck.
>
> Replace the `TraversalPass` compute dispatch with `vkCmdTraceRaysKHR`. Leaf
> intersection functions become callable shaders bound via the SBT.
> `SpatialDomain::gpu_serialize()` populates the SBT entry for each domain. The
> `SurfaceInteraction` ABI is unchanged — the output of the RT pipeline writes
> the same `SurfaceInteraction` struct to the same HitBuffer. The `ShadingPass`
> reads from the same buffer regardless of which traversal path produced it.
>
> The Vulkan RT extension references (§1.13) and Embree (§1.9) are the primary
> guides for this migration.

______________________________________________________________________

## 7. End-to-End Frame Walkthrough

A concrete trace of one frame from scene load to displayed pixel.

### Scene Load (CPU, once)

```
1. Parse scene description (glTF/OBJ)
2. Create LeafNode instances for each object
3. Build BVH tree from LeafNode AABBs (SAH, 32 bins, parallel)
4. Flatten BVH to GpuBvhNode array (near-child-first ordering)
5. Discover lights: walk LeafNode list, collect emissives → LightNode[]
6. Flatten materials: tree → Vec<GpuMaterialNode> (DAG serialization)
7. Upload to GPU:
   - BVH nodes      → VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
   - Leaf types     → VK_BUFFER_USAGE_STORAGE_BUFFER_BIT (u32 tags)
   - Leaf data      → VK_BUFFER_USAGE_STORAGE_BUFFER_BIT (serialized)
   - Materials      → VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
   - Lights         → VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
   - Camera params  → VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
```

### Frame Render (GPU, every frame)

```
RenderGraph::compile():
  1. Build DAG from pass reads()/writes() declarations
  2. Topological sort (Kahn's algorithm)
  3. Insert barriers between adjacent passes with conflicts
  4. Single GRAPHICS|COMPUTE queue

RenderGraph::execute():

  Pass 1: SimulationPass
    vkCmdDispatch(thermal_kernel, domain_count)
    Writes: thermal_state_buffer

  Barrier: NONE → SHADER_READ on thermal_state_buffer

  Pass 2: BvhCullPass
    vkCmdDispatch(cull_shader, node_count)
    Input: camera frustum, BVH nodes
    Writes: foam_draw_list, mesh_draw_list

  Barrier: SHADER_WRITE → INDIRECT_READ on draw lists

  Pass 3: MeshRasterPass
    vkCmdDrawIndexed(indirect_buffer)
    Writes: GBuffer (albedo, normal, depth, material_id)

  Barrier: COLOR_ATTACHMENT_WRITE → SHADER_READ on GBuffer

  Pass 4: PowerFoamRasterPass
    vkCmdDraw(indirect_buffer)
    Writes: GBuffer (depth-tested, overwrites if closer)

  Barrier: COLOR_ATTACHMENT_WRITE → SHADER_READ on GBuffer

  Pass 5: TraversalPass
    vkCmdDispatch(ceil(width/8), ceil(height/8), 1)
    [Stage 7+ optimization: dispatch only for non-GBuffer pixels]
    For each pixel:
      1. Generate primary ray from camera params
      2. traverse_bvh() → Option<SurfaceInteraction>
      3. Write SurfaceInteraction to HitBuffer
    Writes: HitBuffer

  Barrier: SHADER_WRITE → SHADER_READ on HitBuffer

  Pass 6: ShadingPass
    vkCmdDispatch(ceil(width/8), ceil(height/8), 1)
    For each pixel:
      1. Read GBuffer (primary) or HitBuffer (primary fallback)
      2. Look up material via material_id
      3. Direct lighting (NEE + shadow ray + MIS)
      4. Indirect lighting (BSDF sample + secondary ray)
      5. Russian roulette (bounce ≥ 5)
      6. Per-type bounce limits
    Writes: output image (rgba32f)

  Barrier: SHADER_WRITE → TRANSFER_SRC on output image

  Pass 7: PostProcessPass
    vkCmdDispatch(post_shader)
    Applies: tone mapping (ACES/Reinhard), gamma correction
    Writes: swapchain image (rgba8)
```

### Data Flow Summary

```
Scene load:
  LeafNode[] → serialize → BVH + LeafTypes + LeafData + Materials + Lights

Per frame:
  Simulation    → thermal_state → HeightField leaves (displacement)
  Camera        → primary rays  → TraversalPass → HitBuffer
  BvhCullPass   → draw lists    → RasterPasses  → GBuffer
  GBuffer + HitBuffer + Lights + Materials → ShadingPass → Output
  Output        → PostProcessPass → Display
```

______________________________________________________________________

## 8. Validation Strategy

The CPU path tracer in raytrace-rs is the reference implementation. Every
rendering algorithm in the GPU `ShadingPass` must produce statistically
identical output to the CPU path tracer for the same scene configuration.

### What Is Validated

| Aspect | CPU Reference | GPU Target | Comparison |
|---|---|---|---|
| BVH traversal | `flat_bvh.rs` iterative | `traversal.rs` shader | Same BVH nodes, same order |
| Material eval | `Material::eval()` | match on `GpuMaterialType` | Same BSDF value within ε |
| Direct lighting | NEE with shadow ray | `ShadingPass` direct loop | Same contribution within variance |
| Shadow rays | `occluded()` | `occluded()` in shader | Boolean occlusion matches |
| MIS weights | Power heuristic | Power heuristic | Same weights for same PDFs |
| Russian roulette | Throughput-proportional | Throughput-proportional | Same survival probability |
| Bounce limits | Per-type | Per-type | Same termination behavior |

### Validation Methodology

1. **Identical scene serialization**: the same `LeafNode` list produces the same
   `GpuBvhNode`, `GpuMaterialNode`, `LightNode` arrays on both paths. The
   serialization is tested independently.

2. **Deterministic comparison**: for the same random seed and sample count, CPU
   and GPU must produce pixel values within floating-point tolerance (ε = 1e-5
   for single-precision). This catches algorithmic divergence without
	   statistical tests.

3. **Visual regression**: side-by-side renders of reference scenes (Cornell box,
   glossy spheres, glass with caustics) must be perceptually identical. A pixel
   difference shader overlays the difference for human inspection.

4. **ABI validation**: at every evolution point that changes
   `SurfaceInteraction` or `GpuLeafType`, a full regression run is required
   before and after the change. Both CPU and GPU outputs must match within ε on
   all reference scenes.

### The CPU Path Is Never Broken

The implementation sequence is additive. Each stage adds GPU capability without
modifying the CPU path tracer. The CPU path continues to render correctly
throughout development, serving as fallback and validation tool at every stage.

______________________________________________________________________

## 9. References

### Primary Architecture

**Starlight / Matryoshka renderer** — inigid. Reddit r/computergraphics, May
2025\. Zig + GLSL Vulkan compute shader prototype, ~2,500 lines.
`reddit.com/r/computergraphics/comments/1skkxrm`

**Radiant Foam** — Govindarajan, Rebain, Yi, Tagliasacchi. ICCV 2025.
arXiv:2502.01157. `radfoam.github.io`

**Power Foam** — Govindarajan, Rebain, Verbin, Yi, Prabhu, Tagliasacchi. arXiv
2604.24994, 2026. `powerfoam.github.io`

**Niri compositor** — YaLTeR. `github.com/YaLTeR/niri`. `niri_render_elements!`
macro and `NiriRenderer` supertrait.

**rust-gpu** — Embark Studios. `github.com/EmbarkStudios/rust-gpu`.

**renderling** — schell. `github.com/schell/renderling`.

**raytrace-rs** — Atan-D-RP4. `github.com/Atan-D-RP4/raytrace-rs`.

### Operating Systems

**Operating Systems: Three Easy Pieces** — Arpaci-Dusseau & Arpaci-Dusseau.
`ostep.org`. Free textbook. The VFS discussion (chapters 39–40) directly informs
the BVH-as-VFS framing in §0.5. The process abstraction informs the spatial
domain sovereignty model.

**seL4 Microkernel** — Klein et al., "seL4: Formal Verification of an OS
Kernel," SOSP 2009. `sel4.systems`. The formally-verified L4 microkernel. The
capability model (object + unforgeable authority token = permission to interact)
is the conceptual target for the portal `WorldHandle` evolution in §2.12.

**Plan 9 from Bell Labs** — Pike et al., Bell Labs. The uniform namespace
architecture: 9P protocol routes all resource access through the same interface.
The VFS-as-architecture framing, not VFS-as-implementation-detail. Informs §0.5.

### Rendering Architecture

**Physically Based Rendering: From Theory to Implementation** — Pharr, Jakob,
Humphreys. `pbrt.org`. BSDF evaluation, importance sampling, MIS,
`SurfaceInteraction` as renderer-agnostic hit description, `Interaction`
hierarchy for volumes. The `ShadingPass` is a GPU implementation of PBRT's
integrator.

**Raytracing in One Weekend** (series) — Peter Shirley. BVH, `Hittable`,
`HitRecord`, `Material`, `Camera` foundations in raytrace-rs.

**Embree** — Wald et al., Intel. `embree.github.io`. The production CPU
reference for traversal/intersection separation. `RTCGeometry` is the CPU-side
`SpatialDomain`. Filter functions are the CPU-side custom intersection shaders.
`rtcSetGeometryIntersectFunction` / `rtcSetGeometryOccludedFunction` are the
full/shadow-ray split implemented in §2.8.

**Mitsuba 3** — Jakob et al. `mitsuba-renderer.org`. The production system
closest to Starlight's goals. `Shape` → `Intersection` → `BSDF` pipeline,
differentiable geometry via Dr.Jit, pluggable integrators. Study for the
`Medium` integration as the reference for `VolumeInteraction` evolution.

**Falcor** — Kallweit et al., NVIDIA Research.
`github.com/NVIDIAGameWorks/Falcor`. Research renderer on a render graph.
`RenderPass::reflect()` maps to `RenderNode::reads()/writes()`. Production
render graph reference for §2.7.

**MoonRay** — DreamWorks Animation. `github.com/dreamworksanimation/moonray`.
Open-sourced 2023. Per-component BSDF (`BsdfComponent`) is the production
reference for Starlight's material dispatch. Geometry plugin system is another
`SpatialDomain` reference at production quality.

**FrameGraph: Extensible Rendering Architecture in Frostbite** — Hammarén. GDC
2017\. `reads()/writes()` declarations, topological sort, automatic barrier
insertion. The pattern implemented in §2.7.

### Hardware Ray Tracing

**Vulkan Ray Tracing** — Khronos. `VK_KHR_acceleration_structure`,
`VK_KHR_ray_tracing_pipeline`. TLAS/BLAS separation, SBT (the hardware dispatch
table equivalent to `match leaf.type_tag`), callable shaders. The migration
target for Stage 13.

**DirectX Raytracing (DXR)** — Microsoft. Any-hit shaders, closest-hit shaders,
miss shaders. The shader type hierarchy maps to Starlight's intersection
function variants. The SBT in DXR formalizes the sovereign-leaf dispatch in
hardware.

**NVIDIA RTX Architecture** — NVIDIA. RT cores traverse BVH nodes; intersection
testing runs on normal shader cores. Hardware-level proof that the
routing/intersection separation is correct: RT cores accelerate the
geometry-agnostic part; programmable cores handle the domain-specific part.

### Libraries and Tools

**ash** — `github.com/ash-rs/ash`. Thin Rust Vulkan bindings.

**gpu-allocator** — Traverse-Research.
`github.com/Traverse-Research/gpu-allocator`.

**gltf** (Rust crate) — `github.com/gltf-rs/gltf`. glTF 2.0 parser.

**tobj** (Rust crate) — `github.com/Twinklebear/tobj`. OBJ + MTL parser.

**image** (Rust crate) — `github.com/image-rs/image`. PNG/JPEG decoding.

**Dynamic Rendering (Vulkan 1.3)** — `VK_KHR_dynamic_rendering`. Eliminates
pre-created `VkRenderPass` objects; used by `MeshRasterPass` and
`PowerFoamRasterPass`.

**glTF 2.0 Specification** — Khronos Group.
`registry.khronos.org/glTF/specs/2.0`. Primary input format for Starlight.

______________________________________________________________________

## 10. Evolution Roadmap

All evolution notes from throughout the spec collected into a single reference.
Each row states the current implementation, the trigger for evolution, and the
target. These are commits, not suggestions.

| Abstraction | Stage Introduced | Current State | Trigger | Target |
|---|---|---|---|---|
| Spatial router | 0 | `FlatBvhNode[]` BVH | Hardware RT available; or profiling shows BVH is bottleneck | `SpatialRouter` trait; BVH is one impl |
| Leaf dispatch | 1 | `LeafNode` enum + `leaf_nodes!` macro | 5th leaf type working, OR plugin/dynamic loading needed | `SpatialDomain` trait in `leaf/domain.rs`; `LeafNode` is one impl |
| Interaction type | 1 | `SurfaceInteraction` (surface-only) | `ConstantMedium` or any volumetric leaf in Stage 8 | `Interaction` base + `SurfaceInteraction` + `VolumeInteraction` |
| Portal targeting | 8 | `world_id: u32` in `PortalFrame` | Streaming, networked, or procedural worlds needed | `WorldHandle` capability + `WorldLoader` trait (Stage 12) |
| Simulation in BVH | 9 | External buffers read by leaves | A simulation domain must be ray-queryable (e.g., fog density) | `SpatialDomain` impl for simulation domains; Path B in §2.11 |
| Material dispatch | 1 (GPU) | `material_id` → global flat buffer | Leaf type needs per-instance material outside shared BSDF model | `SurfaceInteraction` inline material payload |
| Light selection | 5 | Uniform random | >50 lights in scene | Power-weighted CDF; then `DLSCache` for many lights |
| Queue model | 7 | Single `GRAPHICS\|COMPUTE` | Profiling shows compute/graphics overlap benefit | Multi-queue + timeline semaphores |
| Traversal dispatch | 4 | GPU compute dispatch | Hardware RT available + Stage 13 schedule | `vkCmdTraceRaysKHR` + SBT |
| TraversalPass coverage | 7 | Unconditional full-screen | Stage 7 render graph work | GBuffer-driven cull: dispatch only for uncovered pixels |
| `SpatialDomain` trait | 6 (trigger) | Not yet present | 5th leaf type reached | Introduce trait; blanket impl for `LeafNode` |
| Portal worlds | 8 | Fixed table at load time | Dynamic world loading | `WorldLoader` trait + lazy world loading |
| Shadow ray path | 5 | Software `occluded()` in traversal shader | Hardware RT (Stage 13) | `vkCmdTraceRaysKHR` with AHS returning `false` immediately |
