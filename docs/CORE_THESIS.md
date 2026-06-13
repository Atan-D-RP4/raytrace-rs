# Starlight Rendering Engine

## Design Specification v2 — Iterated

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

## 0. Thesis

> **The spatial hierarchy is the universal. The leaf is sovereign over rendering
> and simulation. The type system is the dispatcher. Rasterization and ray
> traversal are two distinct execution paths over the same spatial data.**

Starlight is a hybrid real-time renderer where heterogeneous scene geometry —
analytic shapes, SDF fields, triangle meshes, foam-based captures, fractals,
participating media, and portals to recursive sub-worlds — coexists in a single
BVH. Two execution paths operate over this BVH simultaneously:

**The ray-driven path** (Starlight core): one GPU compute dispatch per frame,
one BVH traversal per pixel. When a ray reaches a leaf, that leaf runs its own
intersection function — analytic formula, SDF ray march, foam graph walk,
fractal distance estimator, or a portal that recurses the traversal into a
sub-BVH. The traversal code is ignorant of leaf contents. A uniform
`SurfaceInteraction` is returned regardless of how the hit was found. Secondary
effects (reflections, AO, soft shadows, refractions) are computed here.

**The raster path**: the BVH is used for view-frustum culling and LOD selection,
not per-pixel traversal. Culled leaf lists feed traditional Vulkan draw calls
into a deferred GBuffer. Primary visibility comes from here. It is fast,
tile-coherent, and occupies a separate graphics pipeline.

Both paths share the same BVH, the same material buffer, the same simulation
state buffers, and the same `SurfaceInteraction` semantics. The `ShadingPass`
unifies their outputs under a single PBR lighting model.

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
to tile- based rasterization and real-time ray tracing from identical data.

**Foundation (Radiant Foam, ICCV 2025)**: space partitioned into Voronoi cells
defined by learnable site positions. Ray traversal hops from cell to cell
through a Delaunay adjacency graph in constant time per step — no BVH needed
inside the foam. The representation is differentiable: cell boundaries move
continuously as site positions change.

**Power Foam's extension**: replaces unbounded Voronoi cells with a bounded
power diagram (weighted Voronoi with per-site radii). Every cell is clipped to a
sphere bound, giving it a finite screen-space projection for tile rasterization.
The adjacency graph becomes the Čech complex (all pairwise-overlapping spheres)
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
sphere bounds as screen- space discs and tile-rasterizes dipole faces into the
GBuffer, alongside any triangle mesh leaves processed by `MeshRasterPass`. Both
raster passes read the same cell data as the traversal shader — no second
representation.

**Differentiability**: Power Foam is the first leaf type optimizable from
observations. Given real images, gradient descent through the foam rendering
reconstructs the scene. This is the neural reconstruction integration path:
captured Power Foam models load and render with correct light transport at
real-time framerates.

Reference: `powerfoam.github.io`

### 1.3 Niri compositor (YaLTeR) — the Rust structural pattern

Two specific patterns are taken from Niri's source code
(`github.com/YaLTeR/niri`):

**The `niri_render_elements!` macro pattern**
(`src/render_helpers/render_elements.rs`): given `VariantName = ConcreteType`
pairs, generates a concrete enum with delegated trait impls and `From<T>` for
each variant. This replaces `Box<dyn Trait>` with zero- cost static dispatch.
The engine uses this pattern for both the `LeafNode` enum (leaf type dispatch)
and the `FramePassNode` enum (render graph pass dispatch).

**The `NiriRenderer` supertrait pattern** (`src/render_helpers/renderer.rs`): a
trait alias that bundles capability bounds under one name, plus an
`AsGlesRenderer` escape hatch for direct hardware access. The engine's
`GpuRenderer` trait and `AsVkDevice` escape hatch follow this exactly.

What Niri does NOT contribute: Niri has no render graph (its frame is a single
linear element list with no inter-pass dependencies). The render graph is built
fresh.

### 1.4 rust-gpu (Embark Studios) — shaders in Rust

`rust-gpu` (`github.com/EmbarkStudios/rust-gpu`) is a Rust compiler backend that
emits SPIR-V. All GPU shaders in the engine are written as Rust functions in a
dedicated shader crate, compiled to SPIR-V at build time.

The architectural consequence is significant: a `no_std`-compatible
`starlight-types` crate defines `SurfaceInteraction`, `GpuBvhNode`,
`GpuMaterial`, `GpuLeafType`, and all other shared types once. Both the CPU host
crate and the GPU shader crate import this crate. The leaf type tags are a Rust
`#[repr(u32)]` enum in `starlight-types`. The traversal shader's `match leaf.type_tag { GpuLeafType::AnalyticSphere => ... }` uses the same enum variant
values as the host code's `LeafNode` serialization — not because a build script
generates a header, but because it is the same Rust code compiled to two
different targets.

This eliminates the entire `leaf_types.glsl` generation approach from v1. Type
synchronization between CPU and GPU is architectural, not procedural.

`raytrace-rs` already exhibits this pattern in embryo: `material/gpu.rs` and
`texture/gpu.rs` define `GpuMaterialType` and `GpuTextureType` as `#[repr(u32)]`
enums with `repr(C)` node structs. Migrating these into the shared types crate
is the first concrete step of the rust-gpu integration.

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
Rust" approach at production scale. The engine's workspace structure and shader
organization are directly informed by it.

### 1.6 raytrace-rs (Atan-D-RP4) — the existing foundations

The engine's CPU path tracer. Already contains the right building blocks:

- `flat_bvh.rs`: GPU-ready flat BVH with `repr(C)` 64-byte nodes, iterative
  traversal with a 64-entry explicit stack, near-child-first ordering for early
  termination.
- `material/gpu.rs`: `GpuMaterialType` (`#[repr(u32)]`), `GpuMaterialNode`
  (`repr(C)`), `GpuMaterialBuffer` serialization. The CPU `match` and the GPU
  `switch` mirror each other.
- `texture/gpu.rs`: Same pattern for textures.
- `hittable.rs`: `HitRecord` with existing `TODO(renderer-agnostic)` calling for
  `SurfaceInteraction`. The refactor is already anticipated.
- `bvh.rs`: Tree BVH using `Arc<dyn Hittable>` — the `Arc<dyn Hittable>` is what
  gets replaced by the `LeafNode` enum.
- `planar/`: Quad, triangle, box, annulus, ellipse, superellipse, rounded rect,
  polygon.
- `transform.rs`, `onb.rs`, `pdf.rs`, `sampler.rs`: acceleration infrastructure.

______________________________________________________________________

## 2. Architecture

### 2.1 The Two Execution Paths

This is the most important architectural clarification from v1. Rasterization
and ray traversal are not the same operation and must not be conflated.

```
						┌─────────────────────────────────────┐
						│            BVH (shared)             │
						│  spatial address space for all      │
						│  rendering AND simulation           │
						└────────────┬──────────┬────────────┘
									 │          │
			   ┌─────────────────────▼──┐   ┌──▼──────────────────────────┐
			   │     RASTER PATH        │   │        RAY PATH             │
			   │  (primary visibility)  │   │   (secondary effects)       │
			   ├────────────────────────┤   ├─────────────────────────────┤
			   │ BvhCullPass            │   │ TraversalPass               │
			   │  frustum + occlusion   │   │  compute dispatch           │
			   │  → draw lists          │   │  one ray per pixel          │
			   │                        │   │  BVH traversal              │
			   │ MeshRasterPass         │   │  match leaf.type_tag        │
			   │  vkCmdDrawIndexed      │   │  → leaf intersection fn     │
			   │  → GBuffer             │   │  → SurfaceInteraction       │
			   │                        │   │  → HitBuffer (2ndary hits)  │
			   │ PowerFoamRasterPass    │   │                             │
			   │  tile splat raster     │   │ Reflections, AO, refractions│
			   │  → GBuffer             │   │ soft shadows, caustics      │
			   └──────────┬─────────────┘   └─────────────┬───────────────┘
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

**The invariant**: the BVH is queried by both paths, but by different
mechanisms. The ray path walks the BVH per pixel in a compute shader. The raster
path uses the BVH to produce a visible-object list (by frustum culling AABB
nodes), then issues traditional draw calls for those objects. The raster path
does not walk the BVH per pixel. It issues draw calls.

### 2.2 The Workspace

Three crates. The boundary between them is the CPU/GPU split:

```
starlight/
├── types/          starlight-types       no_std, shared CPU+GPU
├── shaders/        starlight-shaders     rust-gpu SPIR-V target
└── engine/         starlight-engine      host, Vulkan, ash
```

**`starlight-types`** is `no_std` + `no_std`-compatible. It contains every type
that must be visible in both host code and shader code: `SurfaceInteraction`,
`GpuBvhNode`, `GpuLeafType`, `GpuLeafData`, `GpuMaterialType`,
`GpuMaterialNode`, `GpuTextureType`, `GpuTextureNode`, `Ray`, and the math
primitives `Vec3`, `Vec4`, `Mat4`, `Aabb`. Alignment and sizing are `repr(C)`
throughout. This crate evolves directly from `raytrace-rs`'s existing
`material/gpu.rs`, `texture/gpu.rs`, and `flat_bvh.rs`.

**`starlight-shaders`** targets `spirv-unknown-vulkan1.2`. It imports
`starlight- types` and uses `spirv-std` for shader builtins. Each leaf
intersection function is a Rust function. The traversal entry point is a Rust
function annotated `#[spirv(compute(threads(8, 8)))]`. No GLSL, no HLSL, no
`build.rs` header emission.

**`starlight-engine`** is the host. It imports `starlight-types` for the shared
types, loads the SPIR-V compiled from `starlight-shaders`, and drives Vulkan via
`ash`. The `LeafNode` host enum, `RenderGraph`, `EngineState`, `ResourcePool`,
and all Vulkan resource management live here.

### 2.3 The Shared Types Crate — Why It Replaces build.rs

In v1, `build.rs` emitted a `leaf_types.glsl` header from Rust enum ordinals so
that the GLSL `switch` statement matched the Rust enum values. This is correct
but fragile: it is a procedural synchronization, not an architectural one.
Adding a leaf type requires updating the macro, recompiling, and trusting that
the emitted ordinals match.

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

The `leaf_nodes!` macro (Niri pattern) operates on the **host-side** `LeafNode`
enum. It is separate from `GpuLeafType` in `starlight-types` — `LeafNode` is the
rich CPU type owning the full Rust data for each leaf. `GpuLeafType` is the thin
`repr(u32)` tag shared with the shader.

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
			LeafNode::AnalyticSphere(_)  => GpuLeafType::AnalyticSphere,
			// ... one arm per variant, enforced exhaustive
		}
	}
}

impl SimLeaf for LeafNode {
	fn simulation_cell(&self) -> Option<&dyn SimulationCell> {
		match self { /* delegates */ }
	}
}

// GpuSerialize writes each variant to the flat leaf data buffer
// for upload to the GPU LeafData SSBO
impl GpuSerialize for LeafNode {
	fn write_bytes(&self, dst: &mut [u8]) {
		match self { /* per-variant serialization */ }
	}
	fn byte_size(&self) -> usize {
		match self { /* per-variant size */ }
	}
}

impl From<AnalyticSphere> for LeafNode { ... }
// ...From<T> for each variant
```

The `gpu_type_tag()` method returns a `GpuLeafType` value from
`starlight-types`. This is how the host tells the GPU which `match` arm to take.
The connection is the shared enum, not an emitted constant.

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
// Used inside pass record() when extension commands are needed
pub trait AsVkDevice {
	fn vk_device(&self)   -> &ash::Device;
	fn vk_physical(&self) -> vk::PhysicalDevice;
	fn vk_queue(&self)    -> vk::Queue;   // single GRAPHICS|COMPUTE queue
}
```

All `unsafe` Vulkan calls are confined to pass `record()` implementations and
this escape hatch. No `unsafe` in the render graph orchestration layer.

### Buffer Layout Convention

**Rule: buffer layout types in `starlight-types` use primitive arrays
`[f32; N]`, not `glam::Vec3`.**

GLSL `vec3` has 16-byte alignment in storage buffers (std140/std430).
`[f32; 3]` compiles to SPIR-V `OpTypeArray float 3` with 4-byte element
alignment — no special `vec3` rule applies. Using `[f32; 3]` for buffer
layout types eliminates the alignment mismatch between CPU `repr(C)` structs
and GPU SSBO reads. No `gpu_layout` crate, no manual padding, no
`scalarBlockLayout` dependency for correctness.

The convention:
- **Buffer layout types** (`LightNode`, `SurfaceInteraction`, `GpuBvhNode`,
  leaf data structs): use `[f32; 3]` for 3-component fields.
- **Shader computation code**: convert to `glam::Vec3` at read time
  (`Vec3::from(array)`). Use `Vec3` freely in math.
- **`bytemuck::Pod + Zeroable`**: derive on all buffer layout types. Upload
  via `bytemuck::cast_slice` — zero-copy, no serialization.

`scalarBlockLayout` is still enabled as a device feature (belt-and-suspenders)
because it enables ergonomic future use and costs nothing. But it is not
load-bearing for correctness.

### 2.7 The Render Graph

Not from Niri. Built fresh following Frostbite's frame graph pattern (GDC 2017).

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
		// 4. No multi-queue for now — single GRAPHICS|COMPUTE queue
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
Vulkan 1.3) — no pre-created `VkRenderPass` objects. The render graph barrier
insertion handles the `COLOR_ATTACHMENT_WRITE → SHADER_READ` transition between
raster passes and the shading pass.

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

// Bindings mirror the host-side ResourcePool handles
#[spirv(compute(threads(8, 8)))]
pub fn traversal_main(
	#[spirv(global_invocation_id)] id: UVec3,
	#[spirv(descriptor_set = 0, binding = 0, storage_buffer)] bvh:         &[GpuBvhNode],
	#[spirv(descriptor_set = 0, binding = 1, storage_buffer)] leaf_types:  &[u32],
	#[spirv(descriptor_set = 0, binding = 2, storage_buffer)] leaf_data:   &[u8],
	#[spirv(descriptor_set = 0, binding = 3, storage_buffer)] materials:   &[GpuMaterialNode],
	#[spirv(descriptor_set = 0, binding = 4, storage_buffer)] sim_buffers: &[f32],
	#[spirv(descriptor_set = 0, binding = 5)] camera_ubo:                  &CameraParams,
	#[spirv(descriptor_set = 0, binding = 6, storage_image)] output:       &Image!(2D, format=rgba32f, sampled=false),
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

**Sky model**: Initially, `sample_sky()` returns a constant background
color (e.g., gradient or uniform). For physically based rendering, the
sky is promoted to a `SkyLight` — a large distant sphere with an HDR
environment map sampled as a light source. The `SkyLight` is added to
the light buffer at scene load and sampled during direct lighting
alongside other lights. Pre-filtered mipmaps for split-sum
approximation are a later optimization. The `sample_sky()` miss path
evaluates the environment map in the traversal shader when no geometry
is hit.

fn traverse_bvh(
	ray: Ray,
	bvh: &[GpuBvhNode],
	leaf_types: &[u32],
	leaf_data: &[u8],
	sim: &[f32],
) -> Option<SurfaceInteraction> {
	// Iterative traversal, explicit stack (32 entries — matches FlatBvhNode design)
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
		Some(GpuLeafType::Portal)         => traverse_portal(ray, leaf_idx, leaf_data, bvh,
															 leaf_types, leaf_data, sim,
															 &mut stack, &mut stack_top),
		Some(GpuLeafType::PowerFoam)      => intersect_foam(ray, leaf_idx, leaf_data),
		None                              => None,
	};
	// update closest...
	closest
}

/// Shadow ray query — short-circuit occlusion test.
/// Returns true if any leaf is hit before t_max.
/// No SurfaceInteraction is computed — early termination on first hit.
fn occluded(
    ray: Ray,
    bvh: &[GpuBvhNode],
    leaf_types: &[u32],
    leaf_data: &[u8],
    t_max: f32,
) -> bool {
    // Same BVH descent as traverse(), but returns true immediately
    // on any leaf hit without computing SurfaceInteraction.
    // Cheaper than full traversal: no normal, no UV, no material.
}
```

The traversal is the Starlight core. It is ignorant of leaf contents except in
the `match` arm. Every leaf function returns `Option<SurfaceInteraction>`.

### 2.9 Primary Visibility Arbitration

The `ShadingPass` reads two buffers:

- **GBuffer** (from `MeshRasterPass` + `PowerFoamRasterPass`): albedo, normal,
  depth, material ID for primary visibility of rasterizable objects.
- **HitBuffer** (from `TraversalPass`): `SurfaceInteraction` for secondary ray
  hits — specular reflections, AO samples, refraction rays, shadow rays.

The arbitration rule:

> **Primary visibility is always the GBuffer.** The raster path writes the
> surface seen by the primary camera ray. The `TraversalPass` does not compete
> for primary visibility; its output is used only for secondary effects. For
> pixels with no raster coverage (SDF-only leaves, fractal leaves, portals —
> which have no raster draw call), the `TraversalPass` also provides primary
> visibility via the HitBuffer. The `ShadingPass` selects: if the GBuffer depth
> for a pixel is valid, use GBuffer as primary; otherwise fall back to HitBuffer
> primary.

For secondary effects, the `ShadingPass` additionally traces a reflection ray or
an AO ray (via a secondary TraversalPass dispatch or baked into the first) for
pixels whose GBuffer material has specular response. These hits come from the
HitBuffer and are composited over the GBuffer primary using a weighted blend.

This is standard deferred shading with a ray-traced secondary effects layer. The
novelty is that the deferred GBuffer can be written by heterogeneous raster
passes (mesh raster, foam splat) and the secondary layer can hit heterogeneous
leaf types (SDF, foam, portal).

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
reads, the render graph can insert a `NONE` barrier between them (no
write-after-read hazard). The `ShadingPass` reads both outputs.

For the foam adjacency walk: start with brute-force iteration over all dipole
faces in the cell's data block. Add graph-walk traversal only after profiling
confirms the brute-force path is the bottleneck. The threshold for switching is
empirical; 128 sites per cell is a reasonable starting estimate.

### 2.11 Simulation — Data-Parallel Kernels

**The v1 problem**: `SimulationPass` iterating all `SimLeaf`s and calling
`step()` on each would produce one compute dispatch per cell, creating CPU-side
overhead proportional to cell count.

**The v2 design**: simulation cells are typed grid kernels. A
`ThermalDiffusionCell` is not one cell in the BVH — it is a simulation domain
that covers a region of space and is referenced by multiple BVH leaves. The
`SimulationPass` dispatches one kernel per simulation type across all active
domains of that type:

```rust
// In starlight-engine/src/graph/passes/simulation.rs
pub struct SimulationPass {
	thermal_domains: Vec<ThermalDomainHandle>,
	fluid_domains:   Vec<FluidDomainHandle>,
	// ... one Vec per simulation type
}

impl RenderNode for SimulationPass {
	fn writes(&self) -> &[ResourceHandle] {
		// all sim output buffers
	}

	fn record(&self, cmd: &mut CommandRecorder, pool: &ResourcePool) {
		// One vkCmdDispatch for ALL thermal domains (they share a kernel,
		// indexed by domain ID in the buffer)
		if !self.thermal_domains.is_empty() {
			cmd.dispatch_thermal_diffusion(
				pool.thermal_params_buffer(),
				pool.thermal_state_buffer(),
				self.thermal_domains.len() as u32,
			);
		}
		// One vkCmdDispatch for ALL fluid domains
		if !self.fluid_domains.is_empty() {
			cmd.dispatch_fluid(/* ... */);
		}
	}
}
```

A `HeightField` leaf references a thermal domain by ID. The leaf data struct
contains the domain ID and a region offset into the thermal state buffer.
`intersect_heightfield` reads the temperature at the leaf's region offset to
compute displacement.

This is O(simulation_types) dispatches per frame, not O(simulation_cells).
Adding a new cell of an existing type costs nothing in dispatch overhead.

### 2.12 Portal Recursion — Bounded Stack and Cycle Detection

The existing `flat_bvh.rs` already uses a 64-entry explicit stack for BVH
traversal. Portal recursion extends this stack model:

**Stack discipline**: the traversal stack holds `TraversalState` entries —
either a BVH node index in the current world, or a portal frame boundary marker.
Total entries: 32 (conservative, sufficient for any practical scene depth).

**Portal frame marker**:

```rust
// In starlight-types/src/bvh.rs
pub struct PortalFrame {
	pub sub_bvh_root:   u32,   // root node index in the flat BVH buffer
	pub ray_transform:  Mat4,  // transforms ray from parent to child space
	pub scale_factor:   f32,   // for camera speed adjustment
	pub world_id:       u32,   // sub-world identifier
}
```

**Maximum portal depth**: 16 nested portals. When the stack would exceed 16
portal frame markers, the portal leaf returns `LeafHit::miss()`. This is the
correct behavior — a ray that has passed through 16 nested worlds is
contributing negligible radiance.

**Cycle detection**: each sub-BVH root node carries a `last_visited_frame`
counter (updated by the CPU each frame). In the traversal shader, before
entering a portal:

```rust
if portal_frame.world_id == current_world_id {
	return LeafHit::miss();  // direct self-portal cycle
}
if bvh_roots[portal_frame.sub_bvh_root].last_frame == current_frame {
	return LeafHit::miss();  // already visited this world this frame
}
```

This prevents infinite loops in portal graphs. The cost is one buffer read per
portal traversal — negligible.

### 2.13 The Material / BSDF System

The material system is geometry-agnostic: every leaf returns
`SurfaceInteraction`, and the `ShadingPass` evaluates the material
referenced by `material_id` without knowing the leaf type. This is the
payoff of leaf sovereignty — adding a new leaf type requires no changes
to the material system.

### Material Dispatch

The `Material` enum dispatches to 9 concrete material types (6
scattering + 3 composition). Each material implements `sample()`,
`eval()`, `pdf()`, and `gpu_node()`. The GPU serialization flattens the
material tree into a flat `Vec<GpuMaterialNode>` buffer, indexed by
`material_id` in `SurfaceInteraction`.

On CPU: `Material` enum match → struct methods (standard Rust dispatch).

On GPU: `GpuMaterialType` (`#[repr(u32)]`) tag in `GpuMaterialNode`
→ `@switch` in the shading shader → per-type BSDF evaluation.

The enum dispatch is confirmed correct for \<10 material types
(Architecture Review comparison with pbrt-v4 TaggedPointer and MoonRay
component architecture — both are equivalent at this scale).

### BSDF Evaluation Operations

Every material implements three core operations:

1. `sample(wo, rng) → BsdfSample`: sample a direction, return
   direction + BSDF×cos + PDF + pdf_kind. The `pdf_kind` field enables
   per-sample delta routing for composition materials (Coated, Mix).

2. `eval(wo, wi) → Color3`: evaluate the BSDF at a direction pair.
   Returns `f × |cos θ_i|` (already multiplied by the cosine factor).

3. `pdf(wo, wi) → f64`: evaluate the PDF for a given direction pair.
   Used for MIS weight computation when the sampled direction came from
   a different distribution (e.g., light sampling).

### Per-Sample Metadata

`BsdfSample` carries `pdf_kind: PdfKind` which indicates whether the
sample is delta (specular reflection/refraction) or non-delta (glossy,
diffuse). This enables the integrator to route delta and non-delta paths
differently without calling `material.is_delta()` — which cannot know
which child a composition material will sample.

For GPU: `pdf_kind` maps to `BxDFFlags` in the shading shader. The
flags indicate reflection/transmission and specular/glossy/diffuse,
enabling the same per-sample routing on GPU.

### Layered Materials

`Coated { substrate, coating }` implements a smooth dielectric
clearcoat over an arbitrary substrate. The current implementation uses
an analytic Fresnel split:

if u < Fresnel(coating_ior, cos_theta) → coating reflection (delta)
else → substrate sample

This is correct for smooth (delta) coating. For rough clearcoat, the
split must be replaced with MIS between coating and substrate lobes
(LuxCore's `GlossyCoating` pattern: `w_coating = 0.5 * (1 + F_avg)`).
Additionally, rough clearcoat requires a GGX NDF evaluation for the
coating lobe — the `Coated` struct currently lacks a `roughness`
parameter. TODO: add `coating_roughness: f32` field and GGX evaluation
when rough coating is needed.

pbrt-v4's `LayeredBxDF` uses a Monte Carlo random walk through layers
(Guo et al. 2018) — more physically general but more expensive. The
analytic approach is adequate for the current learning scope; the MC
walk is a future extension if rough coating is needed.

### GPU Material Serialization

The existing `GpuMaterialBuffer` in  `raytrace-rs/material/gpu.rs`
flattens the material tree into a flat buffer:

GpuMaterialNode {
material_type: GpuMaterialType, // #[repr(u32)] tag
data: GpuMaterialData, // repr(C) per-type data
child_a: u32, // index into buffer (composition)
child_b: u32, // index into buffer (composition)
}

Composition materials (Mix, Coated) reference children by index,
forming a valid DAG. The GPU shader reads this buffer and dispatches
on `material_type` via `@switch`.

This serialization is tested (6 tests in  `material/mod.rs`) and
already produces correct output. The migration to `starlight-types` is
a move, not a rewrite.

### 2.14 Direct Lighting — Shadow Rays and MIS

Direct lighting is the most impactful rendering quality subsystem.
The current raytrace-rs uses next-event estimation without explicit
shadow rays — sampling a direction toward a light and tracing a ray
to see if it gets there. This is correct but extremely noisy when
occluders sit between hit points and lights.

### The Direct Lighting Algorithm

The `ShadingPass` computes direct lighting via next-event estimation
with explicit shadow rays, following pbrt-v4's `SampleLd` pattern:

1. Sample a light from the light buffer:
   light = light_buffer[random_light_index]

2. Sample a point on the light surface:
   light_point, light_pdf = light.sample(hit_point, rng)

3. Evaluate the BSDF toward the light point:
   wi = normalize(light_point - hit_point)
   f = material.eval(wo, wi) // already × cos

4. Trace a shadow ray (occlusion test):
   shadow_ray = Ray(hit_point, wi, t_max = distance_to_light - ε)
   visible = TraversalPass.occluded(shadow_ray) // boolean

5. Compute MIS weight (power heuristic):
   p_light = light_pdf // solid-angle PDF from hit point to light
   p_bsdf = material.pdf(wo, wi) // BSDF PDF for this direction
   weight = p_light² / (p_light² + p_bsdf²)

6. Accumulate contribution:
   if visible:
   L += f * light.radiance * weight / p_light

### Shadow Ray Execution

Shadow rays are short-circuit occlusion tests — they trace a ray
through the BVH and return only a boolean (hit or miss), not a
`SurfaceInteraction`. This is cheaper than full intersection because:

- No shading normal computation
- No UV interpolation
- No material lookup
- Early termination on first hit

The `TraversalPass` exposes two entry points:

- `traverse(ray) → Option<SurfaceInteraction>` — full traversal
- `occluded(ray) → bool` — shadow ray, early termination

Both share the same BVH traversal code; `occluded` simply returns
`true` as soon as any leaf is hit, without computing the full
`SurfaceInteraction`.

### MIS Strategy

The power heuristic replaces fixed mixture weights with adaptive
weights that respond to the relative quality of each sampling technique:

weight_i = pdf_i² / Σ(pdf_j²)

When light sampling has a much higher PDF than BSDF sampling (narrow
cone from hit point to small light), the power heuristic gives most
weight to light sampling — correct because light sampling is more
efficient in this regime. When BSDF sampling is better (glossy
surfaces, large lights), BSDF gets more weight.

This is the standard Veach MIS approach used by pbrt-v4, LuxCore, and
MoonRay. The fixed `[1/3, 2/3]` weights in raytrace-rs are unbiased
but suboptimal — they don't adapt to the scene.

### 2.15 Bounce Control and Russian Roulette

### Per-Type Bounce Limits

The `ShadingPass` enforces per-type bounce limits to control noise
and performance:

  max_diffuse_depth  = 5   // diffuse bounces (noisy, low contribution)
  max_glossy_depth   = 8   // glossy bounces (moderate noise)
  max_specular_depth = 12  // specular bounces (noiseless, need many)

When a bounce exceeds its type limit, the path terminates. This is
standard practice — LuxCore and MoonRay both use per-type limits.

The bounce type is determined by  `BsdfSample.flags`:
- DIFFUSE flag → diffuse bounce counter
- GLOSSY flag → glossy bounce counter
- SPECULAR flag → specular bounce counter

### Russian Roulette

After the minimum bounce threshold (5), survival probability is
proportional to the maximum component of the accumulated throughput:

```rust
if bounce >= 5:
	survival = max(throughput.r, throughput.g, throughput.b)
	survival = clamp(survival, 0.05, 1.0)
	if random > survival: terminate path
	throughput /= survival  // unbias
```

This is the standard Russian roulette from pbrt-v4 and LuxCore. The
floor clamp (0.05) prevents paths from being terminated too early in
dark regions, which would bias the result.

### 2.16 Light Buffer — Storage and Sampling

Direct lighting requires a flat, GPU-readable buffer of all light
sources in the scene. The `ShadingPass` reads this buffer to sample
light positions and evaluate light PDFs.

### Light Discovery

Lights are not declared separately. At scene load time, the host
walks the `LeafNode` list and identifies emissive leaves — those whose
material has `is_emissive() == true`. Each emissive leaf becomes a
`LightNode` in the flat light buffer. This eliminates the manual
`add_light()` ceremony from raytrace-rs.

```
Scene::build():
    for (i, leaf) in leaf_nodes.iter().enumerate():
        if leaf.material().is_emissive():
            let r = leaf.material().emitted();
            light_buffer.push(LightNode {
                leaf_index:  i as u32,
                area:        leaf.aabb().surface_area(),  // approx
                radiance:    [r.x, r.y, r.z],
                luminance:   0.2126 * r.x + 0.7152 * r.y + 0.0722 * r.z,
            })
```

### LightNode Layout

```rust
// In starlight-types/src/light.rs
//
// Buffer layout convention: use [f32; 3] instead of Vec3 for any type
// that lives in a GPU buffer. See Section 2.6 for why.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightNode {
    pub leaf_index: u32,          // offset 0  — index into leaf buffer
    pub area:       f32,          // offset 4  — surface area of the light (for PDF)
    pub radiance:   [f32; 3],     // offset 8  — emitted radiance (RGB)
    pub luminance:  f32,          // offset 20 — pre-computed for power-based selection
}
// 24 bytes, repr(C), bytemuck::Pod.
// [f32; 3] has 4-byte alignment — no std430 vec3 alignment surprise.
// Upload via bytemuck::cast_slice(&[LightNode]) → &[u8]. Zero-copy.
```

**Why `[f32; 3]` instead of `Vec3`**: GLSL `vec3` has 16-byte alignment in
SSBOs (std140/std430). A `repr(C)` struct with `Vec3` packs tightly on CPU
(24 bytes) but the GPU expects padding (32 bytes). Using `[f32; 3]` avoids
this entirely — `[f32; 3]` compiles to SPIR-V `OpTypeArray float 3` with
4-byte element alignment. No alignment surprise, no `gpu_layout` crate, no
manual padding. See the buffer layout convention in Section 2.6.

### Light Sampling

The `ShadingPass` samples lights uniformly at first (adequate for
<10 lights). Power-based selection is a later optimization:

```
// Uniform selection (initial)
fn sample_light(hit_point: Vec3, rng: Rng) -> (Vec3, f32, f32) {
    let idx = rng.u32(0, light_count);
    let light = light_buffer[idx];
    let light_point = sample_point_on_light(light, rng);
    let to_light = light_point - hit_point;
    let dist = to_light.length();
    let wi = to_light / dist;
    // Solid-angle PDF: probability of selecting this light ×
    // probability of sampling this point on the light surface
    let p_light = (1.0 / light_count as f32) * (dist * dist / light.area);
    (light_point, light.radiance, p_light)
}
```

### Light Buffer Upload

The light buffer is built once at scene load and uploaded to a
`VK_BUFFER_USAGE_STORAGE_BUFFER_BIT` SSBO. The `ShadingPass` reads
it as `&[GpuLightNode]`. When simulation displaces geometry (e.g.,
thermal diffusion moves a HeightField), the light buffer is rebuilt
only if a light leaf was displaced — not every frame.

### Future: Power-Based Selection

For scenes with many lights (50+), uniform selection wastes samples
on dim lights. Power-based selection weights by `light.area *
light.radiance.luminance()`:

```
weight[i] = light[i].area * luminance(light[i].radiance)
cdf[i]    = sum(weight[0..=i]) / sum(weight[0..N])
sample:   binary search on CDF
```

This is LuxCore's `Power` strategy. The `DLSCache` (direct light
sampling cache) is more sophisticated but deferred — power-based
selection covers the common case.

### 2.17 Transform System — Completing the Foundation

raytrace-rs currently has only `Translate` and `RotateY`. The
`Transform` trait requires implementing `hit`, `bbox`, `ray`,
`object_to_world_direction` — heavy per-transform. Every new
transform duplicates the same pattern.

### The Problem

Each `TransformObject<T: Transform, O: Hittable>` wraps an inner
object with a transform. Adding `RotateX`, `RotateZ`, `Scale`, and
composition means 4+ new types, each with identical boilerplate.

### The Solution: Mat4 Everywhere

pbrt-v4 stores a single `Transform` (4×4 matrix pair:
`renderFromObject` and `objectFromRender`) on every shape. No type
parameter per operation. Starlight follows this pattern:

On the **host side**: transforms are `Mat4` stored in leaf data
structs. `AnalyticSphere` gets a `center: Vec3` field (already has
this). `TriangleMesh` gets a `transform: Mat4` field. The inverse is
computed once at scene load and stored alongside the forward matrix.

On the **GPU side**: leaf data structs store a `TransformPair` — both
the forward and inverse matrices. The traversal shader reads
`leaf.transform.object_from_world` directly instead of inverting at
intersection time. Matrix inversion is ~20 FLOPs and a branch; with
millions of rays per frame, this is significant:

```rust
// In starlight-types/src/leaf.rs
#[repr(C)]
pub struct TransformPair {
    pub world_from_object: Mat4,   // forward — for transforming points out
    pub object_from_world: Mat4,   // inverse — for transforming rays in
}
```

The shader code becomes:

```rust
fn intersect_sphere(ray: Ray, leaf: &SphereLeaf) -> Option<SurfaceInteraction> {
    // Transform ray to object space — read pre-computed inverse, no inversion
    let local_origin = leaf.transform.object_from_world * ray.origin.extend(1.0);
    let local_dir = leaf.transform.object_from_world * ray.direction.extend(0.0);
    // ... standard sphere intersection in object space ...
    // Transform hit back to world space
    let world_point = leaf.transform.world_from_object * hit_point.extend(1.0);
    // Normal transform: inverse transpose, but since inverse is stored,
    // use transpose of the inverse (or equivalently, the adjugate).
    // For uniform-scale transforms, world_from_object.transpose() suffices.
    let world_normal = leaf.transform.world_from_object.transpose() * normal.extend(0.0);
}
```

### What This Eliminates

- `Transform` trait with 5 methods → gone
- `TransformObject<T, O>` generic → gone
- `RotateX`, `RotateZ`, `Scale` types → gone
- Composition helpers → not needed (matrix multiplication)

### What This Requires

- Every leaf data struct that supports transforms stores a `TransformPair`
  (128 bytes: two `Mat4`). For analytic shapes this is optional; for
  meshes it is required.
- The inverse matrix is computed once at scene load and stored in the
  `TransformPair`. GPU shaders receive both matrices — no runtime
  inversion.
- `onb.rs` (orthonormal basis from normal) stays — it is used by
  BSDF evaluation, not by transforms.

### Migration Path

The `Transform` trait and `TransformObject` wrapper are removed in
Stage 0 (workspace setup). Existing scenes that use `Translate` and
`RotateY` are rewritten to store the composed `Mat4` directly in
leaf data. This is a one-time migration with no behavioral change.

### 2.18 Unified CPU/GPU Abstraction Layer

The renderer maintains two execution paths — CPU path tracer and GPU
compute shader — that must produce identical output for the same scene.
The abstraction layer ensures both paths implement the same algorithmic
interface, so algorithmic changes propagate to both paths by
construction, not by discipline.

### The Problem

Without a shared abstraction, CPU and GPU implementations diverge:
the CPU path tracer evaluates materials via `Material::eval()`, while
the GPU shader evaluates via `@switch` on `GpuMaterialType`. If one
changes and the other doesn't, they silently produce different output.
The divergence is invisible until a visual regression appears.

### The Solution: Trait-Defined Interface, Convention-Bound GPU

Define the algorithmic interface as Rust traits. CPU implements these
traits directly. GPU implements these as shader functions that follow
the same interface by convention — same function signatures, same
semantics, same data types. The shared `starlight-types` crate enforces
data type compatibility at compile time.

```rust
// In starlight-engine/src/traits.rs

/// Ray-scene intersection. CPU: BVH traversal.
/// GPU: same BVH traversal in traversal.rs shader.
pub trait Intersectable {
    fn intersect(&self, ray: &Ray) -> Option<SurfaceInteraction>;
    fn occluded(&self, ray: &Ray, t_max: f32) -> bool;
}

/// Material evaluation. CPU: Material enum match.
/// GPU: @switch on GpuMaterialType in shading.rs shader.
pub trait Shading {
    fn eval(&self, wo: Vec3, wi: Vec3) -> Color3;
    fn sample(&self, wo: Vec3, rng: &mut Rng) -> BsdfSample;
    fn pdf(&self, wo: Vec3, wi: Vec3) -> f64;
    fn emitted(&self) -> Color3;
    fn is_emissive(&self) -> bool;
}

/// Light source sampling. CPU: LightNode buffer iteration.
/// GPU: same buffer, same iteration in shading.rs shader.
pub trait LightSampling {
    fn sample(&self, point: Vec3, rng: &mut Rng) -> (Vec3, Color3, f32);
    fn pdf(&self, point: Vec3, wi: Vec3) -> f32;
    fn area(&self) -> f32;
    fn radiance(&self) -> Color3;
}

/// Texture evaluation. CPU: Texture enum match.
/// GPU: @switch on GpuTextureType in shading.rs shader.
pub trait Texturing {
    fn sample(&self, uv: [f32; 2]) -> Color3;
    fn sample_normal(&self, uv: [f32; 2]) -> Vec3;
    fn sample_roughness(&self, uv: [f32; 2]) -> f32;
}
```

### CPU Implementations

These are the current raytrace-rs types, which already implement the
correct algorithms:

```
Intersectable → BvhNode<LeafNode> (iterative traversal)
Shading       → Material enum (match → struct methods)
LightSampling → Vec<LightNode> (uniform selection, solid-angle PDF)
Texturing     → Texture enum (match → image sample)
```

### GPU Implementations

The GPU "implements" these traits as shader functions that read from
serialized buffers. The function signatures match the trait signatures
by convention:

```
gpu_intersect(ray, bvh, leaf_types, leaf_data)
    → Option<SurfaceInteraction>     // same return type

gpu_shade(material, wo, rng)
    → BsdfSample                     // same return type

gpu_sample_light(lights, point, rng)
    → (Vec3, Color3, f32)            // same return type

gpu_sample_texture(texture, uv)
    → Color3                         // same return type
```

The GPU functions cannot implement Rust traits directly (rust-gpu
doesn't support trait objects in SPIR-V). But they follow the same
interface, operate on the same data types (from `starlight-types`), and
produce the same results. The shared types crate enforces this at
compile time — if `SurfaceInteraction` changes, both CPU and GPU code
must adapt.

### Data Format Conversion

The conversion from CPU scene representation to GPU buffer layout is
the bridge between the two execution paths. Every CPU type that
participates in rendering has a `Gpu*` counterpart:

```
CPU Type                GPU Type                Conversion
──────────────────────  ──────────────────────  ──────────────────
Scene                   GpuSceneBuffers         Scene::flatten()
Material (enum)         GpuMaterialNode          Material::gpu_node()
Texture (enum)          GpuTextureNode           Texture::gpu_node()
LeafNode (enum)         GpuLeafType + LeafData   LeafNode::serialize()
BvhNode<LeafNode>       GpuBvhNode[]             BvhNode::flatten()
LightNode               LightNode (repr(C))      direct copy
HitRecord               SurfaceInteraction       HitRecord::to_interaction()
```

The `Scene::flatten()` method produces all GPU buffers in one pass:

```rust
pub struct GpuSceneBuffers {
    pub bvh_nodes:      Vec<GpuBvhNode>,
    pub leaf_types:     Vec<u32>,
    pub leaf_data:      Vec<u8>,
    pub materials:      Vec<GpuMaterialNode>,
    pub textures:       Vec<GpuTextureNode>,
    pub texture_images: Vec<GpuImageHandle>, // VkImage handles, one per texture
    pub lights:         Vec<LightNode>,
    pub camera:         GpuCameraParams,
}
```

**Hardware texture sampling**: The `texture_atlas` approach (SSBO with
software sampling) is replaced by `VkImage` objects with bindless
descriptors. Each texture is uploaded as a separate `VkImage` with
mipmaps, backed by `VkImageView` and `VkSampler`. The
`GpuTextureNode.image_index` field indexes into a bindless texture
array bound via `VK_EXT_descriptor_indexing`. The traversal shader
calls `texture(sampler2DArray, uv)` — hardware bilinear filtering,
anisotropic filtering, and the GPU texture cache all work. The atlas
approach gives up all of this and is not suitable for a real-time
renderer.

The `ResourcePool` manages the `VkImage` lifecycle. Scene loading
uploads textures as staged transfers; the `GpuImageHandle` is an
opaque handle into the pool. The `GpuTextureNode` stores the
`sampler_index` and `image_index` for bindless access.

Each `RenderNode` receives the buffers it needs via `ResourcePool`.
The `TraversalPass` receives BVH + leaf + material buffers. The
`ShadingPass` receives material + light + texture buffers. The render
graph barrier model ensures correct ordering.

### Why This Matters

1. **Algorithmic consistency**: changing `Shading::eval()` on CPU
   reminds you to update `gpu_shade()` in the shader. The shared types
   enforce data compatibility.

2. **Testing**: the CPU path is the reference. Any algorithmic change
   can be validated on CPU first (deterministic, debuggable), then
   ported to GPU with confidence.

3. **Extensibility**: adding a new material type means adding a variant
   to the CPU `Material` enum AND a `@switch` arm in the GPU shader.
   The shared `GpuMaterialType` enum enforces exhaustiveness in both.

4. **Scene format support**: loaders produce the CPU `Scene`
   representation. The `flatten()` method converts to GPU buffers.
   Loaders don't know about GPU specifics — they produce typed Rust
   structs that serialize cleanly.

### 2.19 Scene Format Support — glTF and OBJ

Production renderers load scenes from standard formats. raytrace-rs
currently has no loader — all geometry is hand-coded. Starlight needs
to load glTF 2.0 (the modern standard) and OBJ (legacy compatibility)
to be comparable to LuxCore, MoonRay, and pbrt-v4.

### glTF 2.0 — The Primary Format

glTF is the "JPEG of 3D" — a compact, vendor-neutral format designed
for runtime delivery. It is the primary input format for Starlight.

**Core features required:**

| Feature | glTF Element | Starlight Mapping |
|---------|-------------|-------------------|
| Triangle meshes | `mesh.primitives` | `TriangleMesh` leaf |
| PBR materials | `material.pbrMetallicRoughness` | `Material::Glossy` + textures |
| Texture maps | `image` + `sampler` + `texture` | `Texture::Image` + atlas |
| Transforms | `node.translation/rotation/scale` | `Mat4` in leaf data |
| Cameras | `camera.perspective/orthographic` | `Camera` params |
| Scenes | `scene.nodes` | `Scene` root |
| Node hierarchy | `node.children` | Parent-child transform composition |

**PBR metallic-roughness mapping to Starlight materials:**

```
glTF Material                          Starlight Material
─────────────────────────────────────  ─────────────────────────────
baseColorFactor = [r,g,b,1]           Lambertian { albedo: Color3 }
baseColorFactor = [r,g,b,1] +         Glossy (GGX dielectric)
  metallicFactor > 0.5                    with metallic/albedo textures
metallicFactor = 1, roughness = 0      Metal (GGX conductor)
emissiveFactor = [r,g,b]             DiffuseLight { color: Color3 }
alphaMode = "BLEND"                   Material with alpha channel
alphaMode = "MASK"                    Alpha test (cutout)
doubleSided = true                    Flip normal if back-face hit
```

**Alpha-tested geometry handling**: Cutout textures (`alphaMode = "MASK"`)
are excluded from BVH traversal. The traversal shader has no mechanism
to discard hits based on alpha — it returns the first geometric hit
that passes the t-interval test. For cutout geometry, the raster path
handles it: `MeshRasterPass` uses `discard` in the fragment shader
when `alpha < cutoff`. This is consistent with the two-execution-path
model: primary visibility of cutout geometry comes from raster, and
secondary rays (reflections, shadows) miss that geometry. For foliage,
grilles, and similar cutout surfaces, this is acceptable. The
`alphaMode = "BLEND"` case (semi-transparent) is deferred — it requires
order-independent transparency or a separate sorting pass, which is a
later extension.

**glTF extensions to support (in priority order):**

| Extension | What It Adds | Priority |
|-----------|-------------|----------|
| `KHR_lights_punctual` | Point, spot, directional lights | P0 — needed for lit scenes |
| `KHR_materials_unlit` | Unlit material (emissive constant) | P1 — simple, useful for UI |
| `KHR_materials_clearcoat` | Clear coating layer | P2 — maps to `Coated` |
| `KHR_materials_transmission` | Transmission (glass-like) | P2 — maps to `Dielectric` |
| `KHR_materials_ior` | Index of refraction override | P2 — configurable IOR |
| `KHR_materials_specular` | Specular color/factor | P3 — fine-tuning |
| `KHR_materials_emissive_strength` | Emissive intensity multiplier | P3 — HDR emissives |

**Deferred extensions** (not needed for initial support):

- `KHR_animation_pointer` — animation system, deferred to Stage 10+
- `KHR_materials_iridescence` — thin-film interference, complex
- `KHR_materials_volume` — volumetric absorption, needs volume rendering
- `KHR_materials_sheen` — fabric/sheen lobe, additional BxDF
- `KHR_materials_dispersion` — chromatic dispersion, spectral rendering

**glTF data layout:**

```
glTF file (JSON + binary)
  ├── .gltf (JSON) — scene graph, materials, accessors
  ├── .bin (binary) — vertex buffers, index buffers, animation data
  └── .png/.jpg — texture images

Loading pipeline:
  1. Parse JSON → scene graph (nodes, meshes, materials, cameras)
  2. Parse binary → raw vertex/index data (via accessors + bufferViews)
  3. Decode images → RGBA pixel data
  4. Build Scene:
     a. Compose node transforms (parent × child → Mat4)
     b. Create TriangleMesh leaves for each mesh primitive
     c. Create Material instances for each glTF material
     d. Create Texture instances for each image
     e. Create LightNode entries for KHR_lights_punctual lights
     f. Create Camera from glTF camera
  5. Build BVH from all leaves
  6. Call Scene::flatten() → GpuSceneBuffers
```

### OBJ — Legacy Compatibility

OBJ is simpler but still useful for quick testing and legacy scenes.

**Core features:**

| Feature | OBJ Directive | Starlight Mapping |
|---------|--------------|-------------------|
| Vertices | `v x y z` | Vertex positions |
| Normals | `vn x y z` | Vertex normals |
| UVs | `vt u v` | Texture coordinates |
| Faces | `f v/vt/vn` | Triangle indices (triangulate ngons) |
| Groups | `g name` / `usemtl name` | Material group boundaries |
| Materials | `mtllib name` | MTL file parsing |
| Smooth shading | `s off/1` | Flat vs smooth normals |

**MTL material mapping:**

```
MTL Field                           Starlight Material
──────────────────────────────────  ─────────────────────────────
Kd r g b                            Lambertian { albedo }
Ks r g b + Ns exp                   Glossy { specular_color, roughness }
Ka r g b                            DiffuseLight { color }
d alpha (or Tr 1-alpha)             Alpha / transparency
map_Kd filename                     Texture::Image (albedo)
map_Ks filename                     Texture::Image (specular)
map_Bump filename                   Normal map (future)
Ni ior                              Dielectric { ior }
```

**OBJ limitations** (vs glTF):
- No PBR metallic-roughness (only Blinn-Phong via Ks/Ns)
- No animation, no scene hierarchy (flat object list)
- No embedded binary data (separate .obj + .mtl + texture files)
- Ngons must be triangulated (fan triangulation)
- No instancing (duplicate geometry)

OBJ support is implemented in Stage 2 alongside triangle mesh support.
glTF support is implemented in a dedicated stage after the shared types
are stable.

### Unified Loading Pipeline

Both formats feed into the same `Scene` representation:

```
                     ┌─────────────┐
                     │  glTF file  │
                     └──────┬──────┘
                            │
                     ┌──────▼──────┐
                     │  GltfLoader  │
                     └──────┬──────┘
                            │
                            ▼
                     ┌─────────────┐
                     │    Scene     │  ← unified CPU representation
                     │  LeafNode[]  │
                     │  Material[]  │
                     │  Texture[]   │
                     │  Light[]     │
                     │  Camera      │
                     └──────┬──────┘
                            │
                     ┌──────▼──────┐
                     │  OBJ file   │
                     └──────┬──────┘
                            │
                     ┌──────▼──────┐
                     │   ObjLoader  │
                     └──────┬──────┘
                            │
                            ▼
                     ┌─────────────┐
                     │    Scene     │  ← same representation
                     └──────┬──────┘
                            │
                     ┌──────▼──────┐
                     │ Scene::flatten() │
                     └──────┬──────┘
                            │
                            ▼
                     ┌─────────────┐
                     │ GpuSceneBuffers │  ← GPU-ready buffers
                     └─────────────┘
```

The loaders produce a `Scene` struct. The `flatten()` method converts
to GPU buffers. Loaders don't know about GPU specifics — they produce
typed Rust structs. The GPU path doesn't know about file formats — it
reads serialized buffers. The `Scene` struct is the single source of
truth.

______________________________________________________________________

## 3. Module Layout

Three-crate workspace. Existing `raytrace-rs` files are mapped explicitly.

```
starlight/
│
├── types/                          starlight-types  [no_std]
│   └── src/
│       ├── lib.rs
│       ├── interaction.rs          SurfaceInteraction       ← from HitRecord TODO
│       ├── ray.rs                  Ray                      ← raytrace-rs ray.rs
│       ├── bvh.rs                  GpuBvhNode, PortalFrame  ← raytrace-rs flat_bvh.rs
│       ├── leaf.rs                 GpuLeafType, GpuLeafData ← new
│       ├── light.rs                LightNode                ← new (Section 2.16)
│       ├── material.rs             GpuMaterialType/Node     ← raytrace-rs material/gpu.rs
│       ├── texture.rs              GpuTextureType/Node      ← raytrace-rs texture/gpu.rs
│       └── math/
│           ├── vec3.rs             Vec3, Vec4               ← raytrace-rs vec3.rs
│           ├── mat4.rs             Mat4                     ← new
│           └── aabb.rs             Aabb                     ← raytrace-rs aabb.rs
│
├── shaders/                        starlight-shaders  [spirv target]
│   └── src/
│       ├── lib.rs                  shader crate root
│       ├── traversal.rs            BVH traversal + leaf dispatch entry point
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
		├── traits.rs              Intersectable, Shading, LightSampling, Texturing
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
		│       ├── bvh_build.rs    BvhBuildPass (BLAS/TLAS if RT enabled)
		│       ├── mesh_raster.rs  MeshRasterPass (dynamic rendering)
		│       ├── foam_raster.rs  PowerFoamRasterPass (tile splat)
		│       ├── traversal.rs    TraversalPass (Starlight compute dispatch)
		│       ├── shading.rs      ShadingPass (unified PBR)
		│       └── post.rs         PostProcessPass (TAA, tonemap)
		│
		├── scene/
		│   ├── mod.rs              Scene: BvhNode<LeafNode>, material buffer, light buffer
		│   ├── bvh.rs              BvhNode<LeafNode> tree  ← raytrace-rs bvh.rs
		│   ├── light.rs            Light discovery + LightNode construction (Section 2.16)
		│   ├── flatten.rs          Scene::flatten() → GpuSceneBuffers (Section 2.18)
		│   ├── gltf.rs             glTF 2.0 loader (Section 2.19)
		│   ├── obj.rs              OBJ + MTL loader (Section 2.19)
		│   └── leaf/
		│       ├── mod.rs          leaf_nodes! macro + LeafNode enum
		│       ├── analytic.rs     AnalyticSphere, AnalyticAabb ← raytrace-rs sphere.rs
		│       ├── trimesh.rs      TriangleMesh             ← raytrace-rs planar/tri.rs
		│       ├── sdf.rs          SdfVolume
		│       ├── heightfield.rs  HeightField
		│       ├── fractal.rs      FractalParams
		│       ├── medium.rs       ConstantMedium           ← raytrace-rs const_medium.rs
		│       ├── portal.rs       Portal
		│       └── power_foam/
		│           ├── mod.rs      PowerFoamCell
		│           ├── diagram.rs  bounded power diagram, Čech complex
		│           ├── surface.rs  dipole face, detail sites
		│           └── radiance.rs spherical Voronoi directional radiance
		│
		├── material/
		│   ├── mod.rs              Material enum      ← raytrace-rs material/mod.rs
		│   ├── interaction.rs      SurfaceInteraction (wraps types/ version)
		│   ├── scatter.rs          BSDF, importance sampling, MIS
		│   └── texture.rs          Texture enum       ← raytrace-rs texture/mod.rs
		│
		└── simulation/
			├── mod.rs              SimulationDomain trait
			├── thermal.rs          ThermalDomain: grid state, GPU buffer
			├── fluid.rs            FluidDomain
			└── fracture.rs         FractureDomain
```

______________________________________________________________________

## 4. The `SurfaceInteraction` Contract

Every leaf intersection function — in both the Rust CPU path tracer and the
rust-gpu shader — returns the same type, defined once in `starlight-types`:

```rust
// In starlight-types/src/interaction.rs
// This is the SurfaceInteraction called for in raytrace-rs hittable.rs TODO
//
// Buffer layout convention: [f32; 3] not Vec3 (see Section 2.6).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SurfaceInteraction {
	pub point:        [f32; 3],  // world-space hit position
	pub normal:       [f32; 3],  // shading normal
	pub geo_normal:   [f32; 3],  // geometric normal (for self-intersection offset)
	pub uv:           [f32; 2],
	pub t:            f32,        // ray parameter at hit
	pub material_id:  u32,        // index into flat material buffer
	pub object_id:    u32,        // per-instance data, simulation coupling
	pub world_id:     u32,        // which sub-world (0 = root, portal depth > 0 otherwise)
}
// 48 bytes. Convert to Vec3 at read time: Vec3::from(interaction.point).
```

**Why no `flags` or `eta` in `SurfaceInteraction`**: These describe the
*result* of a scattering event, not the geometry. A leaf doesn't know
whether the surface it hit is specular or diffuse — that depends on the
material parameters, which are evaluated by the `ShadingPass`. The
`ShadingPass` reads `material_id` from `SurfaceInteraction`, looks up
the material, evaluates the BSDF, and produces a `BsdfSample` that
contains `flags` (BxDFFlags) and `eta` (IOR ratio). This matches
pbrt-v4's design where `flags` and `eta` live in `BSDFSample`, not in
the hit record. The only exception is `eta` for nested media
transmission, which is tracked in the path state (`current_eta`), not
in the hit record.

The `ShadingPass` operates entirely on `SurfaceInteraction`. It does not know
whether the hit came from an analytic sphere, an SDF ray march, a foam cell
walk, or a fractal distance estimator. PBRT's BSDF evaluation, importance
sampling, and MIS apply uniformly. This is the payoff of leaf sovereignty:
adding a new leaf type requires no changes to the shading system.

The CPU path tracer converts `HitRecord` to `SurfaceInteraction` at the call
site. `HitRecord` retains its borrowed `&'rec Material` reference for the CPU
path; the GPU path uses `material_id` to index the flat material SSBO. Both
routes produce the same `SurfaceInteraction` for identical scene configurations
— the CPU path tracer is the reference validator.

BxDFFlags (for reference — stored in `BsdfSample`, not here):
BxDFFlags::REFLECTION = 0x01
BxDFFlags::TRANSMISSION = 0x02
BxDFFlags::DIFFUSE = 0x04
BxDFFlags::GLOSSY = 0x08
BxDFFlags::SPECULAR = 0x10

The `pdf_is_proportional` flag from pbrt-v4 is not needed — all
materials in raytrace-rs produce exact PDF values. If layered materials
with MC random walk are added later, this flag can be added to
`BsdfSample` without breaking the `SurfaceInteraction` contract.

______________________________________________________________________

## 5. Design Invariants

Eight invariants. These are not preferences — breaking any one collapses a
load-bearing assumption.

1. **The BVH is the universal spatial address space** for both rendering and
   simulation. No second acceleration structure, no separate raster scene graph.

2. **Every leaf returns `SurfaceInteraction`**. The shading pass is completely
   agnostic to leaf type. No leaf has a special shading path.

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
   grid- style GPU kernels. `SimulationPass` dispatches O(simulation_types)
   kernels per frame, not O(simulation_cells).

8. **Primary visibility arbitration is explicit.** GBuffer is primary; HitBuffer
   is secondary effects. `ShadingPass` documents the selection rule for pixels
   with and without GBuffer coverage.

______________________________________________________________________

## 6. Implementation Sequence

Each stage produces a runnable, visually verifiable output. The existing CPU
path tracer is never broken — it runs in parallel as the reference
implementation.

The sequence follows Review 1's recommendation: **CPU-only stages first**,
then **incremental GPU stages** with raster added *after* the full GPU path
tracer. This ensures the compute path tracer is the reference for every pixel
before raster is introduced as an optimization.

### Stage 0 — Arena refactor + workspace setup + shared types crate

**Arena refactor** (from `docs/arena-refactor-plan.md`, ~361 lines, 8 files):
Replace `Arc<dyn Hittable>` with `Vec<Box<dyn Hittable>>` in Scene. The BVH
lifetime-borrows from the Scene's object list. Automatic light detection via
`is_emissive()` — no more manual `add_light()` ceremony. Light BVH built from
emissive indices in the sorted objects. This eliminates Arc overhead, removes
scattered allocations, and moves storage toward GPU readiness.

**Transform system cleanup**: Remove `Transform` trait and `TransformObject<T,O>`
wrapper. Store `Mat4` directly in leaf data structs (Section 2.17). Rewrite
existing `Translate` and `RotateY` scenes to composed `Mat4`.

**Workspace setup**: Create the three-crate workspace. Move `material/gpu.rs`,
`texture/gpu.rs`, and `flat_bvh.rs` GPU-facing types from `raytrace-rs` into
`starlight-types`. Verify that `starlight-engine` re-imports them and the
existing CPU renderer still compiles and produces identical output.

### Stage 1 — `LeafNode` enum + `SurfaceInteraction` (CPU only)

Introduce the `leaf_nodes!` macro and `LeafNode` enum in `starlight-engine`.
Replace `Arc<dyn Hittable>` in `BvhNode` leaves with `LeafNode`. Extract
`SurfaceInteraction` from `HitRecord` (the `TODO(renderer-agnostic)` in
`hittable.rs` has been waiting for this). Verify identical render output from
the CPU path tracer. This eliminates vtable dispatch from the BVH hot path.

### Stage 2 — Triangle mesh support + OBJ loader (CPU only)

The biggest geometric gap. Add `TriangleMesh` leaf type with indexed vertex
buffers (positions, normals, UVs) and per-triangle index triples. Implement
Möller–Trumbore ray-triangle intersection on CPU. Add a minimal OBJ parser on
the host side — positions (`v`), normals (`vn`), UVs (`vt`), faces (`f`), and
material groups (`usemtl`). Verify by loading an OBJ file and rendering it on
the CPU path tracer.

This unlocks all real-world scenes. The mesh data is serialized into the flat
leaf data buffer as a `GpuTriangleMesh` struct (vertex count, index count,
offsets into vertex/index sub-buffers). The `GpuLeafType::TriangleMesh` variant
in the traversal shader reads this struct and dispatches to `intersect_trimesh`.

### Stage 3 — rust-gpu shader crate baseline (first GPU leap)

Add `starlight-shaders` to the workspace. Configure the rust-gpu toolchain.
Write a trivial compute shader (one thread writes a gradient to a storage image)
using `spirv-std` builtins. Load the compiled SPIR-V in a minimal `ash` Vulkan
context and display the result. This validates the rust-gpu build pipeline
before shader logic is added. Test each feature incrementally — avoid heavy use
of closures, iterators, or dynamic dispatch inside shaders.

### Stage 4 — Triangle mesh intersection on GPU

Port Möller–Trumbore to the shader. Upload the flat BVH, leaf data, and
material buffers using `starlight-engine`'s `ResourcePool`. Output matches the
CPU reference render for analytic spheres and triangle meshes. This is the first
`TraversalPass` implementation. All GPU types come from `starlight-types` — the
same structs used in both `raytrace-rs` serialization and the shader.

### Stage 5 — Direct lighting + shadow rays on GPU

Add the `ShadingPass` as a GPU compute shader. For each pixel:
1. Read HitBuffer (primary hit from TraversalPass)
2. Look up material via `material_id`
3. Sample light from `LightNode` buffer
4. Eval BSDF toward light
5. Shadow ray via `TraversalPass.occluded()` — separate entry point that shares
   the traversal loop but returns `bool` on first hit without computing
   `SurfaceInteraction`
6. Power heuristic MIS weight
7. Accumulate: `L += f * radiance * weight / p_light`

The `occluded()` function must not be a full `traverse_bvh()` that returns a
`SurfaceInteraction`. Implement it as a separate shader function that shares
the traversal loop but returns `bool` on first hit. The `match leaf_type`
inside `occluded()` can be a fast path that only checks `is_leaf` and returns
`true` without computing normals/UVs.

Also extend the CPU path tracer to use explicit shadow rays with the same
algorithm. Compare output pixel-by-pixel with the GPU. This is the first
validation of the shading pipeline.

### Stage 6 — Indirect lighting, Russian roulette, bounce limits (complete GPU path tracer)

Complete the GPU path tracer with indirect lighting:
1. Sample BSDF direction
2. Trace secondary ray (another `traverse_bvh`)
3. Accumulate: `L += f_cos * 1/p_mixture`
4. Russian roulette: survival probability = `max(throughput)`, start at bounce 5
5. Enforce per-type bounce limits: `max_diffuse_depth`, `max_glossy_depth`,
   `max_specular_depth`

Use a shared RNG implementation in `starlight-types` with
`#[cfg(not(target_arch = "spirv"))]` for CPU and
`#[cfg(target_arch = "spirv")]` for GPU. Deterministic comparison requires
identical random number sequences on both sides.

This is the Starlight equivalent of pbrt-v4's `PathIntegrator`, operating on
`SurfaceInteraction` independently of geometry type. The CPU path tracer in
raytrace-rs serves as the reference implementation — identical algorithm,
different execution target.

### Stage 7 — Render graph + raster path

Now that the full GPU path tracer works for every pixel, add raster as an
optimization for primary visibility. Introduce `RenderGraph` with
`FramePassNode`. Add `BvhCullPass` (compute, outputs draw lists) and
`MeshRasterPass` (dynamic rendering, writes GBuffer) for triangle mesh leaves.
Add `ShadingPass` reading GBuffer + HitBuffer, applying the primary visibility
arbitration rule. The first frame where both raster and ray contributions are
unified in one lighting model.

Since the compute path tracer already produces correct output for every pixel,
you can validate that the GBuffer matches the ray-traced primary within
tolerance. This reduces the risk of debugging two execution paths at once.

### Stage 8 — Additional leaf types in shader

Add `GpuLeafType::SdfVolume` (ray march a hardcoded sphere SDF). Add
`GpuLeafType::Portal` with the bounded stack and cycle detection. At this point:
one dispatch, multiple sovereign leaf types, portals working. This is the
Starlight architecture running.

### Stage 9 — Simulation coupling

Add `SimulationPass` with `ThermalDomain` as the first domain type. Implement
the thermal diffusion kernel in `starlight-shaders/src/simulation/thermal.rs`.
Wire `HeightField` leaf to read the domain's output buffer. The render graph
schedules `SimulationPass` before `TraversalPass` and inserts the barrier.
Surface displaces from simulation in real time.

### Stage 10 — Power Foam leaf

Implement `RadiantFoamCell` first (unbounded Voronoi, ray tracing only).
Validate the cell-walk traversal against the Power Foam paper's algorithm.
Extend to `PowerFoamCell` with sphere bounds (the bounded power diagram). Add
`PowerFoamRasterPass`. The same leaf now works in both execution paths from
identical data. Do not attempt the Čech complex adjacency walk and dipole face
rasterization simultaneously — get brute-force working first.

Add the differentiable training loop. Loss from image comparisons. Gradients
through the foam rendering. Adam optimizer updating site positions, radii,
directional radiance. Engine can reconstruct Power Foam captures from real-world
images.

### Stage 11 — glTF loader (final front-end)

At this point you have a working engine with a full GPU path tracer, raster
path, simulation, and Power Foam. Adding glTF is just another front-end.
Implement `gltf.rs` parsing: JSON scene graph, binary buffer accessors, image
decoding. Map glTF PBR metallic-roughness to Starlight materials
(Section 2.19). Support `KHR_lights_punctual` for point/spot/directional
lights.

Implement `Scene::flatten()` — the unified conversion from CPU `Scene` to
`GpuSceneBuffers` (Section 2.18). This method produces all GPU buffers in one
pass: BVH nodes, leaf types, leaf data, materials, textures, lights. Both
execution paths consume the same flattened output.

Verify by loading the glTF spec sample models (Box, Duck, FlightHelmet) and
rendering. The scene graph, materials, textures, and lights must all load
correctly.

This stage also defines the trait interfaces (Section 2.18) —
`Intersectable`, `Shading`, `LightSampling`, `Texturing` — and implements
them for the CPU types. The GPU shader functions follow the same interface by
convention.

______________________________________________________________________

## 7. End-to-End Frame Walkthrough

A concrete trace of one frame from scene load to displayed pixel. This
ties together every subsystem described in Sections 2–4.

### Scene Load (CPU, once)

```
1. Parse scene description (JSON/custom format)
2. Create LeafNode instances for each object
   - AnalyticSphere { center, radius, material_id, transform }
   - TriangleMesh { vertices, normals, uvs, indices, material_id }
   - SdfVolume { sdf_type, params, material_id }
   - ...
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
┌─ RenderGraph::compile() ─────────────────────────────────────────┐
│                                                                   │
│  1. Build DAG from pass reads()/writes() declarations             │
│  2. Topological sort (Kahn's algorithm)                           │
│  3. Insert barriers between adjacent passes with conflicts        │
│  4. Single GRAPHICS|COMPUTE queue — no multi-queue                │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘

┌─ RenderGraph::execute() ─────────────────────────────────────────┐
│                                                                   │
│  Pass 1: SimulationPass                                           │
│    vkCmdDispatch(thermal_kernel, domain_count)                    │
│    Writes: thermal_state_buffer                                   │
│                                                                   │
│  Barrier: NONE→SHADER_READ on thermal_state_buffer                │
│                                                                   │
│  Pass 2: BvhCullPass                                              │
│    vkCmdDispatch(cull_shader, node_count)                         │
│    Input: camera frustum, BVH nodes                               │
│    Writes: foam_draw_list, mesh_draw_list                         │
│                                                                   │
│  Barrier: SHADER_WRITE→INDIRECT_READ on draw lists                │
│                                                                   │
│  Pass 3: MeshRasterPass                                           │
│    vkCmdDrawIndexed(indirect_buffer)                              │
│    Input: mesh_draw_list, vertex/index buffers                    │
│    Writes: GBuffer (albedo, normal, depth, material_id)           │
│                                                                   │
│  Barrier: COLOR_ATTACHMENT_WRITE→SHADER_READ on GBuffer           │
│                                                                   │
│  Pass 4: PowerFoamRasterPass                                      │
│    vkCmdDraw(indirect_buffer)                                     │
│    Input: foam_draw_list, PowerFoamCell data                      │
│    Writes: GBuffer (same targets, depth-tested write, overwrites     │
│             if closer)                                               │
│                                                                   │
│  Barrier: COLOR_ATTACHMENT_WRITE→SHADER_READ on GBuffer           │
│                                                                   │
│  Pass 5: TraversalPass                                            │
│    vkCmdDispatch(ceil(width/8), ceil(height/8), 1)                │
│    Runs for ALL pixels unconditionally.                            │
│    For each pixel:                                                │
│      1. Generate primary ray from camera params                   │
│      2. traverse_bvh() → Option<SurfaceInteraction>               │
│      3. Write SurfaceInteraction to HitBuffer                     │
│    Writes: HitBuffer (SurfaceInteraction per pixel)               │
│    Note: For pixels with valid GBuffer coverage, the traversal    │
│    result is used only for secondary effects. The ShadingPass     │
│    uses GBuffer depth as arbiter. Running everywhere wastes       │
│    compute for fully-rasterized frames but avoids a GBuffer       │
│    dependency before traversal. This is a known optimization      │
│    opportunity — later, add a cull pass that reads GBuffer depth  │
│    and only dispatches traversal for pixels with no coverage.     │
│                                                                   │
│  Barrier: SHADER_WRITE→SHADER_READ on HitBuffer                   │
│                                                                   │
│  Pass 6: ShadingPass                                              │
│    vkCmdDispatch(ceil(width/8), ceil(height/8), 1)                │
│    For each pixel:                                                │
│      1. Read GBuffer (primary) or HitBuffer (primary fallback)    │
│      2. Look up material via material_id                          │
│      3. Direct lighting:                                          │
│         a. Sample light from LightNode buffer                     │
│         b. Sample point on light surface                          │
│         c. Eval BSDF toward light                                 │
│         d. Shadow ray: TraversalPass.occluded(shadow_ray)         │
│         e. Power heuristic MIS weight                             │
│         f. Accumulate: L += f * radiance * weight / p_light       │
│      4. Indirect lighting:                                        │
│         a. Sample BSDF direction                                  │
│         b. Trace secondary ray (another traverse_bvh)             │
│         c. Accumulate: L += f_cos * 1/p_mixture                   │
│      5. Russian roulette (bounce ≥ 5)                             │
│      6. Enforce per-type bounce limits                            │
│    Writes: output image (rgba32f)                                 │
│                                                                   │
│  Barrier: SHADER_WRITE→TRANSFER_SRC on output image               │
│                                                                   │
│  Pass 7: PostProcessPass                                          │
│    vkCmdDispatch(post_shader)                                     │
│    Input: output image (rgba32f)                                  │
│    Applies: tone mapping (ACES/Reinhard), gamma correction        │
│    Writes: swapchain image (rgba8)                                │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

### Data Flow Summary

```
Scene load:
  LeafNode[] ──serialize──→ BVH + LeafTypes + LeafData + Materials + Lights

Per frame:
  Simulation ──→ thermal_state ──→ HeightField leaves (displacement)
  Camera ──→ primary rays ──→ TraversalPass ──→ HitBuffer
  BvhCullPass ──→ draw lists ──→ RasterPasses ──→ GBuffer
  GBuffer + HitBuffer + Lights + Materials ──→ ShadingPass ──→ Output
  Output ──→ PostProcessPass ──→ Display
```

The CPU path tracer performs the same algorithm (traverse BVH, evaluate
material, direct lighting with shadow ray, indirect lighting with
Russian roulette) but in a single-threaded loop instead of a GPU
dispatch. It serves as the reference implementation — identical math,
different execution target.

______________________________________________________________________

## 8. Validation Strategy

The CPU path tracer in raytrace-rs is the reference implementation.
Every rendering algorithm in the GPU ShadingPass must produce
statistically identical output to the CPU path tracer for the same
scene configuration.

### What Is Validated

| Aspect | CPU Reference | GPU Target | Comparison |
|--------|--------------|------------|------------|
| BVH traversal | `flat_bvh.rs` iterative | `traversal.rs` shader | Same BVH nodes, same traversal order |
| Material eval | `Material::eval()` | `@switch` on `GpuMaterialType` | Same BSDF value within ε |
| Direct lighting | `ray_color()` NEE | `ShadingPass` direct loop | Same contribution within variance |
| Shadow rays | (new in CPU) | `occluded()` | Boolean occlusion matches |
| MIS weights | Power heuristic | Power heuristic | Same weights for same PDFs |
| Russian roulette | Throughput-proportional | Throughput-proportional | Same survival probability |
| Bounce limits | Per-type | Per-type | Same termination behavior |

### Validation Methodology

1. **Identical scene serialization**: the same `LeafNode` list produces
   the same `GpuBvhNode`, `GpuMaterialNode`, `LightNode` arrays on
   both CPU and GPU paths. The serialization is tested independently
   (existing 6 tests in `material/mod.rs`).

2. **Deterministic comparison**: for the same random seed and sample
   count, CPU and GPU must produce pixel values within floating-point
   tolerance (ε = 1e-5 for single-precision). This catches algorithmic
   divergence without requiring statistical tests.

3. **Visual regression**: side-by-side renders of reference scenes
   (cornell box, glossy spheres, glass with caustics) must be
   perceptually identical. A pixel difference shader overlays the
   difference for human inspection.

4. **Per-stage validation**: each Implementation Sequence stage
   produces a runnable output that can be compared against the CPU
   reference. Stage 4 (first GPU path tracer) validates BVH traversal
   + AnalyticSphere intersection. Stage 9 (PBR shading) validates the
   full light transport algorithm.

### The CPU Path Is Never Broken

The implementation sequence is additive — each stage adds GPU
capability without modifying the CPU path tracer. The CPU path
continues to render correctly throughout development. When the GPU
path produces identical output, the CPU path remains as a fallback
and validation tool.

______________________________________________________________________

## 9. References

**Matryoshka renderer** — inigid. Reddit r/computergraphics, May 2025. *"What if
every part of the world could choose how to render itself?"* Zig + GLSL Vulkan
compute shader prototype, ~2,500 lines.
`reddit.com/r/computergraphics/comments/1skkxrm`

**Radiant Foam** — Govindarajan, Rebain, Yi, Tagliasacchi. ICCV 2025
(Highlight). *"Radiant Foam: Real-Time Differentiable Ray Tracing."*
arXiv:2502.01157. `radfoam.github.io`

**Power Foam** — Govindarajan, Rebain, Verbin, Yi, Prabhu, Tagliasacchi. arXiv
2026\. *"Power Foam: Unifying Real-Time Differentiable Ray Tracing and
Rasterization."* arXiv:2604.24994. `powerfoam.github.io`

**Niri compositor** — YaLTeR. `github.com/YaLTeR/niri` Rust Wayland compositor.
`niri_render_elements!` macro in `src/render_helpers/render_elements.rs`.
`NiriRenderer` supertrait in `src/render_helpers/renderer.rs`.

**Smithay** — `smithay.github.io`. Rust Wayland compositor framework. Provides
the `Element` and `RenderElement<R>` traits that `niri_render_elements!`
generates impls for. Informs the trait structure of the engine's `RenderLeaf`
and `RenderNode` traits.

**rust-gpu** — Embark Studios. `github.com/EmbarkStudios/rust-gpu` Rust compiler
backend emitting SPIR-V. Enables shaders written in Rust, shared types between
host and GPU, and exhaustive `match` dispatch in the traversal shader compiled
to SPIR-V switch constructs. Foundation of the `starlight-shaders` crate.

**renderling** — schell. `github.com/schell/renderling` GPU-driven Rust renderer
using rust-gpu for all shaders. Reference implementation for: multi-crate
workspace structure with shared types, `spirv-std` builtins usage, shader entry
point organization, and bindless resource patterns under rust-gpu.

**raytrace-rs** — Atan-D-RP4. `github.com/Atan-D-RP4/raytrace-rs` CPU path
tracer; the engine's starting point. Already contains: `FlatBvhNode` (64-byte
repr(C), 64-entry iterative stack), `GpuMaterialType`/`GpuTextureType`
(`#[repr(u32)]` tags with `repr(C)` node structs), `SurfaceInteraction` TODO in
`hittable.rs`, and a rich set of planar primitives. The `starlight-types` shared
crate evolves from its existing `gpu.rs` modules.

**Raytracing in One Weekend** (series) — Peter Shirley. BVH, `Hittable`,
`HitRecord`, `Material`, `Camera` foundations in `raytrace-rs`. The mental model
for leaf intersection maps directly to GPU traversal.

**Physically Based Rendering: From Theory to Implementation** — Pharr, Jakob,
Humphreys. `pbrt.org`. Ground truth for BSDF evaluation, importance sampling,
MIS, and `SurfaceInteraction` as the renderer-agnostic hit description. The
`ShadingPass` is essentially a GPU implementation of PBRT's integrator,
operating on `SurfaceInteraction` independently of geometry type.

**FrameGraph: Extensible Rendering Architecture in Frostbite** — Hammarén. GDC
2017\. The render graph design pattern. `reads()`/`writes()` declarations,
topological sort, automatic barrier insertion.

**ash** — `github.com/ash-rs/ash`. Thin Rust Vulkan bindings. All Vulkan API
calls in `starlight-engine`. Explicit, zero-overhead, `unsafe`-confined.

**gpu-allocator** — Traverse-Research.
`github.com/Traverse-Research/gpu-allocator`. GPU memory allocator for
`ResourcePool`. Handles suballocation, alignment, and memory type selection for
BVH scratch buffers, simulation state buffers, and the GBuffer image suite.

**Dynamic Rendering (Vulkan 1.3 / VK_KHR_dynamic_rendering)** — Khronos. Used by
`MeshRasterPass` and `PowerFoamRasterPass`. Eliminates pre-created
`VkRenderPass` objects; render pass metadata is specified at command recording
time. Simplifies the render graph's pass lifecycle management.

**glTF 2.0** — Khronos Group. `registry.khronos.org/glTF/specs/2.0`. The
"JPEG of 3D" — compact, vendor-neutral format for 3D scene delivery. PBR
metallic-roughness material model, node hierarchy, cameras, lights via
extensions. Primary input format for Starlight.

**gltf** (Rust crate) — `github.com/gltf-rs/gltf`. Rust parser for glTF 2.0.
Streaming iteration over accessors, buffer views, and images. Used by
`starlight-engine/src/scene/gltf.rs` for scene loading.

**tobj** (Rust crate) — `github.com/Twinklebear/tobj`. Lightweight OBJ + MTL
parser. Used by `starlight-engine/src/scene/obj.rs` for legacy format support.

**image** (Rust crate) — `github.com/image-rs/image`. PNG/JPEG decoding for
glTF texture images. Produces RGBA pixel data for hardware texture upload.
