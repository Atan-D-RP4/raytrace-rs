//! Texture system architecture.
//!
//! Pipeline:
//! 1. Geometry writes hit data into [`TextureCoords`].
//! 2. [`TextureMapping`] transforms coordinates (world/mapping/texture/uv).
//! 3. [`Texture::value`] evaluates final color from the mapped context.
//!
//! The design keeps coordinate generation (geometry), coordinate transformation
//! (mapping), and color evaluation (texture) as separate responsibilities.
//!
//! TODO(renderer-agnostic): replace direct path-tracer handoff with
//! `SurfaceInteraction` so rasterizer/GPU/hybrid/SDF backends share this API.
//! TODO(mapping-2d3d): split mapping APIs into explicit 2D/3D channels
//! (e.g. `TextureMapping2D` for UV generation and `TextureMapping3D` for point transforms).

mod gpu;
mod impls;
mod mapping;

pub use gpu::{GPU_TEX_NONE, GpuTextureBuffer, GpuTextureNode, GpuTextureType};
pub use impls::{CheckerTexture, ImageTexture, MappedTexture, NoiseTexture, SolidColor};
pub use mapping::TextureMapping;

use crate::vec3::{Color3, Point3, Vec3};

/// Coordinate spaces carried through the texture pipeline.
#[derive(Debug, Clone, Copy)]
pub struct TexturePoints {
    /// The immutable hit position in world space
    pub world: Point3,
    /// The geometry-provided mapping space (for spheres, the unit-sphere point)
    pub mapping: Point3,
    /// The mutable 3D coordinate that 3D mappings shape for procedural textures.
    pub texture: Point3,
}

impl TexturePoints {
    /// Creates a new coordinate bundle.
    ///
    /// `texture` starts as `world` and can be reshaped by mappings.
    pub fn new(world: Point3, mapping: Point3) -> Self {
        Self {
            world,
            mapping,
            texture: world,
        }
    }

    /// Returns a copy with an updated texture-space point.
    pub fn with_texture(mut self, texture: Point3) -> Self {
        self.texture = texture;
        self
    }
}

/// Screen-space partial derivatives for texture filtering and LOD.
///
/// When populated, these enable anisotropic filtering and mipmap selection
/// on GPU. Currently zeroed — ray/pixel differentials are not yet computed
/// by the path tracer.
#[derive(Debug, Clone, Copy, Default)]
pub struct TextureDerivatives {
    /// Screen-space derivative of the hit position with respect to screen x.
    pub dpdx: Vec3,
    /// Screen-space derivative of the hit position with respect to screen y.
    pub dpdy: Vec3,
    /// Screen-space derivative of U with respect to screen x.
    pub dudx: f64,
    /// Screen-space derivative of U with respect to screen y.
    pub dudy: f64,
    /// Screen-space derivative of V with respect to screen x.
    pub dvdx: f64,
    /// Screen-space derivative of V with respect to screen y.
    pub dvdy: f64,
}

/// Full texture evaluation context passed to mappings and textures.
///
/// Built by [`HitRecord::texture_coords`](crate::hittable::HitRecord::texture_coords)
/// from geometry hit data. Flows through [`TextureMapping::map`] then into
/// [`Texture::value`].
///
/// TODO(mapping-2d3d): when 2D/3D mapping channels are split, move UV-only and
/// point-only fields into dedicated sub-contexts to prevent accidental cross-use.
#[derive(Debug, Clone, Copy)]
pub struct TextureCoords {
    /// Surface U coordinate in [0, 1] from primitive parameterization.
    pub u: f64,
    /// Surface V coordinate in [0, 1] from primitive parameterization.
    pub v: f64,
    /// Coordinate spaces (world, mapping, texture) carried through the pipeline.
    pub tex_points: TexturePoints,
    /// Outward geometric normal at the hit point (before shading adjustments).
    pub geometry_normal: Vec3,
    /// Screen-space derivatives for filtering (currently zeroed).
    pub derivatives: TextureDerivatives,
}

impl TextureCoords {
    /// Constructs a texture context from geometry-provided hit data.
    pub fn new(
        u: f64,
        v: f64,
        world_point: Point3,
        mapping_point: Point3,
        geometry_normal: Vec3,
    ) -> Self {
        Self {
            u,
            v,
            tex_points: TexturePoints::new(world_point, mapping_point),
            geometry_normal,
            derivatives: TextureDerivatives::default(),
        }
    }

    /// Returns a copy with a different texture-space point.
    pub fn with_texture_point(mut self, point: Point3) -> Self {
        self.tex_points = self.tex_points.with_texture(point);
        self
    }

    /// Returns a copy with remapped UV coordinates.
    /// TODO(mapping-2d3d): move to a UV-specific mapping context once 2D and 3D
    /// mappings are represented by separate types.
    pub fn with_uv(mut self, u: f64, v: f64) -> Self {
        self.u = u;
        self.v = v;
        self
    }
}

/// A texture that evaluates a color at a surface point.
///
/// Textures are the leaf nodes in the material tree — they provide spatially
/// varying color (albedo, emission, etc.) that materials sample during
/// shading. The pipeline is: geometry → [`TextureCoords`] →
/// [`TextureMapping::map`] → `Texture::value` → [`Color3`].
pub trait Texture: Send + Sync {
    /// Evaluate the texture at the given coordinate context.
    ///
    /// For image textures, this samples the image at `(u, v)`. For procedural
    /// textures (checker, noise), this evaluates the function at
    /// `tex_points.texture`.
    fn value(&self, coords: &TextureCoords) -> Color3;

    /// Serialize this texture node for the GPU buffer.
    ///
    /// Returns the node index in the buffer. The default implementation
    /// returns [`GPU_TEX_NONE`], indicating no GPU representation.
    /// Concrete texture types override this when GPU serialization is needed.
    fn serialize_gpu(&self, _buf: &mut gpu::GpuTextureBuffer) -> u32 {
        GPU_TEX_NONE
    }
}
