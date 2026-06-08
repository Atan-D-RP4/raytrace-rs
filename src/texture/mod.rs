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

/// Optional derivatives for future filtering/LOD work.
///
/// These are currently initialized to zero until ray/pixel differentials are added.
#[derive(Debug, Clone, Copy, Default)]
pub struct TextureDerivatives {
    pub dpdx: Vec3,
    pub dpdy: Vec3,
    pub dudx: f64,
    pub dudy: f64,
    pub dvdx: f64,
    pub dvdy: f64,
}

/// Full texture evaluation context passed to mappings and textures.
///
/// Carries UVs, coordinate spaces, geometric normal, and derivative slots.
/// TODO(mapping-2d3d): when 2D/3D mapping channels are split, move UV-only and
/// point-only fields into dedicated sub-contexts to prevent accidental cross-use.
#[derive(Debug, Clone, Copy)]
pub struct TextureCoords {
    pub u: f64,
    pub v: f64,
    pub tex_points: TexturePoints,
    pub geometry_normal: Vec3,
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

/// A texture evaluates a color from a fully prepared texture context.
pub trait Texture: Send + Sync {
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
