//! Texture system architecture.
//!
//! Pipeline:
//! 1. Geometry writes hit data into [`TextureCoords`].
//! 2. Mappings transform coordinates ([`TextureMapping3D`] → point,
//!    [`UvGen`] → UV, [`TextureMapping2D`] → UV).
//! 3. [`Texture::value`] evaluates final color from the mapped context.
//!
//! The design keeps coordinate generation (geometry), coordinate transformation
//! (mapping), and color evaluation (texture) as separate responsibilities.
//!
//! TODO(renderer-agnostic): replace direct path-tracer handoff with
//! `SurfaceInteraction` so rasterizer/GPU/hybrid/SDF backends share this API.

mod gpu;
mod impls;
pub mod mapping;

pub use gpu::{GpuTextureBuffer, GpuTextureNode, GpuTextureType, GPU_TEX_NONE};
pub use impls::{
    CheckerTexture, ImageTexture, MappedTexture, NoiseTexture, SolidColor, SphericalUvMapping,
    TriplanarMapping, UvCheckerTexture, WorldSpaceMapping,
};

use crate::vec3::{Color3, Direction3, Point3};

pub trait UVDifferentiable {
    /// Returns the UV coordinates and their screen-space derivatives.
    ///
    /// (∂u/∂p, ∂v/∂p) at a mapping-space point.
    /// Default: zeroed → textures fall back to point sampling (volumes, no-UV).
    fn uv_gradient(&self, _mapping_point: &Point3) -> (Direction3, Direction3) {
        // Default: no derivatives (zeroed)
        (Direction3::ZERO, Direction3::ZERO)
    }
}

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
/// When populated, these enable anisotropic filtering and mipmap selection.
/// The path tracer computes them via ray differentials (`RayDifferentials`).
#[derive(Debug, Clone, Copy, Default)]
pub struct TextureDerivatives {
    /// Screen-space derivative of the hit position with respect to screen x.
    pub dpdx: Direction3,
    /// Screen-space derivative of the hit position with respect to screen y.
    pub dpdy: Direction3,
    /// Screen-space derivative of U with respect to screen x.
    pub dudx: f32,
    /// Screen-space derivative of U with respect to screen y.
    pub dudy: f32,
    /// Screen-space derivative of V with respect to screen x.
    pub dvdx: f32,
    /// Screen-space derivative of V with respect to screen y.
    pub dvdy: f32,
}

impl TextureDerivatives {
    /// Creates a new derivative bundle.
    pub fn new(
        dpdx: Direction3,
        dpdy: Direction3,
        dudx: f32,
        dudy: f32,
        dvdx: f32,
        dvdy: f32,
    ) -> Self {
        Self {
            dpdx,
            dpdy,
            dudx,
            dudy,
            dvdx,
            dvdy,
        }
    }

    /// Build UV derivatives from world-space position derivatives and the
    /// surface's UV Jacobian (∂u/∂p, ∂v/∂p).
    pub fn from_surface(
        dpdx: Direction3,
        dpdy: Direction3,
        dudp: Direction3,
        dvdp: Direction3,
    ) -> Self {
        Self {
            dpdx,
            dpdy,
            dudx: dpdx.dot(dudp.into_inner()),
            dudy: dpdy.dot(dudp.into_inner()),
            dvdx: dpdx.dot(dvdp.into_inner()),
            dvdy: dpdy.dot(dvdp.into_inner()),
        }
    }

    /// Returns true if all derivatives are zero.
    pub fn is_zero(&self) -> bool {
        const EPS: f32 = 1e-12;
        self.dpdx.length() < EPS
            && self.dpdy.length() < EPS
            && self.dudx.abs() < EPS
            && self.dudy.abs() < EPS
            && self.dvdx.abs() < EPS
            && self.dvdy.abs() < EPS
    }
}

/// Full texture evaluation context passed to mappings and textures.
///
/// Built by [`HitRecord::texture_coords`](crate::hittable::HitRecord::texture_coords)
/// from geometry hit data. Flows through 3D mapping → UV generation → 2D mapping
/// then into [`Texture::value`].
#[derive(Debug, Clone, Copy)]
pub struct TextureCoords {
    /// Surface U coordinate in [0, 1] from primitive parameterization.
    pub u: f32,
    /// Surface V coordinate in [0, 1] from primitive parameterization.
    pub v: f32,
    /// Coordinate spaces (world, mapping, texture) carried through the pipeline.
    pub tex_points: TexturePoints,
    /// Outward geometric normal at the hit point (before shading adjustments).
    pub geometry_normal: Direction3,
    /// Screen-space derivatives for filtering (currently zeroed).
    pub derivatives: TextureDerivatives,
}

impl TextureCoords {
    /// Constructs a texture context from geometry-provided hit data.
    pub fn new(
        u: f32,
        v: f32,
        world_point: Point3,
        mapping_point: Point3,
        geometry_normal: Direction3,
        derivatives: Option<TextureDerivatives>,
    ) -> Self {
        Self {
            u,
            v,
            tex_points: TexturePoints::new(world_point, mapping_point),
            geometry_normal,
            derivatives: derivatives.unwrap_or_default(),
        }
    }

    /// Returns a copy with a different texture-space point.
    pub fn with_texture_point(mut self, point: Point3) -> Self {
        self.tex_points = self.tex_points.with_texture(point);
        self
    }

    /// Returns a copy with remapped UV coordinates.
    pub fn with_uv(mut self, u: f32, v: f32) -> Self {
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
/// `MappedTexture` (3D mapping → UV generation → 2D mapping) → `Texture::value` → [`Color3`].
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
