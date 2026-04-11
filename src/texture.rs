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

use std::f64::consts::PI;
use std::path::Path;
use std::sync::Arc;

use image::Rgba32FImage;

use crate::interval::Interval;
use crate::perlin::Perlin;

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

/// Coordinate mappings applied before evaluating an underlying texture.
/// TODO(mapping-2d3d): split this enum into `TextureMapping2D` and
/// `TextureMapping3D` to make UV remaps vs 3D point remaps explicit.
pub enum TextureMapping {
    /// No coordinate change.
    Identity,
    /// Uniform scale in 3D texture space.
    PointScale { inv_scale: Vec3 },
    /// Converts mapping-space unit-sphere position into UVs.
    Spherical,
}

impl TextureMapping {
    /// Builds a uniform point-scale mapping.
    ///
    /// `scale` is cell size; smaller values increase frequency.
    pub fn point_scale_uniform(scale: f64) -> Self {
        assert!(scale > 0.0, "texture scale must be positive");
        let inv_scale = 1.0 / scale;

        Self::PointScale {
            inv_scale: Vec3::from(inv_scale, inv_scale, inv_scale),
        }
    }

    /// Applies this mapping to a texture context and returns the mapped copy.
    /// TODO(mapping-2d3d): return distinct mapped outputs for 2D and 3D paths
    /// instead of mutating a single mixed context.
    pub fn map(&self, coords: TextureCoords) -> TextureCoords {
        match self {
            TextureMapping::Identity => coords,
            TextureMapping::PointScale { inv_scale } => {
                coords.with_texture_point(coords.tex_points.texture * *inv_scale)
            }
            TextureMapping::Spherical => {
                // p: point on unit sphere centered at origin (mapping space).
                let p = coords.tex_points.mapping.unit_vector();
                let theta = (-p.y).acos();
                let phi = -p.z.atan2(p.x) + PI;

                let u = phi / (2.0 * PI); // u: angle around +Y axis from X = -1.
                let v = theta / PI; // v: angle from Y = -1 to Y = +1.
                //
                // Examples:
                //  <p, u, v>
                //  <1, 0, 0> -> (0.50, 0.50), <-1, 0, 0> -> (0.00, 0.50)
                //  <0, 1, 0> -> (0.50, 1.00), < 0,-1, 0> -> (0.50, 0.00)
                //  <0, 0, 1> -> (0.25, 0.50), < 0, 0,-1> -> (0.75, 0.50)

                coords.with_uv(u, v)
            }
        }
    }
}

/// A texture evaluates a color from a fully prepared texture context.
pub trait Texture: Send + Sync {
    fn value(&self, coords: &TextureCoords) -> Color3;
}

/// Compositional wrapper for mapping coordinates first, then evaluating the wrapped texture.
pub struct MappedTexture {
    mapping: TextureMapping,
    texture: Arc<dyn Texture>,
}

impl MappedTexture {
    /// Creates a texture that applies `mapping` before sampling `texture`.
    pub fn new(mapping: TextureMapping, texture: Arc<dyn Texture>) -> Self {
        Self { mapping, texture }
    }
}

impl Texture for MappedTexture {
    fn value(&self, coords: &TextureCoords) -> Color3 {
        let mapped_coords = self.mapping.map(*coords);
        self.texture.value(&mapped_coords)
    }
}

/// Constant color texture
pub struct SolidColor {
    albedo: Color3,
}

impl SolidColor {
    /// Construct from a `Color3` value.
    pub fn new(albedo: Color3) -> Self {
        Self { albedo }
    }

    /// Construct RGB components.
    pub fn from_rgb(r: f64, g: f64, b: f64) -> Self {
        Self {
            albedo: Color3::from(r, g, b),
        }
    }
}

impl Texture for SolidColor {
    fn value(&self, _coords: &TextureCoords) -> Color3 {
        self.albedo
    }
}

/// Alternates between two child textures using integer parity in texture space.
pub struct CheckerTexture {
    even: Arc<dyn Texture>,
    odd: Arc<dyn Texture>,
}

impl CheckerTexture {
    pub fn new(even: Arc<dyn Texture>, odd: Arc<dyn Texture>) -> Self {
        Self { even, odd }
    }

    /// Convenience checker from two solid colors.
    pub fn from_color(c1: Color3, c2: Color3) -> Self {
        Self {
            even: Arc::new(SolidColor::new(c1)),
            odd: Arc::new(SolidColor::new(c2)),
        }
    }
}

impl Texture for CheckerTexture {
    fn value(&self, coords: &TextureCoords) -> Color3 {
        let x = coords.tex_points.texture.x.floor() as i32;
        let y = coords.tex_points.texture.y.floor() as i32;
        let z = coords.tex_points.texture.z.floor() as i32;

        if (x + y + z) % 2 == 0 {
            self.even.value(coords)
        } else {
            self.odd.value(coords)
        }
    }
}

/// Loads an image texture and stores it in float RGBA format.
pub struct ImageTexture {
    image: Rgba32FImage,
}

impl ImageTexture {
    pub fn new<P: AsRef<Path>>(filename: P) -> image::ImageResult<Self> {
        let image = image::open(filename)?.to_rgba32f();
        Ok(Self { image })
    }
}

impl Texture for ImageTexture {
    fn value(&self, coords: &TextureCoords) -> Color3 {
        if self.image.height() == 0 {
            return Color3::from(0., 1., 1.);
        }

        let u = Interval::from(0., 1.).clamp(coords.u);
        let v = 1.0 - Interval::from(0., 1.).clamp(coords.v);

        let i = (u * self.image.width() as f64).min((self.image.width() - 1) as f64);
        let j = (v * self.image.height() as f64).min((self.image.height() - 1) as f64);
        let pixel = self.image.get_pixel(i as u32, j as u32);

        Color3::from(pixel[0] as f64, pixel[1] as f64, pixel[2] as f64)
    }
}

/// Procedural Perlin-noise texture source.
pub struct NoiseTexture {
    noise: Perlin,
}

impl Default for NoiseTexture {
    fn default() -> Self {
        Self::new()
    }
}

impl NoiseTexture {
    pub fn new() -> Self {
        Self {
            noise: Perlin::new(),
        }
    }
}

impl Texture for NoiseTexture {
    fn value(&self, coords: &TextureCoords) -> Color3 {
        let point = coords.tex_points.texture;
        // Color3::from(1., 1., 1.) * 0.5 * (1.0 + self.noise.noise(&point)) // Smooth Perlin Texture
        // Color3::from(1., 1., 1.) * self.noise.turbulence(&point, 7) // Turbulent Perlin Texture
        Color3::from(0.5, 0.5, 0.5)
            * (1.0 + (point.z + (10.0 * self.noise.turbulence(&point, 7))).sin())
        // Marbled Perlin Texture
    }
}
