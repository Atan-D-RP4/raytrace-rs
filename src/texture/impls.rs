//! Concrete texture implementations.
//!
//! Each type implements the [`Texture`] trait and evaluates a color from
//! the coordinate context. These are the leaf nodes that materials sample
//! during shading.

use std::path::Path;
use std::sync::Arc;

use image::Rgba32FImage;

use crate::interval::Interval;
use crate::perlin::Perlin;
use crate::texture::mapping::{TextureMapping2D, TextureMapping3D, UvGen};
use crate::texture::{Texture, TextureCoords};
use crate::vec3::Color3;

/// Compositional wrapper for mapping coordinates first, then evaluating the wrapped texture.
///
/// The mapping pipeline is: 3D mapping → UV generation → 2D mapping → texture evaluation.
pub struct MappedTexture<T: Texture> {
    /// 2D mapping applied to UV coordinates after UV generation.
    mapping2d: TextureMapping2D,
    /// 3D mapping applied to the texture-space point before UV generation.
    mapping3d: TextureMapping3D,
    /// UV generation applied to the texture-space point after 3D mapping.
    uv_gen: UvGen,
    /// The wrapped texture to evaluate after mapping.
    texture: T,
}

impl<T: Texture> MappedTexture<T> {
    /// Creates a texture with identity mapping pipeline (3D identity, no UV gen, 2D identity).
    /// Apply mappings via [`with_mapping3d`](Self::with_mapping3d),
    /// [`with_uv_gen`](Self::with_uv_gen), and [`with_mapping2d`](Self::with_mapping2d).
    pub fn new(texture: T) -> Self {
        Self {
            mapping2d: TextureMapping2D::Identity,
            mapping3d: TextureMapping3D::Identity,
            uv_gen: UvGen::None,
            texture,
        }
    }

    pub fn with_mapping2d(mut self, mapping: TextureMapping2D) -> Self {
        self.mapping2d = mapping;
        self
    }

    pub fn with_mapping3d(mut self, mapping: TextureMapping3D) -> Self {
        self.mapping3d = mapping;
        self
    }

    pub fn with_uv_gen(mut self, uv_gen: UvGen) -> Self {
        self.uv_gen = uv_gen;
        self
    }
}

impl<T: Texture> Texture for MappedTexture<T> {
    fn value(&self, coords: &TextureCoords) -> Color3 {
        // Apply 3D point mapping first (transforms the texture space).
        let tex_point = self.mapping3d.map_point(coords.tex_points.texture);

        let (u, v) = self
            .uv_gen
            .map_to_uv(coords.tex_points.mapping)
            .unwrap_or((coords.u, coords.v));

        let (su, sv) = self.mapping2d.map_uv(u, v);

        let mapped = coords.with_texture_point(tex_point).with_uv(su, sv);
        self.texture.value(&mapped)
    }
}

/// Uniform color texture — returns the same [`Color3`] at every point.
///
/// Used as the fallback when no texture is provided (e.g. `Material::lambertian_color`),
/// and as the GPU serialization color for textured materials.
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
            albedo: Color3::new(r, g, b),
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
    /// Creates a checker from two arbitrary child textures.
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
    /// Loads an image from disk and converts it to float RGBA.
    pub fn new<P: AsRef<Path>>(filename: P) -> image::ImageResult<Self> {
        let image = image::open(filename)?.to_rgba32f();
        Ok(Self { image })
    }
}

impl Texture for ImageTexture {
    fn value(&self, coords: &TextureCoords) -> Color3 {
        if self.image.height() == 0 {
            return Color3::new(0., 1., 1.);
        }

        let u = Interval::from(0., 1.).clamp(coords.u);
        let v = 1.0 - Interval::from(0., 1.).clamp(coords.v);

        let i = (u * self.image.width() as f64).min((self.image.width() - 1) as f64);
        let j = (v * self.image.height() as f64).min((self.image.height() - 1) as f64);
        let pixel = self.image.get_pixel(i as u32, j as u32);

        Color3::new(pixel[0] as f64, pixel[1] as f64, pixel[2] as f64)
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
    /// Creates a new noise texture with random Perlin permutation tables.
    pub fn new() -> Self {
        Self {
            noise: Perlin::new(),
        }
    }
}

impl Texture for NoiseTexture {
    /// Marbled Perlin texture: combines turbulence with a sinusoidal warp
    /// for a natural stone-like appearance.
    ///
    /// Other variants (for reference):
    /// - Smooth: `Color3::from(1., 1., 1.) * 0.5 * (1.0 + self.noise.noise(&point))`
    /// - Turbulent: `Color3::from(1., 1., 1.) * self.noise.turbulence(&point, 7)`
    fn value(&self, coords: &TextureCoords) -> Color3 {
        let point = coords.tex_points.texture;
        Color3::new(0.5, 0.5, 0.5)
            * (1.0 + (point.z + (10.0 * self.noise.turbulence(point, 7))).sin())
    }
}
