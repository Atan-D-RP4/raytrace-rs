use std::path::Path;
use std::sync::Arc;

use image::Rgba32FImage;

use super::{Texture, TextureCoords, TextureMapping};
use crate::interval::Interval;
use crate::perlin::Perlin;
use crate::vec3::Color3;

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
            * (1.0 + (point.z + (10.0 * self.noise.turbulence(point, 7))).sin())
        // Marbled Perlin Texture
    }
}
