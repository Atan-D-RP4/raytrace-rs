use std::path::Path;
use std::sync::Arc;

use image::Rgba32FImage;

use crate::interval::Interval;
use crate::vec3::{Color3, Point3, Vec3};

/// Geometry fills this with the coordinates it knows how to produce at a hit point.
///
/// UV-mapped textures read `u` and `v`, while procedural textures can use the
/// full 3D hit position stored in `point`.
#[derive(Debug, Clone, Copy)]
pub struct TextureCoords {
    pub u: f64,
    pub v: f64,
    pub point: Point3,
}

impl TextureCoords {
    pub fn new(u: f64, v: f64, point: Point3) -> Self {
        Self { u, v, point }
    }

    pub fn with_point(mut self, point: Point3) -> Self {
        self.point = point;
        self
    }
}

/// Mappings adapt geometry-space coordinates into the coordinate space expected
/// by a particular texture.
pub trait TextureMapping: Send + Sync {
    fn map(&self, coords: TextureCoords) -> TextureCoords;
}

/// The default mapping passes coordinates through unchanged.
pub struct IdentityMapping;

impl TextureMapping for IdentityMapping {
    fn map(&self, coords: TextureCoords) -> TextureCoords {
        coords
    }
}

/// Scales the 3D hit point before a procedural texture sees it.
///
/// This keeps cell sizing logic in the mapping layer instead of baking it into
/// each procedural texture implementation.
pub struct PointScaleMapping {
    inv_scale: Vec3,
}

impl PointScaleMapping {
    pub fn from_uniform(scale: f64) -> Self {
        assert!(scale > 0.0, "texture scale must be positive");
        let inv_scale = 1.0 / scale;

        Self {
            inv_scale: Vec3::from(inv_scale, inv_scale, inv_scale),
        }
    }
}

impl TextureMapping for PointScaleMapping {
    fn map(&self, coords: TextureCoords) -> TextureCoords {
        coords.with_point(coords.point * self.inv_scale)
    }
}

pub trait Texture: Send + Sync {
    fn value(&self, coords: &TextureCoords) -> Color3;
}

/// A compositional texture wrapper that applies a mapping before delegating to
/// another texture.
pub struct MappedTexture {
    mapping: Arc<dyn TextureMapping>,
    texture: Arc<dyn Texture>,
}

impl MappedTexture {
    pub fn new(mapping: Arc<dyn TextureMapping>, texture: Arc<dyn Texture>) -> Self {
        Self { mapping, texture }
    }
}

impl Texture for MappedTexture {
    fn value(&self, coords: &TextureCoords) -> Color3 {
        let mapped_coords = self.mapping.map(*coords);
        self.texture.value(&mapped_coords)
    }
}

pub struct SolidColor {
    albedo: Color3,
}

impl SolidColor {
    pub fn new(albedo: Color3) -> Self {
        Self { albedo }
    }

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

pub struct CheckerTexture {
    even: Arc<dyn Texture>,
    odd: Arc<dyn Texture>,
}

impl CheckerTexture {
    pub fn new(even: Arc<dyn Texture>, odd: Arc<dyn Texture>) -> Self {
        Self { even, odd }
    }

    pub fn from_color(c1: Color3, c2: Color3) -> Self {
        Self {
            even: Arc::new(SolidColor::new(c1)),
            odd: Arc::new(SolidColor::new(c2)),
        }
    }
}

impl Texture for CheckerTexture {
    fn value(&self, coords: &TextureCoords) -> Color3 {
        let x = coords.point.x.floor() as i32;
        let y = coords.point.y.floor() as i32;
        let z = coords.point.z.floor() as i32;

        if (x + y + z) % 2 == 0 {
            self.even.value(coords)
        } else {
            self.odd.value(coords)
        }
    }
}

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
