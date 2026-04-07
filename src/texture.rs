use std::path::Path;
use std::sync::Arc;

use image::{Pixel, Rgba32FImage};

use crate::interval::Interval;
use crate::vec3::{Color3, Point3};

pub trait Texture: Send + Sync {
    fn value(&self, u: f64, v: f64, point: &Point3) -> Color3;
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
    fn value(&self, _u: f64, _v: f64, _point: &Point3) -> Color3 {
        self.albedo
    }
}

pub struct CheckerTexture {
    inv_scale: f64,
    even: Arc<dyn Texture>,
    odd: Arc<dyn Texture>,
}

impl CheckerTexture {
    pub fn new(scale: f64, even: Arc<dyn Texture>, odd: Arc<dyn Texture>) -> Self {
        Self {
            inv_scale: 1.0 / scale,
            even,
            odd,
        }
    }

    pub fn from_color(scale: f64, c1: Color3, c2: Color3) -> Self {
        Self {
            inv_scale: 1.0 / scale,
            even: Arc::new(SolidColor::new(c1)),
            odd: Arc::new(SolidColor::new(c2)),
        }
    }
}

impl Texture for CheckerTexture {
    fn value(&self, u: f64, v: f64, point: &Point3) -> Color3 {
        let x = (self.inv_scale * point.x).floor() as i32;
        let y = (self.inv_scale * point.y).floor() as i32;
        let z = (self.inv_scale * point.z).floor() as i32;

        if (x + y + z) % 2 == 0 {
            self.even.value(u, v, point)
        } else {
            self.odd.value(u, v, point)
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
    fn value(&self, u: f64, v: f64, _point: &Point3) -> Color3 {
        if self.image.height() == 0 {
            return Color3::from(0., 1., 1.);
        }

        let u = Interval::from(0., 1.).clamp(u);
        let v = 1.0 - Interval::from(0., 1.).clamp(v);

        let i = (u * self.image.width() as f64).min((self.image.width() - 1) as f64);
        let j = (v * self.image.height() as f64).min((self.image.height() - 1) as f64);
        let pixel = self.image.get_pixel(i as u32, j as u32);

        Color3::from(pixel[0] as f64, pixel[1] as f64, pixel[2] as f64)
    }
}
