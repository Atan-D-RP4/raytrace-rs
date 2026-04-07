use std::sync::Arc;

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
