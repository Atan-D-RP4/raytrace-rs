use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub};

use rand::RngExt;

#[derive(Default, Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Position in 3D space (glam Vec3 backed).
///
/// TODO(type-safety): convert to `newtype Point3(Vec3)` to prevent accidental
/// swaps with Vec3/Color3. Add `From`, `Into`, and explicit constructor like
/// `Point3::new(x, y, z)` to preserve ergonomics.
pub type Point3 = glam::Vec3;

/// RGB color value (glam Vec3 backed).
///
/// TODO(type-safety): convert to `newtype Color3(Vec3)` to prevent accidental
/// swaps with Vec3/Point3. Add `From`, `Into`, and explicit constructor like
/// `Color3::new(r, g, b)` to preserve ergonomics.
pub type Color3 = glam::Vec3;

impl Vec3 {
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);

    pub const ONE: Self = Self::new(1.0, 1.0, 1.0);

    pub const X: Self = Self::new(1.0, 0.0, 0.0);
    pub const Y: Self = Self::new(0.0, 1.0, 0.0);
    pub const Z: Self = Self::new(0.0, 0.0, 1.0);

    #[inline(always)]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    #[inline(always)]
    pub fn random() -> Self {
        let mut rng = rand::rng();
        Self::new(rng.random(), rng.random(), rng.random())
    }

    #[inline(always)]
    pub fn random_range(min: f32, max: f32) -> Self {
        let mut rng = rand::rng();
        Self::new(
            rng.random_range(min..max),
            rng.random_range(min..max),
            rng.random_range(min..max),
        )
    }

    #[inline(always)]
    pub const fn length_squared(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    #[inline(always)]
    pub fn length(&self) -> f32 {
        self.length_squared().sqrt()
    }

    #[inline(always)]
    pub const fn near_zero(&self) -> bool {
        let s = 1e-8;
        self.x.abs() < s && self.y.abs() < s && self.z.abs() < s
    }

    #[inline(always)]
    pub fn unit_vector(&self) -> Self {
        *self / self.length()
    }

    #[inline(always)]
    pub const fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

impl Vec3 {
    #[inline(always)]
    pub const fn dot(&self, v: &Vec3) -> f32 {
        self.x * v.x + self.y * v.y + self.z * v.z
    }

    #[inline(always)]
    pub const fn cross(&self, v: &Vec3) -> Vec3 {
        Vec3 {
            x: self.y * v.z - self.z * v.y,
            y: self.z * v.x - self.x * v.z,
            z: self.x * v.y - self.y * v.x,
        }
    }
}

impl Add for Vec3 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Sub for Vec3 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Neg for Vec3 {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl Mul for Vec3 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
        }
    }
}

impl Add<f32> for Vec3 {
    type Output = Vec3;

    #[inline(always)]
    fn add(self, t: f32) -> Vec3 {
        Self {
            x: self.x + t,
            y: self.y + t,
            z: self.z + t,
        }
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, t: f32) -> Self {
        Self {
            x: self.x * t,
            y: self.y * t,
            z: self.z * t,
        }
    }
}

// Enables `t * v` in addition to `v * t`
impl Mul<Vec3> for f32 {
    type Output = Vec3;

    fn mul(self, v: Vec3) -> Vec3 {
        v * self
    }
}

impl Div<f32> for Vec3 {
    type Output = Self;

    #[inline(always)]
    fn div(self, t: f32) -> Self {
        self * (1.0 / t)
    }
}

impl AddAssign for Vec3 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl MulAssign<f32> for Vec3 {
    #[inline(always)]
    fn mul_assign(&mut self, t: f32) {
        self.x *= t;
        self.y *= t;
        self.z *= t;
    }
}

impl DivAssign<f32> for Vec3 {
    fn div_assign(&mut self, t: f32) {
        *self *= 1.0 / t;
    }
}

impl Index<usize> for Vec3 {
    type Output = f32;

    #[inline(always)]
    fn index(&self, i: usize) -> &f32 {
        match i {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Vec3 index out of bounds: {}", i),
        }
    }
}

impl IndexMut<usize> for Vec3 {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut f32 {
        match i {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Vec3 index out of bounds: {}", i),
        }
    }
}

impl std::fmt::Display for Vec3 {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {} {}", self.x, self.y, self.z)
    }
}

impl From<glam::Vec3> for Vec3 {
    #[inline(always)]
    fn from(v: glam::Vec3) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }
}

impl From<Vec3> for glam::Vec3 {
    #[inline(always)]
    fn from(v: Vec3) -> Self {
        Self::new(v.x, v.y, v.z)
    }
}

#[inline(always)]
pub fn random_in_unit_disk_with_rng<R: rand::Rng + ?Sized>(rng: &mut R) -> Vec3 {
    loop {
        let point = Vec3::new(
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
            0.0,
        );
        if point.length_squared() < 1.0 {
            return point;
        }
    }
}

/// Rejection sampling to generate a random unit vector uniformly distributed on the surface of the
/// unit sphere.
#[inline(always)]
pub fn random_unit_vector_with_rng<R: rand::Rng + ?Sized>(rng: &mut R) -> Vec3 {
    loop {
        let point = Vec3::new(
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
        );
        let len_squared = point.length_squared();
        if 1e-160 < len_squared && len_squared <= 1.0 {
            return point / len_squared.sqrt();
        }
    }
}

#[inline(always)]
pub fn random_cosine_direction<R: rand::Rng + ?Sized>(rng: &mut R) -> Vec3 {
    let r1: f32 = rng.random();
    let r2: f32 = rng.random();

    let phi = 2.0 * std::f32::consts::PI * r1;
    let (sin_phi, cos_phi) = phi.sin_cos();
    let x = cos_phi * r2.sqrt();
    let y = sin_phi * r2.sqrt();
    let z = (1.0 - r2).sqrt();

    Vec3::new(x, y, z)
}

#[inline(always)]
pub fn random_cosine_direction2<R: rand::Rng + ?Sized>(rng: &mut R) -> Vec3 {
    let (mut u, mut v) = (rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0));
    while u * u + v * v >= 1.0 {
        (u, v) = (rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0));
    }

    Vec3::new(u, v, (1.0 - u.powi(2) - v.powi(2)).sqrt())
}

/// Shirley concentric disk mapping: maps `(u, v)` in `[0, 1)²` to a point on the
/// unit disk. Zero rejection, minimal distortion.
#[inline(always)]
pub fn concentric_disk(u: f32, v: f32) -> (f32, f32) {
    // Map to [-1, 1]²
    let sx = 2.0 * u - 1.0;
    let sy = 2.0 * v - 1.0;

    // Handle degenerate case at origin
    if sx == 0.0 && sy == 0.0 {
        return (0.0, 0.0);
    }

    let (x, y) = if sx.abs() > sy.abs() {
        let r = sx;
        let phi = std::f32::consts::FRAC_PI_4 * (sy / sx);
        (r * phi.cos(), r * phi.sin())
    } else {
        let r = sy;
        let phi = std::f32::consts::FRAC_PI_4 * (sx / sy);
        (r * phi.cos(), r * phi.sin())
    };

    (x, y)
}

#[inline(always)]
pub fn random_on_hemisphere<R: rand::Rng + ?Sized>(rng: &mut R, normal: Vec3) -> Vec3 {
    let on_unit_sphere = random_unit_vector_with_rng(rng);
    if on_unit_sphere.dot(&normal) > 0. {
        on_unit_sphere
    } else {
        -on_unit_sphere
    }
}

#[inline(always)]
pub fn reflect(v: &Vec3, n: &Vec3) -> Vec3 {
    *v - 2.0 * v.dot(n) * *n
}

#[inline(always)]
pub fn refract(uv: &Vec3, n: &Vec3, etai_over_etat: f32) -> Vec3 {
    let cos_theta = (-*uv).dot(n).min(1.0);
    let r_out_perp = etai_over_etat * (*uv + cos_theta * *n);
    let r_out_parallel = -(1.0 - r_out_perp.length_squared()).abs().sqrt() * *n;
    r_out_perp + r_out_parallel
}
