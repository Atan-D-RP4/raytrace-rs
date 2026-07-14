use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use derive_more::{Display, From};
use rand::RngExt;
use rand::distr::{Distribution, StandardUniform};

// ── OLD custom Vec3 struct (dead code, kept for compilation of helper fns below) ──

#[derive(Debug, Clone, Copy, Default)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    pub const ZERO: Self = Self::new(0., 0., 0.);
    pub const ONE: Self = Self::new(1., 1., 1.);
    pub fn random() -> Self {
        let mut rng = rand::rng();
        Self::new(rng.random(), rng.random(), rng.random())
    }
    pub fn random_range(min: f32, max: f32) -> Self {
        let mut rng = rand::rng();
        Self::new(
            rng.random_range(min..max),
            rng.random_range(min..max),
            rng.random_range(min..max),
        )
    }
    pub fn length_squared(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }
    pub fn length(&self) -> f32 {
        self.length_squared().sqrt()
    }
    pub fn near_zero(&self) -> bool {
        self.x.abs() < 1e-8 && self.y.abs() < 1e-8 && self.z.abs() < 1e-8
    }
    pub fn dot(&self, other: &Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
    pub fn cross(&self, other: &Vec3) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
    pub fn normalize(&self) -> Self {
        let len = self.length();
        if len < 1e-12 {
            Self::ZERO
        } else {
            Self::new(self.x / len, self.y / len, self.z / len)
        }
    }
}

impl Add for Vec3 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}
impl AddAssign for Vec3 {
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}
impl Sub for Vec3 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}
impl SubAssign for Vec3 {
    fn sub_assign(&mut self, other: Self) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}
impl Mul for Vec3 {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Self::new(self.x * other.x, self.y * other.y, self.z * other.z)
    }
}
impl Mul<f32> for Vec3 {
    type Output = Self;
    fn mul(self, t: f32) -> Self {
        Self::new(self.x * t, self.y * t, self.z * t)
    }
}
impl Mul<Vec3> for f32 {
    type Output = Vec3;
    fn mul(self, v: Vec3) -> Vec3 {
        v * self
    }
}
impl MulAssign<f32> for Vec3 {
    fn mul_assign(&mut self, t: f32) {
        self.x *= t;
        self.y *= t;
        self.z *= t;
    }
}
impl Div<f32> for Vec3 {
    type Output = Self;
    fn div(self, t: f32) -> Self {
        self * (1.0 / t)
    }
}
impl DivAssign<f32> for Vec3 {
    fn div_assign(&mut self, t: f32) {
        self.x /= t;
        self.y /= t;
        self.z /= t;
    }
}
impl Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}
impl Index<usize> for Vec3 {
    type Output = f32;
    fn index(&self, idx: usize) -> &f32 {
        match idx {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Vec3 index {idx} out of range"),
        }
    }
}
impl IndexMut<usize> for Vec3 {
    fn index_mut(&mut self, idx: usize) -> &mut f32 {
        match idx {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Vec3 index {idx} out of range"),
        }
    }
}

impl std::fmt::Display for Vec3 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} {} {}", self.x, self.y, self.z)
    }
}

impl From<glam::Vec3> for Vec3 {
    fn from(v: glam::Vec3) -> Self {
        Self::new(v.x, v.y, v.z)
    }
}
impl From<Vec3> for glam::Vec3 {
    fn from(v: Vec3) -> Self {
        Self::new(v.x, v.y, v.z)
    }
}

// ── Helper functions (operate on old Vec3, dead code but kept for now) ─────

pub fn reflect(v: &Vec3, n: &Vec3) -> Vec3 {
    *v - 2.0 * v.dot(n) * *n
}

pub fn refract(uv: &Vec3, n: &Vec3, etai_over_etat: f32) -> Vec3 {
    let cos_theta = (-*uv).dot(n).min(1.0);
    let r_out_perp = etai_over_etat * (*uv + cos_theta * *n);
    let r_out_parallel = -(1.0 - r_out_perp.length_squared()).abs().sqrt() * *n;
    r_out_perp + r_out_parallel
}

pub fn random_in_unit_disk_with_rng<R: rand::Rng + ?Sized>(rng: &mut R) -> Vec3 {
    loop {
        let point = Vec3::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0), 0.);
        if point.length_squared() < 1.0 {
            return point;
        }
    }
}

pub fn random_unit_vector_with_rng<R: rand::Rng + ?Sized>(rng: &mut R) -> Vec3 {
    random_on_sphere_with_rng(rng)
}

pub fn random_on_sphere_with_rng<R: rand::Rng + ?Sized>(rng: &mut R) -> Vec3 {
    let y = rng.random_range(-1.0..1.0);
    let theta = rng.random_range(0.0..std::f32::consts::TAU);
    let r = (1.0_f32 - y * y).sqrt();
    Vec3::new(r * theta.cos(), y, r * theta.sin())
}

/// Shirley concentric disk mapping: maps `(u, v)` in `[0, 1)²` to a point on the
/// unit disk. Zero rejection, minimal distortion.
#[inline(always)]
pub fn concentric_disk(u: f32, v: f32) -> (f32, f32) {
    let sx = 2.0 * u - 1.0;
    let sy = 2.0 * v - 1.0;
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

pub fn random_on_hemisphere_with_rng<R: rand::Rng + ?Sized>(rng: &mut R, normal: &Vec3) -> Vec3 {
    let on_unit_sphere = random_unit_vector_with_rng(rng);
    if on_unit_sphere.dot(normal) > 0.0 {
        on_unit_sphere
    } else {
        -on_unit_sphere
    }
}

// ── Newtype wrappers for glam::Vec3 ─────────────────────────────────────────

/// RGB color value wrapping `glam::Vec3`.
///
/// Operations and constants mirror `Vec3`; use `.r()`, `.g()`, `.b()` for channel access.
#[derive(Debug, Clone, Copy, PartialEq, From, Display)]
#[display("{_0}")]
pub struct Color3(pub glam::Vec3);

impl Default for Color3 {
    fn default() -> Self {
        Self(glam::Vec3::ZERO)
    }
}

impl std::iter::Sum<Color3> for Color3 {
    fn sum<I: Iterator<Item = Color3>>(iter: I) -> Color3 {
        iter.fold(Color3::ZERO, |a, b| a + b)
    }
}

impl Distribution<Color3> for StandardUniform {
    #[inline]
    fn sample<R: RngExt + ?Sized>(&self, rng: &mut R) -> Color3 {
        Color3(rng.random())
    }
}

impl Distribution<Direction3> for StandardUniform {
    #[inline]
    fn sample<R: RngExt + ?Sized>(&self, rng: &mut R) -> Direction3 {
        Direction3(rng.random())
    }
}

/// 3D position in space wrapping `glam::Vec3`.
#[derive(Debug, Clone, Copy, PartialEq, From, Display)]
#[display("{_0}")]
pub struct Point3(pub glam::Vec3);

impl Default for Point3 {
    fn default() -> Self {
        Self(glam::Vec3::ZERO)
    }
}

/// 3D direction vector wrapping `glam::Vec3`.
#[derive(Debug, Clone, Copy, PartialEq, From, Display)]
#[display("{_0}")]
pub struct Direction3(pub glam::Vec3);

impl Default for Direction3 {
    fn default() -> Self {
        Self(glam::Vec3::ZERO)
    }
}

// ── Macro helpers ──────────────────────────────────────────────────────────

macro_rules! impl_deref {
    ($ty:ident) => {
        impl std::ops::Deref for $ty {
            type Target = glam::Vec3;
            #[inline(always)]
            fn deref(&self) -> &glam::Vec3 {
                &self.0
            }
        }
        impl std::ops::DerefMut for $ty {
            #[inline(always)]
            fn deref_mut(&mut self) -> &mut glam::Vec3 {
                &mut self.0
            }
        }
    };
}

macro_rules! impl_binop {
    ($ty:ident, $trait:ident, $fn:ident, $op:tt) => {
        impl std::ops::$trait for $ty {
            type Output = $ty;
            #[inline(always)]
            fn $fn(self, rhs: $ty) -> $ty {
                $ty(self.0 $op rhs.0)
            }
        }
    };
    ($ty:ident, $trait:ident, $fn:ident, $op:tt, $rhs:ty) => {
        impl std::ops::$trait<$rhs> for $ty {
            type Output = $ty;
            #[inline(always)]
            fn $fn(self, rhs: $rhs) -> $ty {
                $ty(self.0 $op rhs)
            }
        }
    };
}

macro_rules! impl_assign_op {
    ($ty:ident, $trait:ident, $fn:ident, $op:tt) => {
        impl std::ops::$trait for $ty {
            #[inline(always)]
            fn $fn(&mut self, rhs: $ty) {
                self.0 $op rhs.0;
            }
        }
    };
    ($ty:ident, $trait:ident, $fn:ident, $op:tt, $rhs:ty) => {
        impl std::ops::$trait<$rhs> for $ty {
            #[inline(always)]
            fn $fn(&mut self, rhs: $rhs) {
                self.0 $op rhs;
            }
        }
    };
}

macro_rules! impl_neg {
    ($ty:ident) => {
        impl std::ops::Neg for $ty {
            type Output = $ty;
            #[inline(always)]
            fn neg(self) -> $ty {
                $ty(-self.0)
            }
        }
    };
}

macro_rules! impl_scalar_mul_reverse {
    ($ty:ident) => {
        impl std::ops::Mul<$ty> for f32 {
            type Output = $ty;
            #[inline(always)]
            fn mul(self, rhs: $ty) -> $ty {
                $ty(self * rhs.0)
            }
        }
    };
}

macro_rules! impl_constructors {
    ($ty:ident) => {
        impl $ty {
            #[inline(always)]
            pub const fn new(x: f32, y: f32, z: f32) -> Self {
                Self(glam::Vec3::new(x, y, z))
            }
            #[inline(always)]
            pub fn splat(v: f32) -> Self {
                Self(glam::Vec3::splat(v))
            }
            #[inline(always)]
            pub fn into_inner(self) -> glam::Vec3 {
                self.0
            }

            pub const ZERO: Self = Self(glam::Vec3::ZERO);
            pub const ONE: Self = Self(glam::Vec3::ONE);
            pub const X: Self = Self(glam::Vec3::X);
            pub const Y: Self = Self(glam::Vec3::Y);
            pub const Z: Self = Self(glam::Vec3::Z);
            pub const NEG_X: Self = Self(glam::Vec3::NEG_X);
            pub const NEG_Y: Self = Self(glam::Vec3::NEG_Y);
            pub const NEG_Z: Self = Self(glam::Vec3::NEG_Z);
        }
    };
}

macro_rules! impl_vec_methods {
    ($ty:ident, $($method:ident),+ $(,)?) => {
        impl $ty {
            $(#[inline(always)] pub fn $method(self) -> Self { Self(self.0.$method()) })+
        }
    };
}

// ── Deref ──────────────────────────────────────────────────────────────────

impl_deref!(Color3);
impl_deref!(Point3);
impl_deref!(Direction3);

// ── Constructors & constants ───────────────────────────────────────────────

impl_constructors!(Color3);
impl_constructors!(Point3);
impl_constructors!(Direction3);

// ── Binary ops (same-type) ─────────────────────────────────────────────────

impl_binop!(Color3, Add, add, +);
impl_binop!(Color3, Sub, sub, -);
impl_binop!(Color3, Mul, mul, *);
impl_binop!(Color3, Mul, mul, *, f32);
impl_binop!(Color3, Div, div, /, f32);

impl_binop!(Point3, Mul, mul, *, f32);
impl_binop!(Point3, Div, div, /, f32);

impl_binop!(Direction3, Add, add, +);
impl_binop!(Direction3, Sub, sub, -);
impl_binop!(Direction3, Mul, mul, *, f32);
impl_binop!(Direction3, Div, div, /, f32);

// ── Negation ───────────────────────────────────────────────────────────────

impl_neg!(Color3);
impl_neg!(Point3);
impl_neg!(Direction3);

// ── Assign ops ─────────────────────────────────────────────────────────────

impl_assign_op!(Color3, AddAssign, add_assign, +=);
impl_assign_op!(Color3, SubAssign, sub_assign, -=);
impl_assign_op!(Color3, MulAssign, mul_assign, *=);
impl_assign_op!(Color3, MulAssign, mul_assign, *=, f32);
impl_assign_op!(Color3, DivAssign, div_assign, /=, f32);

impl_assign_op!(Point3, AddAssign, add_assign, +=);
impl_assign_op!(Point3, SubAssign, sub_assign, -=);
impl_assign_op!(Point3, MulAssign, mul_assign, *=, f32);
impl_assign_op!(Point3, DivAssign, div_assign, /=, f32);

impl_assign_op!(Direction3, AddAssign, add_assign, +=);
impl_assign_op!(Direction3, SubAssign, sub_assign, -=);
impl_assign_op!(Direction3, MulAssign, mul_assign, *=, f32);
impl_assign_op!(Direction3, DivAssign, div_assign, /=, f32);

// ── Reverse scalar multiply ────────────────────────────────────────────────

impl_scalar_mul_reverse!(Color3);
impl_scalar_mul_reverse!(Point3);
impl_scalar_mul_reverse!(Direction3);

// ── Vec3 methods returning Self ────────────────────────────────────────────

impl_vec_methods!(
    Color3,
    normalize,
    normalize_or_zero,
    abs,
    sqrt,
    floor,
    fract
);
impl_vec_methods!(
    Point3,
    normalize,
    normalize_or_zero,
    abs,
    sqrt,
    floor,
    fract
);
impl_vec_methods!(
    Direction3,
    normalize,
    normalize_or_zero,
    abs,
    sqrt,
    floor,
    fract
);

// ── Min / max / clamp (take Vec3 for flexibility with splat) ───────────────

impl Color3 {
    #[inline]
    pub fn min(self, other: glam::Vec3) -> Self {
        Self(self.0.min(other))
    }
    #[inline]
    pub fn max(self, other: glam::Vec3) -> Self {
        Self(self.0.max(other))
    }
    #[inline]
    pub fn clamp(self, min: glam::Vec3, max: glam::Vec3) -> Self {
        Self(self.0.clamp(min, max))
    }
}
impl Point3 {
    #[inline]
    pub fn min(self, other: glam::Vec3) -> Self {
        Self(self.0.min(other))
    }
    #[inline]
    pub fn max(self, other: glam::Vec3) -> Self {
        Self(self.0.max(other))
    }
    #[inline]
    pub fn clamp(self, min: glam::Vec3, max: glam::Vec3) -> Self {
        Self(self.0.clamp(min, max))
    }
}
impl Direction3 {
    #[inline]
    pub fn min(self, other: glam::Vec3) -> Self {
        Self(self.0.min(other))
    }
    #[inline]
    pub fn max(self, other: glam::Vec3) -> Self {
        Self(self.0.max(other))
    }
    #[inline]
    pub fn clamp(self, min: glam::Vec3, max: glam::Vec3) -> Self {
        Self(self.0.clamp(min, max))
    }
}

// ── Color3-specific ──────────────────────────────────────────────────────

impl Color3 {
    #[inline(always)]
    pub fn r(self) -> f32 {
        self.0.x
    }
    #[inline(always)]
    pub fn g(self) -> f32 {
        self.0.y
    }
    #[inline(always)]
    pub fn b(self) -> f32 {
        self.0.z
    }
}

// ── Point3-specific (affine ops) ─────────────────────────────────────────

impl std::ops::Add<glam::Vec3> for Point3 {
    type Output = Point3;
    #[inline(always)]
    fn add(self, rhs: glam::Vec3) -> Point3 {
        Point3(self.0 + rhs)
    }
}
impl std::ops::AddAssign<glam::Vec3> for Point3 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: glam::Vec3) {
        self.0 += rhs;
    }
}
impl std::ops::Sub<glam::Vec3> for Point3 {
    type Output = Point3;
    #[inline(always)]
    fn sub(self, rhs: glam::Vec3) -> Point3 {
        Point3(self.0 - rhs)
    }
}
impl std::ops::SubAssign<glam::Vec3> for Point3 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: glam::Vec3) {
        self.0 -= rhs;
    }
}
impl std::ops::Sub<Point3> for Point3 {
    type Output = glam::Vec3;
    #[inline(always)]
    fn sub(self, rhs: Point3) -> glam::Vec3 {
        self.0 - rhs.0
    }
}

// ── Point3 + Direction3 (affine combination) ────────────────────────────

impl std::ops::Add<Direction3> for Point3 {
    type Output = Point3;
    #[inline(always)]
    fn add(self, rhs: Direction3) -> Point3 {
        Point3(self.0 + rhs.0)
    }
}
impl std::ops::AddAssign<Direction3> for Point3 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Direction3) {
        self.0 += rhs.0;
    }
}

// ── Direction3-specific ──────────────────────────────────────────────────

impl Direction3 {
    #[inline(always)]
    pub fn dot(self, other: glam::Vec3) -> f32 {
        self.0.dot(other)
    }
    #[inline(always)]
    pub fn cross(self, other: glam::Vec3) -> Direction3 {
        Direction3(self.0.cross(other))
    }
    #[inline(always)]
    pub fn reflect(self, normal: glam::Vec3) -> Self {
        Self(self.0.reflect(normal))
    }
    #[inline(always)]
    pub fn refract(self, normal: glam::Vec3, eta: f32) -> Self {
        Self(self.0.refract(normal, eta))
    }
}
