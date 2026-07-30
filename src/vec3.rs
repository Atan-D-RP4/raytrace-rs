use derive_more::{Display, From};
use glam::Vec3;
use rand::RngExt;
use rand::distr::{Distribution, StandardUniform};

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

// ── Newtype wrappers for glam::Vec3 ─────────────────────────────────────────

/// RGB color value wrapping `glam::Vec3`.
///
/// Operations and constants mirror `Vec3`; use `.r()`, `.g()`, `.b()` for channel access.
#[derive(Debug, Clone, Copy, PartialEq, From, Display)]
#[display("{_0}")]
pub struct Color3(pub Vec3);

impl Default for Color3 {
    fn default() -> Self {
        Self(Vec3::ZERO)
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
pub struct Point3(pub Vec3);

impl Default for Point3 {
    fn default() -> Self {
        Self(Vec3::ZERO)
    }
}

/// 3D direction vector wrapping `glam::Vec3`.
#[derive(Debug, Clone, Copy, PartialEq, From, Display)]
#[display("{_0}")]
pub struct Direction3(pub Vec3);

impl Default for Direction3 {
    fn default() -> Self {
        Self(Vec3::ZERO)
    }
}

// ── Macro helpers ──────────────────────────────────────────────────────────

macro_rules! impl_accessors {
    ($ty:ident) => {
        impl $ty {
            #[inline(always)]
            pub fn x(self) -> f32 {
                self.0.x
            }
            #[inline(always)]
            pub fn y(self) -> f32 {
                self.0.y
            }
            #[inline(always)]
            pub fn z(self) -> f32 {
                self.0.z
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
                Self(Vec3::new(x, y, z))
            }
            #[inline(always)]
            pub fn splat(v: f32) -> Self {
                Self(Vec3::splat(v))
            }
            #[inline(always)]
            pub fn into_inner(self) -> Vec3 {
                self.0
            }

            pub const ZERO: Self = Self(Vec3::ZERO);
            pub const ONE: Self = Self(Vec3::ONE);
            pub const X: Self = Self(Vec3::X);
            pub const Y: Self = Self(Vec3::Y);
            pub const Z: Self = Self(Vec3::Z);
            pub const NEG_X: Self = Self(Vec3::NEG_X);
            pub const NEG_Y: Self = Self(Vec3::NEG_Y);
            pub const NEG_Z: Self = Self(Vec3::NEG_Z);
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

macro_rules! impl_vec_index {
    ($ty:ident) => {
        impl std::ops::Index<usize> for $ty {
            type Output = f32;
            #[inline(always)]
            fn index(&self, index: usize) -> &f32 {
                &self.0[index]
            }
        }

        impl std::ops::IndexMut<usize> for $ty {
            #[inline(always)]
            fn index_mut(&mut self, index: usize) -> &mut f32 {
                &mut self.0[index]
            }
        }
    };
}

// ── Accessor ───────────────────────────────────────────────────────────────
impl_accessors!(Color3);
impl_accessors!(Point3);
impl_accessors!(Direction3);

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

// ── Indexing ───────────────────────────────────────────────────────────────

impl_vec_index!(Color3);
impl_vec_index!(Point3);
impl_vec_index!(Direction3);

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
    pub fn min(self, other: Vec3) -> Self {
        Self(self.0.min(other))
    }

    #[inline]
    pub fn max(self, other: Vec3) -> Self {
        Self(self.0.max(other))
    }

    #[inline]
    pub fn clamp(self, min: Vec3, max: Vec3) -> Self {
        Self(self.0.clamp(min, max))
    }
}

impl Point3 {
    #[inline]
    pub fn min(self, other: Vec3) -> Self {
        Self(self.0.min(other))
    }

    #[inline]
    pub fn max(self, other: Vec3) -> Self {
        Self(self.0.max(other))
    }

    #[inline]
    pub fn clamp(self, min: Vec3, max: Vec3) -> Self {
        Self(self.0.clamp(min, max))
    }
}

impl Direction3 {
    #[inline]
    pub fn min(self, other: Vec3) -> Self {
        Self(self.0.min(other))
    }

    #[inline]
    pub fn max(self, other: Vec3) -> Self {
        Self(self.0.max(other))
    }

    #[inline]
    pub fn clamp(self, min: Vec3, max: Vec3) -> Self {
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

impl std::ops::Add<Vec3> for Point3 {
    type Output = Point3;

    #[inline(always)]
    fn add(self, rhs: Vec3) -> Point3 {
        Point3(self.0 + rhs)
    }
}

impl std::ops::AddAssign<Vec3> for Point3 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Vec3) {
        self.0 += rhs;
    }
}

impl std::ops::Sub<Vec3> for Point3 {
    type Output = Point3;

    #[inline(always)]
    fn sub(self, rhs: Vec3) -> Point3 {
        Point3(self.0 - rhs)
    }
}

impl std::ops::SubAssign<Vec3> for Point3 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Vec3) {
        self.0 -= rhs;
    }
}

impl std::ops::Sub<Point3> for Point3 {
    type Output = Direction3;

    #[inline(always)]
    fn sub(self, rhs: Point3) -> Direction3 {
        Direction3(self.0 - rhs.0)
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

// ── Point3 - Direction3 (affine combination) ────────────────────────────

impl std::ops::Sub<Direction3> for Point3 {
    type Output = Point3;

    #[inline(always)]
    fn sub(self, rhs: Direction3) -> Point3 {
        Point3(self.0 - rhs.0)
    }
}

impl std::ops::SubAssign<Direction3> for Point3 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Direction3) {
        self.0 -= rhs.0;
    }
}

// ── Direction3-specific ──────────────────────────────────────────────────

impl Direction3 {
    /// [`glam::Vec3::dot`]
    #[inline(always)]
    pub fn dot(self, other: Vec3) -> f32 {
        self.0.dot(other)
    }

    /// [`Vec3::cross`]
    #[inline(always)]
    pub fn cross(self, other: Vec3) -> Direction3 {
        Direction3(self.0.cross(other))
    }

    /// [`glam::Vec3::reflect`]
    #[inline(always)]
    pub fn reflect(self, normal: Vec3) -> Self {
        Self(self.0.reflect(normal))
    }

    /// [`glam::Vec3::refract`]
    #[inline(always)]
    pub fn refract(self, normal: Vec3, eta: f32) -> Self {
        Self(self.0.refract(normal, eta))
    }

    /// [`glam::Vec3::length_squared`]
    #[inline(always)]
    pub fn length_squared(self) -> f32 {
        self.0.length_squared()
    }

    /// [`glam::Vec3::length`]
    #[inline(always)]
    pub fn length(self) -> f32 {
        self.0.length()
    }
}
