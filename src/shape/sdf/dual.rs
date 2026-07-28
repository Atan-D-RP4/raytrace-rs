use std::cmp::Ordering;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Forward-mode dual number with N tangent lanes and generic scalar type `T`.
///
/// N=3 tracks ∂f/∂x, ∂f/∂y, ∂f/∂z simultaneously.
/// N=1 is a single-tangent dual (standard forward AD).
///
/// All ops implement the chain rule for exact derivatives.
/// Can be nested to track higher-order derivatives, e.g. `Dual<Dual<f32, 3>, 3>`
#[derive(Clone, Copy, Debug)]
pub struct Dual<T: Scalar, const N: usize> {
    pub v: T,
    pub d: [T; N],
}

impl<T: Scalar, const N: usize> Dual<T, N> {
    /// Seed tangent lane `idx` = 1.0, value = `v`.
    /// Call with `idx=0,1,2` for x, y, z gradients.
    #[inline(always)]
    pub fn variable(idx: usize, v: T) -> Self {
        let mut d = [T::constant(0.); N];
        d[idx] = T::constant(1.);
        Self { v, d }
    }

    #[inline(always)]
    pub fn constant(v: T) -> Self {
        Self {
            v,
            d: [T::constant(0.); N],
        }
    }

    #[inline(always)]
    pub fn value(&self) -> T {
        self.v
    }

    #[inline(always)]
    pub fn tangent(&self, idx: usize) -> T {
        self.d[idx]
    }

    #[inline(always)]
    pub fn splat(v: T) -> Self {
        Self::constant(v)
    }
}

// Addition: v = a + b,  d[i] = a.d[i] + b.d[i]
impl<T: Scalar, const N: usize> std::ops::Add for Dual<T, N> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        let mut d = [T::constant(0.); N];
        // for i in 0..N {
        //     d[i] = self.d[i] + rhs.d[i];
        // }
        d.iter_mut()
            .zip(self.d.iter().zip(rhs.d.iter()))
            .for_each(|(di, (&self_di, &rhs_di))| {
                *di = self_di + rhs_di;
            });
        Self {
            v: self.v + rhs.v,
            d,
        }
    }
}

// Multiplication: v = a * b,  d[i] = a.d[i]*b + b.d[i]*a  (product rule)
impl<T: Scalar, const N: usize> std::ops::Mul for Dual<T, N> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let mut d = [T::constant(0.); N];
        d.iter_mut()
            .zip(self.d.iter().zip(rhs.d.iter()))
            .for_each(|(di, (&self_di, &rhs_di))| {
                *di = (self_di * rhs.v) + (rhs_di * self.v);
            });
        Self {
            v: self.v * rhs.v,
            d,
        }
    }
}

// Subtraction: v = a - b,  d[i] = a.d[i] - b.d[i]
impl<T: Scalar, const N: usize> std::ops::Sub for Dual<T, N> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        let mut d = [T::constant(0.); N];
        d.iter_mut()
            .zip(self.d.iter().zip(rhs.d.iter()))
            .for_each(|(di, (&self_di, &rhs_di))| {
                *di = self_di - rhs_di;
            });
        Self {
            v: self.v - rhs.v,
            d,
        }
    }
}

/// Division: v = a / b,  d[i] = (a.d[i]*b - b.d[i]*a) / (b*b)  (quotient rule)
impl<T: Scalar, const N: usize> std::ops::Div for Dual<T, N> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        let mut d = [T::constant(0.); N];
        d.iter_mut()
            .zip(self.d.iter().zip(rhs.d.iter()))
            .for_each(|(di, (&self_di, &rhs_di))| {
                *di = ((self_di * rhs.v) - (rhs_di * self.v)) / (rhs.v * rhs.v);
            });
        Self {
            v: self.v / rhs.v,
            d,
        }
    }
}

// Negation: v = -a,  d[i] = -a.d[i]
impl<T: Scalar, const N: usize> std::ops::Neg for Dual<T, N> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            v: -self.v,
            d: self.d.map(|x| -x),
        }
    }
}

/// PartialEq only compares the value, not the tangent lanes.
impl<T: Scalar, const N: usize> PartialEq for Dual<T, N> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.v == other.v
    }
}

/// PartialOrd only compares the value, not the tangent lanes.
impl<T: Scalar, const N: usize> PartialOrd for Dual<T, N> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.v.partial_cmp(&other.v)
    }
}

/// Forward-mode AD math operations for Dual<T, N> (sqrt/sin/cos/powi).
impl<T: Scalar, const N: usize> Dual<T, N> {
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        let s = self.v.sqrt();
        let mut d = [T::constant(0.); N];
        let inv2s = T::constant(0.5) / s; // d(sqrt(v))/dv = 1/(2√v)
        d.iter_mut().zip(self.d.iter()).for_each(|(di, &self_di)| {
            *di = self_di * inv2s;
        });
        Self { v: s, d }
    }

    #[inline(always)]
    pub fn sin(self) -> Self {
        let s = self.v.sin();
        let c = self.v.cos();
        let mut d = [T::constant(0.); N];
        // d/dv sin(v) = cos(v)
        d.iter_mut().zip(self.d.iter()).for_each(|(di, &self_di)| {
            *di = self_di * c;
        });
        Self { v: s, d }
    }

    #[inline(always)]
    pub fn cos(self) -> Self {
        let s = self.v.cos();
        let n = self.v.sin();
        let mut d = [T::constant(0.); N];
        // d/dv cos(v) = -sin(v)
        d.iter_mut().zip(self.d.iter()).for_each(|(di, &self_di)| {
            *di = -self_di * n;
        });
        Self { v: s, d }
    }

    #[inline(always)]
    pub fn powi(self, n: i32) -> Self {
        // d/dv vⁿ = n·vⁿ⁻¹.  For n=0 the gradient is always zero (constant).
        // Guard: `0 * v.powi(-1)` for v=0 gives inf, not NaN, so is_nan() won't catch it.
        let p = self.v.powi(n);
        if n == 0 {
            return Self::constant(p); // v^0 = 1, gradient = 0
        }
        let factor = T::constant(n as f32) * self.v.powi(n - 1);
        let mut d = [T::constant(0.); N];
        d.iter_mut().zip(self.d.iter()).for_each(|(di, &self_di)| {
            *di = self_di * factor;
        });
        Self { v: p, d }
    }

    #[inline(always)]
    pub fn ln(self) -> Self {
        // d/dv ln(v) = 1/v
        let s = self.v.ln();
        let inv_v = T::constant(1.0) / self.v;
        let mut d = [T::constant(0.); N];
        d.iter_mut().zip(self.d.iter()).for_each(|(di, &self_di)| {
            *di = self_di * inv_v;
        });
        Self { v: s, d }
    }
}

/// A scalar type usable in SDF expressions.
///
/// Implemented for `f32` (zero-cost value path) and `Dual<N>` (forward-AD
/// gradient path). Write SDF functions once generic over `T: Scalar`.
pub trait Scalar:
    Copy
    + Send
    + Sync
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + PartialEq
    + PartialOrd
{
    fn constant(v: f32) -> Self;
    fn from_f32(v: f32) -> Self {
        Self::constant(v)
    }

    // Math operations — sqrt/sin/cos/powi have no std trait equivalents and
    // need custom AD rules.  abs/min/max are provided via PartialOrd + Neg.
    fn sqrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn ln(self) -> Self;

    #[inline(always)]
    fn abs(self) -> Self {
        if self >= Self::zero() { self } else { -self }
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        if self <= other { self } else { other }
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        if self >= other { self } else { other }
    }

    // Misc
    fn zero() -> Self {
        Self::constant(0.0)
    }
    fn one() -> Self {
        Self::constant(1.0)
    }
}

macro_rules! impl_scalar {
    ($t:ty) => {
        impl Scalar for $t {
            #[inline(always)]
            fn constant(v: f32) -> Self {
                v as $t
            }

            #[inline(always)]
            fn sqrt(self) -> Self {
                self.sqrt()
            }

            #[inline(always)]
            fn sin(self) -> Self {
                self.sin()
            }

            #[inline(always)]
            fn cos(self) -> Self {
                self.cos()
            }

            #[inline(always)]
            fn powi(self, n: i32) -> Self {
                self.powi(n)
            }

            #[inline(always)]
            fn ln(self) -> Self {
                self.ln()
            }
        }
    };
}

macro_rules! impl_scalar_forward_ad {
    ($t:ty) => {
        impl<T: Scalar, const N: usize> Scalar for $t {
            #[inline(always)]
            fn constant(v: f32) -> Self {
                Self::constant(T::constant(v))
            }

            #[inline(always)]
            fn sqrt(self) -> Self {
                self.sqrt()
            }

            #[inline(always)]
            fn sin(self) -> Self {
                self.sin()
            }

            #[inline(always)]
            fn cos(self) -> Self {
                self.cos()
            }

            #[inline(always)]
            fn powi(self, n: i32) -> Self {
                self.powi(n)
            }

            #[inline(always)]
            fn ln(self) -> Self {
                self.ln()
            }
        }
    };
}

impl_scalar!(f32);
impl_scalar!(f64);
impl_scalar_forward_ad!(Dual<T, N>);
