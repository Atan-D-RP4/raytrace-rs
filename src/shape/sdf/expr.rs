use crate::shape::sdf::{DynEval, SdfFn, dispatch::DynSdfFn, dual::Scalar};

pub enum SdfExpr {
    // ── CSG operators ──
    Union(Box<SdfExpr>, Box<SdfExpr>),
    Intersect(Box<SdfExpr>, Box<SdfExpr>),
    Subtract(Box<SdfExpr>, Box<SdfExpr>),
    SmoothUnion {
        k: f32,
        a: Box<SdfExpr>,
        b: Box<SdfExpr>,
    },

    // ── Custom escape hatch (dynamic, dispatch preserves gradients) ──
    Custom(Box<dyn DynSdfFn>),
}

impl SdfFn for SdfExpr {
    fn eval<T: Scalar + DynEval>(&self, x: T, y: T, z: T) -> T {
        match self {
            SdfExpr::Union(a, b) => {
                let da = a.eval(x, y, z);
                let db = b.eval(x, y, z);
                da.min(db)
            }
            SdfExpr::Intersect(a, b) => {
                let da = a.eval(x, y, z);
                let db = b.eval(x, y, z);
                da.max(db)
            }
            SdfExpr::Subtract(a, b) => {
                let da = a.eval(x, y, z);
                let db = b.eval(x, y, z);
                da.max(-db)
            }
            SdfExpr::SmoothUnion { k, a, b } => {
                let da = a.eval(x, y, z);
                let db = b.eval(x, y, z);
                let half = T::constant(0.5);
                let h = (half + half * (db - da) / T::from_f32(*k))
                    .clamp(T::constant(0.), T::constant(1.));
                let smooth = T::from_f32(*k) * h * h * (T::constant(3.) - T::from_f32(2.) * h);
                da.min(db) - smooth
            }
            SdfExpr::Custom(sdf_fn) => DynEval::eval_dyn(&**sdf_fn, x, y, z),
        }
    }
}

impl std::ops::BitOr for SdfExpr {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        SdfExpr::Union(Box::new(self), Box::new(rhs))
    }
}

impl std::ops::BitAnd for SdfExpr {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self {
        SdfExpr::Intersect(Box::new(self), Box::new(rhs))
    }
}

impl std::ops::Sub for SdfExpr {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        SdfExpr::Subtract(Box::new(self), Box::new(rhs))
    }
}
