use std::sync::Arc;

use crate::shape::sdf::{
    BoxSdf, CapsuleSdf, CylinderSdf, MandelbulbSdf, RoundBoxSdf, SphereSdf, TorusSdf,
};
use crate::shape::sdf::{DynEval, SdfFn, dispatch::DynSdfFn, dual::Scalar};

/// A data-only SDF expression tree.
///
/// Kind codes are a stable contract: the variant tag is the GPU kernel kind code,
/// the payload is the parameter record. The primitive leaves wrap the data-only
/// struct kit from `impls.rs` (which holds the eval math); CSG operators compose
/// them. `Custom` is the CPU-only escape hatch for closure-based SDFs (same class
/// as `FunctionRegion`) — never required in shipped scenes.
#[derive(Clone)]
#[repr(u8)]
pub enum SdfExpr {
    // ── Primitive leaves (data-only) ──
    Sphere(SphereSdf) = 0,
    Box(BoxSdf) = 1,
    RoundBox(RoundBoxSdf) = 2,
    Torus(TorusSdf) = 3,
    Capsule(CapsuleSdf) = 4,
    Cylinder(CylinderSdf) = 5,
    Mandelbulb(MandelbulbSdf) = 6,

    // ── CSG operators ──
    Union(Box<SdfExpr>, Box<SdfExpr>) = 7,
    Intersect(Box<SdfExpr>, Box<SdfExpr>) = 8,
    Subtract(Box<SdfExpr>, Box<SdfExpr>) = 9,
    SmoothUnion {
        k: f32,
        a: Box<SdfExpr>,
        b: Box<SdfExpr>,
    } = 10,

    // ── Custom escape hatch (closure-SDFs, CPU-only) ──
    Custom(Arc<dyn DynSdfFn>) = 11,
}

macro_rules! impl_from_sdf {
    ($($variant:ident, $kit:ty),+ $(,)?) => {
        $(
            impl From<$kit> for SdfExpr {
                fn from(sdf: $kit) -> Self {
                    SdfExpr::$variant(sdf)
                }
            }
        )+
    };
}

impl_from_sdf!(
    Sphere,
    SphereSdf,
    Box,
    BoxSdf,
    RoundBox,
    RoundBoxSdf,
    Torus,
    TorusSdf,
    Capsule,
    CapsuleSdf,
    Cylinder,
    CylinderSdf,
    Mandelbulb,
    MandelbulbSdf,
);

impl SdfFn for SdfExpr {
    fn eval<T: Scalar + DynEval>(&self, x: T, y: T, z: T) -> T {
        match self {
            // ── Primitive leaves — delegate to the data-only struct kit ──
            SdfExpr::Sphere(s) => s.eval(x, y, z),
            SdfExpr::Box(b) => b.eval(x, y, z),
            SdfExpr::RoundBox(b) => b.eval(x, y, z),
            SdfExpr::Torus(t) => t.eval(x, y, z),
            SdfExpr::Capsule(c) => c.eval(x, y, z),
            SdfExpr::Cylinder(c) => c.eval(x, y, z),
            SdfExpr::Mandelbulb(m) => m.eval(x, y, z),
            // ── CSG operators ──
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::vec3::Point3;
    use glam::Vec3;

    fn val(expr: &SdfExpr, x: f32, y: f32, z: f32) -> f32 {
        expr.eval::<f32>(x, y, z)
    }

    #[test]
    fn leaf_kinds_delegate_to_kit() {
        let sphere = SdfExpr::Sphere(SphereSdf::new(1.0));
        assert!(val(&sphere, 0.0, 0.0, 0.0) < 0.0);
        assert_eq!(val(&sphere, 1.0, 0.0, 0.0), 0.0);
        assert!(val(&sphere, 2.0, 0.0, 0.0) > 0.0);

        let cylinder = SdfExpr::Cylinder(CylinderSdf::new(Point3::ZERO, 0.5, 2.0));
        assert!(val(&cylinder, 0.0, 0.0, 0.0) < 0.0);
        assert!(val(&cylinder, 2.0, 0.0, 0.0) > 0.0);
    }

    #[test]
    fn from_impls_wrap_kit_structs() {
        let s: SdfExpr = SphereSdf::new(1.0).into();
        assert!(val(&s, 0.0, 0.0, 0.0) < 0.0);
    }

    #[test]
    fn csg_composes_leaves() {
        // Classic sphere with a box carved out: sphere - box.
        let carved =
            SdfExpr::Sphere(SphereSdf::new(1.0)) - SdfExpr::Box(BoxSdf::new(Vec3::splat(0.4)));
        assert!(val(&carved, 0.0, 0.0, 0.0) > 0.0); // carved out: inside box, outside sphere
        assert!(val(&carved, 0.9, 0.0, 0.0) < 0.0); // still inside the sphere
        assert!(val(&carved, 2.0, 0.0, 0.0) > 0.0); // outside everything
    }
}
