use crate::shape::sdf::SdfFn;
use crate::shape::sdf::dual::Dual;

/// Object-safe SDF evaluation for dynamic dispatch.
///
/// The expression tree stores `Custom(Box<dyn DynSdfFn>)` — a type-erased SDF.  Object safety
/// requires concrete (non-generic) evaluation methods. There is exactly one method per evaluation
/// context that `SdfShape` uses:
///
/// - `eval_f32`       — value path (sphere tracing), `T = f32`
/// - `eval_dual`      — value + gradient (normals), `T = Dual<f32, 3>`
/// - `eval_curvature` — value + first + second derivatives (mean curvature), `T = Dual<Dual<f32, 3>, 3>`
///
/// The blanket impl routes all three to the generic `SdfFn::eval`, so a custom SDF composed into an
/// expression tree keeps full dual-number gradients — identical to the monomorphized path.
pub trait DynSdfFn: Send + Sync {
    fn eval_f32(&self, x: f32, y: f32, z: f32) -> f32;
    fn eval_dual(&self, x: Dual<f32, 3>, y: Dual<f32, 3>, z: Dual<f32, 3>) -> Dual<f32, 3>;
    fn eval_curvature(
        &self,
        x: Dual<Dual<f32, 3>, 3>,
        y: Dual<Dual<f32, 3>, 3>,
        z: Dual<Dual<f32, 3>, 3>,
    ) -> Dual<Dual<f32, 3>, 3>;
}

impl<F: SdfFn + 'static> DynSdfFn for F {
    fn eval_f32(&self, x: f32, y: f32, z: f32) -> f32 {
        self.eval(x, y, z)
    }

    fn eval_dual(&self, x: Dual<f32, 3>, y: Dual<f32, 3>, z: Dual<f32, 3>) -> Dual<f32, 3> {
        self.eval(x, y, z)
    }

    fn eval_curvature(
        &self,
        x: Dual<Dual<f32, 3>, 3>,
        y: Dual<Dual<f32, 3>, 3>,
        z: Dual<Dual<f32, 3>, 3>,
    ) -> Dual<Dual<f32, 3>, 3> {
        self.eval(x, y, z)
    }
}

/// Dispatch to the correct `DynSdfFn` method for a concrete scalar type.
///
/// Implemented for exactly the three scalar types the SDF pipeline uses.
/// `SdfFn::eval<T: Scalar + DynEval>` calls this in the `Custom` branch, so
/// a type-erased SDF in an expression tree evaluates with the same dual
/// semantics as a monomorphized one.
pub trait DynEval: Sized {
    fn eval_dyn(sdf: &dyn DynSdfFn, x: Self, y: Self, z: Self) -> Self;
}

impl DynEval for f32 {
    fn eval_dyn(sdf: &dyn DynSdfFn, x: f32, y: f32, z: f32) -> f32 {
        sdf.eval_f32(x, y, z)
    }
}

impl DynEval for Dual<f32, 3> {
    fn eval_dyn(
        sdf: &dyn DynSdfFn,
        x: Dual<f32, 3>,
        y: Dual<f32, 3>,
        z: Dual<f32, 3>,
    ) -> Dual<f32, 3> {
        sdf.eval_dual(x, y, z)
    }
}

impl DynEval for Dual<Dual<f32, 3>, 3> {
    fn eval_dyn(
        sdf: &dyn DynSdfFn,
        x: Dual<Dual<f32, 3>, 3>,
        y: Dual<Dual<f32, 3>, 3>,
        z: Dual<Dual<f32, 3>, 3>,
    ) -> Dual<Dual<f32, 3>, 3> {
        sdf.eval_curvature(x, y, z)
    }
}
