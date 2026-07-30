use crate::shape::sdf::dual::Dual;

use super::SdfFn;

pub trait DynSdfFn: Send + Sync {
    fn eval_f32(&self, x: f32, y: f32, z: f32) -> f32;
    fn eval_dual(&self, x: Dual<f32, 3>, y: Dual<f32, 3>, z: Dual<f32, 3>) -> Dual<f32, 3>;
}

impl<F: SdfFn + 'static> DynSdfFn for F {
    fn eval_f32(&self, x: f32, y: f32, z: f32) -> f32 {
        self.eval(x, y, z)
    }

    fn eval_dual(&self, x: Dual<f32, 3>, y: Dual<f32, 3>, z: Dual<f32, 3>) -> Dual<f32, 3> {
        self.eval(x, y, z)
    }
}
