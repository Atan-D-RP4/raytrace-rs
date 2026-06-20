use crate::vec3::{Point3, Vec3};

#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Point3,
    /// TODO: refactor to `Direction(Vec3)` newtype when Vec3/Color3/Point3 get
    /// proper newtypes — this field must never be zero (see debug_assert in constructors).
    pub direction: Vec3,
    pub time: f64,
    pub inverse_direction: Vec3,
}

impl Ray {
    pub fn new(origin: Point3, direction: Vec3) -> Self {
        debug_assert!(
            !direction.near_zero(),
            "zero-direction ray — produces inf/nan in inverse_direction and AABB tests"
        );
        Self {
            origin,
            direction,
            time: 0.,
            inverse_direction: Vec3::from(1. / direction.x, 1. / direction.y, 1. / direction.z),
        }
    }

    pub fn new_with_time(origin: Point3, direction: Vec3, time: f64) -> Self {
        debug_assert!(
            !direction.near_zero(),
            "zero-direction ray — produces inf/nan in inverse_direction and AABB tests"
        );
        Self {
            origin,
            direction,
            time,
            inverse_direction: Vec3::from(1. / direction.x, 1. / direction.y, 1. / direction.z),
        }
    }

    /// Construct without direction validation — for non-intersection uses
    /// (parametric curves, transform stubs) where zero direction is valid.
    #[inline(always)]
    pub fn new_raw(origin: Point3, direction: Vec3, time: f64) -> Self {
        Self {
            origin,
            direction,
            time,
            inverse_direction: Vec3::from(1. / direction.x, 1. / direction.y, 1. / direction.z),
        }
    }

    pub fn at(&self, t: f64) -> Point3 {
        let origin: Vec3 = self.origin;
        let direction = self.direction;
        origin + direction * t
    }
}
