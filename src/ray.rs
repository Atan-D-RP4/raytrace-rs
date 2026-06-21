use crate::aabb::Aabb;
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

    pub fn at(&self, t: f64) -> Point3 {
        let origin: Vec3 = self.origin;
        let direction = self.direction;
        origin + direction * t
    }
}

/// A linearly-interpolated point over time: at(t) = origin + velocity * t
///
/// Used for moving primitives (sphere center interpolation).
/// Unlike Ray, this has no direction validation or inverse_direction garbage.
/// When velocity is zero, the point is stationary (no motion).
#[derive(Debug, Clone, Copy)]
pub struct ParametricCurve {
    /// Starting point of the curve (e.g., ray origin).
    pub origin: Point3,
    /// Velocity vector of the curve (e.g., ray direction). Can be zero for stationary curves.
    pub velocity: Vec3,
}

impl ParametricCurve {
    pub fn new(origin: Point3, velocity: Vec3) -> Self {
        Self { origin, velocity }
    }

    /// Evaluate the curve at time t ∈ [0, 1]
    pub fn at(&self, t: f64) -> Point3 {
        self.origin + self.velocity * t
    }

    /// Compute the AABB swept by `bbox` moving along this curve from t=0 to t=1.
    ///
    /// For stationary curves (velocity = 0), returns the original bbox translated
    /// to origin. For moving curves, merges the AABB at both endpoints.
    pub fn sweep_aabb(&self, bbox: &Aabb) -> Aabb {
        let box0 = bbox.translate(self.origin);
        let box1 = bbox.translate(self.origin + self.velocity);

        box0.merge(&box1)
    }

    pub fn is_moving(&self) -> bool {
        self.velocity.length_squared() > 0.0
    }
}
