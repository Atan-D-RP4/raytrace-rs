use crate::aabb::Aabb;
use crate::vec3::{Point3, Vec3, reflect, refract};

#[derive(Debug, Clone, Copy)]
pub struct RayDifferentials {
    pub rx_origin: Point3,
    pub ry_origin: Point3,
    pub rx_direction: Vec3,
    pub ry_direction: Vec3,
}

#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Point3,
    /// TODO: refactor to `Direction(Vec3)` newtype when Vec3/Color3/Point3 get
    /// proper newtypes — this field must never be zero (see debug_assert in constructors).
    pub direction: Vec3,
    pub time: f64,
    pub inverse_direction: Vec3,
    pub differentials: Option<RayDifferentials>,
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
            inverse_direction: Vec3::new(1. / direction.x, 1. / direction.y, 1. / direction.z),
            differentials: None,
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
            inverse_direction: Vec3::new(1. / direction.x, 1. / direction.y, 1. / direction.z),
            differentials: None,
        }
    }

    pub fn new_with_differentials(
        origin: Point3,
        direction: Vec3,
        time: f64,
        differentials: Option<RayDifferentials>,
    ) -> Self {
        debug_assert!(
            !direction.near_zero(),
            "zero-direction ray — produces inf/nan in inverse_direction and AABB tests"
        );
        Self {
            origin,
            direction,
            time,
            inverse_direction: Vec3::new(1. / direction.x, 1. / direction.y, 1. / direction.z),
            differentials,
        }
    }

    /// Evaluate the ray at parameter t: returns the point along the ray at distance t from the
    /// origin.
    pub fn at(&self, t: f64) -> Point3 {
        let origin: Vec3 = self.origin;
        let direction = self.direction;
        origin + direction * t
    }

    /// World-space footprint of one pixel (dpdx) at a surface hit, using the
    /// tangent-plane projection (Igehy 1999 / pbrt `ComputeDifferentials`):
    /// intersect the offset ray with the tangent plane at the hit point rather
    /// than evaluating it at the primary ray's own hit distance. This accounts
    /// for surface foreshortening, which the `t_hit`-scaled estimate cannot.
    ///
    /// Falls back to the bounded `t_hit` estimate when the offset ray is nearly
    /// parallel to the tangent plane (grazing angles on curved surfaces), where
    /// the tangent-plane intersection is ill-conditioned and would blow up to
    /// infinity — producing extreme LOD and warping at the limb.
    pub(crate) fn differential_footprint(
        rx_origin: Point3,
        rx_direction: Vec3,
        hit_point: Point3,
        normal: Vec3,
        t_hit: f64,
        primary_origin: Point3,
        primary_direction: Vec3,
    ) -> Vec3 {
        let denom = normal.dot(&rx_direction);
        if denom.abs() < 1e-4 {
            // Grazing angle: tangent-plane formula is ill-conditioned.
            // Fall back to the bounded t_hit estimate.
            return (rx_origin - primary_origin) + t_hit * (rx_direction - primary_direction);
        }
        let t = normal.dot(&(hit_point - rx_origin)) / denom;
        (rx_origin + rx_direction * t) - hit_point
    }

    /// Propagate ray differentials through a surface scatter event.
    pub fn propagate_differentials(
        &self,
        normal: Vec3,
        hit_time: f64,
        eta: Option<f64>,
        hit_point: Point3,
    ) -> Option<RayDifferentials> {
        if let Some(rd) = self.differentials {
            // Preserve the spatial footprint: offset the new ray origins by
            // the incoming position derivatives (dpdx / dpdy at the hit).
            let dpdx = Ray::differential_footprint(
                rd.rx_origin,
                rd.rx_direction,
                hit_point,
                normal,
                hit_time,
                self.origin,
                self.direction,
            );
            let dpdy = Ray::differential_footprint(
                rd.ry_origin,
                rd.ry_direction,
                hit_point,
                normal,
                hit_time,
                self.origin,
                self.direction,
            );

            // Regenerate the ray differentials for the scattered ray. For reflection, the direction derivatives
            // are reflected. For refraction, the direction derivatives are refracted.
            let (rx_direction, ry_direction) = if let Some(eta) = eta {
                (
                    refract(&rd.rx_direction, &normal, eta),
                    refract(&rd.ry_direction, &normal, eta),
                )
            } else {
                (
                    reflect(&rd.rx_direction, &normal),
                    reflect(&rd.ry_direction, &normal),
                )
            };

            Some(RayDifferentials {
                rx_origin: hit_point + dpdx,
                ry_origin: hit_point + dpdy,
                rx_direction,
                ry_direction,
            })
        } else {
            None
        }
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
