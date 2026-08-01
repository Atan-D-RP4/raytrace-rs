use crate::bvh::aabb::Aabb;
use crate::vec3::{Direction3, Point3};

/// Ray differentials for computing pixel footprints and texture filtering.
#[derive(Debug, Clone, Copy)]
pub struct RayDifferentials {
    /// The origin of the ray differential in the x direction.
    pub rx_origin: Point3,
    /// The origin of the ray differential in the y direction.
    pub ry_origin: Point3,
    /// The direction of the ray differential in the x direction.
    pub rx_direction: Direction3,
    /// The direction of the ray differential in the y direction.
    pub ry_direction: Direction3,
}

/// A ray in 3D space, defined by an origin point and a direction vector, with optional ray
/// differentials for computing pixel footprints and texture filtering.
///
/// The ray can also have an associated time value, which is useful for motion blur and
/// time-dependent effects.
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    /// The origin point of the ray.
    pub origin: Point3,
    /// The direction vector of the ray. Should be normalized.
    pub direction: Direction3,
    /// The time at which the ray exists, useful for motion blur and time-dependent effects.
    pub time: f32,
    /// The inverse of the direction vector, used for efficient ray-box intersection tests.
    pub inverse_direction: Direction3,
    /// Optional ray differentials for computing pixel footprints and texture filtering.
    pub differentials: Option<RayDifferentials>,
}

impl Ray {
    pub fn new(origin: Point3, direction: Direction3) -> Self {
        debug_assert!(
            direction.length_squared() >= 1e-8,
            "zero-direction ray — produces inf/nan in inverse_direction and AABB tests"
        );
        Self {
            origin,
            direction,
            time: 0.,
            inverse_direction: Direction3(direction.into_inner().recip()),
            differentials: None,
        }
    }

    pub fn new_with_time(origin: Point3, direction: Direction3, time: f32) -> Self {
        debug_assert!(
            direction.length_squared() >= 1e-8,
            "zero-direction ray — produces inf/nan in inverse_direction and AABB tests"
        );
        Self {
            origin,
            direction,
            time,
            inverse_direction: Direction3(direction.into_inner().recip()),
            differentials: None,
        }
    }

    pub fn new_with_differentials(
        origin: Point3,
        direction: Direction3,
        time: f32,
        differentials: Option<RayDifferentials>,
    ) -> Self {
        debug_assert!(
            direction.length_squared() >= 1e-8,
            "zero-direction ray — produces inf/nan in inverse_direction and AABB tests"
        );
        Self {
            origin,
            direction,
            time,
            inverse_direction: Direction3(direction.into_inner().recip()),
            differentials,
        }
    }

    /// Evaluate the ray at parameter t: returns the point along the ray at distance t from the
    /// origin.
    /// P(t) = O + t * D
    /// where t is the distance along the ray, O is the origin, and D is the direction.
    pub fn at(&self, t: f32) -> Point3 {
        self.origin + self.direction * t
    }

    /// World-space pixel footprint (dpdx) using tangent-plane projection (Igehy 2000 / pbrt ComputeDifferentials):
    /// intersect offset ray with tangent plane at hit point to account for surface foreshortening that
    /// t_hit-scaled estimates miss.
    ///
    /// Fallback to bounded t_hit estimate when offset ray is nearly parallel to tangent plane (denom < 1e-4),
    /// preventing extreme LOD and warping at grazing angles where intersection becomes ill-conditioned.
    pub(crate) fn differential_footprint(
        &self,
        rx_origin: Point3,
        rx_direction: Direction3,
        hit_point: Point3,
        normal: Direction3,
        t_hit: f32,
    ) -> Direction3 {
        let primary_origin = self.origin;
        let primary_direction = self.direction;
        let denom = normal.dot(rx_direction.into_inner());
        if denom.abs() < 1e-4 {
            // Grazing angle: tangent-plane formula is ill-conditioned.
            // Fall back to the bounded t_hit estimate.
            return (rx_origin - primary_origin) + t_hit * (rx_direction - primary_direction);
        }
        let t = normal.dot((hit_point - rx_origin).into_inner()) / denom;
        (rx_origin + rx_direction * t) - hit_point
    }

    /// Igehy curvature correction for specular scattering off curved surfaces.
    ///
    /// On a curved surface the normal changes across the pixel footprint, so reflecting
    /// (or refracting) the differential direction by a single normal understates the
    /// angular spread of the scattered ray bundle. This function computes the
    /// curvature-dependent terms needed to correct the scattered differential directions.
    ///
    /// **Reflection** (`eta = None`, Igehy 1999 eq. 5):
    /// Returns a correction term to add to the basic reflection of the *offset* direction:
    ///   dωᵣ/dx += -2(ωᵢ·dn/dx)n - 2(ωᵢ·n)dn/dx
    /// where ωᵢ is the incident direction and dn/dx = (dpdx - (dpdx·n)n) · curvature.
    ///
    /// **Refraction** (`eta = Some(η)`, Igehy 1999 eqs. 16–19 / pbrt-v4):
    /// Returns the *full* transmitted differential directions, following pbrt-v4's formula:
    ///   dωt/dx = wi − η·dwodx + μ·dn/dx + dμ/dx·n
    /// where wi is the basic refraction of the center incident direction, dwodx = d − rxd,
    /// μ = wo·n/η − |wi·n|, and dμ/dx depends on the normal derivative from curvature.
    pub fn curvature_correction(
        &self,
        (dpdx, dpdy): (Direction3, Direction3),
        normal: Direction3,
        curvature: f32,
        eta: Option<f32>,
        (rx_direction, ry_direction): (Direction3, Direction3),
    ) -> (Direction3, Direction3) {
        if curvature == 0.0 {
            return if let Some(eta) = eta {
                // Flat surface: just refract the offset direction (no curvature correction).
                (
                    rx_direction.refract(normal.into_inner(), eta),
                    ry_direction.refract(normal.into_inner(), eta),
                )
            } else {
                (Direction3::ZERO, Direction3::ZERO)
            };
        }

        let n = normal;
        let wi = self.direction;
        let wi_dot_n = wi.dot(n.into_inner());

        // dn/dx = (dpdx − (dpdx·n)n) · curvature  (tangent-plane projection × curvature)
        let dpdx_tan = dpdx - dpdx.dot(n.into_inner()) * n;
        let dn_dx = dpdx_tan * curvature;

        // dn/dy = (dpdy - (dpdy.n)n) · curvature  (tangent-plane projection × curvature)
        let dpdy_tan = dpdy - dpdy.dot(n.into_inner()) * n;
        let dn_dy = dpdy_tan * curvature;

        //
        if let Some(eta) = eta {
            // --- Refraction correction (Igehy 1999 eqs. 16–19 / pbrt-v4) ---
            let n_raw = n;
            let wo = -wi;

            // Orient normal so that wo·n ≥ 0 (pbrt convention for transmission).
            let (n_orient, dn_dx_adj, dn_dy_adj) = if wo.dot(n_raw.into_inner()) < 0.0 {
                (-n_raw, -dn_dx, -dn_dy)
            } else {
                (n_raw, dn_dx, dn_dy)
            };

            // Basic refraction of center incident direction through oriented normal.
            let wic = wi.refract(n_orient.into_inner(), eta);

            if wic.length_squared() < 1e-8 {
                // Total internal reflection — no meaningful transmitted differential.
                return (Direction3::ZERO, Direction3::ZERO);
            }

            // dwodx = −∂d/∂x = self.direction − rx_direction
            let d = wi;
            let dwodx = d - rx_direction;
            let dwody = d - ry_direction;

            // d(wo·n)/dx = dwodx·n + wo·dn/dx
            let dwo_dot_n_dx = dwodx.dot(n_orient.into_inner()) + wo.dot(dn_dx_adj.into_inner());
            let dwo_dot_n_dy = dwody.dot(n_orient.into_inner()) + wo.dot(dn_dy_adj.into_inner());

            // μ = wo·n/η − |wi·n|
            let wo_dot_n = wo.dot(n_orient.into_inner());
            let wic_dot_n = wic.dot(n_orient.into_inner());
            let mu = wo_dot_n / eta - wic_dot_n.abs();

            // dμ/dx = d(wo·n)/dx · (1/η + wo·n/(η²·wi·n))
            let mu_factor = if wic_dot_n.abs() < 1e-6 {
                1.0 / eta // Grazing-angle guard: near total internal reflection.
            } else {
                1.0 / eta + wo_dot_n / (eta * eta * wic_dot_n)
            };

            // dμ/dx = d(wo·n)/dx · (1/η + wo·n/(η²·wi·n))
            let dmudx = dwo_dot_n_dx * mu_factor;
            let dmudy = dwo_dot_n_dy * mu_factor;

            // dωt/dx = wi − η·dwodx + μ·dn/dx + dμ/dx·n
            let rx_dir = wic - eta * dwodx + mu * dn_dx_adj + dmudx * n_orient;
            let ry_dir = wic - eta * dwody + mu * dn_dy_adj + dmudy * n_orient;

            (rx_dir.normalize(), ry_dir.normalize())
        } else {
            // --- Reflection correction (Igehy 1999 eq. 5) ---
            // dωᵣ/dx correction: -2(ωᵢ·dn/dx)n - 2(ωᵢ·n)dn/dx
            let dx_correction = -2.0 * wi.dot(dn_dx.into_inner()) * n - 2.0 * wi_dot_n * dn_dx;
            let dy_correction = -2.0 * wi.dot(dn_dy.into_inner()) * n - 2.0 * wi_dot_n * dn_dy;

            (dx_correction, dy_correction)
        }
    }

    /// Propagate ray differentials through a surface scatter event.
    ///
    /// Computes the new position derivatives (`dpdx`/`dpdy`) via [`differential_footprint`],
    /// then transforms the direction derivatives by the scatter event
    /// (reflection or refraction). For reflection off curved surfaces, applies the
    /// Igehy curvature correction ([`curvature_correction`]) to account for the
    /// normal change across the pixel footprint.
    pub fn propagate_differentials(
        &self,
        normal: Direction3,
        hit_time: f32,
        eta: Option<f32>,
        hit_point: Point3,
        curvature: f32,
    ) -> Option<RayDifferentials> {
        self.differentials.as_ref().map(|rd| {
            // Preserve the spatial footprint: offset the new ray origins by
            // the incoming position derivatives (dpdx / dpdy at the hit).
            let dpdx = self.differential_footprint(
                rd.rx_origin,
                rd.rx_direction,
                hit_point,
                normal,
                hit_time,
            );
            let dpdy = self.differential_footprint(
                rd.ry_origin,
                rd.ry_direction,
                hit_point,
                normal,
                hit_time,
            );

            // Regenerate the ray differentials for the scattered ray.
            let (rx_direction, ry_direction) = if let Some(eta) = eta {
                // Refraction: curvature_correction returns the full transmitted
                // differential directions (Igehy 1999 eqs. 16–19 / pbrt-v4).
                self.curvature_correction(
                    (dpdx, dpdy),
                    normal,
                    curvature,
                    Some(eta),
                    (rd.rx_direction, rd.ry_direction),
                )
            } else {
                // Reflection: reflect offset direction, add curvature correction.
                let (mut rx_direction, mut ry_direction) = (
                    rd.rx_direction.reflect(normal.into_inner()),
                    rd.ry_direction.reflect(normal.into_inner()),
                );

                let (dx_correction, dy_correction) = self.curvature_correction(
                    (dpdx, dpdy),
                    normal,
                    curvature,
                    None,
                    (rd.rx_direction, rd.ry_direction),
                );

                rx_direction += dx_correction;
                ry_direction += dy_correction;
                (rx_direction.normalize(), ry_direction.normalize())
            };

            RayDifferentials {
                rx_origin: hit_point + dpdx,
                ry_origin: hit_point + dpdy,
                rx_direction,
                ry_direction,
            }
        })
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
    pub velocity: Direction3,
}

impl ParametricCurve {
    pub fn new(origin: Point3, velocity: Direction3) -> Self {
        Self { origin, velocity }
    }

    /// Evaluate the curve at time t ∈ [0, 1]
    /// P(t) = O + t * V
    /// where, t is the normalized time parameter, O is the origin, and V is the velocity
    pub fn at(&self, t: f32) -> Point3 {
        self.origin + self.velocity * t
    }

    /// Compute the AABB swept by `bbox` moving along this curve from t=0 to t=1.
    ///
    /// For stationary curves (velocity = 0), returns the original bbox translated
    /// to origin. For moving curves, merges the AABB at both endpoints.
    pub fn sweep_aabb(&self, bbox: &Aabb) -> Aabb {
        let box0 = bbox.translate(self.origin.into_inner());
        let box1 = bbox.translate((self.origin + self.velocity).into_inner());

        box0.merge(&box1)
    }

    pub fn is_moving(&self) -> bool {
        self.velocity.length_squared() > 0.0
    }
}
