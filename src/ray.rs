//! SIMD-packed ray types with an axis-major SoA layout (the same pattern as
//! [`AabbPacked`]), plus scalar aliases that replace the old AoS ray types.
//!
//! # Layout
//!
//! Every vector field is stored as `[[f32; N]; 3]` — axis-major, so the N
//! x-components are contiguous and load into one `Simd<f32, N>` register via
//! `Simd::from_array`. An AoS layout (`[Point3; N]`) would stride lanes by
//! 16 bytes and require gathers, which is exactly what the packed AABB avoids.
//!
//! # Two API layers
//!
//! **Scalar API (lane 0)** — the drop-in replacement for the old `Ray`:
//! constructors broadcast one ray across all lanes, accessors and methods
//! (`at`, `differential_footprint`, `curvature_correction`,
//! `propagate_differentials`) operate on lane 0. The aliases
//! [`Ray`], [`RayDifferentials`], and [`ParametricCurve`] are the N=1
//! specializations the rest of the crate uses.
//!
//! **Packed API (all lanes)** — the `*_packed` methods process every lane and
//! are the reference implementations for the SIMD kernels: the kernels load
//! whole axes into registers and replace the per-lane loops with masked
//! operations, verified against the scalar methods lane-by-lane.
//!
//! # Differentials
//!
//! Ray differentials are all-or-nothing per pack: either every lane carries
//! them or none do. The gather (`From<[RayPacked<1>; N]>`) sets `None` if any
//! lane lacks them. Per-lane `Option`s would waste N×48 bytes and complicate
//! the SIMD kernels for a case that never occurs in practice (the perspective
//! camera always produces differentials).

use crate::bvh::aabb::AabbPacked;
use crate::math::vec3::{Direction3, Point3};
use glam::Vec3;

/// Ray differentials for computing pixel footprints and texture filtering,
/// packed for SIMD processing.
#[derive(Debug, Clone, Copy)]
pub struct RayDifferentialsPacked<const N: usize> {
    /// The origin points of the ray differential in the x direction.
    pub rx_origin: [[f32; N]; 3],
    /// The origin points of the ray differential in the y direction.
    pub ry_origin: [[f32; N]; 3],
    /// The direction vectors of the ray differential in the x direction.
    pub rx_direction: [[f32; N]; 3],
    /// The direction vectors of the ray differential in the y direction.
    pub ry_direction: [[f32; N]; 3],
}

impl<const N: usize> RayDifferentialsPacked<N> {
    /// Scalar constructor: broadcasts one ray differential across all lanes.
    pub fn new(
        rx_origin: Point3,
        ry_origin: Point3,
        rx_direction: Direction3,
        ry_direction: Direction3,
    ) -> Self {
        let rx_o = rx_origin.into_inner();
        let ry_o = ry_origin.into_inner();
        let rx_d = rx_direction.into_inner();
        let ry_d = ry_direction.into_inner();
        Self {
            rx_origin: [[rx_o.x; N], [rx_o.y; N], [rx_o.z; N]],
            ry_origin: [[ry_o.x; N], [ry_o.y; N], [ry_o.z; N]],
            rx_direction: [[rx_d.x; N], [rx_d.y; N], [rx_d.z; N]],
            ry_direction: [[ry_d.x; N], [ry_d.y; N], [ry_d.z; N]],
        }
    }

    /// Lane-0 accessor: the x-direction origin point.
    pub fn rx_origin(&self) -> Point3 {
        Point3(lane_vec3(&self.rx_origin, 0))
    }

    /// Lane-0 accessor: the y-direction origin point.
    pub fn ry_origin(&self) -> Point3 {
        Point3(lane_vec3(&self.ry_origin, 0))
    }

    /// Lane-0 accessor: the x-direction direction vector.
    pub fn rx_direction(&self) -> Direction3 {
        Direction3(lane_vec3(&self.rx_direction, 0))
    }

    /// Lane-0 accessor: the y-direction direction vector.
    pub fn ry_direction(&self) -> Direction3 {
        Direction3(lane_vec3(&self.ry_direction, 0))
    }
}

/// A ray in 3D space, defined by an origin point and a direction vector, with
/// optional ray differentials for computing pixel footprints and texture
/// filtering, packed for SIMD processing.
#[derive(Debug, Clone, Copy)]
pub struct RayPacked<const N: usize> {
    /// The origin points of the rays.
    pub origin: [[f32; N]; 3],
    /// The direction vectors of the rays. Should be normalized.
    pub direction: [[f32; N]; 3],
    /// The time values at which the rays exist, useful for motion blur and
    /// time-dependent effects.
    pub time: [f32; N],
    /// The inverse of the direction vectors, used for efficient ray-box
    /// intersection tests.
    pub inverse_direction: [[f32; N]; 3],
    /// Optional ray differentials for computing pixel footprints and texture
    /// filtering. All-or-nothing per pack — see the module docs.
    pub differentials: Option<RayDifferentialsPacked<N>>,
}

// ── Per-lane helpers ────────────────────────────────────────────────────────
// glam is an AoS library (no SoA gather/scatter API — verified against its
// source: only from_array/to_array/from_slice/Index, all single-vector), so
// the lane gather below is necessary. The per-lane math uses glam (already
// 3-wide SIMD); the axis-major layout is what enables cross-lane SIMD later.

/// Gather lane `i` from an axis-major array into a `Vec3`.
#[inline]
fn lane_vec3<const N: usize>(arr: &[[f32; N]; 3], i: usize) -> Vec3 {
    Vec3::new(arr[0][i], arr[1][i], arr[2][i])
}

/// Scatter a `Vec3` into lane `i` of an axis-major array.
#[inline]
fn set_lane_vec3<const N: usize>(arr: &mut [[f32; N]; 3], i: usize, v: Vec3) {
    arr[0][i] = v.x;
    arr[1][i] = v.y;
    arr[2][i] = v.z;
}

/// Transpose per-lane results back into axis-major SoA layout.
#[inline]
fn transpose<const N: usize>(lanes: [Vec3; N]) -> [[f32; N]; 3] {
    core::array::from_fn(|axis| core::array::from_fn(|i| lanes[i][axis]))
}

#[inline]
fn add_packed<const N: usize>(a: &[[f32; N]; 3], b: &[[f32; N]; 3]) -> [[f32; N]; 3] {
    core::array::from_fn(|axis| core::array::from_fn(|i| a[axis][i] + b[axis][i]))
}

impl<const N: usize> RayPacked<N> {
    // ── Scalar constructors (broadcast one ray across all lanes) ────────────

    /// Creates a new ray with the given origin and direction at time 0.
    /// The inverse direction is computed for efficient ray-box intersection
    /// tests. No ray differentials.
    pub fn new(origin: Point3, direction: Direction3) -> Self {
        Self::new_with_differentials(origin, direction, 0.0, None)
    }

    /// Creates a new ray with the given origin, direction, and time.
    /// The inverse direction is computed for efficient ray-box intersection
    /// tests. No ray differentials.
    pub fn new_with_time(origin: Point3, direction: Direction3, time: f32) -> Self {
        Self::new_with_differentials(origin, direction, time, None)
    }

    /// Creates a new ray with the given origin, direction, time, and optional
    /// ray differentials. The inverse direction is computed for efficient
    /// ray-box intersection tests.
    pub fn new_with_differentials(
        origin: Point3,
        direction: Direction3,
        time: f32,
        differentials: Option<RayDifferentialsPacked<N>>,
    ) -> Self {
        debug_assert!(
            direction.length_squared() >= 1e-8,
            "zero-direction ray — produces inf/nan in inverse_direction and AABB tests"
        );
        let o = origin.into_inner();
        let d = direction.into_inner();
        Self {
            origin: [[o.x; N], [o.y; N], [o.z; N]],
            direction: [[d.x; N], [d.y; N], [d.z; N]],
            time: [time; N],
            inverse_direction: [[d.x.recip(); N], [d.y.recip(); N], [d.z.recip(); N]],
            differentials,
        }
    }

    // ── Lane-0 accessors (scalar API) ───────────────────────────────────────

    /// The origin point of lane 0.
    pub fn origin(&self) -> Point3 {
        Point3(lane_vec3(&self.origin, 0))
    }

    /// The direction vector of lane 0.
    pub fn direction(&self) -> Direction3 {
        Direction3(lane_vec3(&self.direction, 0))
    }

    /// The time of lane 0.
    pub fn time(&self) -> f32 {
        self.time[0]
    }

    /// The inverse direction of lane 0.
    pub fn inverse_direction(&self) -> Direction3 {
        Direction3(lane_vec3(&self.inverse_direction, 0))
    }

    // ── Scalar methods (lane 0) ─────────────────────────────────────────────

    /// Evaluate lane 0 at parameter `t`: returns the point along the ray at
    /// distance `t` from the origin.
    /// P(t) = O + t * D
    pub fn at(&self, t: f32) -> Point3 {
        self.at_lane(0, t)
    }

    /// World-space pixel footprint (dpdx) of lane 0 using tangent-plane
    /// projection (Igehy 2000 / pbrt ComputeDifferentials): intersect the
    /// offset ray with the tangent plane at the hit point to account for
    /// surface foreshortening that `t_hit`-scaled estimates miss.
    ///
    /// Falls back to the bounded `t_hit` estimate when the offset ray is
    /// nearly parallel to the tangent plane (`|denom| < 1e-4`), preventing
    /// extreme LOD and warping at grazing angles where the intersection
    /// becomes ill-conditioned.
    pub(crate) fn differential_footprint(
        &self,
        rx_origin: Point3,
        rx_direction: Direction3,
        hit_point: Point3,
        normal: Direction3,
        t_hit: f32,
    ) -> Direction3 {
        Direction3(self.differential_footprint_lane(
            0,
            rx_origin.into_inner(),
            rx_direction.into_inner(),
            hit_point.into_inner(),
            normal.into_inner(),
            t_hit,
        ))
    }

    /// Igehy curvature correction for specular scattering off curved surfaces
    /// (lane 0).
    ///
    /// On a curved surface the normal changes across the pixel footprint, so
    /// reflecting (or refracting) the differential direction by a single
    /// normal understates the angular spread of the scattered ray bundle.
    /// This function computes the curvature-dependent terms needed to correct
    /// the scattered differential directions.
    ///
    /// **Reflection** (`eta = None`, Igehy 1999 eq. 5):
    /// Returns a correction term to add to the basic reflection of the *offset*
    /// direction: dωᵣ/dx += −2(ωᵢ·dn/dx)n − 2(ωᵢ·n)dn/dx.
    ///
    /// **Refraction** (`eta = Some(η)`, Igehy 1999 eqs. 16–19 / pbrt-v4):
    /// Returns the *full* transmitted differential directions.
    pub fn curvature_correction(
        &self,
        (dpdx, dpdy): (Direction3, Direction3),
        normal: Direction3,
        curvature: f32,
        eta: Option<f32>,
        (rx_direction, ry_direction): (Direction3, Direction3),
    ) -> (Direction3, Direction3) {
        let (rx, ry) = self.curvature_correction_lane(
            0,
            dpdx.into_inner(),
            dpdy.into_inner(),
            normal.into_inner(),
            curvature,
            eta,
            rx_direction.into_inner(),
            ry_direction.into_inner(),
        );
        (Direction3(rx), Direction3(ry))
    }

    /// Propagate ray differentials of lane 0 through a surface scatter event.
    ///
    /// Computes the new position derivatives (`dpdx`/`dpdy`) via
    /// [`differential_footprint`](Self::differential_footprint), then
    /// transforms the direction derivatives by the scatter event (reflection
    /// or refraction). For reflection off curved surfaces, applies the Igehy
    /// curvature correction ([`curvature_correction`](Self::curvature_correction))
    /// to account for the normal change across the pixel footprint.
    ///
    /// Lane 0 is fully propagated; other lanes pass through unchanged (the
    /// scalar path is N=1, so this is exact there).
    pub fn propagate_differentials(
        &self,
        normal: Direction3,
        hit_time: f32,
        eta: Option<f32>,
        hit_point: Point3,
        curvature: f32,
    ) -> Option<RayDifferentialsPacked<N>> {
        let rd = self.differentials?;
        let mut out = rd;

        // Preserve the spatial footprint: offset the new ray origins by
        // the incoming position derivatives (dpdx / dpdy at the hit).
        let dpdx = self.differential_footprint(
            rd.rx_origin(),
            rd.rx_direction(),
            hit_point,
            normal,
            hit_time,
        );
        let dpdy = self.differential_footprint(
            rd.ry_origin(),
            rd.ry_direction(),
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
                (rd.rx_direction(), rd.ry_direction()),
            )
        } else {
            // Reflection: reflect offset direction, add curvature correction.
            let (mut rx_direction, mut ry_direction) = (
                rd.rx_direction().reflect(normal.into_inner()),
                rd.ry_direction().reflect(normal.into_inner()),
            );

            let (dx_correction, dy_correction) = self.curvature_correction(
                (dpdx, dpdy),
                normal,
                curvature,
                None,
                (rd.rx_direction(), rd.ry_direction()),
            );

            rx_direction += dx_correction;
            ry_direction += dy_correction;
            (rx_direction.normalize(), ry_direction.normalize())
        };

        set_lane_vec3(&mut out.rx_origin, 0, (hit_point + dpdx).into_inner());
        set_lane_vec3(&mut out.ry_origin, 0, (hit_point + dpdy).into_inner());
        set_lane_vec3(&mut out.rx_direction, 0, rx_direction.into_inner());
        set_lane_vec3(&mut out.ry_direction, 0, ry_direction.into_inner());
        Some(out)
    }

    // ── Per-lane core (shared by the scalar and packed methods) ─────────────

    /// Evaluate lane `i` at parameter `t`.
    fn at_lane(&self, i: usize, t: f32) -> Point3 {
        Point3::new(
            self.origin[0][i] + self.direction[0][i] * t,
            self.origin[1][i] + self.direction[1][i] * t,
            self.origin[2][i] + self.direction[2][i] * t,
        )
    }

    /// World-space pixel footprint of lane `i` (see the scalar wrapper).
    fn differential_footprint_lane(
        &self,
        i: usize,
        rx_origin: Vec3,
        rx_direction: Vec3,
        hit_point: Vec3,
        normal: Vec3,
        t_hit: f32,
    ) -> Vec3 {
        let denom = normal.dot(rx_direction);
        if denom.abs() < 1e-4 {
            // Grazing angle: tangent-plane formula is ill-conditioned.
            // Fall back to the bounded t_hit estimate.
            return (rx_origin - lane_vec3(&self.origin, i))
                + t_hit * (rx_direction - lane_vec3(&self.direction, i));
        }
        let t = normal.dot(hit_point - rx_origin) / denom;
        (rx_origin + rx_direction * t) - hit_point
    }

    /// Igehy curvature correction of lane `i` (see the scalar wrapper).
    #[allow(clippy::too_many_arguments)] // private per-lane core; mirrors the scalar Ray API
    fn curvature_correction_lane(
        &self,
        i: usize,
        dpdx: Vec3,
        dpdy: Vec3,
        normal: Vec3,
        curvature: f32,
        eta: Option<f32>,
        rx_direction: Vec3,
        ry_direction: Vec3,
    ) -> (Vec3, Vec3) {
        let n = normal;
        let wi = lane_vec3(&self.direction, i);
        let wi_dot_n = wi.dot(n);
        let k = curvature;

        // Flat surface (k == 0): no curvature terms — just refract the
        // offset directions, or zero correction for reflection.
        let flat = match eta {
            Some(eta) => (rx_direction.refract(n, eta), ry_direction.refract(n, eta)),
            None => (Vec3::ZERO, Vec3::ZERO),
        };

        // dn/dx = (dpdx − (dpdx·n)n) · curvature  (tangent-plane projection × curvature)
        let dpdx_tan = dpdx - dpdx.dot(n) * n;
        let dn_dx = dpdx_tan * k;
        let dpdy_tan = dpdy - dpdy.dot(n) * n;
        let dn_dy = dpdy_tan * k;

        let general = match eta {
            Some(eta) => {
                // --- Refraction correction (Igehy 1999 eqs. 16–19 / pbrt-v4) ---
                let wo = -wi;

                // Orient normal so that wo·n ≥ 0 (pbrt convention for transmission).
                let (n_orient, dn_dx_adj, dn_dy_adj) = if wo.dot(n) < 0.0 {
                    (-n, -dn_dx, -dn_dy)
                } else {
                    (n, dn_dx, dn_dy)
                };

                // Basic refraction of center incident direction through oriented normal.
                let wic = wi.refract(n_orient, eta);

                if wic.length_squared() < 1e-8 {
                    // Total internal reflection — no meaningful transmitted differential.
                    (Vec3::ZERO, Vec3::ZERO)
                } else {
                    // dwodx = −∂d/∂x = wi − rx_direction
                    let dwodx = wi - rx_direction;
                    let dwody = wi - ry_direction;

                    // d(wo·n)/dx = dwodx·n + wo·dn/dx
                    let dwo_dot_n_dx = dwodx.dot(n_orient) + wo.dot(dn_dx_adj);
                    let dwo_dot_n_dy = dwody.dot(n_orient) + wo.dot(dn_dy_adj);

                    // μ = wo·n/η − |wi·n|
                    let wo_dot_n = wo.dot(n_orient);
                    let wic_dot_n = wic.dot(n_orient);
                    let mu = wo_dot_n / eta - wic_dot_n.abs();

                    // dμ/dx = d(wo·n)/dx · (1/η + wo·n/(η²·wi·n))
                    let mu_factor = if wic_dot_n.abs() < 1e-6 {
                        1.0 / eta // Grazing-angle guard: near total internal reflection.
                    } else {
                        1.0 / eta + wo_dot_n / (eta * eta * wic_dot_n)
                    };
                    let dmudx = dwo_dot_n_dx * mu_factor;
                    let dmudy = dwo_dot_n_dy * mu_factor;

                    // dωt/dx = wi − η·dwodx + μ·dn/dx + dμ/dx·n
                    let rx_dir = wic - eta * dwodx + mu * dn_dx_adj + dmudx * n_orient;
                    let ry_dir = wic - eta * dwody + mu * dn_dy_adj + dmudy * n_orient;

                    (rx_dir.normalize(), ry_dir.normalize())
                }
            }
            None => {
                // --- Reflection correction (Igehy 1999 eq. 5) ---
                // dωᵣ/dx correction: −2(ωᵢ·dn/dx)n − 2(ωᵢ·n)dn/dx
                let dx_correction = -2.0 * wi.dot(dn_dx) * n - 2.0 * wi_dot_n * dn_dx;
                let dy_correction = -2.0 * wi.dot(dn_dy) * n - 2.0 * wi_dot_n * dn_dy;
                (dx_correction, dy_correction)
            }
        };

        if k == 0.0 {
            flat
        } else {
            general
        }
    }

    // ── Packed reference methods (all lanes; SIMD kernels replace these) ────

    /// Evaluate all rays at per-lane parameter `t`: returns the points along
    /// the rays at distance `t[i]` from each origin.
    /// P(t) = O + t * D
    pub fn at_packed(&self, t: [f32; N]) -> [Point3; N] {
        core::array::from_fn(|i| self.at_lane(i, t[i]))
    }

    /// World-space pixel footprints (dpdx) using tangent-plane projection
    /// (Igehy 2000 / pbrt ComputeDifferentials): intersect the offset rays
    /// with the tangent plane at the hit points to account for surface
    /// foreshortening that `t_hit`-scaled estimates miss.
    ///
    /// Per-lane fallback to the bounded `t_hit` estimate when the offset ray
    /// is nearly parallel to the tangent plane (`|denom| < 1e-4`), preventing
    /// extreme LOD and warping at grazing angles where the intersection
    /// becomes ill-conditioned. Matches the scalar
    /// [`differential_footprint`](Self::differential_footprint) lane-by-lane.
    pub fn differential_footprint_packed(
        &self,
        rx_origin: &[[f32; N]; 3],
        rx_direction: &[[f32; N]; 3],
        hit_point: &[[f32; N]; 3],
        normal: &[[f32; N]; 3],
        t_hit: &[f32; N],
    ) -> [[f32; N]; 3] {
        let lanes: [Vec3; N] = core::array::from_fn(|i| {
            self.differential_footprint_lane(
                i,
                lane_vec3(rx_origin, i),
                lane_vec3(rx_direction, i),
                lane_vec3(hit_point, i),
                lane_vec3(normal, i),
                t_hit[i],
            )
        });
        transpose(lanes)
    }

    /// Igehy curvature correction for specular scattering off curved surfaces,
    /// all lanes. Matches the scalar
    /// [`curvature_correction`](Self::curvature_correction) lane-by-lane.
    #[allow(clippy::too_many_arguments)] // mirrors the scalar Ray API; bundling would obscure the per-lane data flow
    pub fn curvature_correction_packed(
        &self,
        dpdx: &[[f32; N]; 3],
        dpdy: &[[f32; N]; 3],
        normal: &[[f32; N]; 3],
        curvature: &[f32; N],
        eta: &[Option<f32>; N],
        rx_direction: &[[f32; N]; 3],
        ry_direction: &[[f32; N]; 3],
    ) -> ([[f32; N]; 3], [[f32; N]; 3]) {
        let lanes: [(Vec3, Vec3); N] = core::array::from_fn(|i| {
            self.curvature_correction_lane(
                i,
                lane_vec3(dpdx, i),
                lane_vec3(dpdy, i),
                lane_vec3(normal, i),
                curvature[i],
                eta[i],
                lane_vec3(rx_direction, i),
                lane_vec3(ry_direction, i),
            )
        });
        (
            transpose(core::array::from_fn(|i| lanes[i].0)),
            transpose(core::array::from_fn(|i| lanes[i].1)),
        )
    }

    /// Propagate ray differentials through a surface scatter event, all lanes.
    /// Matches the scalar
    /// [`propagate_differentials`](Self::propagate_differentials) lane-by-lane.
    pub fn propagate_differentials_packed(
        &self,
        normal: &[[f32; N]; 3],
        hit_time: &[f32; N],
        eta: &[Option<f32>; N],
        hit_point: &[[f32; N]; 3],
        curvature: &[f32; N],
    ) -> Option<RayDifferentialsPacked<N>> {
        let rd = self.differentials.as_ref()?;

        // Preserve the spatial footprint: offset the new ray origins by the
        // incoming position derivatives (dpdx / dpdy at the hit).
        let dpdx = self.differential_footprint_packed(
            &rd.rx_origin,
            &rd.rx_direction,
            hit_point,
            normal,
            hit_time,
        );
        let dpdy = self.differential_footprint_packed(
            &rd.ry_origin,
            &rd.ry_direction,
            hit_point,
            normal,
            hit_time,
        );

        // Regenerate the ray differentials for the scattered rays.
        // Refraction lanes: curvature_correction returns the full transmitted
        // differential directions. Reflection lanes: reflect the offset
        // directions, then add the correction terms.
        let (cc_rx, cc_ry) = self.curvature_correction_packed(
            &dpdx,
            &dpdy,
            normal,
            curvature,
            eta,
            &rd.rx_direction,
            &rd.ry_direction,
        );
        let lanes: [(Vec3, Vec3); N] = core::array::from_fn(|i| {
            let n = lane_vec3(normal, i);
            let rx = lane_vec3(&rd.rx_direction, i);
            let ry = lane_vec3(&rd.ry_direction, i);
            match eta[i] {
                Some(_) => (lane_vec3(&cc_rx, i), lane_vec3(&cc_ry, i)),
                None => (
                    (rx.reflect(n) + lane_vec3(&cc_rx, i)).normalize(),
                    (ry.reflect(n) + lane_vec3(&cc_ry, i)).normalize(),
                ),
            }
        });

        Some(RayDifferentialsPacked {
            rx_origin: add_packed(hit_point, &dpdx),
            ry_origin: add_packed(hit_point, &dpdy),
            rx_direction: transpose(core::array::from_fn(|i| lanes[i].0)),
            ry_direction: transpose(core::array::from_fn(|i| lanes[i].1)),
        })
    }
}

// ── Gather / scatter ────────────────────────────────────────────────────────
// The stage-boundary glue: pack N scalar rays from the wavefront batch into a
// SIMD group, and unpack a group back into scalar rays.

impl<const N: usize> From<[RayPacked<1>; N]> for RayPacked<N> {
    /// Gathers N scalar rays into a packed group. Differentials are
    /// all-or-nothing: if any lane lacks them, the whole pack is `None`.
    fn from(rays: [RayPacked<1>; N]) -> Self {
        // Pack one vector field: axis-major from per-lane Vec3s.
        let pack = |get: fn(&RayPacked<1>) -> Vec3| {
            core::array::from_fn(|axis| core::array::from_fn(|i| get(&rays[i])[axis]))
        };
        let pack_diff = |get: fn(&RayDifferentialsPacked<1>) -> Vec3| {
            core::array::from_fn(|axis| {
                core::array::from_fn(|i| get(rays[i].differentials.as_ref().unwrap())[axis])
            })
        };

        // The unwrap above is safe: guarded by the all() check below.
        let differentials =
            rays.iter()
                .all(|r| r.differentials.is_some())
                .then(|| RayDifferentialsPacked {
                    rx_origin: pack_diff(|rd| rd.rx_origin().into_inner()),
                    ry_origin: pack_diff(|rd| rd.ry_origin().into_inner()),
                    rx_direction: pack_diff(|rd| rd.rx_direction().into_inner()),
                    ry_direction: pack_diff(|rd| rd.ry_direction().into_inner()),
                });

        Self {
            origin: pack(|r| r.origin().into_inner()),
            direction: pack(|r| r.direction().into_inner()),
            time: core::array::from_fn(|i| rays[i].time()),
            inverse_direction: pack(|r| r.inverse_direction().into_inner()),
            differentials,
        }
    }
}

impl<const N: usize> From<RayPacked<N>> for [RayPacked<1>; N] {
    /// Scatters a packed group back into N scalar rays. The whole-pack
    /// differentials become per-lane `Some`/`None` uniformly.
    fn from(packed: RayPacked<N>) -> Self {
        core::array::from_fn(|i| {
            let differentials = packed.differentials.map(|rd| {
                RayDifferentialsPacked::new(
                    Point3::new(rd.rx_origin[0][i], rd.rx_origin[1][i], rd.rx_origin[2][i]),
                    Point3::new(rd.ry_origin[0][i], rd.ry_origin[1][i], rd.ry_origin[2][i]),
                    Direction3(Vec3::new(
                        rd.rx_direction[0][i],
                        rd.rx_direction[1][i],
                        rd.rx_direction[2][i],
                    )),
                    Direction3(Vec3::new(
                        rd.ry_direction[0][i],
                        rd.ry_direction[1][i],
                        rd.ry_direction[2][i],
                    )),
                )
            });
            RayPacked::new_with_differentials(
                Point3::new(
                    packed.origin[0][i],
                    packed.origin[1][i],
                    packed.origin[2][i],
                ),
                Direction3(Vec3::new(
                    packed.direction[0][i],
                    packed.direction[1][i],
                    packed.direction[2][i],
                )),
                packed.time[i],
                differentials,
            )
        })
    }
}

/// A linearly-interpolated point over time: at(t) = origin + velocity * t
///
/// Used for moving primitives (sphere center interpolation).
/// Unlike Ray, this has no direction validation or inverse_direction garbage.
/// When velocity is zero, the point is stationary (no motion).
///
/// This struct is packed for SIMD processing.
#[derive(Debug, Clone, Copy)]
pub struct ParametricCurvePacked<const N: usize> {
    /// Starting points of the curves (e.g., ray origin).
    pub origin: [[f32; N]; 3],
    /// Velocity vectors of the curves (e.g., ray direction). Can be zero for
    /// stationary curves.
    pub velocity: [[f32; N]; 3],
}

impl<const N: usize> ParametricCurvePacked<N> {
    /// Scalar constructor: broadcasts one curve across all lanes.
    pub fn new(origin: Point3, velocity: Direction3) -> Self {
        let o = origin.into_inner();
        let v = velocity.into_inner();
        Self {
            origin: [[o.x; N], [o.y; N], [o.z; N]],
            velocity: [[v.x; N], [v.y; N], [v.z; N]],
        }
    }

    /// Evaluate lane 0 at time t ∈ [0, 1]
    /// P(t) = O + t * V
    pub fn at(&self, t: f32) -> Point3 {
        self.at_lane(0, t)
    }

    /// Evaluate lane `i` at time `t`.
    fn at_lane(&self, i: usize, t: f32) -> Point3 {
        Point3::new(
            self.origin[0][i] + self.velocity[0][i] * t,
            self.origin[1][i] + self.velocity[1][i] * t,
            self.origin[2][i] + self.velocity[2][i] * t,
        )
    }

    /// Whether lane 0 is moving.
    pub fn is_moving(&self) -> bool {
        lane_vec3(&self.velocity, 0).length_squared() > 0.0
    }

    /// Per-lane: whether each curve is moving.
    pub fn are_moving(&self) -> [bool; N] {
        core::array::from_fn(|i| lane_vec3(&self.velocity, i).length_squared() > 0.0)
    }

    /// Compute the AABBs swept by `bbox` moving along the curves from t0 to t1.
    ///
    /// For stationary curves (velocity = 0), returns the bbox translated to
    /// the origin.
    pub fn sweep_aabb(&self, bbox: &AabbPacked<N>, t0: f32, t1: f32) -> AabbPacked<N> {
        let mut min = [[0.0; N]; 3];
        let mut max = [[0.0; N]; 3];
        for axis in 0..3 {
            for i in 0..N {
                let o0 = self.origin[axis][i] + self.velocity[axis][i] * t0;
                let o1 = self.origin[axis][i] + self.velocity[axis][i] * t1;
                min[axis][i] = (bbox.min[axis][i] + o0).min(bbox.min[axis][i] + o1);
                max[axis][i] = (bbox.max[axis][i] + o0).max(bbox.max[axis][i] + o1);
            }
        }
        AabbPacked::new(min, max)
    }
}

// ── Scalar aliases ──────────────────────────────────────────────────────────
// The N=1 specializations replace the old AoS ray types crate-wide; the
// generic `RayPacked<N>` is what the SIMD kernels operate on.

/// The scalar ray: a single ray, the N=1 specialization of [`RayPacked`].
pub type Ray = RayPacked<1>;

/// The scalar ray differentials.
pub type RayDifferentials = RayDifferentialsPacked<1>;

/// The scalar parametric curve.
pub type ParametricCurve = ParametricCurvePacked<1>;

#[cfg(test)]
mod tests {
    use super::*;

    fn test_ray(i: usize) -> RayPacked<1> {
        let f = i as f32;
        RayPacked::new_with_differentials(
            Point3::new(1.0 + f, 2.0, 3.0),
            Direction3(Vec3::new(0.1 + f * 0.1, 0.2, 0.3).normalize()),
            0.5 + f,
            Some(RayDifferentialsPacked::new(
                Point3::new(1.1 + f, 2.1, 3.1),
                Point3::new(1.2 + f, 2.2, 3.2),
                Direction3(Vec3::new(0.4, 0.5, 0.6).normalize()),
                Direction3(Vec3::new(0.7, 0.8, 0.9).normalize()),
            )),
        )
    }

    /// The packed type must round-trip scalar rays losslessly (bit-exact).
    #[test]
    fn packed_roundtrip_preserves_rays() {
        let rays: [RayPacked<1>; 4] = core::array::from_fn(test_ray);
        let packed: RayPacked<4> = rays.into();
        let unpacked: [RayPacked<1>; 4] = packed.into();
        for (a, b) in rays.iter().zip(unpacked.iter()) {
            assert_eq!(a.origin(), b.origin());
            assert_eq!(a.direction(), b.direction());
            assert_eq!(a.time(), b.time());
            assert_eq!(a.inverse_direction(), b.inverse_direction());
            let (ad, bd) = (a.differentials.unwrap(), b.differentials.unwrap());
            assert_eq!(ad.rx_origin(), bd.rx_origin());
            assert_eq!(ad.ry_origin(), bd.ry_origin());
            assert_eq!(ad.rx_direction(), bd.rx_direction());
            assert_eq!(ad.ry_direction(), bd.ry_direction());
        }
    }

    /// A pack with any differential-less lane must gather to `None`.
    #[test]
    fn packed_gather_drops_differentials_when_any_lane_lacks_them() {
        let rays: [RayPacked<1>; 2] = [
            RayPacked::new_with_differentials(
                Point3::new(0.0, 0.0, 0.0),
                Direction3(Vec3::new(0.0, 0.0, 1.0)),
                0.0,
                Some(RayDifferentialsPacked::new(
                    Point3::new(0.0, 0.0, 0.0),
                    Point3::new(0.0, 0.0, 0.0),
                    Direction3(Vec3::new(0.0, 0.0, 1.0)),
                    Direction3(Vec3::new(0.0, 0.0, 1.0)),
                )),
            ),
            RayPacked::new(
                Point3::new(1.0, 1.0, 1.0),
                Direction3(Vec3::new(0.0, 1.0, 0.0)),
            ),
        ];
        let packed: RayPacked<2> = rays.into();
        assert!(packed.differentials.is_none());
    }

    /// N=1 packed must match the scalar `differential_footprint` exactly,
    /// including the grazing-angle fallback.
    #[test]
    fn footprint_matches_scalar() {
        let ray = RayPacked::new_with_differentials(
            Point3::new(0.0, 0.0, 4.0),
            Direction3(Vec3::new(0.0, 0.0, -1.0)),
            0.0,
            Some(RayDifferentialsPacked::new(
                Point3::new(0.1, 0.0, 4.0),
                Point3::new(0.0, 0.1, 4.0),
                Direction3(Vec3::new(0.0, 0.0, -1.0)),
                Direction3(Vec3::new(0.0, 0.0, -1.0)),
            )),
        );
        let rd = ray.differentials.unwrap();

        // Normal case: hit a plane at z=0 with normal +z.
        let hit_point = Point3::new(0.0, 0.0, 0.0);
        let normal = Direction3(Vec3::new(0.0, 0.0, 1.0));
        let t_hit = 4.0;
        let scalar =
            ray.differential_footprint(rd.rx_origin(), rd.rx_direction(), hit_point, normal, t_hit);
        let packed_out = ray.differential_footprint_packed(
            &rd.rx_origin,
            &rd.rx_direction,
            &[[hit_point.x()], [hit_point.y()], [hit_point.z()]],
            &[[normal.x()], [normal.y()], [normal.z()]],
            &[t_hit],
        );
        assert_eq!(scalar.x(), packed_out[0][0]);
        assert_eq!(scalar.y(), packed_out[1][0]);
        assert_eq!(scalar.z(), packed_out[2][0]);

        // Grazing case: offset ray nearly parallel to the tangent plane.
        let grazing = RayPacked::new_with_differentials(
            Point3::new(0.0, 0.0, 4.0),
            Direction3(Vec3::new(0.0, 0.0, -1.0)),
            0.0,
            Some(RayDifferentialsPacked::new(
                Point3::new(0.0, 0.0, 4.0),
                Point3::new(0.0, 0.0, 4.0),
                Direction3(Vec3::new(1.0, 0.0, 0.0)), // parallel to plane
                Direction3(Vec3::new(0.0, 1.0, 0.0)),
            )),
        );
        let grd = grazing.differentials.unwrap();
        let s = grazing.differential_footprint(
            grd.rx_origin(),
            grd.rx_direction(),
            hit_point,
            normal,
            t_hit,
        );
        let p = grazing.differential_footprint_packed(
            &grd.rx_origin,
            &grd.rx_direction,
            &[[hit_point.x()], [hit_point.y()], [hit_point.z()]],
            &[[normal.x()], [normal.y()], [normal.z()]],
            &[t_hit],
        );
        assert_eq!(s.x(), p[0][0]);
        assert_eq!(s.y(), p[1][0]);
        assert_eq!(s.z(), p[2][0]);
    }

    /// N=1 packed must match the scalar `curvature_correction` for both the
    /// reflection and refraction paths.
    #[test]
    fn curvature_correction_matches_scalar() {
        let ray = RayPacked::new_with_differentials(
            Point3::new(0.0, 0.0, 4.0),
            Direction3(Vec3::new(0.0, 0.0, -1.0)),
            0.0,
            Some(RayDifferentialsPacked::new(
                Point3::new(0.1, 0.0, 4.0),
                Point3::new(0.0, 0.1, 4.0),
                Direction3(Vec3::new(0.05, 0.0, -1.0).normalize()),
                Direction3(Vec3::new(0.0, 0.05, -1.0).normalize()),
            )),
        );
        let rd = ray.differentials.unwrap();

        let dpdx = Direction3(Vec3::new(0.1, 0.0, 0.0));
        let dpdy = Direction3(Vec3::new(0.0, 0.1, 0.0));
        let normal = Direction3(Vec3::new(0.0, 0.0, 1.0));
        let curvature = 0.5;

        for eta in [None, Some(1.5)] {
            let (s_rx, s_ry) = ray.curvature_correction(
                (dpdx, dpdy),
                normal,
                curvature,
                eta,
                (rd.rx_direction(), rd.ry_direction()),
            );
            let (p_rx, p_ry) = ray.curvature_correction_packed(
                &[[dpdx.x()], [dpdx.y()], [dpdx.z()]],
                &[[dpdy.x()], [dpdy.y()], [dpdy.z()]],
                &[[normal.x()], [normal.y()], [normal.z()]],
                &[curvature],
                &[eta],
                &[
                    [rd.rx_direction().x()],
                    [rd.rx_direction().y()],
                    [rd.rx_direction().z()],
                ],
                &[
                    [rd.ry_direction().x()],
                    [rd.ry_direction().y()],
                    [rd.ry_direction().z()],
                ],
            );
            assert_eq!(s_rx.x(), p_rx[0][0]);
            assert_eq!(s_rx.y(), p_rx[1][0]);
            assert_eq!(s_rx.z(), p_rx[2][0]);
            assert_eq!(s_ry.x(), p_ry[0][0]);
            assert_eq!(s_ry.y(), p_ry[1][0]);
            assert_eq!(s_ry.z(), p_ry[2][0]);
        }
    }

    /// N=1 packed must match the scalar `propagate_differentials` for both
    /// reflection and refraction.
    #[test]
    fn propagate_matches_scalar() {
        let ray = RayPacked::new_with_differentials(
            Point3::new(0.0, 0.0, 4.0),
            Direction3(Vec3::new(0.0, 0.0, -1.0)),
            0.0,
            Some(RayDifferentialsPacked::new(
                Point3::new(0.1, 0.0, 4.0),
                Point3::new(0.0, 0.1, 4.0),
                Direction3(Vec3::new(0.05, 0.0, -1.0).normalize()),
                Direction3(Vec3::new(0.0, 0.05, -1.0).normalize()),
            )),
        );

        let normal = Direction3(Vec3::new(0.0, 0.0, 1.0));
        let hit_time = 4.0;
        let hit_point = Point3::new(0.0, 0.0, 0.0);
        let curvature = 0.5;

        for eta in [None, Some(1.5)] {
            let s = ray
                .propagate_differentials(normal, hit_time, eta, hit_point, curvature)
                .unwrap();
            let p = ray
                .propagate_differentials_packed(
                    &[[normal.x()], [normal.y()], [normal.z()]],
                    &[hit_time],
                    &[eta],
                    &[[hit_point.x()], [hit_point.y()], [hit_point.z()]],
                    &[curvature],
                )
                .unwrap();
            assert_eq!(s.rx_origin().x(), p.rx_origin[0][0]);
            assert_eq!(s.rx_origin().y(), p.rx_origin[1][0]);
            assert_eq!(s.rx_origin().z(), p.rx_origin[2][0]);
            assert_eq!(s.ry_origin().x(), p.ry_origin[0][0]);
            assert_eq!(s.ry_origin().y(), p.ry_origin[1][0]);
            assert_eq!(s.ry_origin().z(), p.ry_origin[2][0]);
            assert_eq!(s.rx_direction().x(), p.rx_direction[0][0]);
            assert_eq!(s.rx_direction().y(), p.rx_direction[1][0]);
            assert_eq!(s.rx_direction().z(), p.rx_direction[2][0]);
            assert_eq!(s.ry_direction().x(), p.ry_direction[0][0]);
            assert_eq!(s.ry_direction().y(), p.ry_direction[1][0]);
            assert_eq!(s.ry_direction().z(), p.ry_direction[2][0]);
        }
    }

    /// The packed `sweep_aabb` must match the hand-computed swept box.
    #[test]
    fn sweep_aabb_matches_hand_computed() {
        let curve = ParametricCurvePacked::<1>::new(
            Point3::new(1.0, 2.0, 3.0),
            Direction3(Vec3::new(0.5, 0.0, 0.0)),
        );
        let bbox =
            AabbPacked::<1>::from_corners(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0));
        let out = curve.sweep_aabb(&bbox, 0.0, 1.0);

        // box0 = bbox + origin = [1,2]×[2,3]×[3,4]; box1 = bbox + origin + velocity = [1.5,2.5]×[2,3]×[3,4]
        assert_eq!(out.min[0][0], 1.0);
        assert_eq!(out.min[1][0], 2.0);
        assert_eq!(out.min[2][0], 3.0);
        assert_eq!(out.max[0][0], 2.5);
        assert_eq!(out.max[1][0], 3.0);
        assert_eq!(out.max[2][0], 4.0);
    }
}
