use glam::Vec3;

use crate::bvh::aabb::Aabb;
use crate::intersect::Bounded;
use crate::intersect::interaction::Hit;
use crate::light::LightSample;
use crate::math::interval::Interval;
use crate::math::vec3::{Color3, Direction3, Point3};
use crate::ray::RayPacked;
use crate::sampling::pdf::AreaPdf;
use crate::shape::regions::{
    AnnulusRegion, EllipseRegion, FunctionRegion, PolygonRegion, QuadRegion, RoundedRectRegion,
    SuperellipseRegion, TriRegion,
};
use crate::shape::{Region2D, Shape3D, ShapeSurfaceSampling};
use crate::texture::UVDifferentiable;

/// A planar shape that lives in a 3D plane, defined by a parallelogram (corner + two side vectors)
/// and a `Region2D` that carves a 2D shape out of that parallelogram.
///
/// The geometry is pure (no material) — wrap via `ShapeObject<PlanarShape, M>` to make it
/// intersectable with a material. This replaces the old `PlanarPatch<R, M>` which combined
/// geometry and material in one type.
#[derive(Clone)]
pub struct PlanarShape {
    /// The corner point of the planar patch, corresponding to (a, b) = (0, 0).
    corner: Point3,
    /// The side vector corresponding to the (a, b) = (1, 0) direction.
    side_a: Vec3,
    /// The side vector corresponding to the (a, b) = (0, 1) direction.
    side_b: Vec3,

    /// Precomputed reciprocal of the squared length of the normal vector (side_a × side_b).
    w: Vec3,
    /// The axis-aligned bounding box of the planar patch, used for acceleration structures.
    bbox: Aabb,
    /// The unit normal vector of the planar patch, precomputed for efficiency.
    normal: Vec3,
    /// The plane constant `d` in the plane equation `normal . P = d`, precomputed for efficiency.
    d: f32,
    /// The area of the planar patch in world space, precomputed for efficiency.
    area: f32,
    /// The 2D region that defines the shape of the patch (e.g. quad, ellipse, triangle).
    region: Region,
}

/// Unifies the 8 region kinds into a single type, resolved with static dispatch.
///
/// `PlanarShape` stores one of these; the `Region2D` impl below routes every
/// method call to the inner variant's `Region2D` impl, so the shape stays
/// non-generic while each region kind keeps its specialized behavior.
#[derive(Clone)]
enum Region {
    Quad(QuadRegion),
    Ellipse(EllipseRegion),
    Tri(TriRegion),
    Annulus(AnnulusRegion),
    RoundedRect(RoundedRectRegion),
    Superellipse(SuperellipseRegion),
    Polygon(PolygonRegion),
    Function(FunctionRegion),
}

impl Region2D for Region {
    fn contains(&self, a: f32, b: f32) -> bool {
        match self {
            Region::Quad(r) => r.contains(a, b),
            Region::Ellipse(r) => r.contains(a, b),
            Region::Tri(r) => r.contains(a, b),
            Region::Annulus(r) => r.contains(a, b),
            Region::RoundedRect(r) => r.contains(a, b),
            Region::Superellipse(r) => r.contains(a, b),
            Region::Polygon(r) => r.contains(a, b),
            Region::Function(r) => r.contains(a, b),
        }
    }

    fn uv(&self, a: f32, b: f32) -> (f32, f32) {
        match self {
            Region::Quad(r) => r.uv(a, b),
            Region::Ellipse(r) => r.uv(a, b),
            Region::Tri(r) => r.uv(a, b),
            Region::Annulus(r) => r.uv(a, b),
            Region::RoundedRect(r) => r.uv(a, b),
            Region::Superellipse(r) => r.uv(a, b),
            Region::Polygon(r) => r.uv(a, b),
            Region::Function(r) => r.uv(a, b),
        }
    }

    fn area(&self) -> f32 {
        match self {
            Region::Quad(r) => r.area(),
            Region::Ellipse(r) => r.area(),
            Region::Tri(r) => r.area(),
            Region::Annulus(r) => r.area(),
            Region::RoundedRect(r) => r.area(),
            Region::Superellipse(r) => r.area(),
            Region::Polygon(r) => r.area(),
            Region::Function(r) => r.area(),
        }
    }

    fn bounding_box_area(&self) -> f32 {
        match self {
            Region::Quad(r) => r.bounding_box_area(),
            Region::Ellipse(r) => r.bounding_box_area(),
            Region::Tri(r) => r.bounding_box_area(),
            Region::Annulus(r) => r.bounding_box_area(),
            Region::RoundedRect(r) => r.bounding_box_area(),
            Region::Superellipse(r) => r.bounding_box_area(),
            Region::Polygon(r) => r.bounding_box_area(),
            Region::Function(r) => r.bounding_box_area(),
        }
    }

    fn sample(&self, u: f32, v: f32) -> (f32, f32) {
        match self {
            Region::Quad(r) => r.sample(u, v),
            Region::Ellipse(r) => r.sample(u, v),
            Region::Tri(r) => r.sample(u, v),
            Region::Annulus(r) => r.sample(u, v),
            Region::RoundedRect(r) => r.sample(u, v),
            Region::Superellipse(r) => r.sample(u, v),
            Region::Polygon(r) => r.sample(u, v),
            Region::Function(r) => r.sample(u, v),
        }
    }
}

/// A hit record for a ray-plane intersection, including the parametric (a, b) coordinates
#[derive(Clone, Copy)]
struct PlanarHit {
    /// The time `t` along the ray where the intersection occurs.
    time: f32,
    /// The intersection point in world space.
    point: Point3,
    /// The parametric coordinate `a` in the planar patch's local space.
    a: f32,
    /// The parametric coordinate `b` in the planar patch's local space.
    b: f32,
}

impl PlanarShape {
    /// Shared geometry setup: computes the precomputed plane fields from the
    /// parallelogram (corner + two side vectors) and the region kind.
    fn from_region(corner: Point3, side_a: Vec3, side_b: Vec3, region: Region) -> Self {
        // Compute the AABB from the two diagonals of the parallelogram.
        // This is tighter than computing the min/max of all four corners separately.
        let bbox_diagonal1 = Aabb::from_corners(corner, corner + side_a + side_b);
        let bbox_diagonal2 = Aabb::from_corners(corner + side_a, corner + side_b);

        // Cross product gives the unnormalized plane normal (= 2× parallelogram area).
        let n = side_a.cross(side_b);
        let normal = n.normalize();

        Self {
            corner,
            side_a,
            side_b,
            // w = n / |n|²: projects a world-space point onto the (a, b) basis
            // via the formula a = (offset × side_b) · w, b = (side_a × offset) · w.
            w: n / n.dot(n),
            bbox: bbox_diagonal1.merge(&bbox_diagonal2),
            normal,
            // Plane constant d = n · P (for any point P on the plane).
            d: normal.dot(corner.into_inner()),
            // World-space area of the parallelogram (not the region shape).
            area: n.length(),
            region,
        }
    }

    /// Construct a full parallelogram (quad) region from corner `Q` and side vectors `u`, `v`.
    ///
    /// Parameter naming matches *Ray Tracing in One Weekend* (RTIOW) notation.
    pub fn quad(corner: Point3, side_a: Vec3, side_b: Vec3) -> Self {
        Self::from_region(corner, side_a, side_b, Region::Quad(QuadRegion))
    }

    /// Construct a triangle region (a ≥ 0, b ≥ 0, a + b ≤ 1) from corner and side vectors.
    pub fn tri(corner: Point3, side_a: Vec3, side_b: Vec3) -> Self {
        Self::from_region(corner, side_a, side_b, Region::Tri(TriRegion))
    }

    /// Construct an ellipse region (unit disk in parametric space) from center and side vectors.
    pub fn ellipse(corner: Point3, side_a: Vec3, side_b: Vec3) -> Self {
        Self::from_region(corner, side_a, side_b, Region::Ellipse(EllipseRegion))
    }

    /// Construct an annulus (ring) region with configurable inner radius.
    pub fn annulus(corner: Point3, side_a: Vec3, side_b: Vec3, inner: f32) -> Self {
        Self::from_region(
            corner,
            side_a,
            side_b,
            Region::Annulus(AnnulusRegion { inner }),
        )
    }

    /// Construct a rounded rectangle region with configurable corner radius.
    pub fn rounded_rect(corner: Point3, side_a: Vec3, side_b: Vec3, radius: f32) -> Self {
        Self::from_region(
            corner,
            side_a,
            side_b,
            Region::RoundedRect(RoundedRectRegion { radius }),
        )
    }

    /// Construct a superellipse region `|a|ⁿ + |b|ⁿ ≤ 1` with configurable exponent `n`.
    pub fn superellipse(corner: Point3, side_a: Vec3, side_b: Vec3, n: f32) -> Self {
        Self::from_region(
            corner,
            side_a,
            side_b,
            Region::Superellipse(SuperellipseRegion::new(n)),
        )
    }

    /// Construct an arbitrary N-gon polygon region from a list of `(a, b)` vertices.
    pub fn polygon(corner: Point3, side_a: Vec3, side_b: Vec3, vertices: Vec<(f32, f32)>) -> Self {
        Self::from_region(
            corner,
            side_a,
            side_b,
            Region::Polygon(PolygonRegion::new(vertices)),
        )
    }

    /// Construct a boolean-predicate function region from a `FunctionRegion`.
    ///
    /// The function defines `contains(a, b)` — any (a, b) that satisfies the predicate
    /// is inside the shape. Useful for analytical or procedural shapes that don't fit
    /// into a fixed parametric form.
    pub fn function(corner: Point3, side_a: Vec3, side_b: Vec3, region: FunctionRegion) -> Self {
        Self::from_region(corner, side_a, side_b, Region::Function(region))
    }

    /// SIMD packet kernel: per-lane plane intersection + interval test.
    ///
    /// For each lane computes the plane parameter `t = (d − n·o)/(n·dir)`, rejects
    /// rays nearly parallel to the plane (`|n·dir| < 1e-8`) and requires
    /// `min ≤ t ≤ max` (inclusive — matching the scalar `Interval::contains_value`).
    /// Returns the per-lane `t` and the hit mask; lanes that miss are marked false
    /// with an unspecified `t`.
    fn plane_hits<const N: usize>(
        &self,
        ray: &RayPacked<N>,
        ray_t: Interval<N>,
    ) -> ([f32; N], [bool; N]) {
        use std::simd::prelude::*;

        let nx = Simd::splat(self.normal.x);
        let ny = Simd::splat(self.normal.y);
        let nz = Simd::splat(self.normal.z);
        let ox = Simd::from_array(ray.origin[0]);
        let oy = Simd::from_array(ray.origin[1]);
        let oz = Simd::from_array(ray.origin[2]);
        let dx = Simd::from_array(ray.direction[0]);
        let dy = Simd::from_array(ray.direction[1]);
        let dz = Simd::from_array(ray.direction[2]);
        let tmin = Simd::from_array(ray_t.min());
        let tmax = Simd::from_array(ray_t.max());

        let denom = nx * dx + ny * dy + nz * dz;
        let parallel = denom.abs().simd_lt(Simd::splat(1e-8));
        let t = (Simd::splat(self.d) - (nx * ox + ny * oy + nz * oz)) / denom;
        let in_interval = t.simd_ge(tmin) & t.simd_le(tmax);
        let hit = !parallel & in_interval;
        (t.to_array(), hit.to_array())
    }

    /// Returns the intersection of the ray with the plane of the patch, if there is any, alongside
    /// the (a, b) coordinates in parametric space.
    fn hit_plane<const N: usize>(
        &self,
        ray: &RayPacked<N>,
        ray_t: Interval<N>,
    ) -> Option<PlanarHit> {
        let denom = self.normal.dot(ray.direction().into_inner());

        if denom.abs() < 1e-8 {
            return None;
        }

        let t = (self.d - self.normal.dot(ray.origin().into_inner())) / denom;
        if !ray_t.contains_value(t) {
            return None;
        }

        let point = ray.at(t);
        let planar_hit_point_vector = point - self.corner;
        let a = self
            .w
            .dot(planar_hit_point_vector.cross(self.side_b).into_inner());
        let b = self
            .w
            .dot(self.side_a.cross(planar_hit_point_vector.into_inner()));

        Some(PlanarHit {
            time: t,
            point,
            a,
            b,
        })
    }
}

impl UVDifferentiable for PlanarShape {
    /// Returns (∂u/∂P, ∂v/∂P) as 3-vectors. Constant across the patch.
    fn uv_gradient(&self, _mapping_point: &Point3) -> (Direction3, Direction3) {
        let a = self.side_a;
        let b = self.side_b;
        let aa = a.dot(a);
        let bb = b.dot(b);
        let ab = a.dot(b);
        let det = (aa * bb - ab * ab).max(1e-12);

        // du_dp = (bb·side_a − ab·side_b) / det
        let du_dp = (b * bb - a * ab) / det;
        // dv_dp = (−ab·side_a + aa·side_b) / det
        let dv_dp = (a * -ab + b * aa) / det;
        (Direction3(du_dp), Direction3(dv_dp))
    }
}

impl Shape3D for PlanarShape {
    fn intersect_shape<const N: usize>(
        &self,
        ray: &RayPacked<N>,
        ray_t: Interval<N>,
    ) -> Option<Hit> {
        // Scalar reference path (lane 0) — kept as the exact-behavior baseline that the
        // packet kernel below is verified against.
        let planar_hit = self.hit_plane(ray, ray_t)?;

        if !self.region.contains(planar_hit.a, planar_hit.b) {
            return None;
        }

        let uv = self.region.uv(planar_hit.a, planar_hit.b);

        Some(Hit::new(
            planar_hit.time,
            planar_hit.point,
            planar_hit.point,
            Direction3(self.normal),
            Some(uv),
            None,
        ))
    }

    fn intersect_shape_packed<const N: usize>(
        &self,
        ray: &RayPacked<N>,
        ray_t: Interval<N>,
    ) -> [Option<Hit>; N] {
        let (ts, hit) = self.plane_hits(ray, ray_t);
        let points: [Point3; N] = ray.at_packed(ts);
        core::array::from_fn(|i| {
            if !hit[i] {
                return None;
            }
            let point = points[i];
            let offset = point - self.corner;
            let a = self.w.dot(offset.cross(self.side_b).into_inner());
            let b = self.w.dot(self.side_a.cross(offset.into_inner()));
            if !self.region.contains(a, b) {
                return None;
            }
            let uv = self.region.uv(a, b);
            Some(Hit::new(
                ts[i],
                point,
                point,
                Direction3(self.normal),
                Some(uv),
                None,
            ))
        })
    }

    fn occluded_shape<const N: usize>(&self, ray: &RayPacked<N>, ray_t: Interval<N>) -> bool {
        // Boolean-only: plane test + region containment, no UV computation.
        let (ts, hit) = self.plane_hits(ray, ray_t);
        if !hit[0] {
            return false;
        }
        let point = ray.at(ts[0]);
        let offset = point - self.corner;
        let a = self.w.dot(offset.cross(self.side_b).into_inner());
        let b = self.w.dot(self.side_a.cross(offset.into_inner()));
        self.region.contains(a, b)
    }

    fn occluded_shape_packed<const N: usize>(
        &self,
        ray: &RayPacked<N>,
        ray_t: Interval<N>,
    ) -> [bool; N] {
        let (ts, hit) = self.plane_hits(ray, ray_t);
        let points: [Point3; N] = ray.at_packed(ts);
        core::array::from_fn(|i| {
            if !hit[i] {
                return false;
            }
            let offset = points[i] - self.corner;
            let a = self.w.dot(offset.cross(self.side_b).into_inner());
            let b = self.w.dot(self.side_a.cross(offset.into_inner()));
            self.region.contains(a, b)
        })
    }
}

impl Bounded for PlanarShape {
    fn bounding_box(&self) -> Aabb {
        self.bbox
    }
}

impl ShapeSurfaceSampling for PlanarShape {
    fn area(&self) -> f32 {
        self.area * self.region.area()
    }

    fn sample(&self, u: f32, v: f32, _time: f32) -> (Point3, Direction3) {
        // Sample (a, b) in parametric space via the region's sampling distribution,
        // then map to world-space point on the parallelogram.
        let (a, b) = self.region.sample(u, v);
        let point = self.corner + self.side_a * a + self.side_b * b;
        (point, Direction3(self.normal))
    }

    fn sample_direction(&self, origin: Point3, u: f32, v: f32, time: f32) -> Direction3 {
        let (point, _) = self.sample(u, v, time);
        (point - origin).normalize()
    }

    fn sample_light(&self, origin: Point3, u: f32, v: f32, _time: f32) -> LightSample {
        let (point, normal) = self.sample(u, v, 0.0);
        let direction = point - origin;
        let distance = direction.length();
        // Area PDF: uniform over the bounding box area (matches pdf_direction's denominator).
        // No back-face culling here — ShapeObject::sample_light handles emission gating.
        let world_area = self.area * self.region.bounding_box_area();
        let area_pdf = if world_area > 1e-20 {
            world_area.recip()
        } else {
            0.0
        };
        LightSample {
            direction,
            normal,
            distance,
            pdf: AreaPdf(area_pdf),
            emission: Color3::ZERO,
        }
    }

    fn pdf_direction(&self, origin: Point3, direction: Direction3, _time: f32) -> f32 {
        // Inline the plane-intersection + containment test from intersect_shape
        // to avoid constructing a full Ray (with differentials) on a per-light-sample
        // path that only needs a PDF value.
        let dir = direction.into_inner();
        let denom = self.normal.dot(dir);

        // Back-face culling: if the ray direction is parallel to the plane (denom ≈ 0)
        // or points the same way as the normal (surface is rear-facing for emission),
        // return 0 PDF.
        if denom.abs() < 1e-8 {
            return 0.0;
        }

        let t = (self.d - self.normal.dot(origin.into_inner())) / denom;
        if t < 0.001 || !t.is_finite() {
            return 0.0;
        }

        let point = origin.into_inner() + dir * t;
        let offset = Point3(point) - self.corner;
        let a = self.w.dot(offset.cross(self.side_b).into_inner());
        let b = self.w.dot(self.side_a.cross(offset.into_inner()));

        if !self.region.contains(a, b) {
            return 0.0;
        }

        // Area-to-solid-angle PDF conversion: p(ω) = d² / (|cos_θ| · A).
        // cos_θ accounts for the foreshortening of the light surface; d² for inverse-square.
        let distance_squared = (point - origin.into_inner()).length_squared();
        let cos_theta = self.normal.dot(-dir).abs();
        let world_area = self.area * self.region.bounding_box_area();
        if cos_theta > 0.0 {
            distance_squared / (cos_theta * world_area)
        } else {
            0.0
        }
    }
}
