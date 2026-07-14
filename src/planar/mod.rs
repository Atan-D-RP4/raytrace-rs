use std::borrow::Borrow;
use std::sync::Arc;

use crate::aabb::Aabb;
use crate::hittable::Bounded;
use crate::hittable::Hit;
use crate::hittable::Intersectable;
use crate::hittable::MaterialHit;
use crate::hittable::Sampleable;
use crate::hittable::SurfaceInteraction;
use crate::interval::Interval;
use crate::material::Material;
use crate::ray::Ray;
use glam::Vec3;

use crate::texture::UVDifferentiable;
use crate::vec3::Point3;

mod annulus;
mod r#box;
mod ellipse;
mod function;
mod polygon;
mod quad;
mod rounded_rect;
mod superellipse;
mod tri;

pub use annulus::AnnulusRegion;
pub use ellipse::EllipseRegion;
pub use function::FunctionRegion;
pub use polygon::PolygonRegion;
pub use quad::QuadRegion;
pub use r#box::box3d;
pub use rounded_rect::RoundedRectRegion;
pub use superellipse::SuperellipseRegion;
pub use tri::TriRegion;

/// Defines the 2D region test, UV mapping, area, and sampling for a planar shape.
///
/// Methods take `&self` so that per-instance data (e.g. annulus inner radius)
/// can be stored in the region type.
pub trait Region2D: Send + Sync {
    /// Returns true if the given `(a, b)` coordinates are inside the region.
    ///
    /// The `(a, b)` coordinates are in the parametric space defined by the planar patch's `side_a`
    /// and `side_b` vectors.
    /// The specific interpretation of `(a, b)` depends on the shape defined by the region.
    fn contains(&self, a: f32, b: f32) -> bool;

    /// Maps `(a, b)` coordinates to UV space for texture mapping.
    ///
    /// The default implementation maps `(a, b)` directly to `(u, v)`.
    /// The specific mapping can be overridden by region types to achieve different texture
    /// coordinate layouts.
    fn uv(&self, a: f32, b: f32) -> (f32, f32) {
        (a, b)
    }

    /// Area of the region in (a,b) parametric space.
    ///
    /// Used to compute the world-space area (= |side_a × side_b| × area)
    /// for correct PDF computation in light importance sampling.
    fn area(&self) -> f32;

    /// Area of the bounding-box that `sample()` draws from.
    ///
    /// For bbox-based samplers (superellipse, rounded rect, etc.) this is larger
    /// than `area()` — the PDF uses this to stay unbiased.  Default: `area()`.
    fn bounding_box_area(&self) -> f32 {
        self.area()
    }

    /// Samples `(a, b)` within the region from `(u, v)` ∈ [0,1]².
    fn sample(&self, u: f32, v: f32) -> (f32, f32);
}

/// A planar patch with an associated 2D region test, UV mapping, area, and sampling.
///
/// The caller chooses how to store the material: owned (`Material`),
/// reference-counted (`Arc<Material>`), or borrowed (`&Material`).
/// Default is `Arc<Material>`.
#[derive(Clone)]
pub struct PlanarPatch<R: Region2D, M: Borrow<Material> = Arc<Material>> {
    /// The corner point of the planar patch, corresponding to (a, b) = (0, 0).
    corner: Point3,
    /// The side vector corresponding to the (a, b) = (1, 0) direction.
    side_a: Vec3,
    /// The side vector corresponding to the (a, b) = (0, 1) direction.
    side_b: Vec3,

    /// Precomputed reciprocal of the squared length of the normal vector (side_a × side_b).
    w: Vec3,
    /// The material of the planar patch.
    material: M,
    /// The axis-aligned bounding box of the planar patch, used for acceleration structures.
    bbox: Aabb,
    /// The unit normal vector of the planar patch, precomputed for efficiency.
    normal: Vec3,
    /// The plane constant `d` in the plane equation `normal . P = d`, precomputed for efficiency.
    d: f32,
    /// The area of the planar patch in world space, precomputed for efficiency.
    area: f32,
    /// The 2D region that defines the shape of the patch (e.g. quad, ellipse, triangle).
    region: R,
}

#[derive(Clone, Copy)]
pub(crate) struct PlanarHit {
    time: f32,
    point: Point3,
    a: f32,
    b: f32,
}

impl<R: Region2D, M: Borrow<Material>> PlanarPatch<R, M> {
    pub(crate) fn new(corner: Point3, side_a: Vec3, side_b: Vec3, material: M, region: R) -> Self {
        let bbox_diagonal1 = Aabb::from_points(&corner, &(corner + side_a + side_b));
        let bbox_diagonal2 = Aabb::from_points(&(corner + side_a), &(corner + side_b));

        let n = side_a.cross(side_b);
        let normal = n.normalize();

        Self {
            corner,
            side_a,
            side_b,
            w: n / n.dot(n),
            material,
            bbox: bbox_diagonal1.merge(&bbox_diagonal2),
            normal,
            d: normal.dot(corner),
            area: n.length(),
            region,
        }
    }

    /// Returns the intersection of the ray with the plane of the patch, if there is any, alongside
    /// the (a, b) coordinates in parametric space.
    pub(crate) fn hit_plane(&self, ray: &Ray, ray_t: Interval) -> Option<PlanarHit> {
        let denom = self.normal.dot(ray.direction);

        if denom.abs() < 1e-8 {
            return None;
        }

        let t = (self.d - self.normal.dot(ray.origin)) / denom;
        if !ray_t.contains(t) {
            return None;
        }

        let point = ray.at(t);
        let planar_hit_point_vector = point - self.corner;
        let a = self.w.dot(planar_hit_point_vector.cross(self.side_b));
        let b = self.w.dot(self.side_a.cross(planar_hit_point_vector));

        Some(PlanarHit {
            time: t,
            point,
            a,
            b,
        })
    }

    pub(crate) fn material(&self) -> &Material {
        self.material.borrow()
    }
}

impl<R: Region2D, M: Borrow<Material>> UVDifferentiable for PlanarPatch<R, M> {
    /// Returns (∂u/∂P, ∂v/∂P) as 3-vectors. Constant across the patch.
    fn uv_gradient(&self, _mapping_point: &Point3) -> (Vec3, Vec3) {
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
        (du_dp, dv_dp)
    }
}

impl<R: Region2D, M: Borrow<Material> + Send + Sync> Intersectable for PlanarPatch<R, M> {
    fn intersect<'a>(&'a self, ray: &Ray, ray_t: Interval) -> Option<MaterialHit<'a>> {
        let hit = self.hit_plane(ray, ray_t)?;

        if !self.region.contains(hit.a, hit.b) {
            return None;
        }

        // Compute the UV coordinates for texture mapping.
        let uv = self.region.uv(hit.a, hit.b);

        // Precompute UV derivatives for texture filtering.
        let uv_gradients = self.uv_gradient(&hit.point);

        Some(MaterialHit {
            hit: Hit::new(
                hit.time,
                hit.point,
                hit.point,
                self.normal,
                Some(uv),
                Some(uv_gradients),
            ),
            material: self.material(),
        })
    }
}

impl<R: Region2D, M: Borrow<Material> + Send + Sync> Bounded for PlanarPatch<R, M> {
    fn bounding_box(&self) -> Aabb {
        self.bbox
    }
}

impl<R: Region2D, M: Borrow<Material> + Send + Sync> Sampleable for PlanarPatch<R, M> {
    fn pdf_value(&self, origin: Vec3, direction: Vec3, _time: f32) -> f32 {
        // Back-face culling - if the ray is coming from behind the patch, it cannot hit the
        // emitting side, so return 0 PDF.
        // This avoids the mismatch where `.abs()` would return a positive PDF but `emitted()`
        // returns zero for the back face, wasting 50% of light-importance samples.
        if self.normal.dot(direction) >= 0.0 {
            return 0.0;
        }

        // Inline plane intersection — avoids constructing a temporary Ray (3
        // divisions for `inverse_direction` that `hit_plane` never uses).
        let denom = self.normal.dot(direction);
        if denom.abs() < 1e-8 {
            return 0.0;
        }

        let t = (self.d - self.normal.dot(origin)) / denom;
        if t <= 0.001 {
            return 0.0;
        }

        let point = origin + direction * t;
        let planar_hit_point_vector = point - self.corner;
        let a = self.w.dot(planar_hit_point_vector.cross(self.side_b));
        let b = self.w.dot(self.side_a.cross(planar_hit_point_vector));

        if !self.region.contains(a, b) {
            return 0.0;
        }

        let distance_squared = t * t * direction.length_squared();
        // The normal is constant for a planar patch; .abs() gives the Jacobian
        // factor for the area-to-solid-angle measure conversion.
        let cosine = self.normal.dot(-direction.normalize()).abs();
        // Use bounding_box_area() to match the actual sampling distribution:
        // bbox-based samplers (RoundedRect, Superellipse, Polygon, Function)
        // rejection-sample from the bounding box, so the PDF denominator must
        // be the bounding box area, not the true shape area.
        let world_area = self.area * self.region.bounding_box_area();

        distance_squared / (cosine * world_area)
    }

    fn random_direction(&self, origin: Vec3, u: f32, v: f32, _time: f32) -> Vec3 {
        let (a, b) = self.region.sample(u, v);
        let random_point = self.corner + (self.side_a * a) + (self.side_b * b);
        random_point - origin
    }

    fn sample_light(
        &self,
        origin: Vec3,
        u: f32,
        v: f32,
        time: f32,
    ) -> crate::hittable::LightSample {
        let direction = self.random_direction(origin, u, v, time);
        let distance = direction.length();
        // Area PDF: uniform over the bounding box area (matches pdf_value's denominator).
        let world_area = self.area * self.region.bounding_box_area();
        let area_pdf = if world_area > 1e-20 {
            1.0 / world_area
        } else {
            0.0
        };
        // Compute the light's emission at the sampled point.
        // The surface is front-facing when the light normal points toward the shaded surface.
        let front_face = self.normal.dot(-direction) > 0.0;
        let point = origin + direction;
        let hit = Hit::new(time, point, point, self.normal, None, None);
        let si = SurfaceInteraction::new(hit, self.normal, front_face, self.material(), None);
        let wo = -direction.normalize();
        let emission = self.material().emitted(wo, &si);

        crate::hittable::LightSample {
            direction,
            normal: self.normal,
            distance,
            pdf: area_pdf,
            emission,
        }
    }
}

/// A full parallelogram region — the most common shape (walls, light quads, etc.).
pub type Quad<M> = PlanarPatch<QuadRegion, M>;
/// A unit-disk ellipse region.
pub type Ellipse<M> = PlanarPatch<EllipseRegion, M>;
/// A triangular region.
pub type Tri<M> = PlanarPatch<TriRegion, M>;
/// An annular (ring) region with configurable inner radius.
pub type Annulus<M> = PlanarPatch<AnnulusRegion, M>;
/// A rounded rectangle region with configurable corner radius.
pub type RoundedRect<M> = PlanarPatch<RoundedRectRegion, M>;
/// A superellipse region `|a|ⁿ + |b|ⁿ ≤ 1` with configurable exponent.
pub type Superellipse<M> = PlanarPatch<SuperellipseRegion, M>;
/// An arbitrary polygon (N-gon) region.
pub type Polygon<M> = PlanarPatch<PolygonRegion, M>;
/// A region defined by an arbitrary `(a, b) -> bool` predicate.
pub type FunctionPatch<M> = PlanarPatch<FunctionRegion, M>;

/// Construct a parallelogram (quad) from corner `Q` and side vectors `u`, `v`.
///
/// Parameter naming matches *Ray Tracing in One Weekend* (RTIOW) notation.
#[allow(non_snake_case)]
pub fn quad<M: Borrow<Material>>(
    Q: Point3,
    u: Vec3,
    v: Vec3,
    material: M,
) -> PlanarPatch<QuadRegion, M> {
    PlanarPatch::new(Q, u, v, material, QuadRegion)
}

pub fn ellipse<M: Borrow<Material>>(
    center: Point3,
    side_a: Vec3,
    side_b: Vec3,
    material: M,
) -> PlanarPatch<EllipseRegion, M> {
    PlanarPatch::new(center, side_a, side_b, material, EllipseRegion)
}

pub fn tri<M: Borrow<Material>>(
    corner: Point3,
    side_a: Vec3,
    side_b: Vec3,
    material: M,
) -> PlanarPatch<TriRegion, M> {
    PlanarPatch::new(corner, side_a, side_b, material, TriRegion)
}

pub fn annulus<M: Borrow<Material>>(
    center: Point3,
    side_a: Vec3,
    side_b: Vec3,
    inner: f32,
    material: M,
) -> PlanarPatch<AnnulusRegion, M> {
    PlanarPatch::new(center, side_a, side_b, material, AnnulusRegion { inner })
}

pub fn rounded_rect<M: Borrow<Material>>(
    corner: Point3,
    side_a: Vec3,
    side_b: Vec3,
    radius: f32,
    material: M,
) -> PlanarPatch<RoundedRectRegion, M> {
    PlanarPatch::new(
        corner,
        side_a,
        side_b,
        material,
        RoundedRectRegion { radius },
    )
}

pub fn superellipse<M: Borrow<Material>>(
    corner: Point3,
    side_a: Vec3,
    side_b: Vec3,
    n: f32,
    material: M,
) -> PlanarPatch<SuperellipseRegion, M> {
    PlanarPatch::new(corner, side_a, side_b, material, SuperellipseRegion::new(n))
}

pub fn polygon<M: Borrow<Material>>(
    corner: Point3,
    side_a: Vec3,
    side_b: Vec3,
    vertices: Vec<(f32, f32)>,
    material: M,
) -> PlanarPatch<PolygonRegion, M> {
    PlanarPatch::new(
        corner,
        side_a,
        side_b,
        material,
        PolygonRegion::new(vertices),
    )
}

pub fn function_patch<M: Borrow<Material>>(
    corner: Point3,
    side_a: Vec3,
    side_b: Vec3,
    region: FunctionRegion,
    material: M,
) -> PlanarPatch<FunctionRegion, M> {
    PlanarPatch::new(corner, side_a, side_b, material, region)
}
