use std::sync::Arc;

use crate::aabb::Aabb;
use crate::hittable::HitRecord;
use crate::hittable::Hittable;
use crate::interval::Interval;
use crate::material::Material;
use crate::ray::Ray;
use crate::sampler::{DimCursor, Sampler};
use crate::vec3::{Point3, Vec3};

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
pub use r#box::box3d;
pub use ellipse::EllipseRegion;
pub use function::FunctionRegion;
pub use polygon::PolygonRegion;
pub use quad::QuadRegion;
pub use rounded_rect::RoundedRectRegion;
pub use superellipse::SuperellipseRegion;
pub use tri::TriRegion;

#[derive(Clone)]
pub struct PlanarPatch<R: Region2D> {
    corner: Point3,
    side_a: Vec3,
    side_b: Vec3,
    w: Vec3,
    material: Arc<Material>,
    bbox: Aabb,
    normal: Vec3,
    d: f64,
    area: f64,
    pub region: R,
}

#[derive(Clone, Copy)]
pub(crate) struct PlanarHit {
    t: f64,
    point: Point3,
    a: f64,
    b: f64,
}

impl<R: Region2D> PlanarPatch<R> {
    pub(crate) fn new(
        corner: Point3,
        side_a: Vec3,
        side_b: Vec3,
        material: Material,
        region: R,
    ) -> Self {
        let bbox_diagonal1 = Aabb::from_points(&corner, &(corner + side_a + side_b));
        let bbox_diagonal2 = Aabb::from_points(&(corner + side_a), &(corner + side_b));

        let n = side_a.cross(&side_b);
        let normal = n.unit_vector();

        Self {
            corner,
            side_a,
            side_b,
            w: n / n.dot(&n),
            material: Arc::new(material),
            bbox: bbox_diagonal1.merge(bbox_diagonal2),
            normal,
            d: normal.dot(&corner),
            area: n.length(),
            region,
        }
    }

    pub(crate) fn hit_plane(&self, ray: &Ray, ray_t: Interval) -> Option<PlanarHit> {
        let denom = self.normal.dot(&ray.direction);

        if denom.abs() < 1e-8 {
            return None;
        }

        let t = (self.d - self.normal.dot(&ray.origin)) / denom;
        if !ray_t.contains(t) {
            return None;
        }

        let point = ray.at(t);
        let planar_hit_point_vector = point - self.corner;
        let a = self.w.dot(&planar_hit_point_vector.cross(&self.side_b));
        let b = self.w.dot(&self.side_a.cross(&planar_hit_point_vector));

        Some(PlanarHit { t, point, a, b })
    }

    pub(crate) fn material(&self) -> &Material {
        self.material.as_ref()
    }

    pub(crate) fn normal(&self) -> &Vec3 {
        &self.normal
    }

    pub(crate) fn bounding_box(&self) -> Aabb {
        self.bbox
    }

    pub(crate) fn make_hit_record(
        &self,
        ray: &Ray,
        hit: PlanarHit,
        u: f64,
        v: f64,
    ) -> HitRecord<'_> {
        let mut hit_rec = HitRecord::new(hit.t, hit.point, hit.point, Vec3::new(), self.material());
        hit_rec.set_face_normal(ray, self.normal());
        hit_rec.u = u;
        hit_rec.v = v;
        hit_rec
    }
}

/// Defines the 2D region test, UV mapping, area, and sampling for a planar shape.
///
/// Methods take `&self` so that per-instance data (e.g. annulus inner radius)
/// can be stored in the region type.
pub trait Region2D: Send + Sync {
    fn contains(&self, a: f64, b: f64) -> bool;

    fn uv(&self, a: f64, b: f64) -> (f64, f64) {
        (a, b)
    }

    /// Area of the region in (a,b) parametric space.
    ///
    /// Used to compute the world-space area (= |side_a × side_b| × area)
    /// for correct PDF computation in light importance sampling.
    fn area(&self) -> f64;

    /// Area of the bounding-box that `sample()` draws from.
    ///
    /// For bbox-based samplers (superellipse, rounded rect, etc.) this is larger
    /// than `area()` — the PDF uses this to stay unbiased.  Default: `area()`.
    fn bounding_box_area(&self) -> f64 {
        self.area()
    }

    /// Samples `(a, b)` within the region from `(u, v)` ∈ [0,1]².
    fn sample(&self, u: f64, v: f64) -> (f64, f64);
}

impl<R: Region2D> Hittable for PlanarPatch<R> {
    fn hit<'a>(&'a self, ray: &Ray, ray_t: Interval) -> Option<HitRecord<'a>> {
        let hit = self.hit_plane(ray, ray_t)?;
        if !self.region.contains(hit.a, hit.b) {
            return None;
        }

        let (u, v) = self.region.uv(hit.a, hit.b);
        Some(self.make_hit_record(ray, hit, u, v))
    }

    fn bounding_box(&self) -> Aabb {
        self.bbox
    }

    fn pdf_value(&self, origin: Vec3, direction: Vec3) -> f64 {
        // Back-face culling - if the ray is coming from behind the patch, it cannot hit the
        // emitting side, so return 0 PDF.
        // This avoids the mismatch where `.abs()` would return a positive PDF but `emitted()`
        // returns zero for the back face, wasting 50% of light-importance samples.
        if self.normal.dot(&direction) >= 0.0 {
            return 0.0;
        }

        if let Some(hit) = self.hit(
            &Ray::new(origin, direction),
            Interval::from(0.001, f64::INFINITY),
        ) {
            let distance_squared = hit.time * hit.time * direction.length_squared();
            // After set_face_normal, hit.normal always faces the incoming ray (negative dot).
            // .abs() gives the Jacobian factor for the area-to-solid-angle measure conversion.
            let cosine = hit.normal.dot(&direction.unit_vector()).abs();
            let world_area = self.area * self.region.bounding_box_area();

            distance_squared / (cosine * world_area)
        } else {
            0.0
        }
    }

    fn random(
        &self,
        origin: Vec3,
        sampler: &dyn Sampler,
        sample_index: u32,
        dim_offset: &mut DimCursor,
    ) -> Vec3 {
        let u = sampler.sample(sample_index, dim_offset.next_dim());
        let v = sampler.sample(sample_index, dim_offset.next_dim());
        let (a, b) = self.region.sample(u, v);
        let random_point = self.corner + (self.side_a * a) + (self.side_b * b);
        random_point - origin
    }
}

/// A full parallelogram region — the most common shape (walls, light quads, etc.).
pub type Quad = PlanarPatch<QuadRegion>;
/// A unit-disk ellipse region.
pub type Ellipse = PlanarPatch<EllipseRegion>;
/// A triangular region.
pub type Tri = PlanarPatch<TriRegion>;
/// An annular (ring) region with configurable inner radius.
pub type Annulus = PlanarPatch<AnnulusRegion>;
/// A rounded rectangle region with configurable corner radius.
pub type RoundedRect = PlanarPatch<RoundedRectRegion>;
/// A superellipse region `|a|ⁿ + |b|ⁿ ≤ 1` with configurable exponent.
pub type Superellipse = PlanarPatch<SuperellipseRegion>;
/// An arbitrary polygon (N-gon) region.
pub type Polygon = PlanarPatch<PolygonRegion>;
/// A region defined by an arbitrary `(a, b) -> bool` predicate.
pub type FunctionPatch = PlanarPatch<FunctionRegion>;

// Free constructor functions (type aliases don't carry inherent methods).

#[allow(non_snake_case)]
pub fn quad(Q: Point3, u: Vec3, v: Vec3, material: Material) -> Quad {
    PlanarPatch::new(Q, u, v, material, QuadRegion)
}

pub fn ellipse(center: Point3, side_a: Vec3, side_b: Vec3, material: Material) -> Ellipse {
    PlanarPatch::new(center, side_a, side_b, material, EllipseRegion)
}

pub fn tri(corner: Point3, side_a: Vec3, side_b: Vec3, material: Material) -> Tri {
    PlanarPatch::new(corner, side_a, side_b, material, TriRegion)
}

pub fn annulus(
    center: Point3,
    side_a: Vec3,
    side_b: Vec3,
    inner: f64,
    material: Material,
) -> Annulus {
    PlanarPatch::new(center, side_a, side_b, material, AnnulusRegion { inner })
}

pub fn rounded_rect(
    corner: Point3,
    side_a: Vec3,
    side_b: Vec3,
    radius: f64,
    material: Material,
) -> RoundedRect {
    PlanarPatch::new(
        corner,
        side_a,
        side_b,
        material,
        RoundedRectRegion { radius },
    )
}

pub fn superellipse(
    corner: Point3,
    side_a: Vec3,
    side_b: Vec3,
    n: f64,
    material: Material,
) -> Superellipse {
    PlanarPatch::new(corner, side_a, side_b, material, SuperellipseRegion::new(n))
}

pub fn polygon(
    corner: Point3,
    side_a: Vec3,
    side_b: Vec3,
    vertices: Vec<(f64, f64)>,
    material: Material,
) -> Polygon {
    PlanarPatch::new(
        corner,
        side_a,
        side_b,
        material,
        PolygonRegion::new(vertices),
    )
}

pub fn function_patch(
    corner: Point3,
    side_a: Vec3,
    side_b: Vec3,
    region: FunctionRegion,
    material: Material,
) -> FunctionPatch {
    PlanarPatch::new(corner, side_a, side_b, material, region)
}
