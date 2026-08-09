use std::borrow::Borrow;

use crate::bvh::aabb::Aabb;
use crate::intersect::interaction::{Hit, MaterialHit, SurfaceInteraction};
use crate::intersect::{Bounded, Intersectable};
use crate::light::{LightSample, Sampleable};
use crate::material::Material;
use crate::math::interval::Interval;
use crate::ray::Ray;
use crate::sampling::pdf::AreaPdf;

use crate::math::vec3::{Color3, Direction3, Point3};
use crate::texture::UVDifferentiable;

mod box3d;
mod constructors;
mod planar;
pub(crate) mod regions;
mod sdf;
mod sphere;

pub use box3d::{Box3D, BoxShape, shape_box3d};
pub use constructors::*;
pub use planar::PlanarShape;
pub use sdf::dual::Scalar;
pub use sdf::{
    BoxSdf, CapsuleSdf, CylinderSdf, DynEval, MandelbulbSdf, RoundBoxSdf, SdfExpr, SdfFn,
    SdfRepeat, SdfShape, SphereSdf, TorusSdf,
};
pub use sphere::{SphereShape, moving_sphere, sphere};

/// Defines the 2D region test, UV mapping, area, and sampling for a planar shape.
///
/// Methods take `&self` so that per-instance data (e.g. annulus inner radius) can be stored in the
/// region type.
pub trait Region2D: Send + Sync {
    /// Returns true if the given `(a, b)` coordinates are inside the region.
    ///
    /// The `(a, b)` coordinates are in the parametric space defined by the planar patch's `side_a`
    /// and `side_b` vectors.
    fn contains(&self, a: f32, b: f32) -> bool;

    /// Maps `(a, b)` coordinates to UV space for texture mapping.
    ///
    /// The default implementation maps `(a, b)` directly to `(u, v)`.
    fn uv(&self, a: f32, b: f32) -> (f32, f32) {
        (a, b)
    }

    /// Area of the region in (a,b) parametric space.
    ///
    /// Used to compute the world-space area (= |side_a × side_b| × area) for correct PDF
    /// computation in light importance sampling.
    fn area(&self) -> f32;

    /// Area of the bounding-box that `sample()` draws from.
    ///
    /// For bbox-based samplers (superellipse, rounded rect, etc.) this is larger than `area()` —
    /// the PDF uses this to stay unbiased.  Default: `area()`.
    fn bounding_box_area(&self) -> f32 {
        self.area()
    }

    /// Samples `(a, b)` within the region from `(u, v)` ∈ [0,1]².
    ///
    /// Used for uniform surface sampling of the shape.
    fn sample(&self, u: f32, v: f32) -> (f32, f32);
}

/// Core shape geometry: ray intersection and bounding.
///
/// Every shape must support intersection and bounding; surface sampling (needed for area-light
/// emission) is in the [`ShapeSurfaceSampling`] super-trait. Shapes that cannot provide correct
/// surface-area-uniform sampling (e.g. generic SDFs) implement [`Shape3D`] only and are excluded
/// from area-light use at compile time.
pub trait Shape3D: UVDifferentiable + Send + Sync {
    /// Intersect a ray in local space. Returns a bare [`Hit`] for the caller to wrap in
    /// [`MaterialHit`].
    fn intersect_shape(&self, ray: &Ray, ray_t: Interval) -> Option<Hit>;

    /// Returns true if the ray is occluded by the shape. Default implementation calls
    /// [`intersect_shape`] and returns true if a hit is found.
    fn occluded_shape(&self, ray: &Ray, ray_t: Interval) -> bool {
        self.intersect_shape(ray, ray_t).is_some()
    }

    /// Conservative AABB in local space.
    fn bounding_box(&self) -> Aabb;
}

/// Surface sampling for shapes — required for area-light support.
///
/// Provides surface-area, point sampling, direction sampling, and PDF evaluation for Monte Carlo
/// light-source sampling. Shapes that cannot implement correct surface-area-uniform sampling (e.g.
/// generic SDFs) do not implement this trait and are excluded from emission at compile time.
///
/// # Defaults
/// * `sample_direction` — falls back to `sample` + normalize.
/// * `pdf_direction` — area-to-solid-angle conversion via `intersect_shape`.
/// * `sample_light` — uniform area sampling via `sample` + `area`.
///
/// Override `sample_direction` and `pdf_direction` for solid-angle-uniform sampling (less noise for
/// small shapes like spheres). Override `sample_light` when the shape needs custom emission-aware
/// light sampling (e.g. sphere cone sampling).
pub trait ShapeSurfaceSampling: Shape3D {
    /// Surface area. Used for area-to-solid-angle PDF conversion.
    fn area(&self) -> f32;

    /// Sample a point on the surface, returning `(point, unit_normal)`.
    ///
    /// `u` and `v` are uniformly distributed in `[0, 1)`.
    fn sample(&self, u: f32, v: f32, time: f32) -> (Point3, Direction3);

    /// Sample a direction toward this shape from `origin`.
    ///
    /// Default: uniform area sampling via [`sample()`]. Override for solid-angle-uniform sampling
    /// (less noise for small shapes).
    fn sample_direction(&self, origin: Point3, u: f32, v: f32, time: f32) -> Direction3 {
        let (point, _normal) = self.sample(u, v, time);
        (point - origin).normalize()
    }

    /// Sample a point on the light and return a [`LightSample`] with direction, surface normal,
    /// distance, and area PDF.
    ///
    /// Default: uniform area sampling via [`sample()`]. Override for solid-angle-uniform sampling
    /// (less noise for small shapes).
    fn sample_light(&self, origin: Point3, u: f32, v: f32, time: f32) -> LightSample {
        let (point, normal) = self.sample(u, v, time);
        let offset = point - origin;
        let distance = offset.length();
        let area_pdf = 1.0 / self.area();
        LightSample {
            direction: Direction3(offset.into_inner()),
            normal,
            distance,
            pdf: AreaPdf(area_pdf),
            emission: Color3::ZERO,
        }
    }

    /// PDF of sampling `direction` from `origin` toward this shape.
    ///
    /// Default: area-to-solid-angle conversion via [`intersect_shape`]:
    ///   p(ω) = distance² / (area · |cos θ|)
    /// Override for solid-angle-uniform PDF (e.g. sphere uniform-cone).
    fn pdf_direction(&self, origin: Point3, direction: Direction3, time: f32) -> f32 {
        let ray = Ray::new_with_time(origin, direction, time);
        let ray_t = Interval::from(0.001, f32::INFINITY);
        match self.intersect_shape(&ray, ray_t) {
            Some(hit) => {
                let dist2 = (hit.point - origin).length_squared();
                let cos_theta = hit.geometric_normal().dot(-direction.into_inner()).abs();
                if cos_theta > 0.0 {
                    dist2 / (self.area() * cos_theta)
                } else {
                    0.0
                }
            }
            None => 0.0,
        }
    }
}

/// A material-wrapped 3D shape — combines a [`Shape3D`] with a [`Material`],
///
/// Combines a shape with a [`Material`], deriving `Intersectable`, `Bounded`, and `Sampleable`
/// generically.
///
/// * `Intersectable` and `Bounded` — require only [`Shape3D`]; every shape qualifies.
/// * `Sampleable` — additionally requires [`ShapeSurfaceSampling`]; shapes like
///   generic SDFs that cannot implement correct surface sampling are excluded
///   from area-light use at compile time.
///
/// The caller chooses how to store the material: owned (`Material`), reference-counted
/// (`Arc<Material>`), or borrowed (`&Material`). Default is `Arc<Material>`.
#[derive(Clone)]
pub struct ShapeObject<Sh: Clone, M: Borrow<Material>> {
    shape: Sh,
    material: M,
    bbox: Aabb,
}

impl<Sh: Shape3D + Clone, M: Borrow<Material>> ShapeObject<Sh, M> {
    /// Creates a new `ShapeObject`. The bounding box is computed once.
    pub fn new(shape: Sh, material: M) -> Self {
        let bbox = shape.bounding_box();
        Self {
            shape,
            material,
            bbox,
        }
    }

    pub fn shape(&self) -> &Sh {
        &self.shape
    }

    pub fn material(&self) -> &Material {
        self.material.borrow()
    }
}

impl<Sh: Shape3D + Clone, M: Borrow<Material> + Send + Sync> Bounded for ShapeObject<Sh, M> {
    fn bounding_box(&self) -> Aabb {
        self.bbox
    }
}

impl<Sh: Shape3D + Clone, M: Borrow<Material> + Send + Sync> Intersectable for ShapeObject<Sh, M> {
    fn intersect<'a>(&'a self, ray: &Ray, ray_t: Interval) -> Option<MaterialHit<'a>> {
        let mut hit = self.shape.intersect_shape(ray, ray_t)?;
        // Precompute UV derivatives for texture filtering.
        hit.uv_gradients = Some(self.shape.uv_gradient(&hit.mapping_point));
        Some(MaterialHit {
            hit,
            material: self.material.borrow(),
        })
    }

    fn occluded(&self, ray: &Ray, ray_t: Interval) -> bool {
        self.shape.occluded_shape(ray, ray_t)
    }
}

impl<Sh: ShapeSurfaceSampling + Clone, M: Borrow<Material> + Send + Sync> Sampleable
    for ShapeObject<Sh, M>
{
    fn pdf_value(&self, origin: Point3, direction: Direction3, time: f32) -> f32 {
        self.shape.pdf_direction(origin, direction, time)
    }

    fn random_direction(&self, origin: Point3, u: f32, v: f32, time: f32) -> Direction3 {
        self.shape.sample_direction(origin, u, v, time)
    }

    fn sample_light(&self, origin: Point3, u: f32, v: f32, time: f32) -> LightSample {
        let mut sample = self.shape.sample_light(origin, u, v, time);
        // Compute the light's emission at the sampled point on the surface.
        // Construct a minimal SurfaceInteraction to call material.emitted().
        let point = origin + sample.direction.into_inner();
        // Ensure normal is unit length — shape implementations may return
        // non-unit normals in edge cases (e.g. very close light intersections).
        if sample.normal.length_squared() < 1e-10 {
            sample.normal = Direction3::ZERO;
        } else {
            sample.normal = sample.normal.normalize();
        }
        let light_unit = sample.direction.normalize();
        // Front face: light's normal faces toward the shaded surface.
        let front_face = sample.normal.dot(-light_unit.into_inner()) > 0.0;
        let hit = Hit::new(
            time,
            point,
            point, // mapping point is the same as the hit point for surface sampling.
            sample.normal,
            None,
            None,
        );
        let si = SurfaceInteraction::new(hit, sample.normal, front_face, self.material(), None);
        // Compute the outgoing direction from the light to the shaded surface.
        let wo = -light_unit;
        // Evaluate the light's emission in that direction.
        sample.emission = si.emitted(wo);
        sample
    }
}
