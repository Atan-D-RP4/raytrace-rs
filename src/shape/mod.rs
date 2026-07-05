use std::borrow::Borrow;
use std::sync::Arc;

use crate::aabb::Aabb;
use crate::hittable::{
    Bounded, Hit, Intersectable, LightSample, MaterialHit, Sampleable, SurfaceInteraction,
};
use crate::interval::Interval;
use crate::material::Material;
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

mod sphere;

pub use sphere::{SphereShape, moving_sphere, sphere};

/// 3D shape interface — the 3D analogue of [`Region2D`].
///
/// Implementations provide pure-geometry intersection, bounding, area, and
/// surface sampling in local space. Material assignment and trait derivation
/// are handled by [`ShapeObject`].
///
/// [Region2D]: crate::planar::Region2D
pub trait Shape3D: Send + Sync {
    /// Intersect a ray in local space. Returns a bare [`Hit`] for the
    /// caller to wrap in [`MaterialHit`].
    fn intersect_shape(&self, ray: &Ray, ray_t: Interval) -> Option<Hit>;

    /// Conservative AABB in local space.
    fn bounding_box(&self) -> Aabb;

    /// Surface area. Used for area-to-solid-angle PDF conversion.
    fn area(&self) -> f64;

    /// Sample a point on the surface, returning `(point, unit_normal)`.
    ///
    /// `u` and `v` are uniformly distributed in `[0, 1)`.
    fn sample(&self, u: f64, v: f64, time: f64) -> (Point3, Vec3);

    /// Sample a direction toward this shape from `origin`.
    ///
    /// Default fallback: uniform area sampling via [`sample()`]. Non-uniform
    /// direction PDF for most shapes — override with solid-angle-uniform
    /// sampling (less noise for small shapes like spheres).
    fn sample_direction(&self, origin: Vec3, u: f64, v: f64, time: f64) -> Vec3 {
        let (point, _normal) = self.sample(u, v, time);
        (point - origin).unit_vector()
    }

    /// Sample a point on the light and return a [`LightSample`] with direction,
    /// surface normal, distance, and area PDF.
    fn sample_light(&self, origin: Vec3, u: f64, v: f64, time: f64) -> LightSample {
        let (point, normal) = self.sample(u, v, time);
        let offset = point - origin;
        let distance = offset.length();
        // Area PDF: p_A(q) = 1 / area for uniform area sampling.
        let area_pdf = 1.0 / self.area();
        LightSample {
            direction: offset,
            normal,
            distance,
            pdf: area_pdf,
            emission: Vec3::ZERO,
        }
    }

    /// PDF of sampling `direction` from `origin` toward this shape.
    ///
    /// Default fallback: area-to-solid-angle conversion via [`intersect_shape`]:
    ///   p(ω) = distance² / (area · |cos θ|)
    /// Only accurate for uniform area sampling — override for solid-angle-uniform
    /// PDF (e.g. sphere uniform-cone).
    fn pdf_direction(&self, origin: Vec3, direction: Vec3, time: f64) -> f64 {
        let ray = Ray::new_with_time(origin, direction, time);
        let ray_t = Interval::from(0.001, f64::INFINITY);
        match self.intersect_shape(&ray, ray_t) {
            Some(hit) => {
                let dist2 = (hit.point - origin).length_squared();
                let cos_theta = hit.geometric_normal().dot(&(-direction)).abs();
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

/// A material-wrapped 3D shape — the 3D analogue of [`PlanarPatch`].
///
/// Combines a [`Shape3D`] with a [`Material`], deriving `Intersectable`,
/// `Bounded`, and `Sampleable` generically. Adding a new shape only
/// requires implementing [`Shape3D`].
///
/// The caller chooses how to store the material: owned (`Material`),
/// reference-counted (`Arc<Material>`), or borrowed (`&Material`).
/// Default is `Arc<Material>`.
///
/// [PlanarPatch]: crate::planar::PlanarPatch
pub struct ShapeObject<Sh: Shape3D, M: Borrow<Material> = Arc<Material>> {
    shape: Sh,
    material: M,
    bbox: Aabb,
}

impl<Sh: Shape3D, M: Borrow<Material>> ShapeObject<Sh, M> {
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

impl<Sh: Shape3D, M: Borrow<Material> + Send + Sync> Bounded for ShapeObject<Sh, M> {
    fn bounding_box(&self) -> Aabb {
        self.bbox
    }
}

impl<Sh: Shape3D, M: Borrow<Material> + Send + Sync> Intersectable for ShapeObject<Sh, M> {
    fn intersect<'a>(&'a self, ray: &Ray, ray_t: Interval) -> Option<MaterialHit<'a>> {
        let hit = self.shape.intersect_shape(ray, ray_t)?;
        Some(MaterialHit {
            hit,
            material: self.material.borrow(),
        })
    }
}

impl<Sh: Shape3D, M: Borrow<Material> + Send + Sync> Sampleable for ShapeObject<Sh, M> {
    fn pdf_value(&self, origin: Vec3, direction: Vec3, time: f64) -> f64 {
        self.shape.pdf_direction(origin, direction, time)
    }

    fn random_direction(&self, origin: Vec3, u: f64, v: f64, time: f64) -> Vec3 {
        self.shape.sample_direction(origin, u, v, time)
    }

    fn sample_light(&self, origin: Vec3, u: f64, v: f64, time: f64) -> LightSample {
        let mut sample = self.shape.sample_light(origin, u, v, time);
        // Compute the light's emission at the sampled point on the surface.
        // Construct a minimal SurfaceInteraction to call material.emitted().
        let point = origin + sample.direction;
        let light_unit = sample.direction.unit_vector();
        // Front face: light normal faces toward the shaded surface.
        let front_face = sample.normal.dot(&(-light_unit)) > 0.0;
        let hit = Hit::new(time, point, point, sample.normal, None);
        let mut si = SurfaceInteraction::new(hit, sample.normal, front_face, self.material());
        // Re-compute emission with the correct outgoing direction (from light toward
        // shaded surface). The constructor uses Vec3::ZERO which skips coating attenuation
        // in Coated::emitted().
        let wo = -light_unit;
        si.set_emission(si.material().emitted(wo, &si));
        sample.emission = si.emission();
        sample
    }
}
