use std::sync::Arc;

use crate::aabb::Aabb;
use crate::interval::Interval;
use crate::material::Material;
use crate::ray::Ray;
use crate::sampler::{DimCursor, Sampler};
use crate::texture::TextureCoords;
use crate::vec3::Vec3;

/// A path-tracer hit payload used to build texture/material evaluation inputs.
///
/// Stores hit time, UVs, world/mapping-space points, normals, and material.
/// Face orientation is resolved by [`HitRecord::set_face_normal`].
///
/// TODO(renderer-agnostic): introduce a `SurfaceInteraction` type and convert
/// from `HitRecord` so rasterizer/GPU/hybrid/SDF backends can share mapping code.
/// TODO(type-safety): Point3/Vec3/Color3 are aliases today, so these fields can
/// still be mixed up accidentally. Typed newtypes would catch that at compile
/// time, but that belongs with the future `SurfaceInteraction` refactor.
pub struct HitRecord<'rec> {
    /// Ray parameter `t` at the intersection point.
    pub time: f64,

    // surface co-ords
    // TODO(mapping-2d3d): move UVs into a dedicated 2D mapping payload.
    /// Surface U coordinate from primitive UV parameterization.
    pub u: f64,
    /// Surface V coordinate from primitive UV parameterization.
    pub v: f64,

    /// World-space intersection position.
    // TODO(mapping-2d3d): move 3D mapping inputs into a dedicated 3D mapping payload.
    pub point: Vec3,
    /// Mapping-space position consumed by texture mappings.
    pub mapping_point: Vec3,
    /// Outward geometric normal before face-orientation or shading adjustments.
    // TODO(displacement): keep pre- and post-displacement geometric normals.
    pub geometry_normal: Vec3,
    /// Shading normal oriented against the incoming ray direction.
    pub normal: Vec3,
    /// Whether the ray hit the outward-facing side of the surface.
    pub front_face: bool,
    /// Material attached to the intersected primitive.
    pub material: &'rec Material,
}

impl<'rec> HitRecord<'rec> {
    /// Creates a hit record with default UVs and unresolved face orientation.
    pub fn new(
        t: f64,
        point: Vec3,
        mapping_point: Vec3,
        normal: Vec3,
        mat: &'rec Material,
    ) -> Self {
        Self {
            time: t,
            u: 0.0,
            v: 0.0,
            point,
            mapping_point,
            geometry_normal: normal,
            normal,
            front_face: false,
            material: mat,
        }
    }

    /// Ensures shading normal opposes the incoming ray direction.
    ///
    /// `geometry_normal` preserves the outward normal from geometry.
    pub fn set_face_normal(&mut self, ray: &Ray, outward_normal: &Vec3) {
        self.geometry_normal = *outward_normal;
        self.front_face = ray.direction.dot(outward_normal) < 0.;
        self.normal = if self.front_face {
            *outward_normal
        } else {
            -(*outward_normal)
        }
    }

    /// Builds the texture-evaluation context derived from this hit.
    pub fn texture_coords(&self) -> TextureCoords {
        // TODO(renderer-agnostic): move this to `impl From<&HitRecord> for SurfaceInteraction`
        // once texture mapping/evaluation is decoupled from path-tracer hit records.
        TextureCoords::new(
            self.u,
            self.v,
            self.point,
            self.mapping_point,
            self.geometry_normal,
        )
    }
}

/// Trait for ray-testable scene primitives with BVH-compatible bounds.
pub trait Hittable<S: Sampler>: Send + Sync {
    /// Returns the closest hit inside `[ray_t.min, ray_t.max]`, if any.
    fn hit<'a>(&'a self, ray: &Ray, ray_t: Interval) -> Option<HitRecord<'a>>;

    /// Returns a conservative world-space AABB for acceleration structures.
    fn bounding_box(&self) -> Aabb;

    /// Returns the PDF value for sampling this hittable from a given origin and direction.
    /// Default returns 0.0 (no contribution to the PDF).
    fn pdf_value(&self, origin: Vec3, direction: Vec3) -> f64 {
        let _ = (origin, direction);
        0.0
    }

    /// Samples a random direction toward this hittable from a given origin.
    /// Default returns (1, 0, 0) as a placeholder.
    fn random(&self, origin: Vec3, dim_offset: &mut DimCursor<S>) -> Vec3 {
        let _ = (origin, dim_offset);
        Vec3::from(1., 0., 0.)
    }
}

/// Blanket impl: Vec of Hittable objects is itself Hittable.
impl<S: Sampler, T: Hittable<S>> Hittable<S> for Vec<T> {
    fn hit<'a>(&'a self, ray: &Ray, ray_t: Interval) -> Option<HitRecord<'a>> {
        let mut closest = ray_t.max;
        let mut result = None;

        for object in self {
            if let Some(record) = object.hit(ray, Interval::from(ray_t.min, closest)) {
                closest = record.time;
                result = Some(record);
            }
        }

        result
    }

    fn bounding_box(&self) -> Aabb {
        self.iter()
            .fold(Aabb::new(), |acc, obj| acc.merge(obj.bounding_box()))
    }

    fn pdf_value(&self, origin: Vec3, direction: Vec3) -> f64 {
        self.iter()
            .map(|obj| obj.pdf_value(origin, direction) * (1.0 / self.len() as f64))
            .fold(0.0, |acc, val| acc + val)
    }

    fn random(&self, origin: Vec3, dim_offset: &mut DimCursor<S>) -> Vec3 {
        if self.is_empty() {
            return Vec3::ZERO;
        }
        let len = self.len();
        let u = dim_offset.next_sample();
        let index = (u * len as f64).min(len as f64 - 1e-15) as usize;
        self[index].random(origin, dim_offset)
    }
}

/// Blanket impl: Arc<T> is Hittable if T is Hittable.
/// Covers Arc<dyn Hittable>, Arc<Quad>, etc.
impl<S: Sampler, T: Hittable<S> + ?Sized> Hittable<S> for Arc<T> {
    fn hit<'a>(&'a self, ray: &Ray, ray_t: Interval) -> Option<HitRecord<'a>> {
        (**self).hit(ray, ray_t)
    }

    fn bounding_box(&self) -> Aabb {
        (**self).bounding_box()
    }

    fn pdf_value(&self, origin: Vec3, direction: Vec3) -> f64 {
        (**self).pdf_value(origin, direction)
    }

    fn random(&self, origin: Vec3, dim_offset: &mut DimCursor<S>) -> Vec3 {
        (**self).random(origin, dim_offset)
    }
}

pub struct Hit {
    /// Ray parameter `t` at the intersection point.
    pub time: f64,
    /// World-space intersection position.
    // TODO(mapping-2d3d): move 3D mapping inputs into a dedicated 3D mapping payload.
    pub point: Vec3,
    /// Outward geometric normal before face-orientation or shading adjustments.
    pub geometric_normal: Vec3,
    /// Optional UV coordinates for the hit point. `None` for Volume or other primitives that may not have UVs.
    pub uv: Option<(f64, f64)>,
}

impl Hit {
    pub fn new(time: f64, point: Vec3, geometric_normal: Vec3, uv: Option<(f64, f64)>) -> Self {
        Self {
            time,
            point,
            geometric_normal,
            uv,
        }
    }
}

pub struct SurfaceInteraction<'si> {
    hit: Hit,
    shading_normal: Vec3,
    front_face: bool,
    material: &'si Material,
}

impl<'si> SurfaceInteraction<'si> {
    pub fn new(hit: Hit, shading_normal: Vec3, front_face: bool, material: &'si Material) -> Self {
        Self {
            hit,
            shading_normal,
            front_face,
            material,
        }
    }

    /// Construct from a MaterialHit, resolving front_face and shading_normal.
    pub fn from_material_hit(mat_hit: MaterialHit<'si>, ray: &Ray) -> Self {
        let geometric_normal = mat_hit.hit.geometric_normal;
        let mut si = Self {
            hit: mat_hit.hit,
            shading_normal: geometric_normal,
            front_face: false,
            material: mat_hit.material,
        };
        si.set_face_normal(ray);
        si
    }

    pub fn set_face_normal(&mut self, ray: &Ray) {
        self.front_face = ray.direction.dot(&self.hit.geometric_normal) < 0.0;
        self.shading_normal = if self.front_face {
            self.hit.geometric_normal
        } else {
            -self.hit.geometric_normal
        };
    }

    pub fn point(&self) -> Vec3 {
        self.hit.point
    }
    pub fn shading_normal(&self) -> Vec3 {
        self.shading_normal
    }
    pub fn front_face(&self) -> bool {
        self.front_face
    }
    pub fn material(&self) -> &'si Material {
        self.material
    }
    pub fn uv(&self) -> Option<(f64, f64)> {
        self.hit.uv
    }
    pub fn u(&self) -> f64 {
        self.hit.uv.map(|(u, _)| u).unwrap_or(0.0)
    }
    pub fn v(&self) -> f64 {
        self.hit.uv.map(|(_, v)| v).unwrap_or(0.0)
    }
    pub fn geometric_normal(&self) -> Vec3 {
        self.hit.geometric_normal
    }
    pub fn time(&self) -> f64 {
        self.hit.time
    }
    pub fn hit(&self) -> &Hit {
        &self.hit
    }

    pub fn texture_coords(&self) -> crate::texture::TextureCoords {
        crate::texture::TextureCoords::new(
            self.u(),
            self.v(),
            self.point(),
            self.point(), // mapping_point eliminated; point is used as fallback
            self.geometric_normal(),
        )
    }
}

pub struct MaterialHit<'a> {
    pub hit: Hit,
    pub material: &'a Material,
}

pub trait Intersectable: Send + Sync + Bounded {
    /// Returns the closest hit inside `[ray_t.min, ray_t.max]`, if any,
    /// along with a reference to the intersected material.
    fn intersect<'a>(&'a self, ray: &Ray, ray_t: Interval) -> Option<MaterialHit<'a>>;
}

impl<T: Intersectable> Intersectable for Vec<T> {
    fn intersect<'a>(&'a self, ray: &Ray, ray_t: Interval) -> Option<MaterialHit<'a>> {
        let mut closest = ray_t.max;
        let mut result = None;

        for object in self {
            if let Some(mat_hit) = object.intersect(ray, Interval::from(ray_t.min, closest)) {
                closest = mat_hit.hit.time;
                result = Some(mat_hit);
            }
        }

        result
    }
}

impl<T: Intersectable + ?Sized> Intersectable for Arc<T> {
    fn intersect<'a>(&'a self, ray: &Ray, ray_t: Interval) -> Option<MaterialHit<'a>> {
        (**self).intersect(ray, ray_t)
    }
}

pub trait Bounded: Send + Sync {
    /// Returns a conservative world-space AABB for acceleration structures.
    fn bounding_box(&self) -> Aabb;
}

impl<T: Bounded> Bounded for Vec<T> {
    fn bounding_box(&self) -> Aabb {
        self.iter()
            .fold(Aabb::new(), |acc, obj| acc.merge(obj.bounding_box()))
    }
}

impl<T: Bounded + ?Sized> Bounded for Arc<T> {
    fn bounding_box(&self) -> Aabb {
        (**self).bounding_box()
    }
}

pub trait Sampleable<S: Sampler>: Intersectable {
    /// Returns the PDF value for sampling this hittable from a given origin and direction.
    /// Default returns 0.0 (no contribution to the PDF).
    fn pdf_value(&self, origin: Vec3, direction: Vec3) -> f64 {
        let _ = (origin, direction);
        0.0
    }

    /// Samples a random direction toward this hittable from a given origin.
    /// Default returns (1, 0, 0) as a placeholder.
    fn random(&self, origin: Vec3, dim_offset: &mut DimCursor<S>) -> Vec3 {
        let _ = (origin, dim_offset);
        Vec3::from(1., 0., 0.)
    }
}

impl<S: Sampler, T: Sampleable<S>> Sampleable<S> for Vec<T> {
    fn pdf_value(&self, origin: Vec3, direction: Vec3) -> f64 {
        self.iter()
            .map(|obj| obj.pdf_value(origin, direction) * (1.0 / self.len() as f64))
            .fold(0.0, |acc, val| acc + val)
    }

    fn random(&self, origin: Vec3, dim_offset: &mut DimCursor<S>) -> Vec3 {
        if self.is_empty() {
            return Vec3::ZERO;
        }
        let len = self.len();
        let u = dim_offset.next_sample();
        let index = (u * len as f64).min(len as f64 - 1e-15) as usize;
        self[index].random(origin, dim_offset)
    }
}

impl<S: Sampler, T: Sampleable<S> + ?Sized> Sampleable<S> for Arc<T> {
    fn pdf_value(&self, origin: Vec3, direction: Vec3) -> f64 {
        (**self).pdf_value(origin, direction)
    }

    fn random(&self, origin: Vec3, dim_offset: &mut DimCursor<S>) -> Vec3 {
        (**self).random(origin, dim_offset)
    }
}
