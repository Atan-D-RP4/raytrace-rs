use std::sync::Arc;

use crate::aabb::Aabb;
use crate::interval::Interval;
use crate::material::Material;
use crate::ray::Ray;
use crate::texture::TextureCoords;
use crate::vec3::{Vec3, dot};

/// A path-tracer hit payload used to build texture/material evaluation inputs.
///
/// Stores hit time, UVs, world/mapping-space points, normals, and material.
/// Face orientation is resolved by [`HitRecord::set_face_normal`].
///
/// TODO(renderer-agnostic): introduce a `SurfaceInteraction` type and convert
/// from `HitRecord` so rasterizer/GPU/hybrid/SDF backends can share mapping code.
pub struct HitRecord<'rec> {
    /// Ray parameter `t` at the intersection point.
    pub time: f64,

    // surface co-ords
    /// Surface U coordinate from primitive UV parameterization.
    // TODO(mapping-2d3d): move UVs into a dedicated 2D mapping payload.
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
        self.front_face = dot(&ray.direction, outward_normal) < 0.;
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
pub trait Hittable: Send + Sync {
    /// Returns the closest hit inside `[ray_t.min, ray_t.max]`, if any.
    fn hit<'a>(&'a self, ray: &Ray, ray_t: Interval) -> Option<HitRecord<'a>>;

    /// Returns a conservative world-space AABB for acceleration structures.
    fn bounding_box(&self) -> Aabb;
}

// /// Blanket impl: any slice of Hittable objects is itself Hittable.
// impl<T: Hittable> Hittable for &[T] {
//     fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord> {
//         let mut closest = ray_t.max;
//         let mut result = None;
//
//         for object in *self {
//             if let Some(record) = object.hit(ray, Interval::from(ray_t.min, closest)) {
//                 closest = record.time;
//                 result = Some(record);
//             }
//         }
//
//         result
//     }
//
//     fn bounding_box(&self) -> Aabb {
//         self.iter()
//             .fold(Aabb::new(), |acc, obj| acc.merge(obj.bounding_box()))
//     }
// }
//
// /// Blanket impl: any mutable slice of Hittable objects is itself Hittable.
// impl<T: Hittable> Hittable for &mut [T] {
//     fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord> {
//         let mut closest = ray_t.max;
//         let mut result = None;
//
//         for object in self.iter() {
//             if let Some(record) = object.hit(ray, Interval::from(ray_t.min, closest)) {
//                 closest = record.time;
//                 result = Some(record);
//             }
//         }
//
//         result
//     }
//
//     fn bounding_box(&self) -> Aabb {
//         self.iter()
//             .fold(Aabb::new(), |acc, obj| acc.merge(obj.bounding_box()))
//     }
// }

/// Blanket impl: Vec of Hittable objects is itself Hittable.
impl<T: Hittable> Hittable for Vec<T> {
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
}

/// Blanket impl: Arc<T> is Hittable if T is Hittable.
/// Covers Arc<dyn Hittable>, Arc<Quad>, etc.
impl<T: Hittable + ?Sized> Hittable for Arc<T> {
    fn hit<'a>(&'a self, ray: &Ray, ray_t: Interval) -> Option<HitRecord<'a>> {
        (**self).hit(ray, ray_t)
    }

    fn bounding_box(&self) -> Aabb {
        (**self).bounding_box()
    }
}
