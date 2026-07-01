use std::sync::Arc;

use crate::aabb::Aabb;
use crate::interval::Interval;
use crate::material::Material;
use crate::ray::Ray;
use crate::vec3::Vec3;

/// Represents a ray-object intersection hit, containing geometric information about the
/// intersection point.
///
/// This is a low-level struct focused on geometric details, used as an intermediate
/// TODO(type-safety): Point3/Vec3/Color3 are aliases today, so these fields can still be mixed up
/// accidentally. Typed newtypes would catch that at compile time.
pub struct Hit {
    /// Ray parameter `t` at the intersection point.
    pub time: f64,
    /// World-space intersection position.
    // TODO(mapping-2d3d): move 3D mapping inputs into a dedicated 3D mapping payload.
    pub point: Vec3,
    /// Mapping-space position. For sphere primitives this is the unit-sphere point
    /// (normalized direction from sphere center), stable under rigid transforms.
    /// For planar primitives this is the world-space hit point (same as `point`).
    /// Procedural textures (NoiseTexture) sample from `mapping` via TexturePoints,
    /// so this decouples world-space translation from the texture coordinate frame.
    pub mapping_point: Vec3,
    /// Outward geometric normal before face-orientation or shading adjustments.
    /// Must be unit length — set_face_normal depends on it.
    geometric_normal: Vec3,
    /// Optional UV coordinates for the hit point. `None` for Volume or other primitives that may not have UVs.
    pub uv: Option<(f64, f64)>,
}

impl Hit {
    pub fn new(
        time: f64,
        point: Vec3,
        mapping_point: Vec3,
        geometric_normal: Vec3,
        uv: Option<(f64, f64)>,
    ) -> Self {
        debug_assert!(
            geometric_normal.near_zero() || (geometric_normal.length_squared() - 1.0).abs() < 1e-6,
            "geometric_normal must be unit length or zero (for volumes)"
        );
        Self {
            time,
            point,
            mapping_point,
            geometric_normal,
            uv,
        }
    }

    /// Sets the geometric normal (must be unit length, or zero for volumes).
    pub(crate) fn set_geometric_normal(&mut self, n: Vec3) {
        debug_assert!(
            n.near_zero() || (n.length_squared() - 1.0).abs() < 1e-6,
            "geometric_normal must be unit length or zero (for volumes)"
        );
        self.geometric_normal = n;
    }

    /// Returns the geometric normal.
    pub fn geometric_normal(&self) -> Vec3 {
        self.geometric_normal
    }
}

/// Represents a surface interaction, containing both geometric and material information about a
/// ray-object intersection.
///
/// This is a higher-level struct that combines geometric details from `Hit` with material
/// information
/// TODO(type-safety): Point3/Vec3/Color3 are aliases today, so these fields can still be mixed up
/// accidentally. Typed newtypes would catch that at compile time.
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
        match self.hit.uv {
            Some((u, _)) => u,
            None => 0.0,
        }
    }
    pub fn v(&self) -> f64 {
        match self.hit.uv {
            Some((_, v)) => v,
            None => 0.0,
        }
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
            self.hit.mapping_point,
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
            .fold(Aabb::new(), |acc, obj| acc.merge(&obj.bounding_box()))
    }
}

impl<T: Bounded + ?Sized> Bounded for Arc<T> {
    fn bounding_box(&self) -> Aabb {
        (**self).bounding_box()
    }
}

/// Result of sampling a direction toward a light source.
///
/// Contains everything needed to evaluate the direct lighting contribution:
/// the (non-normalized) direction from the surface point to the light,
/// the light's surface normal at the sampled point, the distance, and
/// the area PDF of the sample.
pub struct LightSample {
    /// Non-normalized direction from surface point to the sampled point on the light.
    /// `.unit_vector()` gives the unit direction; `.length()` gives the distance.
    pub direction: Vec3,
    /// Light's outward surface normal at the sampled point (unit length).
    pub normal: Vec3,
    /// Distance from the surface point to the sampled point on the light.
    pub distance: f64,
    /// Area PDF of this sample (probability density per unit area on the light surface).
    pub pdf: f64,
    /// Emission color of the light at the sampled point (radiance).
    pub emission: Vec3,
}

pub trait Sampleable: Intersectable + Send + Sync {
    /// Returns the PDF value for sampling this hittable from a given origin and direction.
    /// Default returns 0.0 (no contribution to the PDF).
    fn pdf_value(&self, origin: Vec3, direction: Vec3, time: f64) -> f64 {
        let _ = (origin, direction, time);
        0.0
    }

    /// Samples a random direction toward this hittable from a given origin.
    /// Takes `(u, v)` in `[0, 1)` for sampling. Default returns Vec3::ZERO.
    fn random_direction(&self, origin: Vec3, u: f64, v: f64, time: f64) -> Vec3 {
        let _ = (origin, u, v, time);
        Vec3::ZERO
    }

    /// Samples a point on the light and returns everything needed for direct lighting:
    /// direction, surface normal, distance, and area PDF.
    ///
    /// The returned [`LightSample`] is self-consistent — the direction points from
    /// `origin` to the sampled point, the normal is the outward normal at that point,
    /// and the distance equals `direction.length()`.
    fn sample_light(&self, origin: Vec3, u: f64, v: f64, time: f64) -> LightSample;
}

impl<T: Sampleable + ?Sized> Sampleable for Arc<T> {
    fn pdf_value(&self, origin: Vec3, direction: Vec3, time: f64) -> f64 {
        (**self).pdf_value(origin, direction, time)
    }

    fn random_direction(&self, origin: Vec3, u: f64, v: f64, time: f64) -> Vec3 {
        (**self).random_direction(origin, u, v, time)
    }

    fn sample_light(&self, origin: Vec3, u: f64, v: f64, time: f64) -> LightSample {
        (**self).sample_light(origin, u, v, time)
    }
}
