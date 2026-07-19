use std::sync::Arc;

use crate::aabb::Aabb;
use crate::interval::Interval;
use crate::material::Material;
use crate::ray::Ray;
use crate::texture::{TextureCoords, TextureDerivatives};
use crate::vec3::{Color3, Direction3, Point3};

/// Represents a ray-object intersection hit, containing geometric information about the
/// intersection point.
///
/// This is a low-level struct focused on geometric details, used as an intermediate
/// representation.
#[derive(Default)]
pub struct Hit {
    /// Ray parameter `t` at the intersection point.
    pub time: f32,
    /// World-space intersection position.
    // TODO(mapping-2d3d): move 3D mapping inputs into a dedicated 3D mapping payload.
    pub point: Point3,
    /// Mapping-space position. For sphere primitives this is the unit-sphere point
    /// (normalized direction from sphere center), stable under rigid transforms.
    /// For planar primitives this is the world-space hit point (same as `point`).
    /// Procedural textures (NoiseTexture) sample from `mapping` via TexturePoints,
    /// so this decouples world-space translation from the texture coordinate frame.
    pub mapping_point: Point3,
    /// Optional UV coordinates for the hit point. `None` for Volume or other primitives that may
    /// not have UVs.
    pub uv: Option<(f32, f32)>,
    // Optional UV gradient for texture filtering. `None` if not computed.
    pub uv_gradients: Option<(Direction3, Direction3)>,

    /// Local surface curvature: how fast the surface normal changes per unit distance
    /// in the tangent plane. For spheres: `1/radius`. For flat surfaces (quads, planes): `0`.
    /// Used by ray differential propagation (Igehy curvature term for curved specular reflection).
    pub curvature: f32,

    /// Outward geometric normal before face-orientation or shading adjustments.
    /// Must be unit length — set_face_normal depends on it.
    geometric_normal: Direction3,
}

impl Hit {
    pub fn new(
        time: f32,
        point: Point3,
        mapping_point: Point3,
        geometric_normal: Direction3,
        uv: Option<(f32, f32)>,
        uv_gradients: Option<(Direction3, Direction3)>,
    ) -> Self {
        debug_assert!(
            {
                // Normals from ray-intersection can have precision loss from the
                // quadratic formula, especially for spheres at FP32 precision
                // when the ray origin is near the surface. Allow 6% tolerance.
                let near_zero = geometric_normal.length_squared() < 1e-8;
                let near_unit = (geometric_normal.length_squared() - 1.0).abs() < 0.06;
                near_zero || near_unit
            },
            "geometric_normal must be near-unit or zero (for volumes): len={} len_sq={}",
            geometric_normal.length(),
            geometric_normal.length_squared()
        );
        Self {
            time,
            point,
            mapping_point,
            uv,
            uv_gradients,
            curvature: 0.0, // default: flat — override in curved shapes (sphere: 1/radius)
            geometric_normal,
        }
    }

    /// Sets the geometric normal (must be unit length, or zero for volumes).
    pub(crate) fn set_geometric_normal(&mut self, n: Direction3) {
        debug_assert!(
            n.length_squared() < 1e-8 || (n.length_squared() - 1.0).abs() < 0.06,
            "geometric_normal must be near-unit or zero (for volumes): len={} len_sq={}",
            n.length(),
            n.length_squared()
        );
        self.geometric_normal = n;
    }

    /// Returns the geometric normal.
    pub fn geometric_normal(&self) -> Direction3 {
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
    /// Geometric hit information, including position, normal, and UV coordinates.
    hit: Hit,
    /// Shading normal at the hit point, which may differ from the geometric normal due to
    /// normal mapping or other shading effects. Must be unit length.
    shading_normal: Direction3,
    /// Indicates whether the ray hit the front face of the surface (true) or the back face (false).
    front_face: bool,
    /// Reference to the material of the intersected surface.
    material: &'si Material,
    /// Optional texture derivatives for texture filtering. `None` if not computed.
    tex_derivatives: Option<TextureDerivatives>,
}

impl<'si> SurfaceInteraction<'si> {
    /// Constructs a new `SurfaceInteraction` from a `Hit`, shading normal, front face flag, and material.
    pub fn new(
        hit: Hit,
        shading_normal: Direction3,
        front_face: bool,
        material: &'si Material,
        tex_derivatives: Option<TextureDerivatives>,
    ) -> Self {
        Self {
            hit,
            shading_normal,
            front_face,
            material,
            tex_derivatives,
        }
    }

    /// Construct from a MaterialHit, resolving front_face and shading_normal.
    #[inline]
    pub fn from_material_hit(mat_hit: MaterialHit<'si>, ray: &Ray) -> Self {
        let geometric_normal = mat_hit.hit.geometric_normal;

        let tex_derivatives = if let Some(gradients) = mat_hit.hit.uv_gradients
            && let Some(rd) = ray.differentials
        {
            let t_hit = mat_hit.hit.time;
            let p = ray.at(t_hit);
            let dpdx = ray.differential_footprint(
                rd.rx_origin,
                rd.rx_direction,
                p,
                geometric_normal,
                t_hit,
            );
            let dpdy = ray.differential_footprint(
                rd.ry_origin,
                rd.ry_direction,
                p,
                geometric_normal,
                t_hit,
            );
            let (du_dp, dv_dp) = gradients;
            Some(TextureDerivatives::from_surface(dpdx, dpdy, du_dp, dv_dp))
        } else {
            None
        };
        let mut si = Self {
            hit: mat_hit.hit,
            shading_normal: geometric_normal,
            front_face: false,
            material: mat_hit.material,
            tex_derivatives,
        };
        si.set_face_normal(ray);

        si
    }

    /// Sets the front_face and shading_normal based on the ray direction and geometric normal.
    pub fn set_face_normal(&mut self, ray: &Ray) {
        self.front_face = ray.direction.dot(self.hit.geometric_normal.into_inner()) < 0.0;
        self.shading_normal = if self.front_face {
            self.hit.geometric_normal
        } else {
            -self.hit.geometric_normal
        };
    }

    /// Returns the world-space intersection point.
    pub fn point(&self) -> Point3 {
        self.hit.point
    }

    /// Returns the shading normal at the intersection point, which may differ from the geometric
    /// normal due to normal mapping or other shading effects.
    pub fn shading_normal(&self) -> Direction3 {
        self.shading_normal
    }

    /// Returns true if the ray hit the front face of the surface, false if it hit the back face.
    pub fn front_face(&self) -> bool {
        self.front_face
    }

    /// Convenience: evaluate emission at this surface point for a given outgoing direction.
    /// Delegates to `Material::emitted()`.
    pub fn emitted(&self, wo: Direction3) -> Color3 {
        self.material.emitted(wo, self)
    }

    /// Returns a reference to the material of the intersected surface.
    pub fn material(&self) -> &'si Material {
        self.material
    }

    /// Returns the UV coordinates of the intersection point, if available.
    pub fn uv(&self) -> Option<(f32, f32)> {
        self.hit.uv
    }

    /// Returns the geometric normal at the intersection point, which is the outward normal before
    /// any shading adjustments.
    pub fn geometric_normal(&self) -> Direction3 {
        self.hit.geometric_normal
    }

    /// Returns the ray parameter `t` at the intersection point.
    pub fn time(&self) -> f32 {
        self.hit.time
    }

    /// Returns the mapping-space position of the intersection point, which is used for procedural
    /// textures.
    pub fn hit(&self) -> &Hit {
        &self.hit
    }

    /// Returns the texture coordinates for this surface interaction, combining the UV coordinates
    /// and the mapping-space position. This is used for texture sampling.
    pub fn texture_coords(&self) -> TextureCoords {
        let (u, v) = self.hit.uv.unwrap_or((0.0, 0.0));
        TextureCoords::new(
            u,
            v,
            self.point(),
            self.hit.mapping_point,
            self.geometric_normal(),
            self.tex_derivatives,
        )
    }
}

/// Represents a ray-object intersection hit along with a reference to the intersected material.
pub struct MaterialHit<'a> {
    /// Geometric hit information, including position, normal, and UV coordinates.
    pub hit: Hit,
    /// Reference to the material of the intersected surface.
    pub material: &'a Material,
}

pub trait Intersectable: Send + Sync + Bounded {
    /// Returns the closest hit inside `[ray_t.min, ray_t.max]`, if any,
    /// along with a reference to the intersected material.
    fn intersect<'a>(&'a self, ray: &Ray, ray_t: Interval) -> Option<MaterialHit<'a>>;

    // Returns bool and short-circuits the moment any primitive/node reports a hit inside the interval,
    // rather than tightening best_t and continuing
    fn occluded(&self, ray: &Ray, ray_t: Interval) -> bool {
        self.intersect(ray, ray_t).is_some()
    }
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
    pub direction: Direction3,
    /// Light's outward surface normal at the sampled point (unit length).
    pub normal: Direction3,
    /// Distance from the surface point to the sampled point on the light.
    pub distance: f32,
    /// Area PDF of this sample (probability density per unit area on the light surface).
    pub pdf: f32,
    /// Emission color of the light at the sampled point (radiance).
    pub emission: Color3,
}

pub trait Sampleable: Intersectable + Send + Sync {
    /// Returns the PDF value for sampling this hittable from a given origin and direction.
    /// Default returns 0.0 (no contribution to the PDF).
    fn pdf_value(&self, origin: Point3, direction: Direction3, time: f32) -> f32 {
        let _ = (origin, direction, time);
        0.0
    }

    /// Samples a random direction toward this hittable from a given origin.
    /// Takes `(u, v)` in `[0, 1)` for sampling. Default returns Vec3::ZERO.
    fn random_direction(&self, origin: Point3, u: f32, v: f32, time: f32) -> Direction3 {
        let _ = (origin, u, v, time);
        Direction3::ZERO
    }

    /// Samples a point on the light and returns everything needed for direct lighting:
    /// direction, surface normal, distance, and area PDF.
    ///
    /// The returned [`LightSample`] is self-consistent — the direction points from
    /// `origin` to the sampled point, the normal is the outward normal at that point,
    /// and the distance equals `direction.length()`.
    fn sample_light(&self, origin: Point3, u: f32, v: f32, time: f32) -> LightSample;
}

impl<T: Sampleable + ?Sized> Sampleable for Arc<T> {
    fn pdf_value(&self, origin: Point3, direction: Direction3, time: f32) -> f32 {
        (**self).pdf_value(origin, direction, time)
    }

    fn random_direction(&self, origin: Point3, u: f32, v: f32, time: f32) -> Direction3 {
        (**self).random_direction(origin, u, v, time)
    }

    fn sample_light(&self, origin: Point3, u: f32, v: f32, time: f32) -> LightSample {
        (**self).sample_light(origin, u, v, time)
    }
}

/// Construct a minimal [`SurfaceInteraction`] for unit tests.
///
/// The geometry is a trivial default (position at origin, zero curvature, no UV
/// gradients). Only `material` and `shading_normal` are meaningful — set them
/// to control what the BSDF code path sees.
#[cfg(test)]
impl<'a> SurfaceInteraction<'a> {
    pub fn test_surface(
        material: &'a Material,
        shading_normal: Direction3,
    ) -> SurfaceInteraction<'_> {
        SurfaceInteraction::new(
            Hit::new(0.0, Point3::ZERO, Point3::ZERO, shading_normal, None, None),
            shading_normal,
            true,
            material,
            None,
        )
    }
}
