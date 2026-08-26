use crate::bvh::mesh::TriangleIndex;
use crate::material::Material;
use crate::math::vec3::{Color3, Direction3, Point3};
use crate::ray::Ray;
use crate::texture::{TextureCoords, TextureDerivatives};

/// Represents a ray-object intersection hit, containing geometric information about the
/// intersection point.
///
/// This is a low-level struct focused on geometric details, used as an intermediate
/// representation.
#[derive(Default, Copy, Clone)]
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
    /// not have UVs. For surfaces with UVs, this is typically in the range [0, 1] for both u and v.
    pub uv: Option<(f32, f32)>,
    /// Optional UV gradient for texture filtering. `None` if not computed. `du_dp` and `dv_dp` are
    /// the partial derivatives of the UV coordinates with respect to the world-space position. Used
    /// for texture filtering (e.g., MIP mapping).
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
    pub fn from_material_hit(mat_hit: &MaterialHit<'si>, ray: &Ray) -> Self {
        let geometric_normal = mat_hit.hit.geometric_normal;

        let tex_derivatives = if let Some(gradients) = mat_hit.hit.uv_gradients
            && let Some(rd) = ray.differentials
        {
            let t_hit = mat_hit.hit.time;
            let p = ray.at(t_hit);
            let dpdx = ray.differential_footprint(
                rd.rx_origin(),
                rd.rx_direction(),
                p,
                geometric_normal,
                t_hit,
            );
            let dpdy = ray.differential_footprint(
                rd.ry_origin(),
                rd.ry_direction(),
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
        self.front_face = ray.direction().dot(self.hit.geometric_normal.into_inner()) < 0.0;
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
#[derive(Copy, Clone)]
pub struct MaterialHit<'a> {
    /// Geometric hit information, including position, normal, and UV coordinates.
    pub hit: Hit,
    /// Reference to the material of the intersected surface.
    pub material: &'a Material,
}

/// A mesh intersection record: triangle identity + geometric hit.
///
/// [`Hit`] carries the geometry; [`TriangleIndex`] carries the mesh-specific
/// identity. No material — material stays outside the geometry BVH so one mesh
/// can be shared by instances with different materials.
#[derive(Copy, Clone)]
pub struct MeshHit {
    pub tri_index: TriangleIndex,
    pub hit: Hit,
}
