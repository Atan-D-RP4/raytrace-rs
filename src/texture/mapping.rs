//! Texture coordinate transformations and UV generation.
//!
//! These are applied to the texture-space point and UV coordinates before sampling a texture.
//! They are composable and can be used together to achieve various effects like scaling, tiling,
//! mirroring, etc.
//!
//! The 3D mappings modify the texture-space point used for procedural textures, while the 2D
//! mappings modify the UV coordinates used for image textures.
//!
//! The UV generation converts a 3D point in mapping space to UV coordinates, which is useful for
//! procedural textures that need UVs but the geometry doesn't provide them.

use crate::texture::TextureCoords;
use crate::vec3::{Point3, Vec3};

/// 3D coordinate transform applied to the texture-space point.
///
/// These modify `tex_points.texture` (the mutable 3D coordinate for
/// procedural textures). They do NOT affect UV coordinates.
pub enum TextureMapping3D {
    /// No transformation — use the texture point as-is.
    Identity,
    /// Scale the texture-space point by the inverse of a uniform scale factor.
    ///
    /// Smaller `scale` values → higher frequency (more detail).
    /// This is equivalent to dividing the point by `scale`.
    PointScale { inv_scale: Vec3 },
}

impl TextureMapping3D {
    /// Builds a uniform point-scale mapping.
    ///
    /// `scale` is cell size; smaller values increase frequency.
    pub fn point_scale_uniform(scale: f64) -> Self {
        assert!(scale > 0.0, "texture scale must be positive");
        let inv_scale = 1.0 / scale;

        Self::PointScale {
            inv_scale: Vec3::from(inv_scale, inv_scale, inv_scale),
        }
    }

    /// Applies this mapping to a texture context and returns the mapped copy.
    pub fn map(&self, coords: TextureCoords) -> TextureCoords {
        let mapped_point = self.map_point(coords.tex_points.texture);
        coords.with_texture_point(mapped_point)
    }

    /// Applies this mapping to a 3D point and returns the mapped copy.
    pub fn map_point(&self, point: Vec3) -> Vec3 {
        match self {
            TextureMapping3D::Identity => point,
            TextureMapping3D::PointScale { inv_scale } => point * *inv_scale,
        }
    }
}

/// 2D coordinate transform applied to the texture-space point.
///
/// These modify the `(u, v)` coordinates after UV generation but before
/// the texture is sampled. Useful for adjusting tiling, mirroring, etc.
pub enum TextureMapping2D {
    /// No transformation — use the UV coordinates as-is.
    Identity,
    ScaleUV {
        su: f64,
        sv: f64,
    },
}

impl TextureMapping2D {
    /// Builds a uniform UV scale mapping.
    ///
    /// `scale` is cell size; smaller values increase frequency (more detail).
    pub fn scale_uv_uniform(scale: f64) -> Self {
        assert!(scale > 0.0, "texture scale must be positive");
        let inv_scale = 1.0 / scale;

        Self::ScaleUV {
            su: inv_scale,
            sv: inv_scale,
        }
    }

    /// Applies this mapping to a texture context and returns the mapped copy.
    pub fn map(&self, coords: TextureCoords) -> TextureCoords {
        let (u, v) = self.map_uv(coords.u, coords.v);
        coords.with_uv(u, v)
    }

    /// Applies this mapping to a pair of UV coordinates and returns the mapped copy.
    pub fn map_uv(&self, u: f64, v: f64) -> (f64, f64) {
        match self {
            TextureMapping2D::Identity => (u, v),
            TextureMapping2D::ScaleUV { su, sv } => {
                debug_assert!(*su > 0.0 && *sv > 0.0, "UV scale factors must be positive");
                (u * su, v * sv)
            }
        }
    }
}

/// Converts a 3D point in mapping space to UV coordinates.
///
/// Reads from `tex_points.mapping` (the immutable canonical geometry coordinate)
/// and writes to `(u, v)`. When set to `None`, the geometry-provided UVs are used
pub enum UvGen {
    /// Use the geometry-provided UV coordinates — no UV generation.
    None,

    /// Spherical projection: maps a unit-sphere direction to (u, v).
    ///
    /// Input: a unit vector from sphere center to hit point (mapping space).
    /// Output: u ∈ [0,1) (longitude), v ∈ [0,1] (latitude).
    /// (1,0,0) → (0.50, 0.50),  (-1,0,0) → (0.00, 0.50)
    /// (0,1,0) → (0.50, 1.00),  (0,-1,0) → (0.50, 0.00)
    Spherical,
    // Cylindrical,
}

impl UvGen {
    /// Converts a 3D point in mapping space to UV coordinates according to this UV generation method.
    pub fn map_to_uv(&self, point: Point3) -> Option<(f64, f64)> {
        match self {
            UvGen::None => None,
            UvGen::Spherical => {
                let point = point.unit_vector();
                let theta = (-point.y).acos();
                let phi = (-point.z).atan2(point.x) + std::f64::consts::PI;
                let u = phi / (2.0 * std::f64::consts::PI);
                let v = theta / std::f64::consts::PI;
                Some((u, v))
            }
        }
    }
}
