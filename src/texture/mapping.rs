//! Coordinate transformations applied before texture evaluation.
//!
//! A mapping converts the raw hit coordinates into the space expected by
//! the texture. For example, [`TextureMapping::Spherical`] converts a
//! unit-sphere position into UV coordinates for image textures.

use std::f64::consts::PI;

use super::TextureCoords;
use crate::vec3::Vec3;

/// Coordinate mappings applied before evaluating an underlying texture.
/// TODO(mapping-2d3d): split this enum into `TextureMapping2D` and
/// `TextureMapping3D` to make UV remaps vs 3D point remaps explicit.
pub enum TextureMapping {
    /// No coordinate change.
    Identity,
    /// Uniform scale in 3D texture space.
    ///
    /// TODO(mapping-2d3d): this currently has only a uniform constructor.
    /// Either add `point_scale_nonuniform(x, y, z)` or simplify `inv_scale`
    /// to `f64` if non-uniform scale stays unused.
    PointScale { inv_scale: Vec3 },
    /// Converts mapping-space unit-sphere position into UVs.
    Spherical,
}

impl TextureMapping {
    /// Builds a uniform point-scale mapping.
    ///
    /// `scale` is cell size; smaller values increase frequency.
    /// TODO(mapping-2d3d): if non-uniform point scale is ever needed, add a
    /// separate constructor instead of widening this uniform-only API.
    pub fn point_scale_uniform(scale: f64) -> Self {
        assert!(scale > 0.0, "texture scale must be positive");
        let inv_scale = 1.0 / scale;

        Self::PointScale {
            inv_scale: Vec3::from(inv_scale, inv_scale, inv_scale),
        }
    }

    /// Applies this mapping to a texture context and returns the mapped copy.
    /// TODO(mapping-2d3d): return distinct mapped outputs for 2D and 3D paths
    /// instead of mutating a single mixed context.
    pub fn map(&self, coords: TextureCoords) -> TextureCoords {
        match self {
            TextureMapping::Identity => coords,
            TextureMapping::PointScale { inv_scale } => {
                coords.with_texture_point(coords.tex_points.texture * *inv_scale)
            }
            TextureMapping::Spherical => {
                // p: point on unit sphere centered at origin (mapping space).
                let p = coords.tex_points.mapping.unit_vector();
                let theta = (-p.y).acos();
                let phi = -p.z.atan2(p.x) + PI;

                let u = phi / (2.0 * PI); // u: angle around +Y axis from X = -1.
                let v = theta / PI; // v: angle from Y = -1 to Y = +1.
                //
                // Examples:
                //  <p, u, v>
                //  <1, 0, 0> -> (0.50, 0.50), <-1, 0, 0> -> (0.00, 0.50)
                //  <0, 1, 0> -> (0.50, 1.00), < 0,-1, 0> -> (0.50, 0.00)
                //  <0, 0, 1> -> (0.25, 0.50), < 0, 0,-1> -> (0.75, 0.50)

                coords.with_uv(u, v)
            }
        }
    }
}
