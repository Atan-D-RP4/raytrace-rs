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

use crate::math::vec3::{Color3, Direction3, Point3};
use crate::texture::gpu::{GPU_TEX_NONE, GpuTextureBuffer, GpuTextureNode, GpuTextureType};
use crate::texture::{Texture, TextureCoords, TextureDerivatives};

trait Packable {
    /// Applies this mapping to a texture context and returns the mapped copy.
    fn pack(&self) -> Vec<f32>;
}

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
    PointScale { inv_scale: f32 },
}

impl TextureMapping3D {
    /// Builds a uniform point-scale mapping.
    ///
    /// `scale` is cell size; smaller values increase frequency.
    pub fn point_scale_uniform(scale: f32) -> Self {
        assert!(scale > 0.0, "texture scale must be positive");
        let inv_scale = 1.0 / scale;

        Self::PointScale { inv_scale }
    }

    /// Applies this mapping to a texture context and returns the mapped copy.
    pub fn map(&self, coords: TextureCoords) -> TextureCoords {
        let mapped_point = self.map_point(coords.tex_points.texture);
        coords.with_texture_point(mapped_point)
    }

    /// Applies this mapping to a 3D point and returns the mapped copy.
    pub fn map_point(&self, point: Point3) -> Point3 {
        match self {
            TextureMapping3D::Identity => point,
            TextureMapping3D::PointScale { inv_scale } => point * *inv_scale,
        }
    }
}

impl Packable for TextureMapping3D {
    fn pack(&self) -> Vec<f32> {
        match self {
            TextureMapping3D::Identity => vec![0.0], // tag for Identity
            TextureMapping3D::PointScale { inv_scale } => vec![1.0, *inv_scale], // tag + inv_scale
        }
    }
}

/// 2D coordinate transform applied to UV coordinates.
///
/// These modify the `(u, v)` coordinates after UV generation but before
/// the texture is sampled. Useful for adjusting tiling, mirroring, etc.
pub enum TextureMapping2D {
    /// No transformation — use the UV coordinates as-is.
    Identity,
    /// Scale the UV coordinates by the given factors.
    ScaleUV {
        /// Scale factor for the U coordinate.
        su: f32,
        /// Scale factor for the V coordinate.
        sv: f32,
    },
}

impl TextureMapping2D {
    /// Builds a uniform UV scale mapping.
    ///
    /// `scale` is cell size; smaller values increase frequency (more detail).
    pub fn scale_uv_uniform(scale: f32) -> Self {
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
    pub fn map_uv(&self, u: f32, v: f32) -> (f32, f32) {
        match self {
            TextureMapping2D::Identity => (u, v),
            TextureMapping2D::ScaleUV { su, sv } => {
                debug_assert!(*su > 0.0 && *sv > 0.0, "UV scale factors must be positive");
                (u * su, v * sv)
            }
        }
    }
}

impl Packable for TextureMapping2D {
    fn pack(&self) -> Vec<f32> {
        match self {
            TextureMapping2D::Identity => vec![0.0], // tag for Identity
            TextureMapping2D::ScaleUV { su, sv } => vec![1.0, *su, *sv], // tag + su + sv
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
    pub fn map_to_uv(&self, point: Point3) -> Option<(f32, f32)> {
        self.map_to_uv_with_gradient(point)
            .map(|((u, v), _)| (u, v))
    }

    /// Like [`map_to_uv`](Self::map_to_uv), but also returns the projection
    /// Jacobian (∂u/∂p, ∂v/∂p), for derivative propagation.
    ///
    /// The Jacobian is with respect to the *raw* mapping point (not the
    /// normalized direction): u is scale-invariant (atan2), and v's
    /// normalization factor is included analytically. Exact when the mapping
    /// point is the raw world-space point; for pre-normalized mapping points
    /// (a sphere's unit direction `(p − c)/r`) the derivatives carry the
    /// shape's scale factor, so keep `uv_gen = None` for shapes whose geometry
    /// UV is already spherical — the geometry path is exact there.
    pub fn map_to_uv_with_gradient(
        &self,
        point: Point3,
    ) -> Option<((f32, f32), (Direction3, Direction3))> {
        match self {
            UvGen::None => None,
            UvGen::Spherical => {
                let len = point.into_inner().length();
                if len < f32::EPSILON {
                    return Some((
                        (0.0, 0.0),
                        (Direction3::new(0., 0., 0.), Direction3::new(0., 0., 0.)),
                    ));
                }
                let dir = point / len;
                let theta = (-dir.y()).acos();
                let phi = (-dir.z()).atan2(dir.x()) + std::f32::consts::PI;
                let u = phi / (2.0 * std::f32::consts::PI);
                let v = theta / std::f32::consts::PI;

                // Jacobian of the spherical projection w.r.t. the raw point p:
                //   ∂u/∂p = (p_z, 0, −p_x) / (2π(p_x² + p_z²))   (atan2 is scale-invariant)
                //   ∂v/∂p = (0, √(p_x² + p_z²) / (π|p|²), 0)
                // Longitude is undefined at the poles (p_x = p_z = 0) — zero it.
                let xz2 = point.x() * point.x() + point.z() * point.z();
                let du_dp = if xz2 < f32::EPSILON {
                    Direction3::new(0., 0., 0.)
                } else {
                    Direction3::new(
                        point.z() / (2.0 * std::f32::consts::PI * xz2),
                        0.0,
                        -point.x() / (2.0 * std::f32::consts::PI * xz2),
                    )
                };
                let dv_dp =
                    Direction3::new(0.0, xz2.sqrt() / (std::f32::consts::PI * len * len), 0.0);
                Some(((u, v), (du_dp, dv_dp)))
            }
        }
    }
}

impl Packable for UvGen {
    fn pack(&self) -> Vec<f32> {
        match self {
            UvGen::None => vec![0.0],      // tag for None
            UvGen::Spherical => vec![1.0], // tag for Spherical
        }
    }
}

/// Compositional wrapper for mapping coordinates first, then evaluating the wrapped texture.
///
/// The mapping pipeline is: 3D mapping → UV generation → 2D mapping → texture evaluation.
pub struct MappedTexture<T: Texture> {
    /// 2D mapping applied to UV coordinates after UV generation.
    mapping2d: TextureMapping2D,
    /// 3D mapping applied to the texture-space point before UV generation.
    mapping3d: TextureMapping3D,
    /// UV generation applied to the texture-space point after 3D mapping.
    uv_gen: UvGen,
    /// The wrapped texture to evaluate after mapping.
    texture: T,
}

impl<T: Texture> MappedTexture<T> {
    /// Creates a texture with identity mapping pipeline (3D identity, no UV gen, 2D identity).
    /// Apply mappings via [`with_mapping3d`](Self::with_mapping3d),
    /// [`with_uv_gen`](Self::with_uv_gen), and [`with_mapping2d`](Self::with_mapping2d).
    pub fn new(texture: T) -> Self {
        Self {
            mapping2d: TextureMapping2D::Identity,
            mapping3d: TextureMapping3D::Identity,
            uv_gen: UvGen::None,
            texture,
        }
    }

    pub fn with_mapping2d(mut self, mapping: TextureMapping2D) -> Self {
        self.mapping2d = mapping;
        self
    }

    pub fn with_mapping3d(mut self, mapping: TextureMapping3D) -> Self {
        self.mapping3d = mapping;
        self
    }

    pub fn with_uv_gen(mut self, uv_gen: UvGen) -> Self {
        self.uv_gen = uv_gen;
        self
    }
}

impl<T: Texture> Texture for MappedTexture<T> {
    fn value(&self, coords: &TextureCoords) -> Color3 {
        // 1. 3D point mapping (transforms the texture-space point).
        let tex_point = self.mapping3d.map_point(coords.tex_points.texture);
        let mut mapped = coords.with_texture_point(tex_point);

        // Propagate the 3D mapping Jacobian: PointScale scales the texture
        // point, so the position derivatives scale by the same factor
        // (matters once procedural textures filter using derivatives).
        if let TextureMapping3D::PointScale { inv_scale } = self.mapping3d {
            let d = mapped.derivatives;
            mapped.derivatives = TextureDerivatives {
                dpdx: d.dpdx * inv_scale,
                dpdy: d.dpdy * inv_scale,
                ..d
            };
        }

        // 2. UV generation. When UVs are regenerated, the geometry-provided
        // derivatives describe the geometry's own UV parameterization, not
        // the generated one — recompute them from the projection Jacobian.
        match self
            .uv_gen
            .map_to_uv_with_gradient(coords.tex_points.mapping)
        {
            Some(((u, v), (du_dp, dv_dp))) => {
                let d = mapped.derivatives;
                mapped = mapped.with_uv(u, v).with_derivatives(TextureDerivatives {
                    dudx: du_dp.into_inner().dot(d.dpdx.into_inner()),
                    dudy: du_dp.into_inner().dot(d.dpdy.into_inner()),
                    dvdx: dv_dp.into_inner().dot(d.dpdx.into_inner()),
                    dvdy: dv_dp.into_inner().dot(d.dpdy.into_inner()),
                    ..d
                });
            }
            None => {
                mapped = mapped.with_uv(coords.u, coords.v);
            }
        }

        // 3. 2D mapping: scale the UVs and their screen-space derivatives by
        // the same factors (the 2D mapping Jacobian).
        if let TextureMapping2D::ScaleUV { su, sv } = self.mapping2d {
            let (u, v) = (mapped.u, mapped.v);
            let d = mapped.derivatives;
            mapped = mapped
                .with_uv(u * su, v * sv)
                .with_derivatives(d.scale_uv(su, sv));
        }

        self.texture.value(&mapped)
    }

    fn serialize_gpu(&self, buf: &mut GpuTextureBuffer) -> u32 {
        let child = self.texture.serialize_gpu(buf);
        let param_offset = buf.params.len() as u32;
        // Fixed wire layout — the shader reads at known offsets:
        // mapping3d: f32 tag (Identity=0 | PointScale=1) + f32 inv_scale
        buf.push_params(self.mapping3d.pack().as_slice());
        // uv_gen:    f32 tag (None=0 | Spherical=1)
        buf.push_params(self.uv_gen.pack().as_slice());
        // mapping2d: f32 tag (Identity=0 | ScaleUV=1) + f32 su + f32 sv
        buf.push_params(self.mapping2d.pack().as_slice());
        buf.nodes.push(GpuTextureNode {
            texture_type: GpuTextureType::Mapped as u32,
            param_offset,
            child_a: child,
            child_b: GPU_TEX_NONE,
            image_index: GPU_TEX_NONE,
            sampler_index: GPU_TEX_NONE,
        });
        buf.nodes.len() as u32 - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// Probe texture that captures the TextureCoords it receives, to observe
    /// derivative propagation through the mapping pipeline.
    struct DerivProbe(Arc<Mutex<Option<TextureCoords>>>);

    impl Texture for DerivProbe {
        fn value(&self, coords: &TextureCoords) -> Color3 {
            *self.0.lock().unwrap() = Some(*coords);
            Color3::new(0.0, 0.0, 0.0)
        }
    }

    /// A MappedTexture wrapping the probe, plus the handle for reading back
    /// the coords the probe received.
    fn probe() -> (MappedTexture<DerivProbe>, Arc<Mutex<Option<TextureCoords>>>) {
        let cell = Arc::new(Mutex::new(None));
        (MappedTexture::new(DerivProbe(cell.clone())), cell)
    }

    fn derivs() -> TextureDerivatives {
        TextureDerivatives::new(
            Direction3::new(1.0, 0.0, 0.0),
            Direction3::new(0.0, 1.0, 0.0),
            0.5,
            0.1,
            0.2,
            0.7,
        )
    }

    /// Mapping point: unit +z (sphere equator, not a pole).
    fn coords_at(mapping: Point3) -> TextureCoords {
        TextureCoords::new(
            0.25,
            0.5,
            Point3::new(1.0, 2.0, 3.0),
            mapping,
            Direction3::new(0.0, 0.0, 1.0),
            Some(derivs()),
        )
    }

    #[test]
    fn identity_pipeline_preserves_derivatives() {
        let coords = coords_at(Point3::new(0.0, 0.0, 1.0));
        let (tex, cell) = probe();
        tex.value(&coords);
        let got = cell.lock().unwrap().expect("probe texture was sampled");
        let d = got.derivatives;
        assert_eq!(d.dpdx.into_inner(), coords.derivatives.dpdx.into_inner());
        assert_eq!(d.dudx, coords.derivatives.dudx);
        assert_eq!(d.dvdy, coords.derivatives.dvdy);
        assert_eq!(got.u, coords.u);
        assert_eq!(got.v, coords.v);
    }

    #[test]
    fn scale_uv_scales_derivatives() {
        // scale_uv_uniform(2) → su = sv = 0.5.
        let coords = coords_at(Point3::new(0.0, 0.0, 1.0));
        let (tex, cell) = probe();
        let tex = tex.with_mapping2d(TextureMapping2D::scale_uv_uniform(2.0));
        tex.value(&coords);
        let got = cell.lock().unwrap().expect("probe texture was sampled");
        let d = got.derivatives;
        assert!((d.dudx - 0.25).abs() < 1e-6); // 0.5 · 0.5
        assert!((d.dudy - 0.05).abs() < 1e-6); // 0.1 · 0.5
        assert!((d.dvdx - 0.10).abs() < 1e-6); // 0.2 · 0.5
        assert!((d.dvdy - 0.35).abs() < 1e-6); // 0.7 · 0.5
        // Position derivatives are untouched by the 2D mapping.
        assert_eq!(d.dpdx.into_inner(), coords.derivatives.dpdx.into_inner());
    }

    #[test]
    fn point_scale_scales_position_derivatives() {
        // point_scale_uniform(2) → inv_scale = 0.5.
        let coords = coords_at(Point3::new(0.0, 0.0, 1.0));
        let (tex, cell) = probe();
        let tex = tex.with_mapping3d(TextureMapping3D::point_scale_uniform(2.0));
        tex.value(&coords);
        let got = cell.lock().unwrap().expect("probe texture was sampled");
        let d = got.derivatives;
        assert!((d.dpdx.x() - 0.5).abs() < 1e-6); // 1.0 · 0.5
        assert!((d.dpdy.y() - 0.5).abs() < 1e-6); // 1.0 · 0.5
        // UV derivatives are untouched by the 3D mapping.
        assert!((d.dudx - 0.5).abs() < 1e-9);
    }

    #[test]
    fn spherical_uv_gen_recomputes_derivatives() {
        // Mapping point (0,0,1): u = (atan2(-1,0) + π)/2π = 1/4, v = acos(0)/π = 1/2.
        // Jacobian: ∂u/∂p = (1,0,0)/(2π), ∂v/∂p = (0,1/π,0) at the unit +z point.
        let coords = coords_at(Point3::new(0.0, 0.0, 1.0));
        let (tex, cell) = probe();
        let tex = tex.with_uv_gen(UvGen::Spherical);
        tex.value(&coords);
        let got = cell.lock().unwrap().expect("probe texture was sampled");
        let d = got.derivatives;
        let inv_2pi = 1.0 / (2.0 * std::f32::consts::PI);
        let inv_pi = 1.0 / std::f32::consts::PI;
        assert!((got.u - 0.25).abs() < 1e-6);
        assert!((got.v - 0.5).abs() < 1e-6);
        assert!((d.dudx - inv_2pi).abs() < 1e-6); // ∂u/∂p · dpdx, dpdx = (1,0,0)
        assert!(d.dudy.abs() < 1e-9); // ∂u/∂p · dpdy = 0
        assert!(d.dvdx.abs() < 1e-9); // ∂v/∂p · dpdx = 0
        assert!((d.dvdy - inv_pi).abs() < 1e-6); // ∂v/∂p · dpdy = 1/π
    }

    #[test]
    fn spherical_uv_gen_pole_guard_yields_zero_gradients() {
        // At the pole (0,1,0), longitude is undefined: u-gradient zeroed, and
        // v is locally constant to first order — no NaN anywhere.
        let coords = coords_at(Point3::new(0.0, 1.0, 0.0));
        let (tex, cell) = probe();
        let tex = tex.with_uv_gen(UvGen::Spherical);
        tex.value(&coords);
        let got = cell.lock().unwrap().expect("probe texture was sampled");
        let d = got.derivatives;
        assert!(d.dudx.is_finite() && d.dudy.is_finite());
        assert!(d.dvdx.is_finite() && d.dvdy.is_finite());
        assert!(d.dudx.abs() < 1e-9 && d.dudy.abs() < 1e-9);
    }
}
