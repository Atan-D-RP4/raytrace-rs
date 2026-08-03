//! Concrete texture implementations.
//!
//! Each type implements the [`Texture`] trait and evaluates a color from
//! the coordinate context. These are the leaf nodes that materials sample
//! during shading.

use std::sync::Arc;

use crate::math::perlin::Perlin;
use crate::math::vec3::Color3;
use crate::texture::mapping::{MappedTexture, TextureMapping3D};
use crate::texture::{Texture, TextureCoords};

/// Uniform color texture — returns the same [`Color3`] at every point.
///
/// The constant case of a material parameter: `DiffuseReflector::new`
/// wraps a plain color in a `SolidColor`. Because it is UV-independent it
/// bakes directly into GPU material parameters, unlike sampled textures.
pub struct SolidColor {
    albedo: Color3,
}

impl SolidColor {
    /// Construct from a `Color3` value.
    pub fn new(albedo: Color3) -> Self {
        Self { albedo }
    }

    /// Construct RGB components.
    pub fn from_rgb(r: f32, g: f32, b: f32) -> Self {
        Self {
            albedo: Color3::new(r, g, b),
        }
    }
}

impl Texture for SolidColor {
    fn value(&self, _coords: &TextureCoords) -> Color3 {
        self.albedo
    }

    fn as_constant(&self) -> Option<Color3> {
        Some(self.albedo)
    }
}

/// Alternates between two child textures using integer parity in texture space.
pub struct CheckerTexture {
    even: Arc<dyn Texture>,
    odd: Arc<dyn Texture>,
}

impl CheckerTexture {
    /// Creates a checker from two arbitrary child textures.
    pub fn new(even: Arc<dyn Texture>, odd: Arc<dyn Texture>) -> Self {
        Self { even, odd }
    }

    /// Convenience checker from two solid colors.
    pub fn from_color(c1: Color3, c2: Color3) -> Self {
        Self {
            even: Arc::new(SolidColor::new(c1)),
            odd: Arc::new(SolidColor::new(c2)),
        }
    }

    /// Creates a checkerboard texture with a 3D scale mapping.
    /// This is the most common pattern: a two-color checker scaled in world space.
    pub fn with_scale(scale: f32, even: Color3, odd: Color3) -> Arc<dyn Texture> {
        let mapped_tex = MappedTexture::new(CheckerTexture::from_color(even, odd));
        let mapped_tex = mapped_tex.with_mapping3d(TextureMapping3D::point_scale_uniform(scale));
        Arc::new(mapped_tex)
    }
}

impl Texture for CheckerTexture {
    fn value(&self, coords: &TextureCoords) -> Color3 {
        let x = coords.tex_points.texture.x().floor() as i32;
        let y = coords.tex_points.texture.y().floor() as i32;
        let z = coords.tex_points.texture.z().floor() as i32;

        if (x + y + z) % 2 == 0 {
            self.even.value(coords)
        } else {
            self.odd.value(coords)
        }
    }
}

/// Procedural Perlin-noise texture source.
pub struct NoiseTexture {
    noise: Perlin,
}

impl Default for NoiseTexture {
    fn default() -> Self {
        Self::new()
    }
}

impl NoiseTexture {
    /// Creates a new noise texture with random Perlin permutation tables.
    pub fn new() -> Self {
        Self {
            noise: Perlin::new(),
        }
    }

    /// Creates a Perlin noise texture with a world-space scale.
    pub fn with_scale(scale: f32) -> Arc<dyn Texture> {
        Arc::new(
            MappedTexture::new(NoiseTexture::new())
                .with_mapping3d(TextureMapping3D::point_scale_uniform(scale)),
        )
    }
}

impl Texture for NoiseTexture {
    /// Marbled Perlin texture: combines turbulence with a sinusoidal warp
    /// for a natural stone-like appearance.
    ///
    /// Other variants (for reference):
    /// - Smooth: `Color3::new(1., 1., 1.) * 0.5 * (1.0 + self.noise.noise(&point))`
    /// - Turbulent: `Color3::new(1., 1., 1.) * self.noise.turbulence(&point, 7)`
    fn value(&self, coords: &TextureCoords) -> Color3 {
        let point = coords.tex_points.texture;
        Color3::splat(0.5) * (1.0 + (point.z() + (10.0 * self.noise.turbulence(point, 7))).sin())
    }
}

// ─── Mapping-comparison textures ───────────────────────────────────────────

/// 2D checkerboard driven by UV coordinates.
///
/// Unlike [`CheckerTexture`] (which uses the 3D texture-space point), this
/// evaluates a standard latitude/longitude checker from the `(u, v)` in
/// [`TextureCoords`].  Useful for comparing UV-mapped vs 3D-mapped procedural
/// patterns on curved surfaces like spheres.
pub struct UvCheckerTexture {
    scale: f32,
    even: Color3,
    odd: Color3,
}

impl UvCheckerTexture {
    pub fn new(scale: f32, even: Color3, odd: Color3) -> Self {
        Self { scale, even, odd }
    }
}

impl Texture for UvCheckerTexture {
    fn value(&self, coords: &TextureCoords) -> Color3 {
        let u_cell = (coords.u * self.scale).floor() as i32;
        let v_cell = (coords.v * self.scale).floor() as i32;
        if (u_cell + v_cell) % 2 == 0 {
            self.even
        } else {
            self.odd
        }
    }
}

/// Triplanar projection wrapper.
///
/// Projects the inner texture from three orthogonal planes (XY, XZ, YZ)
/// and blends the results using the surface normal as weights.  This
/// eliminates pole/distortion artifacts from UV-based mappings on spheres
/// and other curved geometry.
///
/// Part of the three generic mapping wrappers alongside
/// [`WorldSpaceMapping`] and [`SphericalUvMapping`].
///
/// `sharpness` controls the blend transition: higher values produce sharper
/// boundaries between projections (typical: 4.0–8.0).
pub struct TriplanarMapping<T: Texture> {
    inner: T,
    sharpness: f32,
    scale: f32,
}

impl<T: Texture> TriplanarMapping<T> {
    pub fn new(inner: T, sharpness: f32, scale: f32) -> Self {
        Self {
            inner,
            sharpness,
            scale,
        }
    }
}

impl<T: Texture> Texture for TriplanarMapping<T> {
    fn value(&self, coords: &TextureCoords) -> Color3 {
        use crate::math::vec3::Point3;
        use crate::texture::TextureCoords;

        let inv_scale = 1.0 / self.scale;
        let point = coords.tex_points.texture * inv_scale;
        let n = coords.geometry_normal.into_inner();

        // Blend weights from the absolute normal components, raised to sharpness.
        let n = n.abs().powf(self.sharpness);
        let total = n.element_sum();
        let inv_total = if total > 1e-10 {
            1.0 / total
        } else {
            1.0 / 3.0
        };

        // Sample the inner texture projected onto each of the three axis-aligned planes.
        // zero_axis: 0 = zero X (YZ plane), 1 = zero Y (XZ plane), 2 = zero Z (XY plane).
        let sample_plane = |zero_axis: usize| -> Color3 {
            let pp = match zero_axis {
                0 => Point3::new(0.0, point.y(), point.z()),
                1 => Point3::new(point.x(), 0.0, point.z()),
                _ => Point3::new(point.x(), point.y(), 0.0),
            };
            let mapped = TextureCoords::new(
                coords.u,
                coords.v,
                coords.tex_points.world,
                coords.tex_points.mapping,
                coords.geometry_normal,
                Some(coords.derivatives),
            )
            .with_texture_point(pp);
            self.inner.value(&mapped)
        };

        let c_xy = sample_plane(2); // XY plane — zero Z
        let c_xz = sample_plane(1); // XZ plane — zero Y
        let c_yz = sample_plane(0); // YZ plane — zero X

        // Blend the three plane samples using the normalized weights from the surface normal.
        n.to_array()
            .iter()
            .zip([c_yz, c_xz, c_xy].iter())
            .map(|(&weight, &color)| color * weight * inv_total)
            .sum()
    }
}

/// 3D world-space projection wrapper.
///
/// Scales the3D texture-space point by `1/scale` before passing it to the
/// inner texture.  This is the standard "cell size" convention: smaller
/// `scale` values produce higher-frequency patterns.
///
/// Use this for procedural textures that evaluate at a 3D point (checker,
/// noise, marble, etc.) when you want world-space-aligned patterns.
pub struct WorldSpaceMapping<T: Texture> {
    inner: T,
    inv_scale: f32,
}

impl<T: Texture> WorldSpaceMapping<T> {
    /// `scale` is the desired cell/feature size in world units.
    pub fn new(inner: T, scale: f32) -> Self {
        assert!(scale > 0.0, "scale must be positive");
        Self {
            inner,
            inv_scale: 1.0 / scale,
        }
    }
}

impl<T: Texture> Texture for WorldSpaceMapping<T> {
    fn value(&self, coords: &TextureCoords) -> Color3 {
        let mapped = coords.with_texture_point(coords.tex_points.texture * self.inv_scale);
        self.inner.value(&mapped)
    }
}

/// Spherical UV projection wrapper.
///
/// Converts the 3D mapping-space point (unit-sphere direction for spheres)
/// into latitude/longitude `(u, v)` coordinates and passes them to the
/// inner texture.  Works with any texture that reads from UV coordinates
/// (image textures, [`UvCheckerTexture`], etc.).
///
/// For 3D procedural textures (checker, noise) that read from
/// `tex_points.texture`, use [`WorldSpaceMapping`] instead — passing them
/// through this wrapper would ignore the generated UVs.
pub struct SphericalUvMapping<T: Texture> {
    inner: T,
}

impl<T: Texture> SphericalUvMapping<T> {
    pub fn new(inner: T) -> Self {
        Self { inner }
    }
}

impl<T: Texture> Texture for SphericalUvMapping<T> {
    fn value(&self, coords: &TextureCoords) -> Color3 {
        use std::f32::consts::PI;

        let p = coords.tex_points.mapping.normalize();
        let theta = (-p.y()).acos();
        let phi = (-p.z()).atan2(p.x()) + PI;
        let u = phi / (2.0 * PI);
        let v = theta / PI;

        let mapped = coords.with_uv(u, v);
        self.inner.value(&mapped)
    }
}
