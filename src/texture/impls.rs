//! Concrete texture implementations.
//!
//! Each type implements the [`Texture`] trait and evaluates a color from
//! the coordinate context. These are the leaf nodes that materials sample
//! during shading.

use std::path::Path;
use std::sync::Arc;

use glam::{Mat2, Vec2};
use image::Rgba32FImage;

use crate::interval::Interval;
use crate::perlin::Perlin;
use crate::texture::TextureDerivatives;
use crate::texture::mapping::{TextureMapping2D, TextureMapping3D, UvGen};
use crate::texture::{Texture, TextureCoords};
use crate::vec3::Color3;

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
        // Apply 3D point mapping first (transforms the texture space).
        let tex_point = self.mapping3d.map_point(coords.tex_points.texture);

        let (u, v) = self
            .uv_gen
            .map_to_uv(coords.tex_points.mapping)
            .unwrap_or((coords.u, coords.v));

        let (su, sv) = self.mapping2d.map_uv(u, v);

        let mapped = coords.with_texture_point(tex_point).with_uv(su, sv);
        self.texture.value(&mapped)
    }
}

/// Uniform color texture — returns the same [`Color3`] at every point.
///
/// Used as the fallback when no texture is provided (e.g. `Material::lambertian_color`),
/// and as the GPU serialization color for textured materials.
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

/// Loads an image texture and stores it in float RGBA format.
pub struct ImageTexture {
    image_mips: Vec<Rgba32FImage>,
}

impl ImageTexture {
    /// Loads an image from disk and converts it to float RGBA.
    pub fn new<P: AsRef<Path>>(filename: P) -> image::ImageResult<Self> {
        let image = image::open(filename)?.to_rgba32f();
        let mut tex = Self {
            image_mips: vec![image.clone()],
        };

        let mut current_mip = image;
        // Box-filter mip chain: each level = 2×2 average of the one above.
        while current_mip.width() > 1 && current_mip.height() > 1 {
            let next_mip = image::imageops::resize(
                &current_mip,
                (current_mip.width() / 2).max(1),
                (current_mip.height() / 2).max(1),
                image::imageops::FilterType::Triangle,
            );
            tex.image_mips.push(next_mip.clone());
            current_mip = next_mip;
        }

        Ok(tex)
    }

    /// Loads an image and wraps it in a MappedTexture then Arc.
    /// Returns Err from the underlying image loader if the file can't be opened.
    pub fn load_arc<P: AsRef<Path>>(filename: P) -> image::ImageResult<Arc<dyn Texture>> {
        let tex = ImageTexture::new(filename)?;
        Ok(Arc::new(MappedTexture::new(tex)))
    }

    pub fn image(&self) -> &Rgba32FImage {
        &self.image_mips[0]
    }

    fn texel(&self, level: usize, x: i32, y: i32) -> Color3 {
        let img = &self.image_mips[level];
        let (w, h) = (img.width() as i32, img.height() as i32);
        // u (longitude) wraps around the seam; v (latitude) must clamp — wrapping it
        // would pull opposite-hemisphere colors across the poles.
        let px = x.rem_euclid(w);
        let py = y.clamp(0, h - 1);
        let p = img.get_pixel(px as u32, py as u32);
        Color3::new(p[0], p[1], p[2])
    }

    fn compute_lod(&self, derivatives: &TextureDerivatives) -> f32 {
        let (w, h) = (self.image().width() as f32, self.image().height() as f32);

        let du_v_dx = Vec2::new(derivatives.dudx * w, derivatives.dvdx * h);
        let du_v_dy = Vec2::new(derivatives.dudy * w, derivatives.dvdy * h);

        let rho2 = du_v_dy.length_squared().max(du_v_dx.length_squared());

        0.5 * rho2.max(1e-8).log2().max(0.0)
    }

    /// Bilinear at one mip level (matches your v-flip: v = 1 − v).
    fn bilinear(&self, level: usize, u: f32, v: f32) -> Color3 {
        let img = &self.image_mips[level];
        let (w, h) = (img.width() as f32, img.height() as f32);
        let u = u.clamp(0.0, 1.0);
        let v = 1.0 - v.clamp(0.0, 1.0);
        let x = u * w - 0.5;
        let y = v * h - 0.5;
        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let (dx, dy) = (x - x0 as f32, y - y0 as f32);
        let c00 = self.texel(level, x0, y0);
        let c10 = self.texel(level, x0 + 1, y0);
        let c01 = self.texel(level, x0, y0 + 1);
        let c11 = self.texel(level, x0 + 1, y0 + 1);
        let cx0 = c00 * (1.0 - dx) + c10 * dx;
        let cx1 = c01 * (1.0 - dx) + c11 * dx;
        cx0 * (1.0 - dy) + cx1 * dy
    }

    /// Tri-linear: bilinear at floor(lod) and floor(lod)+1, lerp between.
    fn trilinear(&self, u: f32, v: f32, lod: f32) -> Color3 {
        let max_level = (self.image_mips.len() - 1) as f32;
        let level = lod.floor().clamp(0.0, max_level) as usize;
        let frac = (lod - level as f32).clamp(0.0, 1.0);
        let c0 = self.bilinear(level, u, v);
        if frac <= 0.0 {
            return c0;
        }
        let c1 = self.bilinear((level + 1).min(self.image_mips.len() - 1), u, v);
        c0 * (1.0 - frac) + c1 * frac
    }

    /// Anisotropy ratio of the texture-space footprint (major/minor axis lengths).
    /// 1.0 = square footprint (isotropic); > 1 = elliptical (needs AF).
    fn anisotropy_ratio(&self, d: &TextureDerivatives) -> f32 {
        let (w, h) = (self.image().width() as f32, self.image().height() as f32);

        let duv_dx = Vec2::new(d.dudx * w, d.dvdx * h);
        let duv_dy = Vec2::new(d.dudy * w, d.dvdy * h);

        let major = duv_dx.length().max(duv_dy.length());
        let minor = duv_dx.length().min(duv_dy.length()).max(1e-8);

        (major / minor).clamp(1.0, 16.0)
    }

    /// Anisotropic filtering: sample along the major axis of the footprint ellipse, using the
    /// minor-axis LOD. Trilinear is the isotropic baseline; AF only matters on grazing/angled
    /// surfaces where the footprint is elliptical.
    fn anisotropic(&self, coords: &TextureCoords) -> Color3 {
        let (w, h) = (self.image().width() as f32, self.image().height() as f32);
        let duv_dx = Vec2::new(coords.derivatives.dudx * w, coords.derivatives.dvdx * h);
        let duv_dy = Vec2::new(coords.derivatives.dudy * w, coords.derivatives.dvdy * h);

        // Structure tensor of the footprint ellipse: G = J·Jᵀ, the covariance of the
        // two edge vectors (J has duv_dx, duv_dy as its columns).
        let g = Mat2::from_cols(duv_dx, duv_dy) * Mat2::from_cols(duv_dx, duv_dy).transpose();
        let (a, b, c) = (g.x_axis.x, g.x_axis.y, g.y_axis.y); // G is symmetric: x_axis.y == y_axis.x

        let trace = a + c;
        let disc = ((a - c) * 0.5).powi(2) + b * b;
        let sqrt_disc = disc.sqrt().max(0.0);
        let lambda_major = 0.5 * (trace + 2.0 * sqrt_disc);
        let lambda_minor = (0.5 * (trace - 2.0 * sqrt_disc)).max(0.0);

        let major_len = lambda_major.sqrt();
        let minor_len = lambda_minor.sqrt();

        // LOD from the minor axis (the narrower dimension controls aliasing).
        let lod = 0.5 * lambda_minor.max(1e-12).log2();

        if minor_len < 1e-8 {
            return self.trilinear(coords.u, coords.v, lod);
        }

        // Major-axis direction = eigenvector of G for lambda_major. The axis angle
        // satisfies tan(2θ) = 2b/(a−c); atan2 keeps the correct quadrant and covers
        // the diagonal (b = 0) cases without branching.
        let theta = 0.5 * (2.0 * b).atan2(a - c);
        let dir = Vec2::from_angle(theta);

        // Anisotropy ratio = major/minor, capped.
        let ratio = (major_len / minor_len).clamp(1.0, 16.0);
        let num_samples = ratio.ceil() as usize;
        if num_samples <= 1 {
            return self.trilinear(coords.u, coords.v, lod);
        }

        // Sample along the major axis, centered on the footprint, averaging the results.
        // dir * major_len is in texels-per-pixel (the tensor eigenvalues use
        // duv_dx/duv_dy in texel space), so divide by (w, h) to convert the offset
        // to UV [0,1] space before adding to coords.u/coords.v.
        (0..num_samples)
            .map(|i| {
                let t = i as f32 / (num_samples - 1) as f32;
                let offset = (t - 0.5) * dir * major_len / Vec2::new(w, h);
                self.trilinear(coords.u + offset.x, coords.v + offset.y, lod)
            })
            .sum::<Color3>()
            / num_samples as f32
    }
}

impl Texture for ImageTexture {
    fn value(&self, coords: &TextureCoords) -> Color3 {
        if self.image().height() == 0 {
            return Color3::new(0., 1., 1.);
        }

        if coords.derivatives.is_zero() {
            let u = Interval::from(0., 1.).clamp(coords.u);
            let v = 1.0 - Interval::from(0., 1.).clamp(coords.v);

            let i = (u * self.image().width() as f32).min((self.image().width() - 1) as f32);
            let j = (v * self.image().height() as f32).min((self.image().height() - 1) as f32);
            let pixel = self.image().get_pixel(i as u32, j as u32);
            Color3::new(pixel[0], pixel[1], pixel[2])
        } else {
            let lod = self.compute_lod(&coords.derivatives);
            // Use anisotropic filtering when the footprint is meaningfully
            // elliptical (grazing/angled surfaces); otherwise trilinear.
            if self.anisotropy_ratio(&coords.derivatives) > 1.0 {
                self.anisotropic(coords)
            } else {
                self.trilinear(coords.u, coords.v, lod)
            }
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
        use crate::texture::TextureCoords;
        use crate::vec3::Point3;

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
