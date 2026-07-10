//! Concrete texture implementations.
//!
//! Each type implements the [`Texture`] trait and evaluates a color from
//! the coordinate context. These are the leaf nodes that materials sample
//! during shading.

use std::path::Path;
use std::sync::Arc;

use image::Rgba32FImage;

use crate::interval::Interval;
use crate::perlin::Perlin;
use crate::texture::mapping::{TextureMapping2D, TextureMapping3D, UvGen};
use crate::texture::{Texture, TextureCoords};
use crate::vec3::Color3;

use crate::texture::TextureDerivatives;

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a * (1.0 - t) + b * t
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
    pub fn from_rgb(r: f64, g: f64, b: f64) -> Self {
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
}

impl Texture for CheckerTexture {
    fn value(&self, coords: &TextureCoords) -> Color3 {
        let x = coords.tex_points.texture.x.floor() as i32;
        let y = coords.tex_points.texture.y.floor() as i32;
        let z = coords.tex_points.texture.z.floor() as i32;

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

    pub fn image(&self) -> &Rgba32FImage {
        &self.image_mips[0]
    }

    fn texel(&self, level: usize, x: i32, y: i32) -> Color3 {
        let img = &self.image_mips[level];
        let (w, h) = (img.width() as i32, img.height() as i32);
        let p = img.get_pixel(x.rem_euclid(w) as u32, y.rem_euclid(h) as u32);
        Color3::new(p[0] as f64, p[1] as f64, p[2] as f64)
    }

    fn compute_lod(&self, derivatives: &TextureDerivatives) -> f64 {
        let (w, h) = (self.image().width() as f64, self.image().height() as f64);

        let du_v_dx = (derivatives.dudx * w, derivatives.dvdx * h);
        let du_v_dy = (derivatives.dudy * w, derivatives.dvdy * h);

        let rho2 = (du_v_dx.0 * du_v_dx.0 + du_v_dx.1 * du_v_dx.1)
            .max(du_v_dy.0 * du_v_dy.0 + du_v_dy.1 * du_v_dy.1);

        0.5 * rho2.max(1e-8).log2().max(0.0)
    }

    /// Bilinear at one mip level (matches your v-flip: v = 1 − v).
    fn bilinear(&self, level: usize, u: f64, v: f64) -> Color3 {
        let img = &self.image_mips[level];
        let (w, h) = (img.width() as f64, img.height() as f64);
        let u = u.clamp(0.0, 1.0);
        let v = 1.0 - v.clamp(0.0, 1.0);
        let x = u * w - 0.5;
        let y = v * h - 0.5;
        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let (dx, dy) = (x - x0 as f64, y - y0 as f64);
        let c00 = self.texel(level, x0, y0);
        let c10 = self.texel(level, x0 + 1, y0);
        let c01 = self.texel(level, x0, y0 + 1);
        let c11 = self.texel(level, x0 + 1, y0 + 1);
        let cx0 = c00 * (1.0 - dx) + c10 * dx;
        let cx1 = c01 * (1.0 - dx) + c11 * dx;
        cx0 * (1.0 - dy) + cx1 * dy
    }

    /// Tri-linear: bilinear at floor(lod) and floor(lod)+1, lerp between.
    fn trilinear(&self, u: f64, v: f64, lod: f64) -> Color3 {
        let max_level = (self.image_mips.len() - 1) as f64;
        let level = lod.floor().clamp(0.0, max_level) as usize;
        let frac = (lod - level as f64).clamp(0.0, 1.0);
        let c0 = self.bilinear(level, u, v);
        if frac <= 0.0 {
            return c0;
        }
        let c1 = self.bilinear((level + 1).min(self.image_mips.len() - 1), u, v);
        c0 * (1.0 - frac) + c1 * frac
    }

    /// Anisotropy ratio of the texture-space footprint (major/minor axis lengths).
    /// 1.0 = square footprint (isotropic); > 1 = elliptical (needs AF).
    fn anisotropy_ratio(&self, d: &TextureDerivatives) -> f64 {
        let (w, h) = (self.image().width() as f64, self.image().height() as f64);
        let duv_dx = (d.dudx * w, d.dvdx * h);
        let duv_dy = (d.dudy * w, d.dvdy * h);
        let major = lerp(duv_dx.0, duv_dy.0, 0.5).hypot(lerp(duv_dx.1, duv_dy.1, 0.5));
        let minor = duv_dx.0.hypot(duv_dx.1).min(duv_dy.0.hypot(duv_dy.1));
        if minor < 1e-8 {
            return 1.0;
        }
        (major / minor).clamp(1.0, 16.0)
    }

    /// Anisotropic filtering: sample along the major axis of the footprint ellipse,
    /// using the minor-axis LOD. Trilinear is the isotropic baseline; AF only matters
    /// on grazing/angled surfaces where the footprint is elliptical.
    fn anisotropic(&self, coords: &TextureCoords) -> Color3 {
        let (w, h) = (self.image().width() as f64, self.image().height() as f64);
        let duv_dx = (coords.derivatives.dudx * w, coords.derivatives.dvdx * h);
        let duv_dy = (coords.derivatives.dudy * w, coords.derivatives.dvdy * h);

        // Structure tensor of the footprint ellipse (covariance of the two edge vectors).
        let a = duv_dx.0 * duv_dx.0 + duv_dy.0 * duv_dy.0;
        let b = duv_dx.0 * duv_dx.1 + duv_dy.0 * duv_dy.1;
        let c = duv_dx.1 * duv_dx.1 + duv_dy.1 * duv_dy.1;

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

        // Major-axis direction = eigenvector of the structure tensor for lambda_major.
        let (mx, my) = if b.abs() > 1e-12 {
            (b, lambda_major - a)
        } else if a >= c {
            (1.0, 0.0)
        } else {
            (0.0, 1.0)
        };
        let mlen = (mx * mx + my * my).sqrt().max(1e-12);
        let dir_x = mx / mlen;
        let dir_y = my / mlen;

        // Anisotropy ratio = major/minor, capped.
        let ratio = (major_len / minor_len).clamp(1.0, 16.0);
        let num_samples = ratio.ceil() as usize;
        if num_samples <= 1 {
            return self.trilinear(coords.u, coords.v, lod);
        }

        // Sample along the major axis, averaging the results.
        let mut color_sum = Color3::ZERO;
        for i in 0..num_samples {
            let t = i as f64 / (num_samples - 1) as f64;
            let u_sample = coords.u + t * dir_x * major_len;
            let v_sample = coords.v + t * dir_y * major_len;
            color_sum += self.trilinear(u_sample, v_sample, lod);
        }

        color_sum / num_samples as f64
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

            let i = (u * self.image().width() as f64).min((self.image().width() - 1) as f64);
            let j = (v * self.image().height() as f64).min((self.image().height() - 1) as f64);
            let pixel = self.image().get_pixel(i as u32, j as u32);
            Color3::new(pixel[0] as f64, pixel[1] as f64, pixel[2] as f64)
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
        Color3::new(0.5, 0.5, 0.5)
            * (1.0 + (point.z + (10.0 * self.noise.turbulence(point, 7))).sin())
    }
}
