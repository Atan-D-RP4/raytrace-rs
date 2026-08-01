use std::path::Path;
use std::sync::Arc;

use glam::{Mat2, Vec2};
use image::Rgba32FImage;

use crate::texture::MappedTexture;
use crate::texture::TextureDerivatives;
use crate::texture::gpu::{
    GPU_TEX_NONE, GpuTextureBuffer, GpuTextureNode, GpuTextureType, ImagePayload,
};
use crate::texture::{Texture, TextureCoords};
use crate::vec3::Color3;

/// Per-axis addressing mode — what happens when a texture-space coordinate
/// leaves the valid range.
///
/// Enforced by [`ImageTexture::texel`] and mirrored by the GPU sampler's
/// address mode (wgpu `AddressMode`, vulkan `VK_SAMPLER_ADDRESS_MODE_*`)
/// so the shader samples exactly like the CPU path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureWrap {
    /// Wrap back into range — `rem_euclid`. Sphere longitude seam.
    Repeat,
    /// Clamp to the edge. Sphere latitude poles.
    ClampToEdge,
}

impl TextureWrap {
    /// Maps a texel coordinate into `[0, size)` under this wrap mode.
    pub fn address(&self, coord: i32, size: i32) -> i32 {
        match self {
            TextureWrap::Repeat => coord.rem_euclid(size),
            TextureWrap::ClampToEdge => coord.clamp(0, size - 1),
        }
    }
}

/// Loads an image texture and stores it in float RGBA format.
pub struct ImageTexture {
    /// Mip chain of the image, level 0 is the original image.
    image_mips: Vec<Rgba32FImage>,
    /// Horizontal addressing mode (default: [`TextureWrap::Repeat`] — sphere longitude seam).
    wrap_u: TextureWrap,
    /// Vertical addressing mode (default: [`TextureWrap::ClampToEdge`] — sphere latitude poles).
    wrap_v: TextureWrap,
}

impl ImageTexture {
    /// Loads an image from disk and converts it to float RGBA.
    pub fn new<P: AsRef<Path>>(filename: P) -> image::ImageResult<Self> {
        let image = image::open(filename)?.to_rgba32f();
        let mut tex = Self {
            image_mips: vec![image.clone()],
            // u (longitude) wraps around the seam; v (latitude) must clamp —
            // wrapping it would pull opposite-hemisphere colors across the poles.
            wrap_u: TextureWrap::Repeat,
            wrap_v: TextureWrap::ClampToEdge,
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

    /// Sets the per-axis addressing modes.
    ///
    /// Defaults are [`TextureWrap::Repeat`] on u (sphere seam) and
    /// [`TextureWrap::ClampToEdge`] on v (poles).
    pub fn with_wraps(mut self, wrap_u: TextureWrap, wrap_v: TextureWrap) -> Self {
        self.wrap_u = wrap_u;
        self.wrap_v = wrap_v;
        self
    }

    /// Horizontal addressing mode.
    pub fn wrap_u(&self) -> TextureWrap {
        self.wrap_u
    }

    /// Vertical addressing mode.
    pub fn wrap_v(&self) -> TextureWrap {
        self.wrap_v
    }

    /// Sample a single texel at the given mip level, applying the wrap modes.
    fn texel(&self, level: usize, x: i32, y: i32) -> Color3 {
        let img = &self.image_mips[level];
        let (w, h) = (img.width() as i32, img.height() as i32);

        // Map the texel coordinates into the valid range using the wrap modes.
        let px = self.wrap_u.address(x, w);
        let py = self.wrap_v.address(y, h);

        // Get the pixel color and convert to Color3 (RGB).
        let p = img.get_pixel(px as u32, py as u32);

        Color3::new(p[0], p[1], p[2])
    }

    /// Compute the level-of-detail (LOD) for the texture sample based on the screen-space
    /// derivatives.
    fn compute_lod(&self, derivatives: &TextureDerivatives) -> f32 {
        let (w, h) = (self.image().width() as f32, self.image().height() as f32);

        // Compute the footprint in texel space by scaling the derivatives by the image size.
        let du_v_dx = Vec2::new(derivatives.dudx * w, derivatives.dvdx * h);
        let du_v_dy = Vec2::new(derivatives.dudy * w, derivatives.dvdy * h);

        // Compute the squared lengths of the footprint vectors and take the maximum.
        let rho2 = du_v_dy.length_squared().max(du_v_dx.length_squared());

        // Compute the LOD as 0.5 * log2(max(rho2, 1e-8)), clamped to a minimum of 0.0.
        0.5 * rho2.max(1e-8).log2().max(0.0)
    }

    /// Bilinear at one mip level (matches the v-flip: v = 1 − v).
    ///
    /// UV is not pre-clamped: out-of-range texel coordinates fall through to [`Self::texel`], which
    /// applies the wrap modes — Repeat tiles across the seam with correct blending, ClampToEdge
    /// saturates at the edge.
    fn bilinear(&self, level: usize, u: f32, v: f32) -> Color3 {
        let img = &self.image_mips[level];
        let (w, h) = (img.width() as f32, img.height() as f32);

        // Compute the floating-point texel coordinates, offset by 0.5 to sample at the texel
        // centers. No UV pre-clamping: out-of-range coordinates fall through to texel(), which
        // applies the wrap modes (Repeat tiles across the seam; ClampToEdge saturates).
        let x = u * w - 0.5;
        let y = (1.0 - v) * h - 0.5;

        // Compute the integer coordinates of the top-left texel.
        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;

        // Fractional parts for interpolation. Use floor-relative (x - x0), NOT fract(): fract() is
        // trunc-relative and goes negative for x < 0, which extrapolates instead of blending when
        // wrapping produces negative texel coordinates.
        let (dx, dy) = (x - x0 as f32, y - y0 as f32);

        // Sample the four neighboring texels and bilinearly interpolate.
        // Bilinear interpolation: c00 * (1 - dx) * (1 - dy) + c10 * dx * (1 - dy) + c01 * (1 - dx) * dy + c11 * dx * dy
        let c = |x: i32, y: i32| self.texel(level, x, y);
        let cx = |y: i32| c(x0, y) * (1.0 - dx) + c(x0 + 1, y) * dx;

        cx(y0) * (1.0 - dy) + cx(y0 + 1) * dy
    }

    /// Trilinear filtering: bilinear at two mip levels, then linear blend between them.
    ///
    /// UV is not pre-clamped: out-of-range texel coordinates fall through to [`Self::texel`], which
    /// applies the wrap modes — Repeat tiles across the seam with correct blending, ClampToEdge
    /// saturates at the edge.
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

        // Eigenvalues of G give the squared lengths of the ellipse axes.
        // The eigenvalues are: λ = 0.5 * (tr(G) ± sqrt(tr(G)² - 4 det(G))).
        // Trace, tr(G) = a + c
        // Determinant, det(G) = ac - b²
        let trace = a + c;

        // Discriminant = tr(G)² - 4 det(G) = (a - c)² + 4b².
        // The eigenvalues are always non-negative because G is positive semi-definite (the squared
        // lengths of the ellipse axes are non-negative).
        let disc = (a - c).powi(2) + 4.0 * b * b;
        let sqrt_disc = disc.sqrt().max(0.0);
        let lambda_major = 0.5 * (trace + sqrt_disc);
        let lambda_minor = (0.5 * (trace - sqrt_disc)).max(0.0);

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
            // Point sample without pre-clamping UV: out-of-range texel
            // coordinates fall through to texel() and its wrap modes.
            let (w, h) = (self.image().width() as i32, self.image().height() as i32);
            let i = self.wrap_u.address((coords.u * w as f32).floor() as i32, w);
            let j = self
                .wrap_v
                .address(((1.0 - coords.v) * h as f32).floor() as i32, h);
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

    fn serialize_gpu(&self, buf: &mut GpuTextureBuffer) -> u32 {
        let image_index = buf.images.len() as u32;
        buf.images.push(ImagePayload {
            mips: self.image_mips.clone(),
            wrap_u: self.wrap_u,
            wrap_v: self.wrap_v,
        });
        let idx = buf.nodes.len() as u32;
        buf.nodes.push(GpuTextureNode {
            texture_type: GpuTextureType::Image as u32,
            param_offset: 0,
            child_a: GPU_TEX_NONE,
            child_b: GPU_TEX_NONE,
            image_index,
            sampler_index: image_index, // one sampler per image for now
        });
        idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec3::{Direction3, Point3};

    #[test]
    fn wrap_repeat_maps_back_into_range() {
        let wrap = TextureWrap::Repeat;
        assert_eq!(wrap.address(5, 4), 1);
        assert_eq!(wrap.address(-1, 4), 3); // rem_euclid stays positive
        assert_eq!(wrap.address(0, 4), 0);
    }

    #[test]
    fn wrap_clamp_to_edge_saturates() {
        let wrap = TextureWrap::ClampToEdge;
        assert_eq!(wrap.address(5, 4), 3);
        assert_eq!(wrap.address(-1, 4), 0);
        assert_eq!(wrap.address(2, 4), 2);
    }

    /// 4×1 image: red, green, blue, yellow columns.
    fn sample_image() -> ImageTexture {
        let mut img = Rgba32FImage::new(4, 1);
        for (x, [r, g, b]) in [
            [1.0, 0.0, 0.0], // red
            [0.0, 1.0, 0.0], // green
            [0.0, 0.0, 1.0], // blue
            [1.0, 1.0, 0.0], // yellow
        ]
        .iter()
        .enumerate()
        {
            img.put_pixel(x as u32, 0, image::Rgba([*r, *g, *b, 1.0]));
        }
        ImageTexture {
            image_mips: vec![img],
            wrap_u: TextureWrap::Repeat,
            wrap_v: TextureWrap::ClampToEdge,
        }
    }

    #[test]
    fn repeat_wraps_positive_uv_in_point_sample() {
        let tex = sample_image();
        // u = 1.125 → texel index 4 → wraps to 0 → red. Pre-wrap behavior
        // clamped to the last column (yellow).
        let coords = TextureCoords::new(
            1.125,
            0.5,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, 0.0),
            Direction3::new(0.0, 0.0, 1.0),
            None,
        );
        let c = tex.value(&coords);
        assert!((c.x() - 1.0).abs() < 1e-6, "expected red, got {c:?}");
        assert!(c.y() < 1e-6);
        assert!(c.z() < 1e-6);
    }

    #[test]
    fn repeat_blends_across_seam_in_bilinear() {
        let tex = sample_image();
        // u = 0.999 with Repeat: x0 = 3, x0 + 1 = 4 → wraps to column 0, so
        // the blend is 0.504·yellow + 0.496·red.
        let c = tex.bilinear(0, 0.999, 0.5);
        assert!((c.x() - 1.0).abs() < 1e-6);
        assert!((c.y() - 0.504).abs() < 1e-3);
        assert!(c.z() < 1e-6);
    }

    #[test]
    fn repeat_blends_across_seam_negative_side() {
        let tex = sample_image();
        // u = -0.001 with Repeat: x0 = -1, dx = 0.496 (floor-relative). fract()
        // would give -0.504 and extrapolate instead of blending. Expect the
        // same 0.504·yellow + 0.496·red blend as the positive side.
        let c = tex.bilinear(0, -0.001, 0.5);
        assert!((c.x() - 1.0).abs() < 1e-6);
        assert!((c.y() - 0.504).abs() < 1e-3);
        assert!(c.z() < 1e-6);
    }

    #[test]
    fn clamp_to_edge_does_not_tile_in_bilinear() {
        let mut tex = sample_image();
        tex.wrap_u = TextureWrap::ClampToEdge;
        // u = 1.05 saturates at the right edge → pure yellow (column 3).
        let c = tex.bilinear(0, 1.05, 0.5);
        assert!((c.x() - 1.0).abs() < 1e-6);
        assert!((c.y() - 1.0).abs() < 1e-6);
        assert!(c.z() < 1e-6);
    }

    #[test]
    fn anisotropic_averages_along_major_axis() {
        let tex = sample_image();
        // Footprint in texel space: duv_dx = (2, 0), duv_dy = (0, 0.5) →
        // G = [[4, 0], [0, 0.25]], axes (2, 0.5), ratio 4 → 4 samples along u
        // at u ∈ {0.25, 0.4167, 0.5833, 0.75}. The analytic average over the
        // 4 distinct columns is (0.25, 0.5, 0.375).
        let coords = TextureCoords::new(
            0.5,
            0.5,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, 0.0),
            Direction3::new(0.0, 0.0, 1.0),
            Some(TextureDerivatives::new(
                Direction3::new(0.0, 0.0, 1.0),
                Direction3::new(0.0, 0.0, 1.0),
                0.5, // dudx * w = 2 texels
                0.0, // dudy
                0.0, // dvdx
                0.5, // dvdy * h = 0.5 texels
            )),
        );
        let c = tex.value(&coords);
        assert!((c.x() - 0.25).abs() < 1e-3, "got {c:?}");
        assert!((c.y() - 0.5).abs() < 1e-3, "got {c:?}");
        assert!((c.z() - 0.375).abs() < 1e-3, "got {c:?}");
    }
}
