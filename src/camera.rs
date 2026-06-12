//! Camera and reference CPU rendering implementation.
//!
//! Current responsibilities:
//! 1. Build camera rays from [`CameraConfig`].
//! 2. Run the CPU Monte-Carlo path-tracing loop.
//! 3. Return an RGB8 image buffer for output.
//!
//! TODO(renderer-abstraction): refactor the rendering loop into a dedicated
//! renderer/pipeline module so camera ray generation can be reused by GPU,
//! raster, hybrid, and other future rendering engines.

use std::sync::{Arc, RwLock};

use rayon::prelude::*;
use tracing::info;

use crate::hittable::Hittable;
use crate::interval::Interval;
use crate::material::PdfKind;
use crate::pdf::{CosinePDF, GgxSamplePDF, HittablePDF, MixturePDF, PDF, UniformSpherePDF};
use crate::ray::Ray;
use crate::sampler::{DimCursor, Sampler, SobolQmcSampler};
use crate::vec3::{Color3, Point3, Vec3};

/// Thread-safe framebuffer shared between UI thread and render thread.
///
/// - Render thread takes write lock, publishes progressive updates.
/// - UI thread takes read lock, blits current snapshot to window surface.
pub type SharedFramebuffer = Arc<RwLock<Framebuffer>>;

/// Shared RGB framebuffer used by live preview path.
///
/// `rgb` layout is tightly packed RGB8 triples:
/// `[R, G, B, R, G, B, ...]`, row-major, top-left origin.
pub struct Framebuffer {
    /// Pixel width of framebuffer.
    pub width: u32,
    /// Pixel height of framebuffer.
    pub height: u32,
    /// Packed RGB8 data, `width * height * 3` bytes.
    pub rgb: Vec<u8>,
    /// Signals render completion to UI redraw loop.
    pub finished: bool,
}

impl Framebuffer {
    /// Creates zero-initialized framebuffer for given dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            rgb: vec![0; (width * height * 3) as usize],
            finished: false,
        }
    }
}

/// User-facing camera configuration.
///
/// This is scene/build-time data. Runtime/precomputed values live in [`Camera`].
#[derive(Default, Clone, Copy)]
pub struct CameraConfig {
    pub image_width: i32,       // Rendered image width in pixels
    pub aspect_ratio: f64,      // Image width / height
    pub samples_per_pixel: i32, // Rays per pixel for anti-aliasing
    pub max_depth: i32,         // Maximum ray bounce depth
    pub vfov: f64,              // Vertical field of view (degrees)
    pub look_from: Point3,      // Camera position
    pub look_at: Point3,        // Look target
    pub vup: Vec3,              // Up direction
    pub defocus_angle: f64,     // Depth of field angle
    pub focus_distance: f64,    // Focal plane distance
    pub background: Color3,     // Background color
    pub exposure: f64,          // Exposure
    pub tone_map: bool,         // Whether to apply tone mapping to final colors
}

impl CameraConfig {
    /// Creates a zero-initialized config; scenes usually set all fields explicitly.
    pub fn new() -> Self {
        Self {
            exposure: 1.0,
            ..Default::default()
        }
    }
}

/// Runtime camera with precomputed sampling and viewport data.
///
/// Construct via [`Camera::from_config`] so derived fields are initialized.
#[derive(Default, Clone)]
pub struct Camera {
    /// Rendered image width in pixels
    image_width: i32,
    /// Computed image height in pixels (derived from width/aspect_ratio)
    image_height: i32,

    /// Image width / height
    aspect_ratio: f64,
    /// Rays per pixel for anti-aliasing
    samples_per_pixel: i32,
    /// Maximum ray bounce depth
    max_depth: i32,
    /// Vertical field of view (degrees)
    vfov: f64,
    /// Camera position
    look_from: Point3,
    /// Look target
    look_at: Point3,
    /// Up direction Vector
    vup: Vec3,
    /// Depth of field angle
    defocus_angle: f64,
    /// Focal plane distance
    focus_distance: f64,
    /// Background color
    background: Color3,
    /// Exposure
    exposure: f64,
    /// Whether to apply tone mapping to final colors
    tone_map: bool,

    /// Defocus disk vector for u-axis (depth of field sampling)
    defocus_disk_u: Vec3,
    /// Defocus disk vector for v-axis (depth of field sampling)
    defocus_disk_v: Vec3,
    /// Location of upper-left pixel in world space
    pixel00_loc: Point3,
    /// Vector from one pixel to the next in horizontal direction
    pixel_delta_u: Point3,
    /// Vector from one pixel to the next in vertical direction
    pixel_delta_v: Point3,
    /// Scale factor for averaging samples (1/samples_per_pixel)
    pixel_samples_scale: f64,
}

impl Camera {
    /// Builds a runtime camera from scene configuration.
    pub fn from_config(config: &CameraConfig) -> Self {
        let mut camera = Self {
            image_width: config.image_width,
            image_height: 0,
            aspect_ratio: config.aspect_ratio,
            samples_per_pixel: config.samples_per_pixel,
            max_depth: config.max_depth,
            vfov: config.vfov,
            look_from: config.look_from,
            look_at: config.look_at,
            vup: config.vup,
            defocus_angle: config.defocus_angle,
            focus_distance: config.focus_distance,
            background: config.background,
            exposure: config.exposure,
            tone_map: config.tone_map,
            ..Default::default()
        };
        camera.initialize();
        camera
    }

    /// Creates a default camera; callers must configure and initialize it before rendering.
    pub fn new() -> Self {
        Default::default()
    }

    /// Returns configured render output dimensions in pixels.
    ///
    /// Useful for pre-sizing shared framebuffer and initial window size.
    pub fn image_dimensions(&self) -> (u32, u32) {
        (self.image_width as u32, self.image_height as u32)
    }

    /// Computes all runtime camera data needed for ray generation.
    ///
    /// This derives image dimensions, viewport basis vectors, per-pixel deltas,
    /// and depth-of-field sampling vectors from the current camera parameters.
    fn initialize(&mut self) {
        self.image_height = ((self.image_width as f64 / self.aspect_ratio) as i32).max(1);

        self.pixel_samples_scale = 1.0 / self.samples_per_pixel as f64;

        let center = self.look_from;

        let theta = self.vfov.to_radians();

        let h = (theta / 2.0).tan();

        // Derive viewport dimensions from vertical FOV and aspect ratio. The viewport is a plane
        // centered at the focal plane, with size determined by the FOV and aspect ratio.
        let viewport_height = 2.0 * h * self.focus_distance;
        let viewport_width = viewport_height * (self.image_width as f64 / self.image_height as f64);

        // Compute camera basis vectors. The camera looks from `look_from` towards `look_at`, with
        // `vup` as the up direction. The viewport is oriented according to these vectors
        let w = (self.look_from - self.look_at).unit_vector();
        let u = self.vup.cross(&w).unit_vector();
        let v = w.cross(&u);

        // Compute pixel deltas by scaling viewport basis vectors by the number of pixels, which
        // represent the world-space vector from pixel to pixel.
        let viewport_u = viewport_width * u; // Vector across viewport horizontal edge
        let viewport_v = viewport_height * -v; // Vector across viewport vertical edge
        // Negated because the v vector points up but the image coordinates increase downwards.

        self.pixel_delta_u = viewport_u / self.image_width as f64;
        self.pixel_delta_v = viewport_v / self.image_height as f64;

        // Compute the world-space location of the upper-left pixel (0,0).
        let viewport_upper_left =
            center - (self.focus_distance * w) - (viewport_u / 2.0) - (viewport_v / 2.0);

        self.pixel00_loc = viewport_upper_left + 0.5 * (self.pixel_delta_u + self.pixel_delta_v);

        // Depth of field: randomize ray origin within a disk of this radius.
        let defocus_radius = self.focus_distance * (self.defocus_angle / 2.0).to_radians().tan();
        self.defocus_disk_u = u * defocus_radius;
        self.defocus_disk_v = v * defocus_radius;
    }

    /// Renders the scene and returns (width, height, RGB pixel data).
    ///
    /// When `framebuffer` is `Some`, publishes progressive intermediate frames
    /// to the shared framebuffer during rendering (live preview mode).
    /// When `None`, renders all samples and returns the final image only.
    pub fn render(
        &mut self,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        framebuffer: Option<SharedFramebuffer>,
    ) -> (u32, u32, Vec<u8>) {
        let _span = tracing::info_span!(
            "render",
            width = self.image_width,
            height = self.image_height,
            spp = self.samples_per_pixel,
            max_depth = self.max_depth,
            background = ?self.background,
            progressive = framebuffer.is_some(),
        )
        .entered();

        info!("camera render started");
        profiling::scope!("camera_render_loop");

        let width = self.image_width as usize;
        let height = self.image_height as usize;
        let total_pixels = width * height;
        let total_samples = self.samples_per_pixel;
        let render_start = std::time::Instant::now();

        let mut accum = vec![Color3::from(0.0, 0.0, 0.0); total_pixels];

        for sample_idx in 0..total_samples {
            let pass_start = std::time::Instant::now();
            let _sample_span =
                tracing::info_span!("render_pass", sample = sample_idx + 1).entered();
            profiling::scope!("render_pass");

            accum
                .par_iter_mut()
                .enumerate()
                .for_each(|(idx, accum_color)| {
                    let i = (idx % width) as f64;
                    let j = (idx / width) as f64;

                    // Deterministic per-pixel seed from pixel coordinates.
                    let sampler = SobolQmcSampler::for_pixel(
                        idx as i32 % width as i32,
                        idx as i32 / width as i32,
                    );
                    let ray = self.get_ray(i, j, &sampler, sample_idx as u32);
                    let sample = self.ray_color(
                        &ray,
                        self.max_depth,
                        world,
                        lights,
                        &sampler,
                        sample_idx as u32,
                    );

                    if sample.x.is_finite() && sample.y.is_finite() && sample.z.is_finite() {
                        *accum_color += sample;
                    }
                });

            // Progressive mode: tonemap and publish intermediate frame.
            if let Some(ref fb) = framebuffer {
                let scale = 1.0 / (sample_idx + 1) as f64;
                profiling::scope!("progressive_tonemap");
                // TODO(opt-preview): reuse scratch buffer across passes to reduce
                // allocator pressure during long renders.
                let mut rgb = vec![0u8; total_pixels * 3];
                for (idx, color) in accum.iter().enumerate() {
                    post_process(*color * scale, self.exposure, self.tone_map)
                        .iter()
                        .enumerate()
                        .for_each(|(c, out)| rgb[idx * 3 + c] = *out);
                }

                // TODO(opt-preview): publish every N passes (adaptive cadence) to reduce
                // lock contention.  Keep frequent updates early, slower cadence late.
                profiling::scope!("progressive_publish");
                if let Ok(mut guard) = fb.write() {
                    guard.width = self.image_width as u32;
                    guard.height = self.image_height as u32;
                    guard.rgb = rgb;
                    guard.finished = sample_idx + 1 == total_samples;
                }
            }

            // Log pass timing every 8 passes and on the final pass.
            if (sample_idx + 1) % 8 == 0 || sample_idx + 1 == total_samples {
                let elapsed = pass_start.elapsed();
                let avg_sec = render_start.elapsed().as_secs_f64() / (sample_idx + 1) as f64;
                info!(
                    sample = sample_idx + 1,
                    total = total_samples,
                    pass_sec = format!("{:.4}", elapsed.as_secs_f64()),
                    avg_sec = format!("{:.4}", avg_sec),
                    eta_sec = format!("{:.1}", avg_sec * (total_samples - sample_idx - 1) as f64),
                    "render pass complete"
                );
            }
        }

        // Convert accumulated linear radiance to RGB8 output.
        profiling::scope!("final_tonemap");
        let mut output = vec![0u8; total_pixels * 3];
        for (idx, color) in accum.iter().enumerate() {
            post_process(
                *color * self.pixel_samples_scale,
                self.exposure,
                self.tone_map,
            )
            .iter()
            .enumerate()
            .for_each(|(c, out)| output[idx * 3 + c] = *out);
        }

        info!(
            elapsed = format!("{:.4}", render_start.elapsed().as_secs_f64()),
            samples_per_sec = format!(
                "{:.0}",
                total_samples as f64 / render_start.elapsed().as_secs_f64()
            ),
            "camera render finished"
        );
        (self.image_width as u32, self.image_height as u32, output)
    }

    /// Camera ray through pixel (i, j). Dims 0-1 = AA jitter, 2-3 = defocus.
    fn get_ray(&self, i: f64, j: f64, sampler: &dyn Sampler, sample_index: u32) -> Ray {
        let offset_u = sampler.sample(sample_index, 0);
        let offset_v = sampler.sample(sample_index, 1);

        let pixel_sample = self.pixel00_loc
            + ((i + offset_u) * self.pixel_delta_u)
            + ((j + offset_v) * self.pixel_delta_v);

        let ray_origin = if self.defocus_angle <= 0. {
            self.look_from
        } else {
            // Sample a point on the defocus disk for depth-of-field ray origins.
            let r = sampler.sample(sample_index, 2).sqrt();
            let theta = sampler.sample(sample_index, 3) * 2.0 * std::f64::consts::PI;
            let px = r * theta.cos();
            let py = r * theta.sin();
            self.look_from + (px * self.defocus_disk_u) + (py * self.defocus_disk_v)
        };

        let ray_direction = pixel_sample - ray_origin;
        // The center of the camera is the ray origin, and the ray direction is the vector from the
        // camera center to the pixel sample location.
        Ray::new_with_time(ray_origin, ray_direction, 0.0)
    }

    /// Reference CPU Monte-Carlo path-tracing integrator.
    ///
    /// Iteratively traces/scatters up to `depth` bounces and multiplies
    /// attenuation along the path.
    ///
    /// Dimension usage: dims 0-3 in `get_ray` (AA jitter + defocus).
    /// The `DimCursor` starts at 4, advancing per bounce for RR, material, and MIS samples.
    ///
    /// TODO(renderer-abstraction): extract this integrator behind a renderer trait
    /// so multiple pipelines (GPU/raster/hybrid/SDF/displacement-aware) can coexist.
    fn ray_color(
        &self,
        initial_ray: &Ray,
        depth: i32,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        sampler: &dyn Sampler,
        sample_index: u32,
    ) -> Color3 {
        let mut ray = *initial_ray;
        let mut accumulated_attenuation = Color3::from(1., 1., 1.);
        let mut accumulated_color = Color3::from(0., 0., 0.);

        let mut dims = DimCursor::new(4); // after ray setup dims (0-3 for AA + defocus)

        for bounce in 0..depth {
            if let Some(record) = world.hit(&ray, Interval::from(0.001, f64::INFINITY)) {
                let emission = record.material.emitted(&record);
                accumulated_color += accumulated_attenuation * emission;

                let max_attenuation = accumulated_attenuation
                    .x
                    .max(accumulated_attenuation.y)
                    .max(accumulated_attenuation.z);

                if max_attenuation < 1e-6 {
                    return accumulated_color;
                }

                let rr = sampler.sample(sample_index, dims.next_dim());
                let u_mat = sampler.sample(sample_index, dims.next_dim());
                let v_mat = sampler.sample(sample_index, dims.next_dim());
                let w_mat = sampler.sample(sample_index, dims.next_dim());
                let x_mat = sampler.sample(sample_index, dims.next_dim());

                // Russian Roulette: survival probability proportional to current
                // path throughput.  The 0.05 floor bounds variance from low-throughput paths.
                if bounce >= 5 {
                    let survival = max_attenuation.clamp(0.05, 1.0);
                    if rr > survival {
                        return accumulated_color;
                    }
                    accumulated_attenuation /= survival;
                }
                // Outgoing direction (away from surface).
                let wo = -ray.direction.unit_vector();

                // TODO(gpu): mirror this boundary in a separate path-trace kernel / WGSL entrypoint.
                if let Some(sample) = record
                    .material
                    .sample(wo, &record, u_mat, v_mat, w_mat, x_mat)
                {
                    // Per-sample delta routing: composed materials (Coated/Mix)
                    // decide via pdf_kind, not a fixed material property.
                    if matches!(sample.pdf_kind, PdfKind::Delta) {
                        accumulated_attenuation = accumulated_attenuation * sample.f_cos;
                        ray = Ray::new_with_time(record.point, sample.wi, ray.time);
                    } else {
                        // Non-delta material: mixture PDF of light + surface sampling.
                        // Weighted 1/3 light, 2/3 surface — surface PDF is usually the better match
                        // for glossy/Lambertian BRDFs.
                        let surface_pdf: &dyn PDF = match sample.pdf_kind {
                            PdfKind::Cosine { normal } => &CosinePDF::new(normal),
                            PdfKind::Ggx {
                                wo: ggx_wo,
                                normal,
                                alpha,
                            } => &GgxSamplePDF::new(ggx_wo, normal, alpha),
                            PdfKind::UniformSphere => &UniformSpherePDF::new(),
                            // Delta is handled by the outer match on pdf_kind.
                            PdfKind::Delta => unreachable!(),
                        };

                        let light_pdf = HittablePDF::new(lights, record.point);
                        let pdfs: &[&dyn PDF] = &[&light_pdf, surface_pdf, surface_pdf];
                        let sampling_pdf = MixturePDF::new(pdfs);

                        let direction = sampling_pdf.generate(sampler, sample_index, &mut dims);
                        // Unitize for BRDF evaluation — PlanarPatch::random() returns a
                        // non-unit vector (distance to light), and BRDFs expect unit length.
                        let direction_unit = direction.unit_vector();
                        let scattered = Ray::new_with_time(record.point, direction, ray.time);
                        let pdf_val = sampling_pdf.value(direction_unit);

                        let f_cos = record.material.eval(wo, direction_unit, &record);

                        // Standard single-sample MIS unbiased estimator: f(x) / p_mixture(x).
                        let weight = 1.0 / pdf_val.max(1e-6);

                        accumulated_attenuation = accumulated_attenuation * f_cos * weight;
                        ray = scattered;
                    }
                } else {
                    return accumulated_color;
                }
            } else {
                // If the ray hits nothing, return the background color
                // let unit_direction = ray.direction.unit_vector();
                // let t = 0.5 * (unit_direction.y + 1.0);

                // The background gradient
                // let background =
                //     ((1.0 - t) * Vec3::from(1.0, 1.0, 1.0)) + (t * Vec3::from(0.5, 0.7, 1.0));
                return accumulated_color + accumulated_attenuation * self.background;
            }
        }

        accumulated_color
    }
}

#[inline(always)]
fn post_process(color: Color3, exposure: f64, tone_map: bool) -> [u8; 3] {
    // Scale by sample count, exposure, and apply gamma correction.
    // Apply tone mapping operator before gamma if enabled, otherwise clamp to [0,1].
    let scaled = if tone_map {
        reinhard_tone_map(exposure, color)
    } else {
        color * exposure
    };

    // Gamma 2: sqrt() converts linear -> sRGB, then scale to [0,255].
    [
        (256.0 * linear_to_gamma(scaled.x).clamp(0.0, 0.999)) as u8,
        (256.0 * linear_to_gamma(scaled.y).clamp(0.0, 0.999)) as u8,
        (256.0 * linear_to_gamma(scaled.z).clamp(0.0, 0.999)) as u8,
    ]
}

#[inline(always)]
const fn reinhard_tone_map(exposure: f64, color: Color3) -> Color3 {
    let mapped = Vec3::from(color.x * exposure, color.y * exposure, color.z * exposure);
    Color3::from(
        mapped.x / (1.0 + mapped.x),
        mapped.y / (1.0 + mapped.y),
        mapped.z / (1.0 + mapped.z),
    )
}

#[inline(always)]
/// Converts a linear color channel to gamma-corrected (gamma=2) space.
fn linear_to_gamma(linear_component: f64) -> f64 {
    if linear_component > 0. {
        linear_component.sqrt()
    } else {
        0.
    }
}
