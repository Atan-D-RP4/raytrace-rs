//! Camera and reference CPU rendering implementation.
//!
//! Current responsibilities:
//! 1. Build camera rays from [`CameraConfig`].
//! 2. Run the CPU Monte-Carlo path-tracing loop.
//! 3. Return an RGB8 image buffer for output.
//!
//! TODO(renderer-abstraction): factor the rendering loop into a dedicated
//! renderer/pipeline module so camera ray generation can be reused by GPU,
//! raster, hybrid, and other future rendering engines.

use std::sync::{Arc, RwLock};

use rand::RngExt;
use rayon::prelude::*;
use tracing::info;

use crate::hittable::Hittable;
use crate::interval::Interval;
use crate::material::PdfKind;
use crate::pdf::{CosinePDF, GgxSamplePDF, HittablePDF, MixturePDF, PDF, UniformSpherePDF};
use crate::ray::Ray;
use crate::vec3::{Color3, Point3, Vec3, random_in_unit_disk_with_rng};

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
    /// Uses parallel iteration over pixel chunks for performance.
    /// Each chunk writes RGB triples directly into the output buffer.
    ///
    /// TODO(renderer-abstraction): move this method and [`Camera::ray_color`]
    /// into a renderer/pipeline component so alternate engines can share camera setup.
    pub fn render(&mut self, world: &impl Hittable, lights: &impl Hittable) -> (u32, u32, Vec<u8>) {
        let _span = tracing::info_span!(
            "render",
            width = self.image_width,
            height = self.image_height,
            spp = self.samples_per_pixel,
            max_depth = self.max_depth,
            background = ?self.background,
        )
        .entered();

        info!("camera render started");
        profiling::scope!("camera_render_loop");

        let image_width = self.image_width;
        let total_pixels = self.image_height * self.image_width;
        let max_depth = self.max_depth;

        let mut output = vec![0u8; total_pixels as usize * 3];

        output
            .par_chunks_mut(3)
            .enumerate()
            .for_each(|(idx, rgb_chunk)| {
                let mut rng = rand::rng();
                // Top-left origin: i → right, j → down.
                let i = idx as i32 % image_width;
                let j = idx as i32 / image_width;

                // Sample every 1000th pixel for profiling without overhead.
                let span = if idx % 1000 == 0 {
                    Some(tracing::info_span!("pixel", i, j))
                } else {
                    None
                };
                let _guard = span.as_ref().map(|s| s.enter());

                // Accumulate samples for anti-aliasing.
                // Using Sobol QMC sequence for better convergence and O(1) sample generation.
                // Sample 0 starts at (0,0) in the Sobol accumulator; each subsequent sample flips
                // one bit (Gray code incremental, O(1) per sample).
                // Per-pixel random digital shift decorrelates adjacent pixels.
                let mut pixel_color = Color3::from(0., 0., 0.);
                let mut sobol_state = [0u32; 5];

                // shift[0] = Sobol accumulator x, [1] = y, [2] = sample index,
                // [3] = shift_x, [4] = shift_y
                sobol_state[3] = rng.random::<u32>();
                sobol_state[4] = rng.random::<u32>();

                for _ in 0..self.samples_per_pixel {
                    let ray = self.get_ray_sobol(i as f64, j as f64, &mut rng, &mut sobol_state);
                    let sample = self.ray_color(&ray, max_depth, &*world, &*lights, &mut rng);
                    // Guard per-sample: a single NaN/Inf corrupts the pixel accumulator (NaN +
                    // anything = NaN), so discard bad samples individually rather than post-hoc
                    // per-pixel.
                    if sample.x.is_finite() && sample.y.is_finite() && sample.z.is_finite() {
                        pixel_color += sample;
                    }
                }

                post_process(
                    pixel_color * self.pixel_samples_scale,
                    self.exposure,
                    self.tone_map,
                )
                .iter()
                .enumerate()
                .for_each(|(c, out)| rgb_chunk[c] = *out);
            });

        info!("camera render finished");

        (self.image_width as u32, self.image_height as u32, output)
    }

    /// Progressive CPU renderer for live preview pipeline.
    ///
    /// Behavior:
    /// - Computes one additional sample per pixel each outer iteration.
    /// - Accumulates linear radiance in `accum`.
    /// - Converts running average to RGB8.
    /// - Publishes full-frame snapshot into `framebuffer` every pass.
    /// - Sets `framebuffer.finished = true` on final pass.
    ///
    /// Threading model:
    /// - Pixel shading inside each pass runs in rayon parallel iterator.
    /// - Publication step takes short write lock only during buffer swap.
    ///
    /// Note: current implementation is full-frame progressive.
    /// Tile-based publication can reduce copy cost and lock contention.
    pub fn render_progressive(
        &mut self,
        world: &impl Hittable,
        lights: &dyn Hittable,
        framebuffer: SharedFramebuffer,
    ) {
        let _span = tracing::info_span!(
            "render_progressive",
            width = self.image_width,
            height = self.image_height,
            spp = self.samples_per_pixel,
            max_depth = self.max_depth,
        )
        .entered();

        info!("camera progressive render started");

        let width = self.image_width as usize;
        let height = self.image_height as usize;
        let total_pixels = width * height;

        let total_samples = self.samples_per_pixel;

        info!(
            width,
            height,
            spp = self.samples_per_pixel,
            "progressive render dimensions"
        );
        // TODO(opt-preview): reuse scratch buffers (`sample_colors`, `rgb`) across passes.
        // Current implementation reallocates each pass and increases allocator pressure.
        let mut accum = vec![Color3::from(0.0, 0.0, 0.0); total_pixels];

        // Initialize per-pixel Sobol states with fixed random digital shifts.
        // States persist across passes so the Sobol sequence progresses through
        // samples 0, 1, 2, ... — each pass advances every pixel by one Sobol point.
        let mut sobol_states: Vec<[u32; 5]> = (0..total_pixels)
            .map(|_| {
                let mut state = [0u32; 5];
                let mut rng = rand::rng();
                state[3] = rng.random::<u32>();
                state[4] = rng.random::<u32>();
                state
            })
            .collect();

        // Using for-loops instead of iterators for better optimization
        for sample_idx in 0..total_samples {
            let _sample_span =
                tracing::info_span!("progressive_pass", sample = sample_idx + 1).entered();
            profiling::scope!("progressive_pass");

            // Advance Sobol sequence by one sample per pixel, accumulate results.
            sobol_states
                .par_iter_mut()
                .zip(accum.par_iter_mut())
                .enumerate()
                .for_each(|(idx, (state, accum_color))| {
                    let mut rng = rand::rng();
                    let i = (idx % width) as f64;
                    let j = (idx / width) as f64;

                    let ray = self.get_ray_sobol(i, j, &mut rng, state);
                    let sample = self.ray_color(&ray, self.max_depth, &*world, &*lights, &mut rng);

                    if sample.x.is_finite() && sample.y.is_finite() && sample.z.is_finite() {
                        *accum_color += sample;
                    }
                });

            let scale = 1.0 / (sample_idx + 1) as f64;
            profiling::scope!("progressive_tonemap");
            let mut rgb = vec![0u8; total_pixels * 3];
            for (idx, color) in accum.iter().enumerate() {
                post_process(*color * scale, self.exposure, self.tone_map)
                    .iter()
                    .enumerate()
                    .for_each(|(c, out)| rgb[idx * 3 + c] = *out);
            }

            profiling::scope!("progressive_publish");
            if let Ok(mut fb) = framebuffer.write() {
                fb.width = self.image_width as u32;
                fb.height = self.image_height as u32;
                fb.rgb = rgb;
                fb.finished = sample_idx + 1 == total_samples;
            }
            // TODO(opt-preview): publish every N passes (adaptive cadence) to reduce lock contention.
            // Keep frequent updates early, slower cadence late for better throughput.

            if (sample_idx + 1) % 8 == 0 || sample_idx + 1 == total_samples {
                info!(
                    sample = sample_idx + 1,
                    total = total_samples,
                    "progressive pass complete"
                );
            }
            info!(
                sample_idx,
                total = total_samples,
                "single render pass complete",
            );
        }

        info!("camera progressive render finished");
    }

    /// Returns a random jitter offset inside the pixel cell.
    /// TODO(sampler-abstraction): extract sampler trait and implementations to decouple from camera.
    fn sample_square<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Vec3 {
        Vec3::from(rng.random::<f64>() - 0.5, rng.random::<f64>() - 0.5, 0.)
    }

    /// Returns a random jitter offset inside the pixel cell, stratified by pixel and sample index.
    /// TODO(sampler-abstraction): extract sampler trait and implementations to decouple from camera.
    fn sample_square_stratified<R: rand::Rng + ?Sized>(
        &self,
        rng: &mut R,
        si: i32,
        sj: i32,
        recip_sqrt_spp: f64,
    ) -> Vec3 {
        // Jitter within [0, 1) cell, then scale to cell size
        let px = (si as f64 + rng.random::<f64>()) * recip_sqrt_spp;
        let py = (sj as f64 + rng.random::<f64>()) * recip_sqrt_spp;

        Vec3::from(px, py, 0.)
    }

    /// Returns a random jitter offset inside the pixel cell, sampled from a Sobol sequence.
    /// TODO(sampler-abstraction): extract sampler trait and implementations to decouple from camera.
    fn sample_square_sobol(&self, state: &mut [u32; 5]) -> Vec3 {
        let [u, v] = sobol_2d_next(state);
        Vec3::from(u, v, 0.)
    }

    /// Constructs a time-sampled camera ray through a jittered pixel sample.
    fn get_ray<R: rand::Rng + ?Sized>(
        &self,
        u: f64,
        v: f64,
        si: i32,
        sj: i32,
        rng: &mut R,
        recip_sqrt_spp: f64,
    ) -> Ray {
        // let offset = self.sample_square(rng);

        // Construct a camera ray originating from the defocus disk and directed at a randomly
        // sampled point around the pixel location i, j for stratified sample square s_i, s_j.
        let offset = self.sample_square_stratified(rng, si, sj, recip_sqrt_spp);

        let pixel_sample = self.pixel00_loc
            + ((u + offset.x) * self.pixel_delta_u)
            + ((v + offset.y) * self.pixel_delta_v);

        let ray_origin = if self.defocus_angle <= 0. {
            self.look_from
        } else {
            self.defocus_disk_sample(rng)
        };
        let ray_direction = pixel_sample - ray_origin;

        // The center of the camera is the ray origin, and the ray direction is the vector from the
        // camera center to the pixel sample location.
        Ray::new_with_time(ray_origin, ray_direction, rng.random::<f64>())
    }

    /// Constructs a time-sampled camera ray through a Sobol-distributed pixel sample.
    fn get_ray_sobol<R: rand::Rng + ?Sized>(
        &self,
        u: f64,
        v: f64,
        rng: &mut R,
        state: &mut [u32; 5],
    ) -> Ray {
        let offset = self.sample_square_sobol(state);

        let pixel_sample = self.pixel00_loc
            + ((u + offset.x) * self.pixel_delta_u)
            + ((v + offset.y) * self.pixel_delta_v);

        let ray_origin = if self.defocus_angle <= 0. {
            self.look_from
        } else {
            self.defocus_disk_sample(rng)
        };
        let ray_direction = pixel_sample - ray_origin;

        Ray::new_with_time(ray_origin, ray_direction, rng.random::<f64>())
    }

    /// Reference CPU Monte-Carlo path-tracing integrator.
    ///
    /// Iteratively traces/scatters up to `depth` bounces and multiplies
    /// attenuation along the path. Returns sky/background gradient on miss.
    ///
    /// TODO(renderer-abstraction): extract this integrator behind a renderer trait
    /// so multiple pipelines (GPU/raster/hybrid/SDF/displacement-aware) can coexist.
    fn ray_color<R: rand::Rng>(
        &self,
        initial_ray: &Ray,
        depth: i32,
        world: &dyn Hittable,
        lights: &dyn Hittable,
        rng: &mut R,
    ) -> Color3 {
        // TODO(gpu): mirror this boundary in a separate path-trace kernel / WGSL entrypoint.
        let mut ray = *initial_ray;
        let mut accumulated_attenuation = Color3::from(1., 1., 1.);
        let mut accumulated_color = Color3::from(0., 0., 0.);

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

                // If we've exceeded the ray bounce limit, no more light is gathered.
                if bounce >= 5 {
                    // Russian Roulette: survival probability proportional to
                    // current path throughput. High-throughput paths (e.g.,
                    // non-absorbing volumes with albedo ≈ 1) survive at
                    // probability 1 — no artificial inflation from compensation.
                    // The 0.05 floor bounds variance from low-throughput paths.
                    let survival = max_attenuation.max(0.05);
                    if rng.random::<f64>() > survival {
                        return accumulated_color;
                    }
                    accumulated_attenuation /= survival;
                }
                // Outgoing direction (away from surface).
                let wo = -ray.direction.unit_vector();

                if let Some(sample) = record.material.sample(wo, &record, rng) {
                    if record.material.is_delta() {
                        // Delta distribution (perfect specular): use sampled
                        // direction directly — no MIS weighting needed.
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
                            // Delta materials are handled by the is_delta() branch above.
                            PdfKind::Delta => unreachable!(),
                        };

                        let light_pdf = HittablePDF::new(lights, record.point);
                        let pdfs: &[&dyn PDF] = &[&light_pdf, surface_pdf, surface_pdf];
                        let sampling_pdf = MixturePDF::new(pdfs);

                        let direction = sampling_pdf.generate(rng);
                        // Unitize direction for BRDF evaluation — PlanarPatch::random() returns a
                        // non-unit vector (distance to light), and BRDFs expect unit-length
                        // incident directions.
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

    /// Samples a point on the defocus disk for depth-of-field ray origins.
    fn defocus_disk_sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Vec3 {
        let point = random_in_unit_disk_with_rng(rng);
        self.look_from + (point.x * self.defocus_disk_u) + (point.y * self.defocus_disk_v)
    }
}

/// Direction numbers for 2D Sobol sequence.
///
/// SOBOL2D_DIRS[dim][bit_pos] = direction number as left-aligned u32.
///
/// Dim 0: Van der Corput (mⱼ = 1 for all j)
/// Dim 1: primitive polynomial x³ + x + 1
const SOBOL2D_DIRS: [[u32; 32]; 2] = [
    // Dim 0: vⱼ = 1/2ʲ⁺¹ → mⱼ = 1 << (31 - j)
    [
        0x80000000, 0x40000000, 0x20000000, 0x10000000, 0x08000000, 0x04000000, 0x02000000,
        0x01000000, 0x00800000, 0x00400000, 0x00200000, 0x00100000, 0x00080000, 0x00040000,
        0x00020000, 0x00010000, 0x00008000, 0x00004000, 0x00002000, 0x00001000, 0x00000800,
        0x00000400, 0x00000200, 0x00000100, 0x00000080, 0x00000040, 0x00000020, 0x00000010,
        0x00000008, 0x00000004, 0x00000002, 0x00000001,
    ],
    // Dim 1: from polynomial x³ + x + 1
    [
        0x80000000, 0xc0000000, 0xa0000000, 0xf0000000, 0x88000000, 0xcc000000, 0xaa000000,
        0xff000000, 0x88800000, 0xccc00000, 0xaaa00000, 0xfff00000, 0x88880000, 0xcccc0000,
        0xaaaa0000, 0xffff0000, 0x88888000, 0xccccc000, 0xaaaaa000, 0xfffff000, 0x88888800,
        0xcccccc00, 0xaaaaaa00, 0xffffff00, 0x88888880, 0xccccccc0, 0xaaaaaaa0, 0xfffffff0,
        0x88888888, 0xcccccccc, 0xaaaaaaaa, 0xffffffff,
    ],
];

// state for 2D Sobol sequence generator, used for stratified sampling in get_ray().
// Accumulated Sobol value for dimension 0 (Van der Corput)
// x - state[0]: u32,
// /// Accumulated Sobol value for dimension 1
// y - state[1]: u32,
// /// Number of samples taken (starts at 0)
// index - state[2]: u32,
// /// Random digital shift for dimension 0
// shift_x - state[3]: u32,
// /// Random digital shift for dimension 1
// shift_y - state[4]: u32,
fn sobol_2d_next(state: &mut [u32; 5]) -> [f64; 2] {
    // For sample k ≥ 1: tzcnt(k) tells us which bit flips in the Gray code
    // from sample k-1 to sample k.
    //
    // CRITICAL: 0u32.trailing_zeros() returns 32, not 0 — that would panic
    // (SOBOL2D_DIRS has indices 0..31). Guard: skip bit-flip for k=0.
    if state[2] > 0 {
        let c = state[2].trailing_zeros() as usize;
        state[0] ^= SOBOL2D_DIRS[0][c];
        state[1] ^= SOBOL2D_DIRS[1][c];
    }
    state[2] += 1;

    // Apply random digital shift (decorrelates adjacent pixels)
    let sx = state[0] ^ state[3];
    let sy = state[1] ^ state[4];

    // Convert u32 fixed-point → f64 in [0, 1)
    let inv = 1.0 / (1u64 << 32) as f64;
    [sx as f64 * inv, sy as f64 * inv]
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
