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

use core::f64;
use std::sync::Arc;

use rand::RngExt;
use rayon::prelude::*;
use tracing::info;

use crate::hittable::Hittable;
use crate::interval::Interval;
use crate::material::Material;
use crate::ray::Ray;
use crate::vec3::{Color3, Point3, Vec3, cross, random_in_unit_disk_with_rng, unit_vector};

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
}

impl CameraConfig {
    /// Creates a zero-initialized config; scenes usually set all fields explicitly.
    pub fn new() -> Self {
        Default::default()
    }
}

/// Runtime camera with precomputed sampling and viewport data.
///
/// Construct via [`Camera::from_config`] so derived fields are initialized.
#[derive(Default, Clone, Copy)]
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
            ..Default::default()
        };
        camera.initialize();
        camera
    }

    /// Creates a default camera; callers must configure and initialize it before rendering.
    pub fn new() -> Self {
        Default::default()
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

        // Determine the viewport dimensions based on the vertical field of view and aspect ratio. The
        // viewport is the plane that the camera rays will be cast through, and is centered at focal_length
        // units in front of the camera origin.
        let viewport_height = 2.0 * h * self.focus_distance;
        let viewport_width = viewport_height * (self.image_width as f64 / self.image_height as f64);

        let w = unit_vector(self.look_from - self.look_at);
        let u = unit_vector(cross(&self.vup, &w));
        let v = cross(&w, &u);

        // Calculate the pixel delta vectors, which are the vectors from one pixel to the next in
        // the u and v directions. These are used to calculate the ray direction for each pixel.
        let viewport_u = viewport_width * u; // Vector across viewport horizontal edge
        let viewport_v = viewport_height * -v; // Vector across viewport vertical edge
        // Negated because the v vector points up but the image coordinates increase downwards.

        // Calculate the pixel delta vectors, which are the vectors from one pixel to the next in
        // the u and v directions. These are used to calculate the ray direction for each pixel
        // by dividing the viewport dimensions by the image dimensions.
        self.pixel_delta_u = viewport_u / self.image_width as f64;
        self.pixel_delta_v = viewport_v / self.image_height as f64;

        // Calculate the location of the upper left pixel, which is the starting point for calculating
        // the ray directions for each pixel. This is done by starting at the camera center,
        // moving forward by focal_length units, and then moving left and up by half the viewport dimensions.
        let viewport_upper_left =
            center - (self.focus_distance * w) - (viewport_u / 2.0) - (viewport_v / 2.0);

        self.pixel00_loc = viewport_upper_left + 0.5 * (self.pixel_delta_u + self.pixel_delta_v);

        // Calculate the defocus disk vectors, which are used to calculate the ray origin for depth
        // of field effects. The defocus disk is a disk centered at the focal plane that represents
        // the area of the scene that will be in focus. The radius of the defocus disk is determined
        // by the focus distance and the defocus angle, which represents how much of the scene
        // should be in focus.
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
    pub fn render(&mut self, world: Arc<dyn Hittable>) -> (u32, u32, Vec<u8>) {
        // No need to initialize it here, `from_config` does it, or caller should do it manually
        // self.initialize();
        let _span = tracing::info_span!(
            "render",
            width = self.image_width,
            height = self.image_height,
            spp = self.samples_per_pixel,
            max_depth = self.max_depth,
            background_r = self.background.x,
            background_g = self.background.y,
            background_b = self.background.z,
        )
        .entered();

        info!("camera render started");
        profiling::scope!("camera_render_loop");

        let camera_snapshot = *self;
        let total_pixels = self.image_height * self.image_width;

        let sqrt_spp = self.samples_per_pixel.isqrt();
        let recip_sqrt_spp = 1.0 / sqrt_spp as f64;

        // Single flat buffer: avoids millions of small Vec allocations.
        // Format: [R, G, B, R, G, B, ...] for each pixel.
        let mut output = vec![0u8; total_pixels as usize * 3];

        // Parallel chunked writes: each thread writes RGB directly to its slice.
        // par_chunks_mut(3) gives mutable slices of exactly 3 bytes (one pixel).
        output
            .par_chunks_mut(3)
            .enumerate()
            .for_each(|(idx, chunk)| {
                let mut rng = rand::rng();
                // Convert flat index to 2D pixel coordinates.
                // Image coordinate origin: top-left (i increases right, j increases down).
                let i = idx as i32 % camera_snapshot.image_width;
                let j = idx as i32 / camera_snapshot.image_width;

                // Sample every 1000th pixel for profiling without overhead.
                let _span = if idx % 1000 == 0 {
                    Some(tracing::info_span!("pixel", i, j))
                } else {
                    None
                };
                let _guard = _span.as_ref().map(|s| s.enter());

                // Accumulate samples for anti-aliasing.
                // Stratified sampling: divide pixel into sqrt(spp) × sqrt(spp) grid,
                // then jitter within each cell. Total samples = spp.
                let mut pixel_color = Color3::from(0., 0., 0.);
                // Using for-loops instead of iterators for better optimization
                for si in 0..sqrt_spp {
                    for sj in 0..sqrt_spp {
                        let ray = camera_snapshot.get_ray(
                            i as f64,
                            j as f64,
                            si,
                            sj,
                            &mut rng,
                            recip_sqrt_spp,
                        );
                        pixel_color += camera_snapshot.ray_color(
                            &ray,
                            camera_snapshot.max_depth,
                            &*world,
                            &mut rng,
                        );
                    }
                }

                if !pixel_color.x.is_finite() {
                    pixel_color.x = 0.0;
                }
                if !pixel_color.y.is_finite() {
                    pixel_color.y = 0.0;
                }
                if !pixel_color.z.is_finite() {
                    pixel_color.z = 0.0;
                }

                // Scale by sample count and apply gamma correction.
                // Gamma 2: sqrt() converts linear -> sRGB, then scale to [0,255].
                let scaled = pixel_color * camera_snapshot.pixel_samples_scale;
                chunk[0] = (256.0 * linear_to_gamma(scaled.x).clamp(0.0, 0.999)) as u8;
                chunk[1] = (256.0 * linear_to_gamma(scaled.y).clamp(0.0, 0.999)) as u8;
                chunk[2] = (256.0 * linear_to_gamma(scaled.z).clamp(0.0, 0.999)) as u8;
            });

        info!("camera render finished");

        (self.image_width as u32, self.image_height as u32, output)
    }

    /// Returns a random jitter offset inside the pixel cell.
    /// TODO(cleanup): remove if stratified-only sampling remains default path.
    fn sample_square<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Vec3 {
        Vec3::from(rng.random::<f64>() - 0.5, rng.random::<f64>() - 0.5, 0.)
    }

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

    /// Reference CPU Monte-Carlo path-tracing integrator.
    ///
    /// Iteratively traces/scatters up to `depth` bounces and multiplies
    /// attenuation along the path. Returns sky/background gradient on miss.
    ///
    /// TODO(renderer-abstraction): extract this integrator behind a renderer trait
    /// so multiple pipelines (GPU/raster/hybrid/SDF/displacement-aware) can coexist.
    fn ray_color<R: rand::Rng + ?Sized>(
        &self,
        initial_ray: &Ray,
        depth: i32,
        world: &dyn Hittable,
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

                if accumulated_attenuation
                    .x
                    .max(accumulated_attenuation.y)
                    .max(accumulated_attenuation.z)
                    < 1e-6
                {
                    return accumulated_color;
                }
                // If we've exceeded the ray bounce limit, no more light is gathered.
                if bounce >= 4 {
                    // Russian Roulette
                    let survival = accumulated_attenuation
                        .x
                        .max(accumulated_attenuation.y)
                        .max(accumulated_attenuation.z)
                        .clamp(0.05, 0.95);
                    if rng.random::<f64>() > survival {
                        return accumulated_color;
                    }
                    accumulated_attenuation /= survival;
                }

                if let Some(scatter) = record.material.scatter(&ray, &record, rng) {
                    let scattering_pdf =
                        record
                            .material
                            .scattering_pdf(&ray, &record, &scatter.scattered);

                    match record.material {
                        Material::Lambertian { tex: _ } => {
                            // RT:ROYL sec 6.2 "uniform PDF instead of perfect match":
                            // sample uniformly over hemisphere => p(omega) = 1 / (2pi).
                            let pdf_val = 1. / (2. * f64::consts::PI);
                            accumulated_attenuation =
                                (accumulated_attenuation * scatter.attenuation * scattering_pdf)
                                    / pdf_val;
                        }
                        _ => {
                            // Non-Lambertian materials in this integrator path are treated as
                            // non-PDF-tracked events (e.g., specular-like scattering).
                            accumulated_attenuation = accumulated_attenuation * scatter.attenuation;
                        }
                    }
                    ray = scatter.scattered;
                } else {
                    return accumulated_color;
                }
            } else {
                // // If the ray hits nothing, return the background color
                // let unit_direction = unit_vector(ray.direction);
                // let t = 0.5 * (unit_direction.y + 1.0);
                // // The background gradient
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

#[inline(always)]
/// Converts a linear color channel to gamma-corrected (gamma=2) space.
fn linear_to_gamma(linear_component: f64) -> f64 {
    if linear_component > 0. {
        linear_component.sqrt()
    } else {
        0.
    }
}
