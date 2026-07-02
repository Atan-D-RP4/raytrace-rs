use crate::ray::Ray;
use crate::vec3::{Color3, Point3, Vec3};

use super::{Camera, CameraRay};

/// User-facing camera configuration.
///
/// This is scene/build-time data. Runtime/precomputed values live in [`Camera`].
#[derive(Default, Clone, Copy)]
pub struct CameraConfig {
    pub image_width: i32,       // Rendered image width in pixels
    pub aspect_ratio: f64,      // Image width / height
    pub samples_per_pixel: i32, // Rays per pixel for anti-aliasing
    pub max_depth: u32,         // Maximum ray bounce depth
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
pub struct PerspectiveCamera {
    /// Rendered image width in pixels
    image_width: i32,
    /// Computed image height in pixels (derived from width/aspect_ratio)
    image_height: i32,

    /// Image width / height
    aspect_ratio: f64,
    /// Rays per pixel for anti-aliasing
    samples_per_pixel: i32,
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

impl PerspectiveCamera {
    /// Creates a new perspective camera from the given configuration, precomputing derived fields for efficient ray generation.
    pub fn from_config(config: &CameraConfig) -> Self {
        let mut cam = Self {
            image_width: config.image_width,
            image_height: 0, // Will be computed in initialize()
            aspect_ratio: config.aspect_ratio,
            samples_per_pixel: config.samples_per_pixel,
            vfov: config.vfov,
            look_from: config.look_from,
            look_at: config.look_at,
            vup: config.vup,
            defocus_angle: config.defocus_angle,
            focus_distance: config.focus_distance,

            defocus_disk_u: Vec3::default(),
            defocus_disk_v: Vec3::default(),
            pixel00_loc: Point3::default(),
            pixel_delta_u: Point3::default(),
            pixel_delta_v: Point3::default(),
            pixel_samples_scale: 1.0 / (config.samples_per_pixel as f64),
        };
        cam.initialize();
        cam
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
}

impl Camera for PerspectiveCamera {
    fn generate_ray(&self, sample: &super::CameraSampler) -> Option<super::CameraRay> {
        let (i, j) = sample.pixel;

        // Anti-Aliasing Jitter: Randomize the ray direction within the pixel by adding a random
        // offset in [0,1) to the pixel coordinates.
        let pixel_sampler = self.pixel00_loc
            + (i as f64 + sample.jitter.0) * self.pixel_delta_u
            + (j as f64 + sample.jitter.1) * self.pixel_delta_v;

        let ray_origin = if self.defocus_angle <= 0. {
            self.look_from
        } else {
            // Depth of field: Randomize ray origin within a disk on the lens.
            // Sampling a point on the defocus disk for it.
            let r = sample.lens.0.sqrt(); // Square root for uniform disk sampling
            let theta = sample.lens.1 * 2.0 * std::f64::consts::PI;
            let (sin_theta, cos_theta) = theta.sin_cos();
            let px = r * cos_theta;
            let py = r * sin_theta;
            self.look_from + (px * self.defocus_disk_u) + (py * self.defocus_disk_v)
        };

        let ray_direction = pixel_sampler - ray_origin;
        Some(CameraRay {
            ray: Ray::new_with_time(ray_origin, ray_direction, sample.time),
            weight: Color3::from(1.0, 1.0, 1.0),
        })
    }

    fn image_resolution(&self) -> (u32, u32) {
        (self.image_width as u32, self.image_height as u32)
    }
}
