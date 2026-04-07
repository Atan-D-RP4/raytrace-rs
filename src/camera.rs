use std::sync::Arc;

use rayon::prelude::*;

use crate::hittable::Hittable;
use crate::interval::Interval;
use crate::ray::Ray;
use crate::vec3::{Color3, Point3, Vec3, cross, random_in_unit_disk, unit_vector};

#[derive(Default, Clone, Copy)]
pub struct CameraConfig {
    pub aspect_ratio: f64,      // Image width / height
    pub image_width: i32,       // Rendered image width in pixels
    pub samples_per_pixel: i32, // Rays per pixel for anti-aliasing
    pub max_depth: i32,         // Maximum ray bounce depth
    pub vfov: f64,              // Vertical field of view (degrees)
    pub look_from: Point3,      // Camera position
    pub look_at: Point3,        // Look target
    pub vup: Vec3,              // Up direction
    pub defocus_angle: f64,     // Depth of field angle
    pub focus_distance: f64,    // Focal plane distance
}

impl CameraConfig {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn aspect_ratio(mut self, ratio: f64) -> Self {
        self.aspect_ratio = ratio;
        self
    }

    pub fn image_width(mut self, width: i32) -> Self {
        self.image_width = width;
        self
    }

    pub fn samples_per_pixel(mut self, samples: i32) -> Self {
        self.samples_per_pixel = samples;
        self
    }

    pub fn max_depth(mut self, depth: i32) -> Self {
        self.max_depth = depth;
        self
    }

    pub fn vfov(mut self, vfov: f64) -> Self {
        self.vfov = vfov;
        self
    }

    pub fn look_from(mut self, point: Point3) -> Self {
        self.look_from = point;
        self
    }

    pub fn look_at(mut self, point: Point3) -> Self {
        self.look_at = point;
        self
    }

    pub fn vup(mut self, vup: Vec3) -> Self {
        self.vup = vup;
        self
    }

    pub fn defocus_angle(mut self, angle: f64) -> Self {
        self.defocus_angle = angle;
        self
    }

    pub fn focus_distance(mut self, distance: f64) -> Self {
        self.focus_distance = distance;
        self
    }
}

#[derive(Default, Clone, Copy)]
pub struct Camera {
    aspect_ratio: f64,
    image_width: i32,
    image_height: i32,
    samples_per_pixel: i32,
    max_depth: i32,
    vfov: f64,
    look_from: Point3,
    look_at: Point3,
    vup: Vec3,
    defocus_angle: f64,
    focus_distance: f64,

    defocus_disk_u: Vec3,
    defocus_disk_v: Vec3,
    pixel00_loc: Point3,
    pixel_delta_u: Point3,
    pixel_delta_v: Point3,
    pixel_samples_scale: f64,
}

impl Camera {
    pub fn from_config(config: &CameraConfig) -> Self {
        let mut camera = Self {
            aspect_ratio: config.aspect_ratio,
            image_width: config.image_width,
            image_height: 0,
            samples_per_pixel: config.samples_per_pixel,
            max_depth: config.max_depth,
            vfov: config.vfov,
            look_from: config.look_from,
            look_at: config.look_at,
            vup: config.vup,
            defocus_angle: config.defocus_angle,
            focus_distance: config.focus_distance,
            ..Default::default()
        };
        camera.initialize();
        camera
    }

    pub fn new() -> Self {
        Default::default()
    }

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
    pub fn render(&mut self, world: Arc<dyn Hittable>) -> (u32, u32, Vec<u8>) {
        self.initialize();
        let camera_snapshot = *self;
        let total_pixels = self.image_height * self.image_width;

        // Single flat buffer: avoids millions of small Vec allocations.
        // Format: [R, G, B, R, G, B, ...] for each pixel.
        let mut output = vec![0u8; total_pixels as usize * 3];

        // Parallel chunked writes: each thread writes RGB directly to its slice.
        // par_chunks_mut(3) gives mutable slices of exactly 3 bytes (one pixel).
        output
            .par_chunks_mut(3)
            .enumerate()
            .for_each(|(idx, chunk)| {
                // Convert flat index to 2D pixel coordinates.
                // Image coordinate origin: top-left (i increases right, j increases down).
                let i = idx as i32 % camera_snapshot.image_width;
                let j = idx as i32 / camera_snapshot.image_width;

                // Accumulate samples for anti-aliasing.
                // Each sample uses a randomly offset ray within the pixel area.
                let pixel_color = (0..camera_snapshot.samples_per_pixel).fold(
                    Color3::from(0., 0., 0.),
                    |acc, _| {
                        // Random offset in [0,1) places sample anywhere in pixel cell.
                        let u = i as f64 + rand::random::<f64>();
                        let v = j as f64 + rand::random::<f64>();

                        let ray = camera_snapshot.get_ray(u, v);
                        acc + camera_snapshot.ray_color(&ray, camera_snapshot.max_depth, &*world)
                    },
                );

                // Scale by sample count and apply gamma correction.
                // Gamma 2: sqrt() converts linear -> sRGB, then scale to [0,255].
                let scaled = pixel_color * camera_snapshot.pixel_samples_scale;
                chunk[0] = (256.0 * linear_to_gamma(scaled.x).clamp(0.0, 0.999)) as u8;
                chunk[1] = (256.0 * linear_to_gamma(scaled.y).clamp(0.0, 0.999)) as u8;
                chunk[2] = (256.0 * linear_to_gamma(scaled.z).clamp(0.0, 0.999)) as u8;
            });

        (self.image_width as u32, self.image_height as u32, output)
    }

    // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
    fn sample_square(&self) -> Vec3 {
        Vec3::from(rand::random::<f64>() - 0.5, rand::random::<f64>() - 0.5, 0.)
    }

    // Construct a camera ray originating from the origin and directed at randomly sampled
    // point around the pixel location i, j.
    fn get_ray(&self, u: f64, v: f64) -> Ray {
        let offset = self.sample_square();
        let pixel_sample = self.pixel00_loc
            + ((u + offset.x) * self.pixel_delta_u)
            + ((v + offset.y) * self.pixel_delta_v);

        let ray_origin = if self.defocus_angle <= 0. {
            self.look_from
        } else {
            self.defocus_disk_sample()
        };
        let ray_direction = pixel_sample - ray_origin;

        // The center of the camera is the ray origin, and the ray direction is the vector from the
        // camera center to the pixel sample location.
        Ray::new_with_time(ray_origin, ray_direction, rand::random::<f64>())
    }

    // Monte-Carlo Path Tracing implementation
    pub fn ray_color(&self, initial_ray: &Ray, depth: i32, world: &dyn Hittable) -> Color3 {
        let mut ray = *initial_ray;
        let mut accumulated_attenuation = Color3::from(1., 1., 1.);

        for _ in 0..depth {
            if let Some(record) = world.hit(&ray, Interval::from(0.001, f64::INFINITY)) {
                if let Some(scatter) = record.material.scatter(&ray, &record) {
                    accumulated_attenuation = accumulated_attenuation * scatter.attenuation;
                    ray = scatter.scattered;
                } else {
                    return Color3::from(0., 0., 0.);
                }
            } else {
                // Background gradient
                let unit_direction = unit_vector(ray.direction);
                let t = 0.5 * (unit_direction.y + 1.0);
                let background =
                    ((1.0 - t) * Vec3::from(1.0, 1.0, 1.0)) + (t * Vec3::from(0.5, 0.7, 1.0));
                return accumulated_attenuation * background;
            }
        }

        Color3::from(0., 0., 0.)
    }

    fn defocus_disk_sample(&self) -> Vec3 {
        let point = random_in_unit_disk();
        self.look_from + (point.x * self.defocus_disk_u) + (point.y * self.defocus_disk_v)
    }
}

#[inline(always)]
fn linear_to_gamma(linear_component: f64) -> f64 {
    if linear_component > 0. {
        linear_component.sqrt()
    } else {
        0.
    }
}
