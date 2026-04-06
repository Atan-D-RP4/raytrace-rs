use rayon::prelude::*;

use crate::Interval;
use crate::hittable::Hittable;
use crate::ray::Ray;
use crate::vec3::{Color3, Point3, Vec3, cross, random_in_unit_disk, unit_vector};

const INTENSITY: Interval = Interval::from(0., 0.999);

#[derive(Default, Clone, Copy)]
pub struct Camera {
    pub aspect_ratio: f64,      // Ratio of the image width to height.
    pub image_width: i32,       // Rendered image width in pixels.
    pub image_height: i32,      // Rendered image height in pixels.
    pub samples_per_pixel: i32, // Number of rays to sample per pixel for anti-aliasing.
    pub max_depth: i32,         // Maximum ray bounce depth for ray color calculations.
    pub vfov: f64,              // Vertical field of view in degrees.
    pub look_from: Point3,      // Camera position in world space.
    pub look_at: Point3,        // Point in world space that the camera is looking at.
    pub vup: Vec3, // "Up" direction for the camera, used to determine the camera's orientation.
    pub defocus_angle: f64, // Angle in degrees representing the amount of defocus for depth of field effects.
    pub focus_distance: f64, // Distance from the camera to the focal plane for depth of field effects.

    defocus_disk_u: Vec3, // Vector representing the horizontal defocus disk for depth of field effects.
    defocus_disk_v: Vec3, // Vector representing the vertical defocus disk for depth
    pixel00_loc: Point3,
    pixel_delta_u: Point3,
    pixel_delta_v: Point3,
    pixel_samples_scale: f64,
}

impl Camera {
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
        // Vector across viewport vertical edge, negated because the v vector points up but the image coordinates increase downwards.
        let viewport_v = viewport_height * -v;

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

    pub fn render(&mut self, world: &dyn Hittable) -> Vec<u8> {
        self.initialize();
        let camera_snapshot = *self;
        let total_pixels = self.image_height * self.image_width;

        // Header size: P6\nwidth height\n255\n (roughly 20-30 bytes)
        // Pixel data: 3 bytes per pixel
        let header_size = 32;
        let pixel_data_size = total_pixels as usize * 3;
        let mut output = Vec::with_capacity(header_size + pixel_data_size);
        write_header(&mut output, self.image_width, self.image_height);

        let pixels: Vec<Vec<u8>> = (0..total_pixels)
            .into_par_iter()
            .map(|idx| {
                let i = idx % camera_snapshot.image_width;
                let j = idx / camera_snapshot.image_width;

                let pixel_color = (0..camera_snapshot.samples_per_pixel).fold(
                    Color3::from(0., 0., 0.),
                    |acc, _| {
                        let u = i as f64 + rand::random::<f64>();
                        let v = j as f64 + rand::random::<f64>();

                        let ray = camera_snapshot.get_ray(u, v);
                        acc + camera_snapshot.ray_color(&ray, camera_snapshot.max_depth, world)
                    },
                );

                let mut buffer = Vec::with_capacity(3);
                write_color(
                    &mut buffer,
                    pixel_color * camera_snapshot.pixel_samples_scale,
                );
                buffer
            })
            .collect();

        pixels.iter().for_each(|pixel| output.extend(pixel));

        output
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

#[inline(always)]
fn write_color(buffer: &mut Vec<u8>, pixel_color: Color3) {
    // Apply a linear to gamma transform for gamma 2
    let mut rbyte = linear_to_gamma(pixel_color.x);
    let mut gbyte = linear_to_gamma(pixel_color.y);
    let mut bbyte = linear_to_gamma(pixel_color.z);

    // Translate the [0,1] component values to the byte range [0,255].
    rbyte = 256.0 * INTENSITY.clamp(rbyte);
    gbyte = 256.0 * INTENSITY.clamp(gbyte);
    bbyte = 256.0 * INTENSITY.clamp(bbyte);

    buffer.push(rbyte as u8);
    buffer.push(gbyte as u8);
    buffer.push(bbyte as u8);
}

fn write_header(buffer: &mut Vec<u8>, width: i32, height: i32) {
    let header = format!("P6\n{} {}\n255\n", width, height);
    buffer.extend(header.as_bytes());
}
