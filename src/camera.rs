use rayon::prelude::*;

use crate::Interval;
use crate::hittable::Hittable;
use crate::ray::Ray;
use crate::vec3::{Color3, Point3, Vec3, random_unit_vector, unit_vector};

const INTENSITY: Interval = Interval::from(0., 0.999);

#[inline(always)]
fn linear_to_gamma(linear_component: f64) -> f64 {
    if linear_component > 0. {
        linear_component.sqrt()
    } else {
        0.
    }
}

#[inline(always)]
fn write_color(buffer: &mut String, pixel_color: Color3) {
    // Apply a linear to gamma transform for gamma 2
    let mut rbyte = linear_to_gamma(pixel_color.x);
    let mut gbyte = linear_to_gamma(pixel_color.y);
    let mut bbyte = linear_to_gamma(pixel_color.z);

    // Translate the [0,1] component values to the byte range [0,255].
    rbyte = 256.0 * INTENSITY.clamp(rbyte);
    gbyte = 256.0 * INTENSITY.clamp(gbyte);
    bbyte = 256.0 * INTENSITY.clamp(bbyte);

    buffer.push_str(format!("{} {} {}\n", rbyte as i32, gbyte as i32, bbyte as i32).as_str());
}

#[derive(Default, Clone, Copy)]
pub struct Camera {
    pub aspect_ratio: f64,
    pub image_width: i32,
    pub image_height: i32,
    pub focal_length: f64,
    pub viewport_height: f64,
    pub samples_per_pixel: i32,
    pub max_depth: i32,
    center: Point3,
    pixel00_loc: Point3,
    pixel_delta_u: Point3,
    pixel_delta_v: Point3,
    pixel_samples_scale: f64,
}

impl Camera {
    pub fn from(
        aspect_ratio: f64,
        image_width: i32,
        viewport_height: f64,
        focal_length: f64,
        max_depth: i32,
    ) -> Self {
        Self {
            aspect_ratio,
            image_width,
            viewport_height,
            focal_length,
            samples_per_pixel: 100,
            max_depth,
            ..Default::default()
        }
    }

    fn initialize(&mut self) {
        self.image_height = ((self.image_width as f64 / self.aspect_ratio) as i32).max(1);

        self.center = Point3::new();

        let viewport_width =
            (self.image_width as f64 / self.image_height as f64) * self.viewport_height;

        let viewport_u = Vec3::from(viewport_width, 0., 0.);
        let viewport_v = Vec3::from(0., -self.viewport_height, 0.);

        self.pixel_delta_u = viewport_u / self.image_width as f64;
        self.pixel_delta_v = viewport_v / self.image_height as f64;

        self.pixel_samples_scale = 1.0 / self.samples_per_pixel as f64;

        let viewport_upper_left = self.center
            - Vec3::from(0., 0., self.focal_length)
            - (viewport_u / 2.0)
            - (viewport_v / 2.0);

        self.pixel00_loc = viewport_upper_left + 0.5 * (self.pixel_delta_u + self.pixel_delta_v);
    }

    pub fn render(&mut self, world: &Vec<Box<dyn Hittable>>) -> String {
        self.initialize();
        let camera_snapshot = *self;
        let image_height = camera_snapshot.image_height;
        let image_width = camera_snapshot.image_width;

        (0..image_height * image_width)
            .into_par_iter()
            .map(|idx| {
                let iter_row = idx % image_width;
                let iter_col = idx / image_width;

                let mut pixel_color = Color3::from(0., 0., 0.);
                (0..camera_snapshot.samples_per_pixel).for_each(|_| {
                    let ray = camera_snapshot.get_ray(iter_row as f64, iter_col as f64);
                    pixel_color += camera_snapshot.ray_color(&ray, self.max_depth, world);
                });
                let mut pixel_buffer = String::new();
                write_color(
                    &mut pixel_buffer,
                    camera_snapshot.pixel_samples_scale * pixel_color,
                );
                pixel_buffer
            })
            .collect()
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

        Ray::new(self.center, pixel_sample - self.center)
    }

    pub fn ray_color(&self, ray: &Ray, depth: i32, world: &Vec<Box<dyn Hittable>>) -> Color3 {
        // If we've exceeded the ray bounce limit, no more light is gathered.
        if depth <= 0 {
            return Color3::from(0., 0., 0.);
        }

        if let Some(record) = world.hit(ray, Interval::from(0.001, f64::INFINITY)) {
            if let Some(scatter) = record.material.scatter(ray, &record) {
                return scatter.attenuation * self.ray_color(&scatter.scattered, depth - 1, world);
            } else {
                return Color3::from(0., 0., 0.);
            }
        }

        let unit_direction = unit_vector(ray.direction);
        let unit_vector = Vec3::from(1.0, 1.0, 1.0);
        let gradient_vector = Vec3::from(0.5, 0.7, 1.0);

        let t = 0.5 * (unit_direction.y + 1.0);
        ((1.0 - t) * unit_vector) + (t * gradient_vector)
    }
}
