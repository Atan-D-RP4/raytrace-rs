use rayon::prelude::*;

use crate::Interval;
use crate::hittable::Hittable;
use crate::ray::Ray;
use crate::vec3::{Color3, Point3, Vec3, unit_vector};

fn write_color(buffer: &mut String, color: Color3) {
    let icolor = color * 255.999;
    buffer.push_str(
        format!(
            "{} {} {}\n",
            icolor.x as i32, icolor.y as i32, icolor.z as i32
        )
        .as_str(),
    );
}

#[derive(Default)]
pub struct Camera {
    pub aspect_ratio: f64,
    pub image_width: i32,
    pub image_height: i32,
    pub focal_length: f64,
    pub viewport_height: f64,
    center: Point3,
    pixel00_loc: Point3,
    pixel_delta_u: Point3,
    pixel_delta_v: Point3,
}

impl Camera {
    pub fn from(
        aspect_ratio: f64,
        image_width: i32,
        viewport_height: f64,
        focal_length: f64,
    ) -> Self {
        Self {
            aspect_ratio,
            image_width,
            viewport_height,
            focal_length,
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

        let viewport_upper_left = self.center
            - Vec3::from(0., 0., self.focal_length)
            - (viewport_u / 2.0)
            - (viewport_v / 2.0);

        self.pixel00_loc = viewport_upper_left + 0.5 * (self.pixel_delta_u + self.pixel_delta_v);
    }

    pub fn render(&mut self, world: &Vec<Box<dyn Hittable>>) -> String {
        self.initialize();

        let image_height = self.image_height;
        let image_width = self.image_width;
        let pixel00_loc = self.pixel00_loc;
        let pixel_delta_u = self.pixel_delta_u;
        let pixel_delta_v = self.pixel_delta_v;
        let camera_center = self.center;

        (0..image_height)
            .into_par_iter()
            .map(|iter_col| {
                let mut row = String::new();
                (0..image_width).for_each(|iter_row| {
                    let pixel_center = pixel00_loc
                        + (iter_row as f64 * pixel_delta_u)
                        + (iter_col as f64 * pixel_delta_v);

                    let ray_direction = pixel_center - camera_center;
                    let ray = Ray::new(camera_center, ray_direction);

                    let color = self.ray_color(&ray, world);
                    write_color(&mut row, color);
                });
                row
            })
            .collect()
    }

    pub fn ray_color(&self, ray: &Ray, world: &Vec<Box<dyn Hittable>>) -> Color3 {
        if let Some(record) = world.hit(ray, Interval::from(0., f64::INFINITY)) {
            return 0.5 * (record.normal + Color3::from(1., 1., 1.));
        }

        let unit_direction = unit_vector(ray.direction);
        let unit_vector = Vec3::from(1.0, 1.0, 1.0);
        let gradient_vector = Vec3::from(0.5, 0.7, 1.0);

        let t = 0.5 * (unit_direction.y + 1.0);
        ((1.0 - t) * unit_vector) + (t * gradient_vector)
    }
}
