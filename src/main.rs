mod camera;
mod hittable;
mod interval;
mod material;
mod ray;
mod sphere;
mod vec3;

use std::fs::OpenOptions;
use std::io::Write;

use hittable::Hittable;
use interval::Interval;
use sphere::Sphere;
use vec3::{Point3, Vec3};

use self::camera::Camera;
use self::material::Material;
use self::vec3::Color3;

fn main() {
    let material_ground = Material::Lambertian {
        albedo: Color3::from(0.8, 0.8, 0.0),
    };
    let material_center = Material::Lambertian {
        albedo: Color3::from(0.1, 0.2, 0.5),
    };
    let material_left = Material::Dielectric {
        refractive_idx: 1.50,
    };
    let material_bubble = Material::Dielectric {
        refractive_idx: 1.0 / 1.50,
    };
    let material_right = Material::Metal {
        albedo: Color3::from(0.8, 0.6, 0.2),
        fuzz: 1.0,
    };

    let world: Vec<Box<dyn Hittable>> = vec![
        Box::new(Sphere::new(
            &Point3::from(0., -100.5, -1.),
            100.,
            &material_ground,
        )),
        Box::new(Sphere::new(
            &Point3::from(0., 0., -1.2),
            0.5,
            &material_center,
        )),
        Box::new(Sphere::new(
            &Point3::from(-1., 0., -1.),
            0.5,
            &material_left,
        )),
        Box::new(Sphere::new(
            &Point3::from(-1., 0., -1.),
            0.4,
            &material_bubble,
        )),
        Box::new(Sphere::new(
            &Point3::from(1., 0., -1.),
            0.5,
            &material_right,
        )),
    ];

    let mut camera = Camera::new();

    camera.samples_per_pixel = 25;

    camera.image_width = 800;
    camera.aspect_ratio = 16.0 / 9.0;
    camera.max_depth = 50;

    camera.vfov = 20.0;
    camera.look_from = Point3::from(-2., 2., 1.);
    camera.look_at = Point3::from(0., 0., -1.);
    camera.vup = Vec3::from(0., 1., 0.);

    camera.defocus_angle = 10.0;
    camera.focus_distance = 3.4;

    let rendered_buffer = camera.render(&world);

    // Create or open a file
    let file_path = "output.ppm";
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(file_path)
        .expect("Unable to create or open file");

    println!();
    file.write_all(&rendered_buffer)
        .expect("Unable to write to file");
}
