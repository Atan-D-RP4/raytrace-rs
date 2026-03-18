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

fn create_randomized_world(camera: &mut Camera) -> Vec<Box<dyn Hittable>> {
    let mut world: Vec<Box<dyn Hittable>> = Vec::new();

    let ground_material = Material::Lambertian {
        albedo: Color3::from(0.5, 0.5, 0.5),
    };
    world.push(Box::new(Sphere::new(
        &Point3::from(0., -1000., 0.),
        1000.,
        &ground_material,
    )));

    (-11..11).for_each(|a| {
        (-11..11).for_each(|b| {
            let material_flag = rand::random::<u8>();
            let center = Point3::from(
                a as f64 + 0.9 * rand::random::<f64>(),
                0.2,
                b as f64 + 0.9 * rand::random::<f64>(),
            );

            if (center - Point3::from(4., 0.2, 0.)).length() > 0.9 {
                let material = if material_flag.is_multiple_of(3) {
                    Material::Lambertian {
                        albedo: Color3::random() * Color3::random(),
                    }
                } else if material_flag % 3 == 1 {
                    Material::Metal {
                        albedo: Color3::random_range(0.5, 1.0),
                        fuzz: rand::random::<f64>() * 0.5,
                    }
                } else {
                    Material::Dielectric {
                        refractive_idx: 1.5,
                    }
                };

                world.push(Box::new(Sphere::new(&center, 0.2, &material)));
            }
        })
    });

    world.push(Box::new(Sphere::new(
        &Point3::from(0., 1., 0.),
        1.,
        &Material::Dielectric {
            refractive_idx: 1.5,
        },
    )));

    world.push(Box::new(Sphere::new(
        &Point3::from(-4., 1., 0.),
        1.,
        &Material::Lambertian {
            albedo: Color3::from(0.4, 0.2, 0.1),
        },
    )));
    world.push(Box::new(Sphere::new(
        &Point3::from(4., 1., 0.),
        1.,
        &Material::Metal {
            albedo: Color3::from(0.7, 0.6, 0.5),
            fuzz: 0.0,
        },
    )));

    camera.aspect_ratio = 16.0 / 9.0;
    camera.image_width = 1200;
    camera.samples_per_pixel = 500;
    camera.max_depth = 50;

    camera.vfov = 20.0;
    camera.look_from = Point3::from(13., 2., 3.);
    camera.look_at = Point3::from(0., 0., 0.);
    camera.vup = Vec3::from(0., 1., 0.);

    camera.defocus_angle = 0.6;
    camera.focus_distance = 10.0;

    world
}

fn create_simple_world(camera: &mut Camera) -> Vec<Box<dyn Hittable>> {
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

    world
}

fn main() {
    let mut camera = Camera::new();

    let world = create_randomized_world(&mut camera);

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
