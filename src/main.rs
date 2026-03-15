mod hittable;
mod interval;
mod ray;
mod sphere;
mod vec3;

use std::fs::OpenOptions;
use std::io::Write;

use hittable::Hittable;
use interval::Interval;
use ray::Ray;
use sphere::Sphere;
use vec3::{Color3, Point3, Vec3, unit_vector};

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

fn ray_color(ray: &Ray, world: &Vec<Box<dyn Hittable>>) -> Color3 {
    if let Some(record) = world.hit(ray, Interval::from(0., f64::INFINITY)) {
        return 0.5 * (record.normal + Color3::from(1., 1., 1.));
    }

    let unit_direction = unit_vector(ray.direction);
    let unit_vector = Vec3::from(1.0, 1.0, 1.0);
    let gradient_vector = Vec3::from(0.5, 0.7, 1.0);

    let t = 0.5 * (unit_direction.y + 1.0);
    ((1.0 - t) * unit_vector) + (t * gradient_vector)
}

fn main() {
    let image_width = 800;
    let aspect_ratio = 16.0 / 9.0;
    let image_height = ((image_width as f64 / aspect_ratio) as usize).max(1);

    let viewport_height = 2.0;
    let viewport_width = (image_width as f64 / image_height as f64) * viewport_height;
    let focal_length = 0.5;
    let camera_center = Point3::new();

    let viewport_u = Vec3::from(viewport_width, 0., 0.);
    let viewport_v = Vec3::from(0., -viewport_height, 0.);

    let pixel_delta_u = viewport_u / image_width as f64;
    let pixel_delta_v = viewport_v / image_height as f64;

    let viewport_upper_left =
        camera_center - Vec3::from(0., 0., focal_length) - (viewport_u / 2.0) - (viewport_v / 2.0);

    let pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    // Create or open a file
    let file_path = "output.ppm";
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(file_path)
        .expect("Unable to create or open file");

    file.write_all(format!("P3\n{image_width} {image_height}\n255\n").as_bytes())
        .expect("Unable to write to file");

    let mut output = String::new();

    let world: Vec<Box<dyn Hittable>> = vec![
        Box::new(Sphere::new(&Point3::from(-1.0, 0., -1.), 0.5)),
        Box::new(Sphere::new(&Point3::from(1., 0., -1.), 0.5)),
    ];

    (0..image_height).for_each(|j| {
        print!("\rScanlines remaining: {}", image_height - j);
        std::io::stdout().flush().unwrap();
        (0..image_width).for_each(|i| {
            let pixel_center =
                pixel00_loc + (i as f64 * pixel_delta_u) + (j as f64 * pixel_delta_v);
            let ray_direction = pixel_center - camera_center;

            let ray = Ray::new(camera_center, ray_direction);
            let color = ray_color(&ray, &world);

            write_color(&mut output, color);
        });
    });

    println!();
    println!("\rDone.");
    file.write_all(output.as_bytes())
        .expect("Unable to write to file");
}
