use std::sync::Arc;

use rand::RngExt;

use crate::camera::CameraConfig;
use crate::const_medium::ConstantMedium;
use crate::hittable::Hittable;
use crate::material::Material;
use crate::quad::{self, Quad};
use crate::sphere::Sphere;
use crate::texture::{
    CheckerTexture, ImageTexture, MappedTexture, NoiseTexture, SolidColor, Texture, TextureMapping,
};
use crate::transform::{RotateY, TransformObject, Translate};
use crate::vec3::{Color3, Point3, Vec3};

fn checker_texture(scale: f64, even: Color3, odd: Color3) -> Arc<dyn Texture> {
    Arc::new(MappedTexture::new(
        TextureMapping::point_scale_uniform(scale),
        Arc::new(CheckerTexture::from_color(even, odd)),
    ))
}

pub struct Scene {
    config: CameraConfig,
    objects: Vec<Arc<dyn Hittable>>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            config: CameraConfig::new(),
            objects: Vec::new(),
        }
    }

    pub fn with_config(mut self, config: CameraConfig) -> Self {
        self.config = config;
        self
    }

    pub fn config(&self) -> &CameraConfig {
        &self.config
    }

    pub fn into_objects(self) -> Vec<Arc<dyn Hittable>> {
        self.objects
    }

    pub fn aspect_ratio(mut self, ratio: f64) -> Self {
        self.config.aspect_ratio = ratio;
        self
    }

    pub fn image_width(mut self, width: i32) -> Self {
        self.config.image_width = width;
        self
    }

    pub fn samples_per_pixel(mut self, samples: i32) -> Self {
        self.config.samples_per_pixel = samples;
        self
    }

    pub fn max_depth(mut self, depth: i32) -> Self {
        self.config.max_depth = depth;
        self
    }

    pub fn vfov(mut self, vfov: f64) -> Self {
        self.config.vfov = vfov;
        self
    }

    pub fn look_from(mut self, point: Point3) -> Self {
        self.config.look_from = point;
        self
    }

    pub fn look_at(mut self, point: Point3) -> Self {
        self.config.look_at = point;
        self
    }

    pub fn vup(mut self, vup: Vec3) -> Self {
        self.config.vup = vup;
        self
    }

    pub fn defocus_angle(mut self, angle: f64) -> Self {
        self.config.defocus_angle = angle;
        self
    }

    pub fn focus_distance(mut self, distance: f64) -> Self {
        self.config.focus_distance = distance;
        self
    }
}

impl Scene {
    pub fn add_sphere(&mut self, center: Point3, radius: f64, material: Material) {
        self.objects
            .push(Arc::new(Sphere::new(&center, radius, material)));
    }

    pub fn add_quad(&mut self, Q: Point3, u: Vec3, v: Vec3, material: Material) {
        self.objects.push(Arc::new(Quad::new(Q, u, v, material)));
    }

    pub fn add_sphere_moving(
        &mut self,
        center_start: Point3,
        center_end: Point3,
        radius: f64,
        material: Material,
    ) {
        self.objects.push(Arc::new(Sphere::new_moving(
            &center_start,
            &center_end,
            radius,
            material,
        )));
    }
}

impl Scene {
    pub fn cornell_box_const_meds() -> Self {
        let mut scene = Scene::empty_cornell_box();

        let white = Material::Lambertian {
            tex: Arc::new(SolidColor::new(Color3::from(0.73, 0.73, 0.73))),
        };

        let box_params = [
            (
                Vec3::from(165., 330., 165.),
                Vec3::from(265., 0., 295.),
                15.,
                white.clone(),
                Material::Isotropic {
                    tex: Arc::new(SolidColor::new(Color3::from(0.00, 0.00, 0.00))),
                },
            ),
            (
                Vec3::from(165., 165., 165.),
                Vec3::from(130., 0., 65.),
                -18.,
                white,
                Material::Isotropic {
                    tex: Arc::new(SolidColor::new(Color3::from(1., 1., 1.))),
                },
            ),
        ];

        let boxes = box_params
            .iter()
            .map(|(size, translate_vec, rotate_angle, mat, phase_fn)| {
                let dx = Vec3::from(size.x, 0., 0.);
                let dy = Vec3::from(0., size.y, 0.);
                let dz = Vec3::from(0., 0., size.z);

                let quad_box: Vec<Arc<dyn Hittable>> = vec![
                    Arc::new(Quad::new(Point3::from(0., 0., size.z), dx, dy, mat.clone())),
                    Arc::new(Quad::new(
                        Point3::from(size.x, 0., size.z),
                        -dz,
                        dy,
                        mat.clone(),
                    )),
                    Arc::new(Quad::new(
                        Point3::from(size.x, 0., 0.),
                        -dx,
                        dy,
                        mat.clone(),
                    )),
                    Arc::new(Quad::new(Point3::from(0., 0., 0.), dz, dy, mat.clone())),
                    Arc::new(Quad::new(
                        Point3::from(0., size.y, size.z),
                        dx,
                        -dz,
                        mat.clone(),
                    )),
                    Arc::new(Quad::new(Point3::from(0., 0., 0.), dx, dz, mat.clone())),
                ];

                let rotated = TransformObject::new(RotateY::new(*rotate_angle), quad_box);
                let wrapped: TransformObject<
                    Translate,
                    TransformObject<RotateY, Vec<Arc<dyn Hittable>>>,
                > = TransformObject::new(Translate::new(*translate_vec), rotated);
                let const_medium =
                    ConstantMedium::new(Arc::new(wrapped), 0.01, Arc::new(phase_fn.clone()));

                Arc::new(const_medium) as Arc<dyn Hittable>
            });

        scene.objects.extend(boxes);

        scene.config.samples_per_pixel = 200;
        scene.config.max_depth = 50;

        scene
    }

    pub fn cornell_box() -> Self {
        let mut scene = Scene::empty_cornell_box();

        let white = Material::Lambertian {
            tex: Arc::new(SolidColor::new(Color3::from(0.73, 0.73, 0.73))),
        };
        let box_params = [
            (
                Vec3::from(165., 330., 165.),
                Vec3::from(265., 0., 295.),
                15.,
                white.clone(),
            ),
            (
                Vec3::from(165., 165., 165.),
                Vec3::from(130., 0., 65.),
                -18.,
                white,
            ),
        ];

        let boxes = box_params
            .iter()
            .map(|(size, translate_vec, rotate_angle, mat)| {
                let dx = Vec3::from(size.x, 0., 0.);
                let dy = Vec3::from(0., size.y, 0.);
                let dz = Vec3::from(0., 0., size.z);

                let quad_box: Vec<Arc<dyn Hittable>> = vec![
                    Arc::new(Quad::new(Point3::from(0., 0., size.z), dx, dy, mat.clone())),
                    Arc::new(Quad::new(
                        Point3::from(size.x, 0., size.z),
                        -dz,
                        dy,
                        mat.clone(),
                    )),
                    Arc::new(Quad::new(
                        Point3::from(size.x, 0., 0.),
                        -dx,
                        dy,
                        mat.clone(),
                    )),
                    Arc::new(Quad::new(Point3::from(0., 0., 0.), dz, dy, mat.clone())),
                    Arc::new(Quad::new(
                        Point3::from(0., size.y, size.z),
                        dx,
                        -dz,
                        mat.clone(),
                    )),
                    Arc::new(Quad::new(Point3::from(0., 0., 0.), dx, dz, mat.clone())),
                ];

                let rotated = TransformObject::new(RotateY::new(*rotate_angle), quad_box);
                let wrapped: TransformObject<
                    Translate,
                    TransformObject<RotateY, Vec<Arc<dyn Hittable>>>,
                > = TransformObject::new(Translate::new(*translate_vec), rotated);

                Arc::new(wrapped) as Arc<dyn Hittable>
            });

        scene.objects.extend(boxes);

        scene.config.samples_per_pixel = 200;

        scene
    }

    pub fn empty_cornell_box() -> Self {
        let mut scene = Self::new();
        let red = Material::Lambertian {
            tex: Arc::new(SolidColor::new(Color3::from(0.65, 0.05, 0.05))),
        };
        let white = Material::Lambertian {
            tex: Arc::new(SolidColor::new(Color3::from(0.73, 0.73, 0.73))),
        };
        let green = Material::Lambertian {
            tex: Arc::new(SolidColor::new(Color3::from(0.12, 0.45, 0.15))),
        };
        let light = Material::DiffuseLight {
            tex: Arc::new(SolidColor::new(Color3::from(15.0, 15.0, 15.0))),
        };

        scene.add_quad(
            Point3::from(555., 0., 0.),
            Vec3::from(0., 555., 0.),
            Vec3::from(0., 0., 555.),
            green,
        );
        scene.add_quad(
            Point3::from(0., 0., 0.),
            Vec3::from(0., 555., 0.),
            Vec3::from(0., 0., 555.),
            red,
        );
        scene.add_quad(
            Point3::from(343., 554., 332.),
            Vec3::from(-130.0, 0., 0.),
            Vec3::from(0., 0., -105.0),
            light,
        );
        scene.add_quad(
            Point3::from(0., 0., 0.),
            Vec3::from(555., 0., 0.),
            Vec3::from(0., 0., 555.),
            white.clone(),
        );
        scene.add_quad(
            Point3::from(555., 555., 555.),
            Vec3::from(-555., 0., 0.),
            Vec3::from(0., 0., -555.),
            white.clone(),
        );
        scene.add_quad(
            Point3::from(0., 0., 555.),
            Vec3::from(555., 0., 0.),
            Vec3::from(0., 555., 0.),
            white,
        );

        scene.config.aspect_ratio = 1.0;
        scene.config.image_width = 800;
        scene.config.samples_per_pixel = 50;
        scene.config.max_depth = 50;

        scene.config.vfov = 40.0;
        scene.config.look_from = Point3::from(278., 278., -800.);
        scene.config.look_at = Point3::from(278., 278., 0.);
        scene.config.vup = Vec3::from(0., 1., 0.);

        scene.config.defocus_angle = 0.0;
        scene.config.focus_distance = 800.0;

        scene.config.background = Color3::from(0.0, 0.0, 0.0);

        scene
    }

    pub fn simple_light() -> Self {
        let mut scene = Self::noisy_spheres();
        //
        // scene.add_quad(
        //     Point3::from(3., 1., -2.),
        //     Vec3::from(2., 0., 0.),
        //     Vec3::from(0., 2., 0.),
        //     Material::DiffuseLight {
        //         tex: Arc::new(SolidColor::new(Color3::from(4.0, 0.4, 0.4))),
        //     },
        // );

        scene.add_sphere(
            Point3::from(0., 7., 0.),
            2.,
            Material::DiffuseLight {
                tex: Arc::new(SolidColor::new(Color3::from(4.0, 4.0, 4.0))),
            },
        );

        scene.config.aspect_ratio = 16. / 9.;
        scene.config.image_width = 800;
        scene.config.samples_per_pixel = 100;
        scene.config.max_depth = 50;
        scene.config.background = Color3::from(0., 0., 0.);

        scene.config.vfov = 20.;
        scene.config.look_from = Point3::from(26., 3., 6.);
        scene.config.look_at = Point3::from(0., 2., 0.);
        scene.config.vup = Vec3::from(0., 1., 0.);

        scene.config.defocus_angle = 0.;

        scene
    }

    pub fn quads() -> Self {
        let mut scene = Self::new();
        let colors = [
            Color3::from(1.0, 0.2, 0.2), // left_red - 1
            Color3::from(0.2, 1.0, 0.2), // back_green - 2
            Color3::from(0.2, 0.2, 1.0), // right_blue - 3
            Color3::from(1.0, 0.5, 0.0), // upper_orange - 4
            Color3::from(0.2, 0.8, 0.8), // lower_teal - 5
        ];

        let quad_vecs = [
            (
                Point3::from(-3., -2., 5.),
                Vec3::from(0., 0., -4.),
                Vec3::from(0., 4., 0.),
            ), // - 1
            (
                Point3::from(-2., -2., 0.),
                Vec3::from(4., 0., 0.),
                Vec3::from(0., 4., 0.),
            ), // - 2
            (
                Point3::from(3., -2., 1.),
                Vec3::from(0., 0., 4.),
                Vec3::from(0., 4., 0.),
            ), // - 3
            (
                Point3::from(-2., 3., 1.),
                Vec3::from(4., 0., 0.),
                Vec3::from(0., 0., 4.),
            ), // - 4
            (
                Point3::from(-2., -3., 5.),
                Vec3::from(4., 0., 0.),
                Vec3::from(0., 0., -4.),
            ), // - 5
        ];

        quad_vecs.iter().zip(colors).for_each(|(vecs, color)| {
            let (Q, u, v) = vecs;
            let material = Material::Lambertian {
                tex: Arc::new(SolidColor::new(color)),
            };
            scene.add_quad(*Q, *u, *v, material);
        });

        scene.config.aspect_ratio = 1.0;
        scene.config.image_width = 800;
        scene.config.samples_per_pixel = 50;
        scene.config.max_depth = 50;

        scene.config.vfov = 80.0;
        scene.config.look_from = Point3::from(0., 0., 9.);
        scene.config.look_at = Point3::from(0., 0., 0.);
        scene.config.vup = Vec3::from(0., 1., 0.);

        scene.config.defocus_angle = 0.0;

        scene.config.focus_distance = 10.0;

        scene.config.background = Color3::from(0.5, 0.7, 1.0);

        scene
    }

    pub fn noisy_spheres() -> Self {
        let mut scene = Self::new();

        let perlin_tex: Arc<dyn Texture> = Arc::new(MappedTexture::new(
            TextureMapping::point_scale_uniform(1. / 4.),
            Arc::new(NoiseTexture::new()),
        ));

        scene.add_sphere(
            Point3::from(0., -1000., 0.),
            1000.,
            Material::Lambertian {
                tex: perlin_tex.clone(),
            },
        );
        scene.add_sphere(
            Point3::from(0., 2., 0.),
            2.,
            Material::Lambertian { tex: perlin_tex },
        );

        scene.config.aspect_ratio = 16.0 / 9.0;
        scene.config.image_width = 800;
        scene.config.samples_per_pixel = 50;
        scene.config.max_depth = 50;
        scene.config.vfov = 20.0;
        scene.config.look_from = Point3::from(13., 2., 3.);
        scene.config.look_at = Point3::from(0., 0., 0.);
        scene.config.vup = Vec3::from(0., 1., 0.);
        scene.config.defocus_angle = 0.6;
        scene.config.focus_distance = 10.0;
        scene.config.background = Color3::from(0.5, 0.7, 1.0);

        scene
    }

    pub fn earth_sphere() -> Self {
        let mut scene = Self::new();

        let image_tex = match ImageTexture::new("./earthmap.jpg") {
            Ok(tex) => tex,
            Err(e) => panic!("Failed to load to image as Texture: {:?}", e),
        };
        let image_tex: Arc<dyn Texture> = Arc::new(MappedTexture::new(
            TextureMapping::Identity,
            Arc::new(image_tex),
        ));
        let checker = Material::Lambertian { tex: image_tex };

        scene.add_sphere(Point3::from(0., 0., 0.), 2., checker);

        scene.config.aspect_ratio = 16.0 / 9.0;
        scene.config.image_width = 800;
        scene.config.samples_per_pixel = 50;
        scene.config.max_depth = 50;
        scene.config.vfov = 20.0;
        scene.config.look_from = Point3::from(13., 2., 3.);
        scene.config.look_at = Point3::from(0., 0., 0.);
        scene.config.vup = Vec3::from(0., 1., 0.);
        scene.config.defocus_angle = 0.6;
        scene.config.focus_distance = 10.0;
        scene.config.background = Color3::from(0.5, 0.7, 1.0);

        scene
    }

    pub fn checkered_spheres() -> Self {
        let mut scene = Self::new();

        let checker = Material::Lambertian {
            tex: checker_texture(
                0.32,
                Color3::from(0.2, 0.4, 0.1),
                Color3::from(0.9, 0.9, 0.9),
            ),
        };
        scene.add_sphere(Point3::from(0., -10., 0.), 10., checker.clone());
        scene.add_sphere(Point3::from(0., 10., 0.), 10., checker);

        scene.config.aspect_ratio = 16.0 / 9.0;
        scene.config.image_width = 800;
        scene.config.samples_per_pixel = 50;
        scene.config.max_depth = 50;
        scene.config.vfov = 20.0;
        scene.config.look_from = Point3::from(13., 2., 3.);
        scene.config.look_at = Point3::from(0., 0., 0.);
        scene.config.vup = Vec3::from(0., 1., 0.);
        scene.config.defocus_angle = 0.6;
        scene.config.focus_distance = 10.0;
        scene.config.background = Color3::from(0.5, 0.7, 1.0);

        scene
    }

    pub fn random_world() -> Self {
        let mut scene = Self::new();

        let ground_material = Material::Lambertian {
            tex: checker_texture(
                0.32,
                Color3::from(0.2, 0.4, 0.1),
                Color3::from(0.9, 0.9, 0.9),
            ),
        };
        scene.add_sphere(Point3::from(0., -1000., 0.), 1000., ground_material);

        for a in -11..11 {
            for b in -11..11 {
                let world_seed = rand::random::<u8>();
                let center = Point3::from(
                    a as f64 + 0.9 * rand::random::<f64>(),
                    0.2,
                    b as f64 + 0.9 * rand::random::<f64>(),
                );

                if (center - Point3::from(4., 0.2, 0.)).length() > 0.9 {
                    let material = if world_seed.is_multiple_of(3) {
                        Material::Lambertian {
                            tex: Arc::new(SolidColor::new(Color3::random() * Color3::random())),
                        }
                    } else if world_seed % 3 == 1 {
                        Material::Metal {
                            albedo: Color3::random_range(0.5, 1.0),
                            fuzz: rand::random::<f64>() * 0.5,
                        }
                    } else {
                        Material::Dielectric {
                            refractive_idx: 1.5,
                        }
                    };

                    if world_seed.is_multiple_of(2) {
                        let target_center =
                            center + Vec3::from(0., rand::rng().random_range(-0.5..0.5), 0.);
                        scene.add_sphere_moving(center, target_center, 0.2, material);
                    } else {
                        scene.add_sphere(center, 0.2, material);
                    }
                }
            }
        }

        scene.add_sphere(
            Point3::from(0., 1., 0.),
            1.,
            Material::Dielectric {
                refractive_idx: 1.5,
            },
        );
        scene.add_sphere(
            Point3::from(-4., 1., 0.),
            1.,
            Material::Lambertian {
                tex: Arc::new(SolidColor::new(Color3::from(0.4, 0.2, 0.1))),
            },
        );
        scene.add_sphere(
            Point3::from(4., 1., 0.),
            1.,
            Material::Metal {
                albedo: Color3::from(0.7, 0.6, 0.5),
                fuzz: 0.0,
            },
        );

        scene.config.aspect_ratio = 16.0 / 9.0;
        scene.config.image_width = 800;
        scene.config.samples_per_pixel = 50;
        scene.config.max_depth = 50;
        scene.config.vfov = 20.0;
        scene.config.look_from = Point3::from(13., 2., 3.);
        scene.config.look_at = Point3::from(0., 0., 0.);
        scene.config.vup = Vec3::from(0., 1., 0.);
        scene.config.defocus_angle = 0.6;
        scene.config.focus_distance = 10.0;
        scene.config.background = Color3::from(0.5, 0.7, 1.0);

        scene
    }

    pub fn simple_world() -> Self {
        let mut scene = Self::new();

        let material_ground = Material::Lambertian {
            tex: Arc::new(SolidColor::new(Color3::from(0.8, 0.8, 0.0))),
        };
        let material_center = Material::Lambertian {
            tex: Arc::new(SolidColor::new(Color3::from(0.1, 0.2, 0.5))),
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

        scene.add_sphere(Point3::from(0., -100.5, -1.), 100., material_ground);
        scene.add_sphere(Point3::from(0., 0., -1.2), 0.5, material_center);
        scene.add_sphere(Point3::from(-1., 0., -1.), 0.5, material_left);
        scene.add_sphere(Point3::from(-1., 0., -1.), 0.4, material_bubble);
        scene.add_sphere(Point3::from(1., 0., -1.), 0.5, material_right);

        scene.config.samples_per_pixel = 25;
        scene.config.image_width = 800;
        scene.config.aspect_ratio = 16.0 / 9.0;
        scene.config.max_depth = 50;
        scene.config.vfov = 20.0;
        scene.config.look_from = Point3::from(-2., 2., 1.);
        scene.config.look_at = Point3::from(0., 0., -1.);
        scene.config.vup = Vec3::from(0., 1., 0.);
        scene.config.defocus_angle = 10.0;
        scene.config.focus_distance = 3.4;
        scene.config.background = Color3::from(0.5, 0.7, 1.0);

        scene
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}
