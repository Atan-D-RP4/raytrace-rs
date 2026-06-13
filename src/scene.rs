use std::sync::Arc;

use rand::RngExt;

use crate::bvh::BvhNode;
use crate::camera::CameraConfig;
use crate::const_medium::ConstantMedium;
use crate::hittable::{Intersectable, Sampleable};
use crate::material::{IsotropicMaterial, LambertianMaterial, Material};
use crate::planar::{box3d, quad};
use crate::sampler::Sampler;
use crate::sphere::Sphere;
use crate::texture::{
    CheckerTexture, ImageTexture, MappedTexture, NoiseTexture, Texture, TextureMapping,
};
use crate::transform::{RotateY, TransformObject, Translate};
use crate::vec3::{Color3, Point3, Vec3};
use tracing::{info, trace};

fn checker_texture(scale: f64, even: Color3, odd: Color3) -> Arc<dyn Texture> {
    Arc::new(MappedTexture::new(
        TextureMapping::point_scale_uniform(scale),
        Arc::new(CheckerTexture::from_color(even, odd)),
    ))
}

pub struct Scene<S: Sampler> {
    config: CameraConfig,
    objects: Vec<Arc<dyn Intersectable>>,
    /// Geometry-only copies of emitting objects, used by the integrator
    /// for light importance sampling (HittablePDF).
    light_objects: Vec<Arc<dyn Sampleable<S>>>,
}

impl<S: Sampler + 'static> Scene<S> {
    pub fn new() -> Self {
        Self {
            config: CameraConfig::new(),
            objects: Vec::new(),
            light_objects: Vec::new(),
        }
    }

    pub fn with_config(mut self, config: CameraConfig) -> Self {
        self.config = config;
        self
    }

    pub fn config(&self) -> &CameraConfig {
        &self.config
    }

    /// Returns (world_objects, light_objects).
    ///
    /// `light_objects` are geometry-only copies of emitting primitives,
    /// used by the integrator for importance sampling (HittablePDF).
    pub fn into_objects(self) -> (Vec<Arc<dyn Intersectable>>, Vec<Arc<dyn Sampleable<S>>>) {
        (self.objects, self.light_objects)
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

impl<S: Sampler + 'static> Scene<S> {
    pub fn add_sphere(&mut self, center: Point3, radius: f64, material: Material) {
        trace!(?center, radius, "add sphere");
        if matches!(material, Material::DiffuseLight { .. }) {
            self.light_objects
                .push(Arc::new(Sphere::new(&center, radius, material.clone())));
        }
        self.objects
            .push(Arc::new(Sphere::new(&center, radius, material)));
    }

    #[allow(non_snake_case)]
    pub fn add_quad(&mut self, Q: Point3, u: Vec3, v: Vec3, material: Material) {
        trace!(?Q, ?u, ?v, "add quad");
        // Auto-detect emitters: add geometry-only copy for light importance sampling.
        if matches!(material, Material::DiffuseLight { .. }) {
            self.light_objects
                .push(Arc::new(quad(Q, u, v, material.clone())));
        }
        self.objects.push(Arc::new(quad(Q, u, v, material)));
    }

    pub fn add_sphere_moving(
        &mut self,
        center_start: Point3,
        center_end: Point3,
        radius: f64,
        material: Material,
    ) {
        trace!(?center_start, ?center_end, radius, "add moving sphere");
        if matches!(material, Material::DiffuseLight { .. }) {
            self.light_objects.push(Arc::new(Sphere::new_moving(
                &center_start,
                &center_end,
                radius,
                material.clone(),
            )));
        }
        self.objects.push(Arc::new(Sphere::new_moving(
            &center_start,
            &center_end,
            radius,
            material,
        )));
    }

    pub fn add_object(&mut self, object: Arc<dyn Intersectable>) {
        self.objects.push(object);
    }

    pub fn add_light(&mut self, light: Arc<dyn Sampleable<S>>) {
        self.objects.push(light.clone());
        self.light_objects.push(light);
    }
}

impl<S: Sampler + 'static> Scene<S> {
    pub fn complex_scene() -> Self {
        profiling::scope!("complex_scene_build");
        let mut scene = Self::new();

        let ground = Material::lambertian_color(0.48, 0.83, 0.53);

        let boxes_per_side = 20;
        let mut boxes1: Vec<Arc<dyn Intersectable>> =
            Vec::with_capacity(boxes_per_side * boxes_per_side);
        for i in 0..boxes_per_side {
            for j in 0..boxes_per_side {
                let w = 100.0;
                let x0 = -1000.0 + (i as f64 * w);
                let z0 = -1000.0 + (j as f64 * w);
                let y0 = 0.0;
                let x1 = x0 + w;
                let y1 = rand::rng().random_range(1.0..101.0);
                let z1 = z0 + w;

                let box_quads = box3d(
                    Point3::from(x0, y0, z0),
                    Point3::from(x1, y1, z1),
                    ground.clone(),
                );

                boxes1.push(Arc::new(box_quads));
            }
        }

        let boxes1_bvh = {
            let mut boxes = boxes1;
            let boxes_len = boxes.len();
            info!(
                box_count = boxes_len,
                "assembled complex_scene ground boxes"
            );
            Arc::new(BvhNode::new(&mut boxes))
        };
        scene.objects.push(boxes1_bvh);

        scene.add_quad(
            Point3::from(123., 554., 147.),
            Vec3::from(300., 0., 0.),
            Vec3::from(0., 0., 265.),
            Material::light(Color3::from(7.0, 7.0, 7.0)),
        );

        let center1 = Point3::from(400., 400., 200.);
        let center2 = center1 + Vec3::from(30., 0., 0.);
        scene.add_sphere_moving(
            center1,
            center2,
            50.,
            Material::lambertian_color(0.7, 0.3, 0.1),
        );

        scene.add_sphere(
            Point3::from(260., 150., 45.),
            50.,
            Material::dielectric(1.5),
        );

        scene.add_sphere(
            Point3::from(0., 150., 145.),
            50.,
            Material::metal(Color3::from(0.8, 0.8, 0.9), 1.0),
        );

        let boundary = Arc::new(Sphere::new(
            &Point3::from(360., 150., 145.),
            70.,
            Material::dielectric(1.5),
        ));
        scene.objects.push(boundary.clone());
        scene.objects.push(Arc::new(ConstantMedium::new_albedo(
            boundary,
            0.2,
            Color3::from(0.2, 0.4, 0.9),
        )));

        let boundary = Arc::new(Sphere::new(
            &Point3::from(0., 0., 0.),
            5000.,
            Material::dielectric(1.5),
        ));
        scene.objects.push(boundary.clone());
        scene.objects.push(Arc::new(ConstantMedium::new_albedo(
            boundary,
            0.0001,
            Color3::from(1., 1., 1.),
        )));

        let emat: Arc<dyn Texture> = match ImageTexture::new("./earthmap.jpg") {
            Ok(tex) => Arc::new(MappedTexture::new(TextureMapping::Identity, Arc::new(tex))),
            Err(e) => panic!("Failed to load earthmap.jpg for complex_scene: {e:?}"),
        };
        scene.add_sphere(
            Point3::from(400., 200., 400.),
            100.,
            Material::lambertian(emat),
        );

        let pertext: Arc<dyn Texture> = Arc::new(MappedTexture::new(
            TextureMapping::point_scale_uniform(0.2),
            Arc::new(NoiseTexture::new()),
        ));
        scene.add_sphere(
            Point3::from(220., 280., 300.),
            80.,
            Material::lambertian(pertext),
        );

        let white = Material::lambertian_color(0.73, 0.73, 0.73);
        let mut boxes2: Vec<Arc<dyn Intersectable>> = Vec::with_capacity(1000);
        for _ in 0..1000 {
            boxes2.push(Arc::new(Sphere::new(
                &Point3::random_range(0., 165.),
                10.,
                white.clone(),
            )));
        }

        let cluster: TransformObject<Translate, TransformObject<RotateY, BvhNode, S>, S> = {
            let mut boxes = boxes2;
            let boxes_len = boxes.len();
            info!(
                sphere_count = boxes_len,
                "assembled complex_scene sphere cluster"
            );
            TransformObject::new(
                Translate::new(Vec3::from(-100., 270., 395.)),
                TransformObject::new(RotateY::new(15.), BvhNode::new(&mut boxes)),
            )
        };
        scene.objects.push(Arc::new(cluster));

        scene.config.aspect_ratio = 1.0;
        scene.config.image_width = 800;
        scene.config.samples_per_pixel = 200;
        scene.config.max_depth = 50;
        scene.config.background = Color3::from(0., 0., 0.);
        scene.config.vfov = 40.0;
        scene.config.look_from = Point3::from(478., 278., -600.);
        scene.config.look_at = Point3::from(278., 278., 0.);
        scene.config.vup = Vec3::from(0., 1., 0.);
        scene.config.defocus_angle = 0.0;
        scene.config.focus_distance = 800.0;

        scene
    }

    pub fn cornell_box_const_meds() -> Self {
        let mut scene = Scene::empty_cornell_box();

        let white = Material::lambertian_color(0.73, 0.73, 0.73);

        let box_params = [
            (
                Vec3::from(165., 330., 165.),
                Vec3::from(265., 0., 295.),
                15.,
                white.clone(),
                Material::isotropic(Color3::from(0.00, 0.00, 0.00)),
            ),
            (
                Vec3::from(165., 165., 165.),
                Vec3::from(130., 0., 65.),
                -18.,
                white,
                Material::isotropic(Color3::from(1., 1., 1.)),
            ),
        ];

        let boxes = box_params
            .iter()
            .map(|(size, translate_vec, rotate_angle, mat, phase_fn)| {
                let quad_box = box3d(Point3::from(0., 0., 0.), *size, mat.clone());

                let rotated = TransformObject::new(RotateY::new(*rotate_angle), quad_box);
                let wrapped: TransformObject<
                    Translate,
                    TransformObject<RotateY, Vec<Arc<dyn Intersectable>>, S>,
                    S,
                > = TransformObject::new(Translate::new(*translate_vec), rotated);
                let const_medium =
                    ConstantMedium::new(Arc::new(wrapped), 0.01, Arc::new(phase_fn.clone()));

                Arc::new(const_medium) as Arc<dyn Intersectable>
            });

        scene.objects.extend(boxes);

        scene.config.samples_per_pixel = 200;
        scene.config.max_depth = 50;

        scene
    }

    pub fn cornell_box() -> Self {
        let mut scene = Scene::empty_cornell_box();

        let white = Material::lambertian_color(0.73, 0.73, 0.73);
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
                white.clone(),
            ),
            // A *smaller* box in front of the taller box and beside the smaller box at the front
            (
                Vec3::from(100., 100., 100.),
                Vec3::from(340., 0., 100.),
                17.,
                Material::dielectric(1.5),
            ),
        ];

        let boxes = box_params
            .iter()
            .map(|(size, translate_vec, rotate_angle, mat)| {
                let quad_box = box3d(Point3::from(0., 0., 0.), *size, mat.clone());

                let rotated = TransformObject::new(RotateY::new(*rotate_angle), quad_box);
                let wrapped: TransformObject<
                    Translate,
                    TransformObject<RotateY, Vec<Arc<dyn Intersectable>>, S>,
                    S,
                > = TransformObject::new(Translate::new(*translate_vec), rotated);

                Arc::new(wrapped) as Arc<dyn Intersectable>
            });

        scene.objects.extend(boxes);

        // Add a small sphere in the center to better visualize the light transport effects.
        scene.add_sphere(
            Point3::from(348., 400., 278.),
            40.,
            Material::metal_with_ior(Color3::from(0.8, 0.8, 0.9), 0.3, 20.0),
        );
        scene.add_sphere(
            Point3::from(200., 350., 200.),
            90.,
            Material::dielectric(1.5),
        );

        scene.config.samples_per_pixel = 200;
        scene.config.tone_map = true;

        scene
    }

    pub fn empty_cornell_box() -> Self {
        let mut scene = Self::new();
        let red = Material::lambertian_color(0.65, 0.05, 0.05);
        let white = Material::lambertian_color(0.73, 0.73, 0.73);
        let green = Material::lambertian_color(0.12, 0.45, 0.15);
        let light = Material::light(Color3::from(15.0, 15.0, 15.0));

        scene.add_quad(
            Point3::from(555., 0., 0.),
            Vec3::from(0., 0., 555.),
            Vec3::from(0., 555., 0.),
            green,
        );
        scene.add_quad(
            Point3::from(0., 555., 555.),
            Vec3::from(0., 0., -555.),
            Vec3::from(0., 555., 0.),
            red,
        );
        scene.add_quad(
            Point3::from(213., 554., 227.),
            Vec3::from(130.0, 0., 0.),
            Vec3::from(0., 0., 105.0),
            light,
        );
        scene.add_quad(
            Point3::from(0., 555., 0.),
            Vec3::from(555., 0., 0.),
            Vec3::from(0., 0., 555.),
            white.clone(),
        );
        scene.add_quad(
            Point3::from(0., 0., 555.),
            Vec3::from(555., 0., 0.),
            Vec3::from(0., 0., -555.),
            white.clone(),
        );
        scene.add_quad(
            Point3::from(555., 0., 555.),
            Vec3::from(-555., 0., 0.),
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

        scene.add_sphere(
            Point3::from(0., 7., 0.),
            2.,
            Material::light(Color3::from(4.0, 4.0, 4.0)),
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
            #[allow(non_snake_case)]
            let (Q, u, v) = vecs;
            let material = Material::lambertian_color(color.x, color.y, color.z);
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
            Material::lambertian(perlin_tex.clone()),
        );
        scene.add_sphere(
            Point3::from(0., 2., 0.),
            2.,
            Material::lambertian(perlin_tex),
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
        let checker = Material::lambertian(image_tex);

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

        let checker = Material::lambertian(checker_texture(
            0.32,
            Color3::from(0.2, 0.4, 0.1),
            Color3::from(0.9, 0.9, 0.9),
        ));
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

        let ground_material = Material::lambertian(checker_texture(
            0.32,
            Color3::from(0.2, 0.4, 0.1),
            Color3::from(0.9, 0.9, 0.9),
        ));
        scene.add_sphere(Point3::from(0., -1000., 0.), 1000., ground_material);

        for a in -21..21 {
            for b in -21..21 {
                let world_seed = rand::random::<u8>();
                let center = Point3::from(
                    a as f64 + 0.9 * rand::random::<f64>(),
                    0.2,
                    b as f64 + 0.9 * rand::random::<f64>(),
                );

                if (center - Point3::from(4., 0.2, 0.)).length() > 0.9 {
                    let rand_albedo = || Color3::random() * Color3::random();
                    let material = match world_seed % 7 {
                        0 => Material::Lambertian(LambertianMaterial {
                            albedo: rand_albedo(),
                            tex: None,
                        }),
                        1 => Material::metal_with_ior(
                            Color3::random_range(0.5, 1.0),
                            rand::random::<f64>() * 0.5,
                            2.5,
                        ),
                        2 => Material::dielectric(1.5),
                        3 => Material::Isotropic(IsotropicMaterial {
                            albedo: Color3::random(),
                            tex: None,
                        }),
                        4 => Material::glossy(Color3::random(), rand::random::<f64>(), 1.5),
                        5 => Material::coated(
                            Material::Lambertian(LambertianMaterial {
                                albedo: rand_albedo(),
                                tex: None,
                            }),
                            Material::metal(
                                Color3::random_range(0.5, 1.0),
                                rand::random::<f64>() * 0.5,
                            ),
                        ),
                        _ => Material::Lambertian(LambertianMaterial {
                            albedo: rand_albedo(),
                            tex: None,
                        })
                        .mix(
                            Material::metal(
                                Color3::random_range(0.5, 1.0),
                                rand::random::<f64>() * 0.5,
                            ),
                            rand::random::<f64>(),
                        ),
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

        scene.add_sphere(Point3::from(0., 1., 0.), 1., Material::dielectric(1.5));
        scene.add_sphere(
            Point3::from(-4., 1., 0.),
            1.,
            Material::lambertian_color(0.4, 0.2, 0.1),
        );
        scene.add_sphere(
            Point3::from(4., 1., 0.),
            1.,
            Material::metal_with_ior(Color3::from(0.7, 0.6, 0.5), 0.0, 2.5),
        );
        scene.add_light(Arc::new(Sphere::new(
            &Point3::from(0., 4.5, 0.),
            1.5,
            Material::light_textured(Arc::new(ImageTexture::new("./earthmap.jpg").unwrap())),
        )));

        scene.config.aspect_ratio = 16.0 / 9.0;
        scene.config.image_width = 800;
        scene.config.samples_per_pixel = 50;
        scene.config.max_depth = 50;
        scene.config.vfov = 40.0;
        scene.config.look_from = Point3::from(13., 2., 6.);
        scene.config.look_at = Point3::from(0., 1., 0.);
        scene.config.vup = Vec3::from(0., 1., 0.);
        scene.config.defocus_angle = 0.6;
        scene.config.focus_distance = 10.0;
        scene.config.background = Color3::from(0.5, 0.7, 1.0);

        scene
    }

    pub fn simple_world() -> Self {
        let mut scene = Self::new();

        let material_ground = Material::lambertian_color(0.8, 0.8, 0.0);
        let material_center = Material::lambertian_color(0.1, 0.2, 0.5);
        let material_left = Material::dielectric(1.50);
        let material_bubble = Material::dielectric(1.0 / 1.50);
        let material_right = Material::metal_with_ior(Color3::from(0.8, 0.6, 0.2), 1.0, 2.5);

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

    /// Test scene for the new material system: demonstrates `Mix` (painted
    /// metal), `Coated` (clear coat over substrate), and `Glossy` (GGX).
    pub fn composition_demo() -> Self {
        let mut scene = Self::new();

        // Ground plane (Lambertian).
        let ground = Material::lambertian_color(0.5, 0.5, 0.5);
        scene.add_quad(
            Point3::from(-5., 0., 0.),
            Vec3::from(10., 0., 0.),
            Vec3::from(0., 0., 10.),
            ground,
        );

        // Sphere 1: plain glossy (roughness 0.2 — tight highlight).
        let glossy = Material::glossy(Color3::from(0.9, 0.9, 0.9), 0.2, 1.5);
        scene.add_sphere(Point3::from(-2.5, 1.0, 4.0), 1.0, glossy);

        // Sphere 2: rough glossy (roughness 0.7 — broad highlight).
        let rough_glossy = Material::glossy(Color3::from(0.7, 0.3, 0.3), 0.7, 1.5);
        scene.add_sphere(Point3::from(0.0, 1.0, 4.0), 1.0, rough_glossy);

        // Sphere 3: 50/50 mix of red Lambertian and silver metal.
        let mixed = Material::lambertian_color(0.8, 0.2, 0.2)
            .mix(Material::metal(Color3::from(0.9, 0.9, 0.9), 0.0), 0.5);
        scene.add_sphere(Point3::from(2.5, 1.0, 4.0), 1.0, mixed);

        // Sphere 4: clear-coated red (dielectric coat over red Lambertian).
        let coated = Material::lambertian_color(0.2, 0.7, 0.2).coated(Material::dielectric(1.5));
        scene.add_sphere(Point3::from(-1.25, 1.0, 6.0), 1.0, coated);

        // Sphere 5: clear-coated blue.
        let coated = Material::lambertian_color(0.2, 0.2, 0.8).coated(Material::dielectric(1.5));
        scene.add_sphere(Point3::from(1.25, 1.0, 6.0), 1.0, coated);

        // Area light above.
        scene.add_quad(
            Point3::from(-3., 8., 2.),
            Vec3::from(6., 0., 0.),
            Vec3::from(0., 0., 6.),
            Material::light(Color3::from(8.0, 8.0, 8.0)),
        );

        scene.config.aspect_ratio = 16.0 / 9.0;
        scene.config.image_width = 800;
        scene.config.samples_per_pixel = 100;
        scene.config.max_depth = 50;
        scene.config.vfov = 40.0;
        scene.config.look_from = Point3::from(0., 4., 10.);
        scene.config.look_at = Point3::from(0., 1., 4.);
        scene.config.vup = Vec3::from(0., 1., 0.);
        scene.config.focus_distance = 10.0;
        scene.config.defocus_angle = 0.0;
        scene.config.background = Color3::from(0.1, 0.1, 0.1);

        scene
    }
}

impl<S: Sampler + 'static> Default for Scene<S> {
    fn default() -> Self {
        Self::new()
    }
}
