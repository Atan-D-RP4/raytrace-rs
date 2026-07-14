use std::sync::Arc;

use glam::Vec3;
use rand::RngExt;
use tracing::{info, trace};

use crate::bvh::BvhNode;
use crate::camera::perspective::CameraConfig;
use crate::const_medium::ConstantMedium;
use crate::environment::{EnvironmentLight, EnvironmentMap};
use crate::flat_bvh::FlatBvh;
use crate::hittable::{Intersectable, Sampleable};
use crate::material::{Bsdf, CoatedMaterial};
use crate::material::{IsotropicMaterial, LambertianMaterial, Material};
use crate::planar::{box3d, quad};
use crate::shape::{moving_sphere, sphere};
use crate::texture::mapping::TextureMapping3D;
use crate::texture::{
    CheckerTexture, ImageTexture, MappedTexture, NoiseTexture, SolidColor, Texture,
};
use crate::transform::{RotateY, TransformObject, Translate};
use crate::vec3::{Color3, Point3};

fn checker_texture(scale: f32, even: Color3, odd: Color3) -> Arc<dyn Texture> {
    let mapped_tex = MappedTexture::new(CheckerTexture::from_color(even, odd));
    let mapped_tex = mapped_tex.with_mapping3d(TextureMapping3D::point_scale_uniform(scale));
    Arc::new(mapped_tex)
}

pub struct Scene {
    /// Camera configuration for the scene.
    config: CameraConfig,
    /// All intersectable objects in the scene, including lights.
    objects: Vec<Arc<dyn Intersectable>>,
    /// Objects whose directions are worth sampling toward (area lights).
    /// Used by `EmitterPDF` for MIS. Delta materials (glass, metal) should
    /// NOT be included — they have no meaningful PDF for importance sampling.
    important_objects: Vec<Arc<dyn Sampleable>>,
    /// Optional environment map for background lighting. If present, used for
    /// rays that miss all scene geometry (includes MIS-weighted contribution
    /// for indirect bounces).
    env_map: Option<Arc<EnvironmentMap>>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            config: CameraConfig::new(),
            objects: Vec::new(),
            important_objects: Vec::new(),
            env_map: None,
        }
    }

    pub fn with_env_map(mut self, env_map: Arc<EnvironmentMap>) -> Self {
        self.important_objects
            .push(Arc::new(EnvironmentLight::new(env_map.clone())));
        self.env_map = Some(env_map);
        self
    }

    pub fn env_map(&self) -> Option<&Arc<EnvironmentMap>> {
        self.env_map.as_ref()
    }

    pub fn with_config(mut self, config: CameraConfig) -> Self {
        self.config = config;
        self
    }

    pub fn config(&self) -> &CameraConfig {
        &self.config
    }

    /// Returns `(objects, important_objects)`.
    ///
    /// `important_objects` are geometry-only copies for importance sampling
    /// via `EmitterPDF` (area lights only — delta materials excluded).
    pub fn into_objects(self) -> (Vec<Arc<dyn Intersectable>>, Vec<Arc<dyn Sampleable>>) {
        (self.objects, self.important_objects)
    }

    pub fn aspect_ratio(mut self, ratio: f32) -> Self {
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

    pub fn vfov(mut self, vfov: f32) -> Self {
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

    pub fn defocus_angle(mut self, angle: f32) -> Self {
        self.config.defocus_angle = angle;
        self
    }

    pub fn focus_distance(mut self, distance: f32) -> Self {
        self.config.focus_distance = distance;
        self
    }
}

impl Scene {
    /// Add an intersectable for intersection, optionally with a separate
    /// importance target for sampling via `EmitterPDF`.
    ///
    /// Use separate objects when the importance target needs a different
    /// material (e.g., `Material::Void` for sampling, `Material::dielectric`
    /// for refraction).
    pub fn add_intersectable(
        &mut self,
        object: Arc<dyn Intersectable>,
        importance_target: Option<Arc<dyn Sampleable>>,
    ) {
        if let Some(target) = importance_target {
            self.important_objects.push(target);
        }
        self.objects.push(object);
    }

    /// Register a sampleable as an importance target.
    ///
    /// Pushes to both `objects` (intersection) and `important_objects`
    /// (importance sampling via `EmitterPDF`).
    pub fn add_importance_target(&mut self, object: Arc<dyn Sampleable>) {
        self.important_objects.push(object.clone());
        self.objects.push(object);
    }

    pub fn add_sphere(&mut self, center: Point3, radius: f32, material: Material) {
        trace!(?center, radius, "add sphere");
        if material.is_emissive() {
            let material = Arc::new(material);
            self.add_intersectable(
                Arc::new(sphere(center, radius, material.clone())),
                Some(Arc::new(sphere(center, radius, material))),
            );
        } else {
            self.add_intersectable(Arc::new(sphere(center, radius, material)), None);
        }
    }

    #[allow(non_snake_case)]
    pub fn add_quad(&mut self, Q: Point3, u: Vec3, v: Vec3, material: Material) {
        trace!(?Q, ?u, ?v, "add quad");
        if material.is_emissive() {
            let material = Arc::new(material);
            self.add_intersectable(
                Arc::new(quad(Q, u, v, material.clone())),
                Some(Arc::new(quad(Q, u, v, material))),
            );
        } else {
            self.add_intersectable(Arc::new(quad(Q, u, v, material)), None);
        }
    }

    pub fn add_sphere_moving(
        &mut self,
        center_start: Point3,
        center_end: Point3,
        radius: f32,
        material: Material,
    ) {
        trace!(?center_start, ?center_end, radius, "add moving sphere");
        if material.is_emissive() {
            let material = Arc::new(material);
            self.add_intersectable(
                Arc::new(moving_sphere(
                    center_start,
                    center_end,
                    radius,
                    material.clone(),
                )),
                Some(Arc::new(moving_sphere(
                    center_start,
                    center_end,
                    radius,
                    material,
                ))),
            );
        } else {
            self.add_intersectable(
                Arc::new(moving_sphere(center_start, center_end, radius, material)),
                None,
            );
        }
    }

    /// Add an intersectable object for intersection only (no importance sampling).
    pub fn add_object(&mut self, object: Arc<dyn Intersectable>) {
        self.objects.push(object);
    }
}

impl Scene {
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
                let x0 = -1000.0 + (i as f32 * w);
                let z0 = -1000.0 + (j as f32 * w);
                let y0 = 0.0;
                let x1 = x0 + w;
                let y1 = rand::rng().random_range(1.0..101.0);
                let z1 = z0 + w;

                let box_quads = box3d(
                    Point3::new(x0, y0, z0),
                    Point3::new(x1, y1, z1),
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
            Arc::new(FlatBvh::from(BvhNode::new(&mut boxes)))
        };
        scene.objects.push(boxes1_bvh);

        scene.add_quad(
            Point3::new(123., 554., 147.),
            Vec3::new(300., 0., 0.),
            Vec3::new(0., 0., 265.),
            Material::light(Color3::new(7.0, 7.0, 7.0)),
        );

        let center1 = Point3::new(400., 400., 200.);
        let center2 = center1 + Vec3::new(30., 0., 0.);
        scene.add_sphere_moving(
            center1,
            center2,
            50.,
            Material::lambertian_color(0.7, 0.3, 0.1),
        );

        scene.add_sphere(Point3::new(260., 150., 45.), 50., Material::dielectric(1.5));

        scene.add_sphere(
            Point3::new(0., 150., 145.),
            50.,
            Material::metal(Color3::new(0.8, 0.8, 0.9), 1.0),
        );

        scene.objects.push(Arc::new(ConstantMedium::new_albedo(
            sphere(
                Point3::new(360., 150., 145.),
                70.,
                Material::dielectric(1.5),
            ),
            0.2,
            Color3::new(0.2, 0.4, 0.9).into_inner(),
        )));

        scene.objects.push(Arc::new(ConstantMedium::new_albedo(
            sphere(Point3::new(0., 0., 0.), 5000., Material::dielectric(1.5)),
            0.0001,
            // Color3::from(1., 1., 1.), // Pure white
            Color3::new(0.7, 0.1, 0.1).into_inner(), // A faint red tint to visualize the volume better
        )));

        let emat: Arc<dyn Texture> = match ImageTexture::new("./earthmap.png") {
            Ok(tex) => {
                let mapped_tex = MappedTexture::new(tex);
                Arc::new(mapped_tex)
            }
            Err(e) => panic!("Failed to load earthmap.png for complex_scene: {e:?}"),
        };
        scene.add_sphere(
            Point3::new(400., 200., 400.),
            100.,
            Material::lambertian(emat),
        );

        let pertext: Arc<dyn Texture> = Arc::new(
            MappedTexture::new(NoiseTexture::new())
                .with_mapping3d(TextureMapping3D::point_scale_uniform(0.2)),
        );
        scene.add_sphere(
            Point3::new(220., 280., 300.),
            80.,
            Material::lambertian(pertext),
        );

        let white = Material::lambertian_color(0.73, 0.73, 0.73);
        let mut boxes2: Vec<Arc<dyn Intersectable>> = Vec::with_capacity(1000);
        for _ in 0..1000 {
            boxes2.push(Arc::new(sphere(
                Point3::new(
                    rand::rng().random_range(0.0..165.),
                    rand::rng().random_range(0.0..165.),
                    rand::rng().random_range(0.0..165.),
                ),
                10.,
                white.clone(),
            )));
        }

        let cluster: TransformObject<Translate, TransformObject<RotateY, BvhNode>> = {
            let mut boxes = boxes2;
            let boxes_len = boxes.len();
            info!(
                sphere_count = boxes_len,
                "assembled complex_scene sphere cluster"
            );
            TransformObject::new(
                Translate::new(Vec3::new(-100., 270., 395.)),
                TransformObject::new(RotateY::new(15.), BvhNode::new(&mut boxes)),
            )
        };
        scene.objects.push(Arc::new(cluster));

        scene.config.aspect_ratio = 1.0;
        scene.config.image_width = 800;
        scene.config.samples_per_pixel = 200;
        scene.config.max_depth = 50;
        scene.config.background = Color3::new(0., 0., 0.);
        scene.config.vfov = 40.0;
        scene.config.look_from = Point3::new(478., 278., -600.);
        scene.config.look_at = Point3::new(278., 278., 0.);
        scene.config.vup = Vec3::new(0., 1., 0.);
        scene.config.defocus_angle = 0.0;
        scene.config.focus_distance = 800.0;

        scene
    }

    pub fn cornell_box_const_meds() -> Self {
        let mut scene = Scene::empty_cornell_box();

        let white = Material::lambertian_color(0.73, 0.73, 0.73);

        let box_params = [
            (
                Vec3::new(165., 330., 165.),
                Vec3::new(265., 0., 295.),
                15.,
                white.clone(),
                Material::isotropic(Color3::new(0.00, 0.00, 0.00)),
            ),
            (
                Vec3::new(165., 165., 165.),
                Vec3::new(130., 0., 65.),
                -18.,
                white,
                Material::isotropic(Color3::new(1., 1., 1.)),
            ),
        ];

        let boxes = box_params
            .iter()
            .map(|(size, translate_vec, rotate_angle, mat, phase_fn)| {
                let quad_box = box3d(Point3::new(0., 0., 0.), Point3(*size), mat.clone());

                let rotated = TransformObject::new(RotateY::new(*rotate_angle), quad_box);
                let wrapped: TransformObject<
                    Translate,
                    TransformObject<RotateY, Vec<Arc<dyn Intersectable>>>,
                > = TransformObject::new(Translate::new(*translate_vec), rotated);
                let const_medium = ConstantMedium::new(Arc::new(wrapped), 0.01, phase_fn.clone());

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
                Vec3::new(165., 330., 165.),
                Vec3::new(265., 0., 295.),
                15.,
                white.clone(),
            ),
            (
                Vec3::new(165., 165., 165.),
                Vec3::new(130., 0., 65.),
                -18.,
                white.clone(),
            ),
            // A *smaller* box in front of the taller box and beside the smaller box at the front
            (
                Vec3::new(100., 100., 100.),
                Vec3::new(340., 0., 100.),
                17.,
                Material::dielectric(1.5),
            ),
        ];

        let boxes = box_params
            .iter()
            .map(|(size, translate_vec, rotate_angle, mat)| {
                let quad_box = box3d(Point3::new(0., 0., 0.), Point3(*size), mat.clone());

                let rotated = TransformObject::new(RotateY::new(*rotate_angle), quad_box);
                let wrapped: TransformObject<
                    Translate,
                    TransformObject<RotateY, Vec<Arc<dyn Intersectable>>>,
                > = TransformObject::new(Translate::new(*translate_vec), rotated);

                Arc::new(wrapped) as Arc<dyn Intersectable>
            });

        scene.objects.extend(boxes);

        // // Add a small sphere in the center to better visualize the light transport effects.
        scene.add_sphere(
            Point3::new(348., 400., 278.),
            40.,
            Material::metal_with_ior(Color3::new(0.8, 0.8, 0.9), 0.3, 20.0),
        );
        scene.add_sphere(
            Point3::new(200., 350., 200.),
            90.,
            // Deeply tinted dielectric to better visualize the caustics and light transport through the box.
            // Values must be <= 1.0 for energy conservation (no light amplification).
            Material::dielectric_tinted(1.5, Color3::new(0.8, 0.8, 1.0)),
        );
        // Glass sphere — delta material, no importance sampling needed.
        scene.add_intersectable(
            Arc::new(sphere(
                Point3::new(200., 90., 200.),
                90.,
                Material::dielectric(1.5),
            )),
            Some(Arc::new(sphere(
                Point3::new(200., 90., 200.),
                90.,
                Material::Void,
            ))),
        );

        scene.config.samples_per_pixel = 200;
        scene.config.tone_map = true;
        scene.config.exposure = 1.8;
        scene.config.image_width = 600;

        scene
    }

    pub fn empty_cornell_box() -> Self {
        let mut scene = Self::new();
        let red = Material::lambertian_color(0.65, 0.05, 0.05);
        let white = Material::lambertian_color(0.73, 0.73, 0.73);
        let green = Material::lambertian_color(0.12, 0.45, 0.15);
        let light = Material::light(Color3::new(16.0, 16.0, 16.0));

        scene.add_quad(
            Point3::new(555., 0., 0.),
            Vec3::new(0., 0., 555.),
            Vec3::new(0., 555., 0.),
            green,
        );
        scene.add_quad(
            Point3::new(0., 0., 555.),
            Vec3::new(0., 0., -555.),
            Vec3::new(0., 555., 0.),
            red,
        );
        scene.add_quad(
            Point3::new(213., 554., 227.),
            Vec3::new(130.0, 0., 0.),
            Vec3::new(0., 0., 105.0),
            light,
        );
        scene.add_quad(
            Point3::new(0., 555., 0.),
            Vec3::new(555., 0., 0.),
            Vec3::new(0., 0., 555.),
            white.clone(),
        );
        scene.add_quad(
            Point3::new(0., 0., 555.),
            Vec3::new(555., 0., 0.),
            Vec3::new(0., 0., -555.),
            white.clone(),
        );
        scene.add_quad(
            Point3::new(555., 0., 555.),
            Vec3::new(-555., 0., 0.),
            Vec3::new(0., 555., 0.),
            white,
        );

        scene.config.aspect_ratio = 1.0;
        scene.config.image_width = 800;
        scene.config.samples_per_pixel = 50;
        scene.config.max_depth = 50;

        scene.config.vfov = 40.0;
        scene.config.look_from = Point3::new(278., 278., -800.);
        scene.config.look_at = Point3::new(278., 278., 0.);
        scene.config.vup = Vec3::new(0., 1., 0.);

        scene.config.defocus_angle = 0.0;
        scene.config.focus_distance = 800.0;

        scene.config.background = Color3::new(0.0, 0.0, 0.0);

        scene
    }

    pub fn simple_light() -> Self {
        let mut scene = Self::noisy_spheres();

        scene.add_sphere(
            Point3::new(0., 7., 0.),
            2.,
            Material::light(Color3::new(4.0, 4.0, 4.0)),
        );

        scene.config.aspect_ratio = 16. / 9.;
        scene.config.image_width = 800;
        scene.config.samples_per_pixel = 100;
        scene.config.max_depth = 50;
        scene.config.background = Color3::new(0., 0., 0.);

        scene.config.vfov = 20.;
        scene.config.look_from = Point3::new(26., 3., 6.);
        scene.config.look_at = Point3::new(0., 2., 0.);
        scene.config.vup = Vec3::new(0., 1., 0.);

        scene.config.defocus_angle = 0.;

        scene
    }

    pub fn quads() -> Self {
        let mut scene = Self::new();
        let colors = [
            Color3::new(1.0, 0.2, 0.2), // left_red - 1
            Color3::new(0.2, 1.0, 0.2), // back_green - 2
            Color3::new(0.2, 0.2, 1.0), // right_blue - 3
            Color3::new(1.0, 0.5, 0.0), // upper_orange - 4
            Color3::new(0.2, 0.8, 0.8), // lower_teal - 5
        ];

        let quad_vecs = [
            (
                Point3::new(-3., -2., 5.),
                Vec3::new(0., 0., -4.),
                Vec3::new(0., 4., 0.),
            ), // - 1
            (
                Point3::new(-2., -2., 0.),
                Vec3::new(4., 0., 0.),
                Vec3::new(0., 4., 0.),
            ), // - 2
            (
                Point3::new(3., -2., 1.),
                Vec3::new(0., 0., 4.),
                Vec3::new(0., 4., 0.),
            ), // - 3
            (
                Point3::new(-2., 3., 1.),
                Vec3::new(4., 0., 0.),
                Vec3::new(0., 0., 4.),
            ), // - 4
            (
                Point3::new(-2., -3., 5.),
                Vec3::new(4., 0., 0.),
                Vec3::new(0., 0., -4.),
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
        scene.config.look_from = Point3::new(0., 0., 9.);
        scene.config.look_at = Point3::new(0., 0., 0.);
        scene.config.vup = Vec3::new(0., 1., 0.);

        scene.config.defocus_angle = 0.0;

        scene.config.focus_distance = 10.0;

        scene.config.background = Color3::new(0.5, 0.7, 1.0);

        scene
    }

    pub fn noisy_spheres() -> Self {
        let mut scene = Self::new();

        let perlin_tex: Arc<dyn Texture> = Arc::new(
            MappedTexture::new(NoiseTexture::new())
                .with_mapping3d(TextureMapping3D::point_scale_uniform(1. / 4.)),
        );

        scene.add_sphere(
            Point3::new(0., -1000., 0.),
            1000.,
            Material::lambertian(perlin_tex.clone()),
        );
        scene.add_sphere(
            Point3::new(0., 2., 0.),
            2.,
            Material::lambertian(perlin_tex),
        );

        scene.config.aspect_ratio = 16.0 / 9.0;
        scene.config.image_width = 800;
        scene.config.samples_per_pixel = 50;
        scene.config.max_depth = 50;
        scene.config.vfov = 20.0;
        scene.config.look_from = Point3::new(13., 2., 3.);
        scene.config.look_at = Point3::new(0., 0., 0.);
        scene.config.vup = Vec3::new(0., 1., 0.);
        scene.config.defocus_angle = 0.6;
        scene.config.focus_distance = 10.0;
        scene.config.background = Color3::new(0.5, 0.7, 1.0);

        scene
    }

    pub fn earth_sphere() -> Self {
        let mut scene = Self::new();

        let image_tex = match ImageTexture::new("./earthmap.png") {
            Ok(tex) => tex,
            Err(e) => panic!("Failed to load to image as Texture: {:?}", e),
        };
        let image_tex: Arc<dyn Texture> = Arc::new(MappedTexture::new(image_tex));
        let checker = Material::lambertian(image_tex);

        scene.add_sphere(Point3::new(0., 0., 0.), 2., checker);

        scene.config.aspect_ratio = 16.0 / 9.0;
        scene.config.image_width = 800;
        scene.config.samples_per_pixel = 50;
        scene.config.max_depth = 50;
        scene.config.vfov = 20.0;
        scene.config.look_from = Point3::new(13., 2., 3.);
        scene.config.look_at = Point3::new(0., 0., 0.);
        scene.config.vup = Vec3::new(0., 1., 0.);
        scene.config.defocus_angle = 0.6;
        scene.config.focus_distance = 10.0;
        scene.config.background = Color3::new(0.5, 0.7, 1.0);

        scene
    }

    pub fn checkered_spheres() -> Self {
        let mut scene = Self::new();

        let checker = Material::lambertian(checker_texture(
            0.32,
            Color3::new(0.2, 0.4, 0.1),
            Color3::new(0.9, 0.9, 0.9),
        ));
        scene.add_sphere(Point3::new(0., -10., 0.), 10., checker.clone());
        scene.add_sphere(Point3::new(0., 10., 0.), 10., checker);

        scene.config.aspect_ratio = 16.0 / 9.0;
        scene.config.image_width = 800;
        scene.config.samples_per_pixel = 50;
        scene.config.max_depth = 50;
        scene.config.vfov = 20.0;
        scene.config.look_from = Point3::new(13., 2., 3.);
        scene.config.look_at = Point3::new(0., 0., 0.);
        scene.config.vup = Vec3::new(0., 1., 0.);
        scene.config.defocus_angle = 0.6;
        scene.config.focus_distance = 10.0;
        scene.config.background = Color3::new(0.5, 0.7, 1.0);

        scene
    }

    pub fn random_world() -> Self {
        let scene = Self::new();

        let mut scene = scene.with_env_map(Arc::new(EnvironmentMap::new(
            image::open("./kiara_1_dawn_4k.hdr").unwrap().into(),
        )));

        let ground_material = Material::coated(
            Material::lambertian(checker_texture(
                0.32,
                Color3::new(0.2, 0.4, 0.1),
                Color3::new(0.9, 0.9, 0.9),
            )),
            Material::dielectric_tinted(1.5, Color3::new(0.8, 0.8, 1.0)),
        );
        scene.add_sphere(Point3::new(0., -1000., 0.), 1000., ground_material);

        for a in -21..21 {
            for b in -21..21 {
                let world_seed = rand::random::<u8>();
                let mut center = Point3::new(
                    a as f32 + 1.4 * rand::random::<f32>(),
                    0.2,
                    b as f32 + 1.4 * rand::random::<f32>(),
                );

                if (center - Point3::new(4., 0.2, 0.)).length() > 1.4 {
                    let rand_albedo = || rand::random::<Vec3>() * rand::random::<Vec3>();
                    fn metal_color() -> Vec3 {
                        Vec3::splat(0.5) + rand::random::<Vec3>() * Vec3::splat(0.5)
                    }
                    let (material, radius) = match world_seed % 7 {
                        0 => (
                            Material::Lambertian(LambertianMaterial {
                                albedo: Color3(rand_albedo()),
                                tex: None,
                            }),
                            0.15,
                        ),
                        1 => (
                            Material::metal_with_ior(
                                Color3(metal_color()),
                                rand::random::<f32>() * 0.5,
                                2.5,
                            ),
                            0.175,
                        ),
                        2 => (Material::dielectric(1.5), 0.2),
                        3 => (
                            Material::Isotropic(IsotropicMaterial {
                                albedo: Color3(rand::random::<Vec3>()),
                                tex: None,
                            }),
                            0.225,
                        ),
                        4 => (
                            Material::glossy(
                                Color3(rand::random::<Vec3>()),
                                rand::random::<f32>(),
                                1.5,
                            ),
                            0.25,
                        ),
                        5 => (
                            Material::coated(
                                Material::Lambertian(LambertianMaterial {
                                    albedo: Color3(rand_albedo()),
                                    tex: None,
                                }),
                                Material::metal(Color3(metal_color()), rand::random::<f32>() * 0.5),
                            ),
                            0.275,
                        ),
                        _ => (
                            Material::lambertian(Arc::new(SolidColor::new(Color3(rand_albedo()))))
                                .mix(
                                    Material::metal(
                                        Color3(metal_color()),
                                        rand::random::<f32>() * 0.5,
                                    ),
                                    rand::random::<f32>(),
                                ),
                            0.3,
                        ),
                    };
                    center.y = radius;

                    if world_seed.is_multiple_of(2) {
                        let target_center =
                            center + Vec3::new(0., rand::rng().random_range(-0.5..0.5), 0.);
                        scene.add_sphere_moving(center, target_center, radius, material);
                    } else {
                        scene.add_sphere(center, radius, material);
                    }
                }
            }
        }

        scene.add_sphere(Point3::new(0., 1., 0.), 1., Material::dielectric(1.5));
        scene.add_sphere(
            Point3::new(-4., 1., 0.),
            1.,
            Material::lambertian_color(0.4, 0.2, 0.1)
                .mix(Material::light(Color3::new(0.4, 0.2, 0.1)), 0.5),
        );
        scene.add_sphere(
            Point3::new(4., 1., 0.),
            1.,
            Material::metal_with_ior(Color3::new(0.7, 0.6, 0.5), 0.0, 2.5),
        );
        scene.add_sphere(
            Point3::new(-2., 4., 2.),
            1.5,
            Material::light_textured(Arc::new(ImageTexture::new("./earthmap.png").unwrap())),
        );
        scene.add_sphere(
            Point3::new(2., 4., -2.),
            1.5,
            Material::light_textured(Arc::new(
                MappedTexture::new(NoiseTexture::new())
                    .with_mapping3d(TextureMapping3D::point_scale_uniform(1. / 4.)),
            )),
        );

        scene.config.aspect_ratio = 16.0 / 9.0;
        scene.config.image_width = 1280;
        scene.config.samples_per_pixel = 50;
        scene.config.max_depth = 50;
        scene.config.vfov = 30.0;
        scene.config.look_from = Point3::new(13., 2., 6.);
        scene.config.look_at = Point3::new(0., 1., 0.);
        scene.config.vup = Vec3::new(0., 1., 0.);
        scene.config.defocus_angle = 0.6;
        scene.config.focus_distance = 10.0;
        scene.config.background = Color3::new(0.5, 0.7, 1.0);

        scene
    }

    pub fn simple_world() -> Self {
        let mut scene = Self::new();

        let material_ground = Material::lambertian_color(0.8, 0.8, 0.0);
        let material_center = Material::lambertian_color(0.1, 0.2, 0.5);
        let material_left = Material::dielectric(1.50);
        let material_bubble = Material::dielectric(1.0 / 1.50);
        let material_right = Material::metal_with_ior(Color3::new(0.8, 0.6, 0.2), 1.0, 2.5);

        scene.add_sphere(Point3::new(0., -100.5, -1.), 100., material_ground);
        scene.add_sphere(Point3::new(0., 0., -1.2), 0.5, material_center);
        scene.add_sphere(Point3::new(-1., 0., -1.), 0.5, material_left);
        scene.add_sphere(Point3::new(-1., 0., -1.), 0.4, material_bubble);
        scene.add_sphere(Point3::new(1., 0., -1.), 0.5, material_right);

        scene.config.samples_per_pixel = 25;
        scene.config.image_width = 800;
        scene.config.aspect_ratio = 16.0 / 9.0;
        scene.config.max_depth = 50;
        scene.config.vfov = 20.0;
        scene.config.look_from = Point3::new(-2., 2., 1.);
        scene.config.look_at = Point3::new(0., 0., -1.);
        scene.config.vup = Vec3::new(0., 1., 0.);
        scene.config.defocus_angle = 10.0;
        scene.config.focus_distance = 3.4;
        scene.config.background = Color3::new(0.5, 0.7, 1.0);

        scene
    }

    /// Test scene for the new material system: demonstrates `Mix` (painted
    /// metal), `Coated` (clear coat over substrate), and `Glossy` (GGX).
    pub fn composition_demo() -> Self {
        let mut scene = Self::new();

        // Ground plane (Lambertian), spans full z-range of the scene.
        let ground = Material::lambertian_color(0.5, 0.5, 0.5);
        scene.add_quad(
            Point3::new(-5., 0., 0.),
            Vec3::new(10., 0., 0.),
            Vec3::new(0., 0., 12.),
            ground,
        );

        // Three rows of three spheres, evenly spaced. All spheres have radius 1.0,
        // so center-to-center distance is 2.05 (0.05 gap avoids precision overlap).
        const SPHERE_GAP: f32 = 2.05;
        const ROW_Z: [f32; 3] = [3.0, 7.0, 11.0];
        const COL_X: [f32; 3] = [-SPHERE_GAP, 0.0, SPHERE_GAP];

        // Row 1 (z=3, front): glossy, rough glossy, mixed Lambertian+metal.
        //   Demonstrates specular highlights at different roughnesses and material blending.
        scene.add_sphere(
            Point3::new(COL_X[0], 1.0, ROW_Z[0]),
            1.0,
            Material::glossy(Color3::new(0.9, 0.9, 0.9), 0.2, 1.5),
        );
        scene.add_sphere(
            Point3::new(COL_X[1], 1.0, ROW_Z[0]),
            1.0,
            Material::glossy(Color3::new(0.7, 0.3, 0.3), 0.7, 1.5),
        );
        scene.add_sphere(
            Point3::new(COL_X[2], 1.0, ROW_Z[0]),
            1.0,
            Material::lambertian_color(0.8, 0.2, 0.2)
                .mix(Material::metal(Color3::new(0.9, 0.9, 0.9), 0.0), 0.5),
        );

        // Row 2 (z=7, middle): clear-coated green, coated glossy, clear-coated blue.
        //   Dielectric shell over diffuse/glossy — secondary specular highlight from the coat.
        scene.add_sphere(
            Point3::new(COL_X[0], 1.0, ROW_Z[1]),
            1.0,
            Material::lambertian_color(0.2, 0.7, 0.2).coated(Material::dielectric(1.5)),
        );
        scene.add_sphere(
            Point3::new(COL_X[1], 1.0, ROW_Z[1]),
            1.0,
            Material::glossy(Color3::new(0.8, 0.2, 0.8), 0.3, 1.5)
                .coated(Material::dielectric(1.5)),
        );
        scene.add_sphere(
            Point3::new(COL_X[2], 1.0, ROW_Z[1]),
            1.0,
            Material::lambertian_color(0.2, 0.2, 0.8).coated(Material::dielectric(1.5)),
        );

        // Row 3 (z=11, back): coated metal, perlin noise, mixed metal+glossy.
        //   Complex materials: shell over specular, 3D texture, dual-material blend.
        scene.add_sphere(
            Point3::new(COL_X[0], 1.0, ROW_Z[2]),
            1.0,
            Material::metal(Color3::new(0.8, 0.8, 0.2), 0.1).coated(Material::dielectric(1.5)),
        );
        let perlin_tex: Arc<dyn Texture> = Arc::new(
            MappedTexture::new(NoiseTexture::new())
                .with_mapping3d(TextureMapping3D::point_scale_uniform(1. / 4.)),
        );
        scene.add_sphere(
            Point3::new(COL_X[1], 1.0, ROW_Z[2]),
            1.0,
            Material::lambertian(perlin_tex),
        );
        scene.add_sphere(
            Point3::new(COL_X[2], 1.0, ROW_Z[2]),
            1.0,
            Material::metal(Color3::new(0.9, 0.9, 0.9), 0.0)
                .mix(Material::glossy(Color3::new(0.2, 0.8, 0.2), 0.5, 1.5), 0.5),
        );

        // Area light above, spanning the full z-range of the scene.
        scene.add_quad(
            Point3::new(-4., 8., 0.),
            Vec3::new(8., 0., 0.),
            Vec3::new(0., 0., 12.),
            Material::light(Color3::new(6.0, 6.0, 6.0)),
        );

        scene.config.aspect_ratio = 16.0 / 9.0;
        scene.config.image_width = 800;
        scene.config.samples_per_pixel = 100;
        scene.config.max_depth = 50;
        scene.config.vfov = 38.0;
        scene.config.look_from = Point3::new(0., 3.5, 16.);
        scene.config.look_at = Point3::new(0., 1., 7.);
        scene.config.vup = Vec3::new(0., 1., 0.);
        scene.config.focus_distance = 10.0;
        scene.config.defocus_angle = 0.0;
        scene.config.background = Color3::new(0.1, 0.1, 0.1);

        scene
    }

    pub fn coated_balls() -> Self {
        let mut scene = Self::new();

        let ground = Material::lambertian_color(0.5, 0.5, 0.5);
        scene.add_quad(
            Point3::new(-5., 0., 0.),
            Vec3::new(10., 0., 0.),
            Vec3::new(0., 0., 12.),
            ground,
        );

        // Sphere 1 — gold metal (low fuzz)
        let coated_metal = Material::metal(Color3::new(0.1, 0.1, 0.7) * 2., 0.1).coated(
            Material::dielectric_tinted(1.4, Color3::new(0.1, 0.7, 0.1) * 2.),
        );
        scene.add_sphere(Point3::new(-2., 1., 4.), 1.0, coated_metal);

        // Sphere 2 — perlin noise (unique pattern)
        let perlin_tex: Arc<dyn Texture> = Arc::new(
            MappedTexture::new(NoiseTexture::new())
                .with_mapping3d(TextureMapping3D::point_scale_uniform(1. / 4.)),
        );
        let coated_perlin =
            Material::lambertian(perlin_tex).coated(Material::light(Color3::new(0.5, 0.3, 0.7)));
        scene.add_sphere(Point3::new(2., 1., 4.), 1.0, coated_perlin);

        // Sphere 3 — blue-emitting glass (light under dielectric coating)
        let coated_glass =
            Material::light(Color3::new(0.2, 0.4, 0.9)).coated(Material::dielectric(1.5));
        scene.add_sphere(Point3::new(0., 1., 8.), 1.0, coated_glass);

        // Sphere 4 — pink-emitting glass (light under dielectric coating)
        let light_coated_glass =
            Material::light(Color3::new(0.9, 0.2, 0.6)).coated(Material::dielectric(1.5));
        scene.add_sphere(Point3::new(0., 3., 8.), 1.0, light_coated_glass);

        // Sphere 5 — red glossy
        let coated_glossy = Material::Coated(CoatedMaterial {
            substrate: Arc::new(Material::glossy(Color3::new(1., 0.0, 0.0), 0.5, 1.5))
                as Arc<dyn Bsdf>,
            coating: Arc::new(Material::dielectric(1.5)) as Arc<dyn Bsdf>,
            coating_tint: Color3::new(1., 0.0, 0.0),
            coating_ior: 1.5,
            thickness: 0.20,
        });
        scene.add_sphere(Point3::new(2., 1., 10.), 1., coated_glossy);

        // Sphere 6 — cyan-teal mix
        let coated_mixed = Material::Coated(CoatedMaterial {
            substrate: Arc::new(
                Material::metal(Color3::new(0.1, 0.7, 0.8), 0.0)
                    .mix(Material::glossy(Color3::new(0.1, 0.9, 0.6), 0.5, 1.5), 0.5),
            ) as Arc<dyn Bsdf>,
            coating: Arc::new(Material::dielectric(1.5)) as Arc<dyn Bsdf>,
            coating_tint: Color3::new(0.1, 0.9, 0.6),
            coating_ior: 1.5,
            thickness: 0.20,
        });
        scene.add_sphere(Point3::new(-2., 1., 10.), 0.8, coated_mixed);

        scene.config.aspect_ratio = 16.0 / 9.0;
        scene.config.image_width = 800;
        scene.config.samples_per_pixel = 100;
        scene.config.max_depth = 50;
        scene.config.vfov = 38.0;
        scene.config.look_from = Point3::new(0., 3.5, 16.);
        scene.config.look_at = Point3::new(0., 1., 7.);
        scene.config.vup = Vec3::new(0., 1., 0.);
        scene.config.focus_distance = 10.0;
        scene.config.defocus_angle = 0.0;
        scene.config.background = Color3::new(0.1, 0.1, 0.1);

        scene
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}
