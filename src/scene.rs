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
use crate::material::{
    DielectricMaterial, DiffuseLightMaterial, GlossyMaterial, MetalMaterial,
    RoughDielectricMaterial,
};
use crate::material::{IsotropicMaterial, LambertianMaterial, Material};
use crate::shape::regions::FunctionRegion;
use crate::shape::{
    annulus, function_patch, moving_sphere, polygon, quad, rounded_rect, shape_box3d, sphere,
    superellipse, tri,
};
use crate::texture::{
    CheckerTexture, ImageTexture, NoiseTexture, SolidColor, SphericalUvMapping, Texture,
    TriplanarMapping, UvCheckerTexture, WorldSpaceMapping,
};
use crate::transform::{RotateY, TransformObject, Translate};
use crate::vec3::{Color3, Direction3, Point3};

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

    pub fn vup(mut self, vup: Direction3) -> Self {
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

    pub fn add_sphere(&mut self, center: Point3, radius: f32, material: impl Into<Material>) {
        let material = material.into();
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
    pub fn add_quad(&mut self, Q: Point3, u: Vec3, v: Vec3, material: impl Into<Material>) {
        let material = material.into();
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

    /// Add an axis-aligned box to the scene with a single material on all 6 faces.
    ///
    /// This creates a single `ShapeObject<BoxShape>` entry in the scene BVH (one BVH leaf
    /// for the entire box). For per-face materials, construct quads individually via `quad()`
    /// and add them with `add_intersectable`.
    pub fn add_box(&mut self, a: Point3, b: Point3, material: impl Into<Material>) {
        let material = material.into();
        trace!(?a, ?b, "add box");
        let material = Arc::new(material);
        let box_3d = Arc::new(shape_box3d(a, b, material.clone()));
        if material.is_emissive() {
            self.add_intersectable(box_3d.clone(), Some(box_3d.clone()));
        } else {
            self.add_intersectable(box_3d, None);
        }
    }

    pub fn add_sphere_moving(
        &mut self,
        center_start: Point3,
        center_end: Point3,
        radius: f32,
        material: impl Into<Material>,
    ) {
        let material = material.into();
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

        let ground: Material = LambertianMaterial::new(Color3::new(0.48, 0.83, 0.53)).into();

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

                let box_quads = shape_box3d(
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
            DiffuseLightMaterial::new(Color3::new(7.0, 7.0, 7.0)),
        );

        let center1 = Point3::new(400., 400., 200.);
        let center2 = center1 + Vec3::new(30., 0., 0.);
        scene.add_sphere_moving(
            center1,
            center2,
            50.,
            Material::from(LambertianMaterial::new(Color3::new(0.7, 0.3, 0.1))),
        );

        scene.add_sphere(
            Point3::new(260., 150., 45.),
            50.,
            DielectricMaterial::new(1.5),
        );

        scene.add_sphere(
            Point3::new(0., 250., 165.),
            50.,
            RoughDielectricMaterial::tinted(1.5, 0.3, Color3::new(1.0, 0.5, 0.5)),
        );

        scene.add_sphere(
            Point3::new(0., 150., 145.),
            50.,
            MetalMaterial::new(Color3::new(0.8, 0.8, 0.9), 1.0),
        );

        scene.objects.push(Arc::new(ConstantMedium::new_albedo(
            sphere(
                Point3::new(360., 150., 145.),
                70.,
                Material::from(DielectricMaterial::new(0.9)),
            ),
            0.2,
            Color3::new(0.2, 0.4, 0.9).into_inner(),
        )));

        scene.objects.push(Arc::new(ConstantMedium::new_albedo(
            sphere(
                Point3::new(0., 0., 0.),
                5000.,
                Material::from(DielectricMaterial::new(0.9)),
            ),
            0.0001,
            // Color3::from(1., 1., 1.), // Pure white
            Color3::new(0.7, 0.1, 0.1).into_inner(), // A faint red tint to visualize the volume better
        )));

        let emat: Arc<dyn Texture> = ImageTexture::load_arc("./earthmap.png")
            .expect("Failed to load earthmap.png for complex_scene");
        scene.add_sphere(
            Point3::new(400., 200., 400.),
            100.,
            LambertianMaterial::with_texture(Color3::ZERO, emat),
        );

        let pertext: Arc<dyn Texture> = NoiseTexture::with_scale(0.7);
        scene.add_sphere(
            Point3::new(220., 280., 300.),
            80.,
            LambertianMaterial::with_texture(Color3::ZERO, pertext),
        );

        let white: Material = LambertianMaterial::new(Color3::new(0.73, 0.73, 0.73)).into();
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
        scene.config.vup = Direction3::new(0., 1., 0.);
        scene.config.defocus_angle = 0.0;
        scene.config.focus_distance = 800.0;

        scene
    }

    pub fn cornell_box_const_meds() -> Self {
        let mut scene = Scene::empty_cornell_box();

        let white: Material = LambertianMaterial::new(Color3::new(0.73, 0.73, 0.73)).into();
        let box_params = [
            (
                Vec3::new(165., 330., 165.),
                Vec3::new(265., 0., 295.),
                15.,
                white.clone(),
                Material::from(IsotropicMaterial::new(Color3::new(0.00, 0.00, 0.00))),
            ),
            (
                Vec3::new(165., 165., 165.),
                Vec3::new(130., 0., 65.),
                -18.,
                white,
                Material::from(IsotropicMaterial::new(Color3::new(1., 1., 1.))),
            ),
        ];

        let boxes = box_params
            .iter()
            .map(|(size, translate_vec, rotate_angle, mat, phase_fn)| {
                let quad_box = shape_box3d(Point3::new(0., 0., 0.), Point3(*size), mat.clone());

                let rotated = TransformObject::new(RotateY::new(*rotate_angle), quad_box);

                let wrapped = TransformObject::new(Translate::new(*translate_vec), rotated);
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

        let white: Material = LambertianMaterial::new(Color3::new(0.73, 0.73, 0.73)).into();
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
                DielectricMaterial::new(1.5).into(),
            ),
        ];

        let boxes = box_params
            .iter()
            .map(|(size, translate_vec, rotate_angle, mat)| {
                let quad_box = shape_box3d(Point3::new(0., 0., 0.), Point3(*size), mat.clone());

                let rotated = TransformObject::new(RotateY::new(*rotate_angle), quad_box);
                let wrapped = TransformObject::new(Translate::new(*translate_vec), rotated);

                Arc::new(wrapped) as Arc<dyn Intersectable>
            });

        scene.objects.extend(boxes);

        // // Add a small sphere in the center to better visualize the light transport effects.
        scene.add_sphere(
            Point3::new(348., 400., 278.),
            40.,
            MetalMaterial::with_ior(Color3::new(0.8, 0.8, 0.9), 0.3, 20.0),
        );
        scene.add_sphere(
            Point3::new(200., 350., 200.),
            90.,
            // Deeply tinted dielectric to better visualize the caustics and light transport through the box.
            // Values must be <= 1.0 for energy conservation (no light amplification).
            // DielectricMaterial::tinted(1.5, Color3::new(0.8, 0.8, 1.0)),
            MetalMaterial::with_ior(Color3::new(0.8, 0.8, 0.9), 0.1, 50.0),
        );

        scene.config.samples_per_pixel = 200;
        scene.config.tone_map = true;
        scene.config.exposure = 1.8;
        scene.config.image_width = 600;

        scene
    }

    pub fn empty_cornell_box() -> Self {
        let mut scene = Self::new();
        let red: Material = LambertianMaterial::new(Color3::new(0.65, 0.05, 0.05)).into();
        let white: Material = LambertianMaterial::new(Color3::new(0.73, 0.73, 0.73)).into();
        let green: Material = LambertianMaterial::new(Color3::new(0.12, 0.45, 0.15)).into();
        let light: Material = DiffuseLightMaterial::new(Color3::new(16.0, 16.0, 16.0)).into();

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
        scene.config.vup = Direction3::new(0., 1., 0.);

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
            DiffuseLightMaterial::new(Color3::new(4.0, 4.0, 4.0)),
        );

        scene.config.aspect_ratio = 16. / 9.;
        scene.config.image_width = 800;
        scene.config.samples_per_pixel = 100;
        scene.config.max_depth = 50;
        scene.config.background = Color3::new(0., 0., 0.);

        scene.config.vfov = 20.;
        scene.config.look_from = Point3::new(26., 3., 6.);
        scene.config.look_at = Point3::new(0., 2., 0.);
        scene.config.vup = Direction3::new(0., 1., 0.);

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
            let material = LambertianMaterial::new(Color3::new(color.x(), color.y(), color.z()));
            scene.add_quad(*Q, *u, *v, material);
        });

        scene.config.aspect_ratio = 1.0;
        scene.config.image_width = 800;
        scene.config.samples_per_pixel = 50;
        scene.config.max_depth = 50;

        scene.config.vfov = 80.0;
        scene.config.look_from = Point3::new(0., 0., 9.);
        scene.config.look_at = Point3::new(0., 0., 0.);
        scene.config.vup = Direction3::new(0., 1., 0.);

        scene.config.defocus_angle = 0.0;

        scene.config.focus_distance = 10.0;

        scene.config.background = Color3::new(0.5, 0.7, 1.0);

        scene
    }

    pub fn noisy_spheres() -> Self {
        let mut scene = Self::new();

        let perlin_tex: Arc<dyn Texture> = NoiseTexture::with_scale(1. / 4.);

        scene.add_sphere(
            Point3::new(0., -1000., 0.),
            1000.,
            LambertianMaterial::with_texture(Color3::ZERO, perlin_tex.clone()),
        );
        scene.add_sphere(
            Point3::new(0., 2., 0.),
            2.,
            LambertianMaterial::with_texture(Color3::ZERO, perlin_tex),
        );

        scene.config.aspect_ratio = 16.0 / 9.0;
        scene.config.image_width = 800;
        scene.config.samples_per_pixel = 50;
        scene.config.max_depth = 50;
        scene.config.vfov = 20.0;
        scene.config.look_from = Point3::new(13., 2., 3.);
        scene.config.look_at = Point3::new(0., 0., 0.);
        scene.config.vup = Direction3::new(0., 1., 0.);
        scene.config.defocus_angle = 0.6;
        scene.config.focus_distance = 10.0;
        scene.config.background = Color3::new(0.5, 0.7, 1.0);

        scene
    }

    pub fn earth_sphere() -> Self {
        let mut scene = Self::new();

        let image_tex: Arc<dyn Texture> = ImageTexture::load_arc("./earthmap.png")
            .expect("Failed to load earthmap.png as Texture");
        let checker: Material = LambertianMaterial::with_texture(Color3::ZERO, image_tex).into();

        scene.add_sphere(Point3::new(0., 0., 0.), 2., checker);

        scene.config.aspect_ratio = 16.0 / 9.0;
        scene.config.image_width = 800;
        scene.config.samples_per_pixel = 50;
        scene.config.max_depth = 50;
        scene.config.vfov = 20.0;
        scene.config.look_from = Point3::new(13., 2., 3.);
        scene.config.look_at = Point3::new(0., 0., 0.);
        scene.config.vup = Direction3::new(0., 1., 0.);
        scene.config.defocus_angle = 0.6;
        scene.config.focus_distance = 10.0;
        scene.config.background = Color3::new(0.5, 0.7, 1.0);

        scene
    }

    pub fn checkered_spheres() -> Self {
        let mut scene = Self::new();

        let checker = LambertianMaterial::with_texture(
            Color3::ZERO,
            CheckerTexture::with_scale(
                0.32,
                Color3::new(0.2, 0.4, 0.1),
                Color3::new(0.9, 0.9, 0.9),
            ),
        );
        scene.add_sphere(Point3::new(0., -10., 0.), 10., checker.clone());
        scene.add_sphere(Point3::new(0., 10., 0.), 10., checker);

        scene.config.aspect_ratio = 16.0 / 9.0;
        scene.config.image_width = 800;
        scene.config.samples_per_pixel = 50;
        scene.config.max_depth = 50;
        scene.config.vfov = 20.0;
        scene.config.look_from = Point3::new(13., 2., 3.);
        scene.config.look_at = Point3::new(0., 0., 0.);
        scene.config.vup = Direction3::new(0., 1., 0.);
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

        let ground_material: Material = LambertianMaterial::with_texture(
            Color3::ZERO,
            CheckerTexture::with_scale(
                0.32,
                Color3::new(0.2, 0.4, 0.1),
                Color3::new(0.9, 0.9, 0.9),
            ),
        )
        .into();
        let ground_material = ground_material
            .coated(DielectricMaterial::tinted(1.2, Color3::new(0.8, 0.8, 1.0)).into());
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
                        0 => (LambertianMaterial::new(Color3(rand_albedo())).into(), 0.15),
                        1 => (
                            Material::from(MetalMaterial::with_ior(
                                Color3(metal_color()),
                                rand::random::<f32>() * 0.5,
                                2.5,
                            )),
                            0.175,
                        ),
                        2 => (DielectricMaterial::new(1.5).into(), 0.2),
                        3 => (
                            IsotropicMaterial::new(Color3(rand::random::<Vec3>())).into(),
                            0.225,
                        ),
                        4 => (
                            Material::from(GlossyMaterial::new(
                                Color3(rand::random::<Vec3>()),
                                rand::random::<f32>(),
                                1.5,
                            )),
                            0.25,
                        ),
                        5 => (
                            Material::from(LambertianMaterial::new(Color3(rand_albedo()))).coated(
                                MetalMaterial::new(
                                    Color3(metal_color()),
                                    rand::random::<f32>() * 0.5,
                                )
                                .into(),
                            ),
                            0.275,
                        ),
                        _ => (
                            Material::from(LambertianMaterial::with_texture(
                                Color3::ZERO,
                                Arc::new(SolidColor::new(Color3(rand_albedo()))),
                            ))
                            .mix(
                                MetalMaterial::new(
                                    Color3(metal_color()),
                                    rand::random::<f32>() * 0.5,
                                )
                                .into(),
                                rand::random::<f32>(),
                            ),
                            0.3,
                        ),
                    };
                    center = Point3::new(center.x(), radius, center.z());

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

        scene.add_sphere(Point3::new(0., 1., 0.), 1., DielectricMaterial::new(1.5));
        scene.add_sphere(
            Point3::new(-4., 1., 0.),
            1.,
            Material::from(LambertianMaterial::new(Color3::new(0.4, 0.2, 0.1))).mix(
                DiffuseLightMaterial::new(Color3::new(0.4, 0.2, 0.1)).into(),
                0.5,
            ),
        );
        scene.add_sphere(
            Point3::new(4., 1., 0.),
            1.,
            MetalMaterial::with_ior(Color3::new(0.7, 0.6, 0.5), 0.0, 2.5),
        );
        scene.add_sphere(
            Point3::new(-2., 4., 2.),
            1.5,
            DiffuseLightMaterial::textured(
                ImageTexture::load_arc("./earthmap.png").expect("Failed to load earthmap.png"),
            ),
        );
        scene.add_sphere(
            Point3::new(2., 4., -2.),
            1.5,
            DiffuseLightMaterial::textured(NoiseTexture::with_scale(1. / 4.)),
        );

        scene.config.aspect_ratio = 16.0 / 9.0;
        scene.config.image_width = 1280;
        scene.config.samples_per_pixel = 50;
        scene.config.max_depth = 50;
        scene.config.vfov = 30.0;
        scene.config.look_from = Point3::new(13., 5., 6.);
        scene.config.look_at = Point3::new(0., 1., 0.);
        scene.config.vup = Direction3::new(0., 1., 0.);
        scene.config.defocus_angle = 0.6;
        scene.config.focus_distance = 10.0;
        scene.config.background = Color3::new(0.5, 0.7, 1.0);

        scene
    }

    pub fn simple_world() -> Self {
        let mut scene = Self::new();

        let material_ground: Material = LambertianMaterial::new(Color3::new(0.8, 0.8, 0.0)).into();
        let material_center: Material = LambertianMaterial::new(Color3::new(0.1, 0.2, 0.5)).into();
        let material_left: Material = DielectricMaterial::new(1.50).into();
        let material_bubble: Material = DielectricMaterial::new(1.0 / 1.50).into();
        let material_right: Material =
            MetalMaterial::with_ior(Color3::new(0.8, 0.6, 0.2), 1.0, 2.5).into();

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
        scene.config.vup = Direction3::new(0., 1., 0.);
        scene.config.defocus_angle = 10.0;
        scene.config.focus_distance = 3.4;
        scene.config.background = Color3::new(0.5, 0.7, 1.0);

        scene
    }

    /// Test scene for the new material system: demonstrates `Mix` (painted
    /// metal), `Coated` (clear coat over substrate), and `Glossy` (GGX).
    pub fn composition_demo() -> Self {
        let scene = Self::new();

        let mut scene = scene.with_env_map(Arc::new(EnvironmentMap::new(
            image::open("./kiara_1_dawn_4k.hdr").unwrap().into(),
        )));

        // Ground plane (Lambertian), spans full z-range of the scene.
        let ground: Material = LambertianMaterial::new(Color3::new(0.5, 0.5, 0.5)).into();
        scene.add_quad(
            Point3::new(-5., 0., 0.),
            Vec3::new(10., 0., 0.),
            Vec3::new(0., 0., 12.),
            ground,
        );

        // Three rows of four spheres, evenly spaced. All spheres have radius 1.0,
        // so center-to-center distance is 2.05 (0.05 gap avoids precision overlap).
        const SPHERE_GAP: f32 = 2.05;
        const ROW_Z: [f32; 3] = [3.0, 7.0, 11.0];
        const COL_X: [f32; 4] = [
            -SPHERE_GAP * 1.5,
            -SPHERE_GAP / 2.0,
            SPHERE_GAP / 2.0,
            SPHERE_GAP * 1.5,
        ];

        // Row 1 (z=3, front): basic material types.
        //   Covers random_world types 0-3: Lambertian, Metal, Dielectric, Isotropic.
        scene.add_sphere(
            Point3::new(COL_X[0], 1.0, ROW_Z[0]),
            1.0,
            LambertianMaterial::new(Color3::new(0.8, 0.2, 0.2)),
        );
        scene.add_sphere(
            Point3::new(COL_X[1], 1.0, ROW_Z[0]),
            1.0,
            MetalMaterial::with_ior(Color3::new(0.9, 0.7, 0.1), 0.1, 2.5),
        );
        scene.add_sphere(
            Point3::new(COL_X[2], 1.0, ROW_Z[0]),
            1.0,
            DielectricMaterial::new(1.5),
        );
        scene.add_sphere(
            Point3::new(COL_X[3], 1.0, ROW_Z[0]),
            1.0,
            IsotropicMaterial::new(Color3::new(0.4, 0.6, 0.3)),
        );

        // Row 2 (z=7, middle): glossy, coated, and emissive materials.
        //   Covers random_world types 4-5 plus Light.
        scene.add_sphere(
            Point3::new(COL_X[0], 1.0, ROW_Z[1]),
            1.0,
            GlossyMaterial::new(Color3::new(0.8, 0.2, 0.8), 0.3, 1.5),
        );
        scene.add_sphere(
            Point3::new(COL_X[1], 1.0, ROW_Z[1]),
            1.0,
            Material::from(LambertianMaterial::new(Color3::new(0.2, 0.7, 0.2)))
                .coated(DielectricMaterial::new(1.5).into()),
        );
        scene.add_sphere(
            Point3::new(COL_X[2], 1.0, ROW_Z[1]),
            1.0,
            // random_world type 5: coated dark Lambertian + metal coating
            Material::from(LambertianMaterial::new(Color3::new(0.3, 0.1, 0.1)))
                .coated(MetalMaterial::new(Color3::new(0.9, 0.7, 0.1), 0.1).into()),
        );
        scene.add_sphere(
            Point3::new(COL_X[3], 1.0, ROW_Z[1]),
            1.0,
            DiffuseLightMaterial::new(Color3::new(0.9, 0.2, 0.6)),
        );

        // Row 3 (z=11, back): composite materials — Mix and Coated combinations.
        //   Covers random_world type 6 plus additional blend and perfect-mirror cases.
        scene.add_sphere(
            Point3::new(COL_X[0], 1.0, ROW_Z[2]),
            1.0,
            Material::from(LambertianMaterial::new(Color3::new(0.8, 0.5, 0.2))).mix(
                MetalMaterial::new(Color3::new(0.9, 0.7, 0.1), 0.1).into(),
                0.5,
            ),
        );
        scene.add_sphere(
            Point3::new(COL_X[1], 1.0, ROW_Z[2]),
            1.0,
            Material::from(LambertianMaterial::new(Color3::new(0.2, 0.2, 0.8))).mix(
                DiffuseLightMaterial::new(Color3::new(0.2, 0.4, 0.9)).into(),
                0.3,
            ),
        );
        scene.add_sphere(
            Point3::new(COL_X[2], 1.0, ROW_Z[2]),
            1.0,
            Material::from(MetalMaterial::new(Color3::new(0.9, 0.7, 0.1), 0.1))
                .coated(DielectricMaterial::new(1.5).into()),
        );
        scene.add_sphere(
            Point3::new(COL_X[3], 1.0, ROW_Z[2]),
            1.0,
            MetalMaterial::with_ior(Color3::new(0.9, 0.9, 0.9), 0.0, 2.5),
        );

        // Area light above, spanning the full z-range of the scene.
        scene.add_quad(
            Point3::new(-5., 8., 0.),
            Vec3::new(10., 0., 0.),
            Vec3::new(0., 0., 12.),
            DiffuseLightMaterial::new(Color3::new(6.0, 6.0, 6.0)),
        );

        scene.config.aspect_ratio = 16.0 / 9.0;
        scene.config.image_width = 800;
        scene.config.samples_per_pixel = 100;
        scene.config.max_depth = 50;
        scene.config.vfov = 38.0;
        // scene.config.look_from = Point3::new(0., 3.5, 16.);
        scene.config.look_from = Point3::new(10., 5., 0.);
        scene.config.look_at = Point3::new(0., 1., 7.);
        scene.config.vup = Direction3::new(0., 1., 0.);
        scene.config.focus_distance = 10.0;
        scene.config.defocus_angle = 0.0;
        scene.config.background = Color3::new(0.1, 0.1, 0.1);

        scene
    }

    pub fn coated_balls() -> Self {
        let mut scene = Self::new();

        let ground: Material = LambertianMaterial::new(Color3::new(0.5, 0.5, 0.5)).into();
        scene.add_quad(
            Point3::new(-5., 0., 0.),
            Vec3::new(10., 0., 0.),
            Vec3::new(0., 0., 12.),
            ground,
        );

        // Sphere 1 — gold metal (low fuzz)
        let coated_metal = Material::from(MetalMaterial::new(Color3::new(0.1, 0.1, 0.7) * 2., 0.1))
            .coated(DielectricMaterial::tinted(1.4, Color3::new(0.1, 0.7, 0.1) * 2.).into());
        scene.add_sphere(Point3::new(-2., 1., 4.), 1.0, coated_metal);

        // Sphere 2 — perlin noise (unique pattern)
        let perlin_tex: Arc<dyn Texture> = NoiseTexture::with_scale(1. / 4.);
        let coated_perlin =
            Material::from(LambertianMaterial::with_texture(Color3::ZERO, perlin_tex))
                .coated(DiffuseLightMaterial::new(Color3::new(0.5, 0.3, 0.7)).into());
        scene.add_sphere(Point3::new(2., 1., 4.), 1.0, coated_perlin);

        // Sphere 3 — blue-emitting glass (light under dielectric coating)
        let coated_glass = Material::from(DiffuseLightMaterial::new(Color3::new(0.2, 0.4, 0.9)))
            .coated(DielectricMaterial::new(1.5).into());
        scene.add_sphere(Point3::new(0., 1., 8.), 1.0, coated_glass);

        // Sphere 4 — pink-emitting glass (light under dielectric coating)
        let light_coated_glass =
            Material::from(DiffuseLightMaterial::new(Color3::new(0.9, 0.2, 0.6)))
                .coated(DielectricMaterial::new(1.5).into());
        scene.add_sphere(Point3::new(0., 3., 8.), 1.0, light_coated_glass);

        // Sphere 5 — red glossy
        let _coated_glossy = Material::Coated(CoatedMaterial {
            substrate: Arc::new(GlossyMaterial::new(Color3::new(1., 0.0, 0.0), 0.5, 1.5))
                as Arc<dyn Bsdf>,
            coating: Arc::new(Material::from(DielectricMaterial::new(1.5))) as Arc<dyn Bsdf>,
            coating_tint: Color3::new(1., 0.0, 0.0),
            coating_ior: 1.5,
            thickness: 0.20,
        });

        // Sphere 6 — cyan-teal mix
        let coated_mixed = Material::Coated(CoatedMaterial {
            substrate: Arc::new(
                Material::from(MetalMaterial::new(Color3::new(0.1, 0.7, 0.8), 0.0)).mix(
                    GlossyMaterial::new(Color3::new(0.1, 0.9, 0.6), 0.5, 1.5).into(),
                    0.5,
                ),
            ) as Arc<dyn Bsdf>,
            coating: Arc::new(Material::from(DielectricMaterial::new(1.5))) as Arc<dyn Bsdf>,
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
        scene.config.vup = Direction3::new(0., 1., 0.);
        scene.config.focus_distance = 10.0;
        scene.config.defocus_angle = 0.0;
        scene.config.background = Color3::new(0.1, 0.1, 0.1);

        scene
    }

    pub fn glass_box() -> Self {
        let scene = Self::new();

        let mut scene = scene.with_env_map(Arc::new(EnvironmentMap::new(
            image::open("./kiara_1_dawn_4k.hdr").unwrap().into(),
        )));

        let glass_material: Material = DielectricMaterial::new(1.5).into();
        let box_size = Vec3::new(2., 2., 2.);
        let box_center = Point3::new(0., 1., 0.);
        let box_min = box_center - box_size / 2.;
        let box_max = box_center + box_size / 2.;

        let glass_box = shape_box3d(box_min, box_max, glass_material);
        scene.objects.push(Arc::new(glass_box));

        scene.config.aspect_ratio = 16.0 / 9.0;
        scene.config.image_width = 800;
        scene.config.samples_per_pixel = 100;
        scene.config.max_depth = 50;
        scene.config.vfov = 38.0;
        scene.config.look_from = Point3::new(0., 3.5, 16.);
        scene.config.look_at = Point3::new(0., 1., 7.);
        scene.config.vup = Direction3::new(0., 1., 0.);
        scene.config.focus_distance = 10.0;
        scene.config.defocus_angle = 0.0;
        scene.config.background = Color3::new(0.1, 0.1, 0.1);

        scene
    }

    /// Master stress test: exercises every material, texture, shape, and
    /// composition feature in the engine.  Each object is labelled in the
    /// source so you can identify it in the rendered image.
    ///
    /// Deterministic — no randomness used anywhere in scene construction.
    pub fn master_stress_test() -> Self {
        let scene = Self::new();

        // Environment map — tests env importance sampling + MIS on miss paths.
        let mut scene = scene.with_env_map(Arc::new(EnvironmentMap::new(
            image::open("./kiara_1_dawn_4k.hdr")
                .expect("Failed to load kiara_1_dawn_4k.hdr for master_stress_test")
                .into(),
        )));

        // ── Cornell box shell (walls, floor, ceiling light) ───────────
        let red: Material = LambertianMaterial::new(Color3::new(0.65, 0.05, 0.05)).into();
        let white: Material = LambertianMaterial::new(Color3::new(0.73, 0.73, 0.73)).into();
        let green: Material = LambertianMaterial::new(Color3::new(0.12, 0.45, 0.15)).into();

        // Floor
        scene.add_quad(
            Point3::new(0., 0., 0.),
            Vec3::new(555., 0., 0.),
            Vec3::new(0., 0., 555.),
            white.clone(),
        );
        // Left wall  (red)
        scene.add_quad(
            Point3::new(0., 0., 0.),
            Vec3::new(0., 555., 0.),
            Vec3::new(0., 0., 555.),
            red,
        );
        // Right wall (green)
        scene.add_quad(
            Point3::new(555., 0., 0.),
            Vec3::new(0., 555., 0.),
            Vec3::new(0., 0., -555.),
            green,
        );
        // Back wall
        scene.add_quad(
            Point3::new(0., 0., 555.),
            Vec3::new(555., 0., 0.),
            Vec3::new(0., 555., 0.),
            white.clone(),
        );
        // Ceiling
        scene.add_quad(
            Point3::new(0., 555., 0.),
            Vec3::new(555., 0., 0.),
            Vec3::new(0., 0., 555.),
            white.clone(),
        );
        // Area light
        scene.add_quad(
            Point3::new(213., 554., 227.),
            Vec3::new(130., 0., 0.),
            Vec3::new(0., 0., 105.),
            DiffuseLightMaterial::new(Color3::new(16., 16., 16.)),
        );

        // ── Vertical grid layout ─────────────────────────────────────
        //
        // The Cornell box is 555 units tall (y: 0→555). Objects are
        // stacked in 5 vertical levels, all centred at z=278 (mid-depth).
        // Overflow shapes sit on the floor at z=100 (near camera).
        //
        //   Level 1 (y=56):   basic materials
        //   Level 2 (y=160):  emissive + isotropic + coated
        //   Level 3 (y=260):  textures
        //   Level 4 (y=360):  composition (Mix, Coated)
        //   Level 5 (y=460):  exotic shapes
        //   Overflow (z=100): additional shapes on the floor

        let zf: f32 = 278.; // z: box centre (front-to-back)
        let zo: f32 = 100.; // z: overflow row (near camera)
        let y1: f32 = 56.; // level 1 — on floor
        let y2: f32 = 160.; // level 2
        let y3: f32 = 260.; // level 3
        let y4: f32 = 360.; // level 4
        let y5: f32 = 460.; // level 5 — near ceiling
        let c1: f32 = 80.;
        let c2: f32 = 170.;
        let c3: f32 = 260.;
        let c4: f32 = 350.;
        let c5: f32 = 440.;
        let r: f32 = 28.;

        // ── Level 1 (y=56) — Basic materials ─────────────────────────
        // C1 — Lambertian (diffuse)
        scene.add_sphere(
            Point3::new(c1, y1, zf),
            r,
            LambertianMaterial::new(Color3::new(0.8, 0.2, 0.2)),
        );
        // C2 — Metal (rough)
        scene.add_sphere(
            Point3::new(c2, y1, zf),
            r,
            MetalMaterial::new(Color3::new(0.8, 0.6, 0.2), 0.3),
        );
        // C3 — Metal (mirror, roughness < 0.01 → delta, explicit IOR)
        scene.add_sphere(
            Point3::new(c3, y1, zf),
            r,
            MetalMaterial::with_ior(Color3::new(0.9, 0.9, 0.9), 0.0, 2.5),
        );
        // C4 — Dielectric (glass)
        scene.add_sphere(Point3::new(c4, y1, zf), r, DielectricMaterial::new(1.5));
        // C5 — Glossy (GGX dielectric BRDF)
        scene.add_sphere(
            Point3::new(c5, y1, zf),
            r,
            GlossyMaterial::new(Color3::new(0.8, 0.2, 0.8), 0.3, 1.5),
        );

        // ── Level 2 (y=160) — Emissive, isotropic, coated ────────────
        // C1 — DiffuseLight (area emitter)
        scene.add_sphere(
            Point3::new(c1, y2, zf),
            r,
            DiffuseLightMaterial::new(Color3::new(4., 4., 4.)),
        );
        // C2 — DiffuseLight (textured emitter — Perlin noise)
        scene.add_sphere(
            Point3::new(c2, y2, zf),
            r,
            DiffuseLightMaterial::textured(NoiseTexture::with_scale(0.5)),
        );
        // C3 — Isotropic (volume-scattering material on a surface)
        scene.add_sphere(
            Point3::new(c3, y2, zf),
            r,
            IsotropicMaterial::new(Color3::new(0.4, 0.6, 0.3)),
        );
        // C4 — RoughDielectric (rough glass)
        scene.add_sphere(
            Point3::new(c4, y2, zf),
            r,
            RoughDielectricMaterial::tinted(1.5, 0.3, Color3::new(0.8, 1.0, 0.8)),
        );
        // C5 — Coated: Lambertian substrate + Metal coating (random_world type 5)
        scene.add_sphere(
            Point3::new(c5, y2, zf),
            r,
            Material::from(LambertianMaterial::new(Color3::new(0.2, 0.7, 0.2)))
                .coated(MetalMaterial::new(Color3::new(0.8, 0.6, 0.2), 0.1).into()),
        );

        // ── Level 3 (y=260) — Textures ───────────────────────────────
        // C1 — Checker texture on Lambertian
        scene.add_sphere(
            Point3::new(c1, y3, zf),
            r,
            LambertianMaterial::with_texture(
                Color3::ZERO,
                CheckerTexture::with_scale(
                    0.32,
                    Color3::new(0.2, 0.4, 0.1),
                    Color3::new(0.9, 0.9, 0.9),
                ),
            ),
        );
        // C2 — Perlin noise texture on Lambertian
        scene.add_sphere(
            Point3::new(c2, y3, zf),
            r,
            LambertianMaterial::with_texture(Color3::ZERO, NoiseTexture::with_scale(1. / 4.)),
        );
        // C3 — Image texture on Lambertian (earth map)
        scene.add_sphere(
            Point3::new(c3, y3, zf),
            r,
            LambertianMaterial::with_texture(
                Color3::ZERO,
                ImageTexture::load_arc("./earthmap.png")
                    .expect("Failed to load earthmap.png for master_stress_test"),
            ),
        );
        // C4 — Image texture on Glossy
        scene.add_sphere(
            Point3::new(c4, y3, zf),
            r,
            GlossyMaterial::textured(
                ImageTexture::load_arc("./earthmap.png")
                    .expect("Failed to load earthmap.png for master_stress_test"),
                0.2,
                1.5,
            ),
        );
        // C5 — SolidColor on Metal (just albedo, no texture object)
        scene.add_sphere(
            Point3::new(c5, y3, zf),
            r,
            MetalMaterial::new(Color3::new(0.7, 0.1, 0.1), 0.1),
        );

        // ── Level 4 (y=360) — Composition (Mix, Coated combos) ───────
        // C1 — Mix: Lambertian + Metal (50/50)
        scene.add_sphere(
            Point3::new(c1, y4, zf),
            r,
            Material::from(LambertianMaterial::new(Color3::new(0.8, 0.5, 0.2))).mix(
                MetalMaterial::new(Color3::new(0.9, 0.7, 0.1), 0.1).into(),
                0.5,
            ),
        );
        // C2 — Mix: Lambertian + DiffuseLight (emissive blend)
        scene.add_sphere(
            Point3::new(c2, y4, zf),
            r,
            Material::from(LambertianMaterial::new(Color3::new(0.2, 0.2, 0.8))).mix(
                DiffuseLightMaterial::new(Color3::new(0.2, 0.4, 0.9)).into(),
                0.3,
            ),
        );
        // C3 — Mix: Metal + Glossy
        scene.add_sphere(
            Point3::new(c3, y4, zf),
            r,
            Material::from(MetalMaterial::new(Color3::new(0.8, 0.2, 0.2), 0.2)).mix(
                GlossyMaterial::new(Color3::new(0.2, 0.8, 0.8), 0.3, 1.5).into(),
                0.5,
            ),
        );
        // C4 — Coated: Metal substrate + Dielectric coating (tinted)
        scene.add_sphere(
            Point3::new(c4, y4, zf),
            r,
            Material::from(MetalMaterial::new(Color3::new(0.9, 0.7, 0.1), 0.1))
                .coated(DielectricMaterial::tinted(1.5, Color3::new(0.9, 0.3, 0.1)).into()),
        );
        // C5 — Coated: Glossy substrate + Dielectric coating
        scene.add_sphere(
            Point3::new(c5, y4, zf),
            r,
            Material::from(GlossyMaterial::new(Color3::new(0.8, 0.2, 0.2), 0.3, 1.5))
                .coated(DielectricMaterial::new(1.5).into()),
        );

        // ── Level 5 (y=460) — Exotic shapes ──────────────────────────
        // C1 — Box (shape_box3d, single material)
        scene.add_box(
            Point3::new(c1 - r, y5 - r * 2., zf - r),
            Point3::new(c1 + r, y5, zf + r),
            LambertianMaterial::new(Color3::new(0.6, 0.6, 0.1)),
        );
        // C2 — Moving Lambertian sphere (motion blur, random_world type 0 moving)
        scene.add_sphere_moving(
            Point3::new(c2, y5, zf),
            Point3::new(c2 + 40., y5, zf),
            r,
            LambertianMaterial::new(Color3::new(0.8, 0.3, 0.1)),
        );
        // C3 — ConstantMedium (volume inside a dielectric sphere)
        scene.add_sphere(
            Point3::new(c3, y5, zf),
            r,
            DielectricMaterial::new(1.0), // invisible boundary
        );
        scene.objects.push(Arc::new(ConstantMedium::new_albedo(
            sphere(Point3::new(c3, y5, zf), r, Material::Void),
            0.5,
            Color3::new(0.8, 0.2, 0.1).into_inner(),
        )));
        // C4 — Triangle (tri constructor)
        let tri_mat: Material = LambertianMaterial::new(Color3::new(0.3, 0.3, 0.9)).into();
        scene.add_object(Arc::new(tri(
            Point3::new(c4 - r, y5, zf - r),
            Vec3::new(r * 2., 0., 0.),
            Vec3::new(0., 0., r * 2.),
            tri_mat,
        )));
        // C5 — Annulus (ring shape)
        let ann_mat: Material = GlossyMaterial::new(Color3::new(0.6, 0.1, 0.6), 0.15, 1.5).into();
        scene.add_object(Arc::new(annulus(
            Point3::new(c5, y5, zf),
            Vec3::new(r, 0., 0.),
            Vec3::new(0., 0., r),
            0.4, // inner radius ratio
            ann_mat,
        )));

        // ── Overflow (floor, z=100) — additional shapes ──────────────
        // Rounded rectangle
        let rr_mat: Material = LambertianMaterial::new(Color3::new(0.1, 0.8, 0.6)).into();
        scene.add_object(Arc::new(rounded_rect(
            Point3::new(c1, 0.1, zo),
            Vec3::new(r, 0., 0.),
            Vec3::new(0., 0., r),
            0.3,
            rr_mat,
        )));
        // Superellipse (n > 2 = squircle, n < 2 = diamond-like)
        let se_mat: Material = GlossyMaterial::new(Color3::new(0.9, 0.5, 0.1), 0.2, 1.5).into();
        scene.add_object(Arc::new(superellipse(
            Point3::new(c2, 0.1, zo),
            Vec3::new(r, 0., 0.),
            Vec3::new(0., 0., r),
            4.0, // squircle exponent
            se_mat,
        )));
        // Polygon (pentagon)
        let pentagon_verts: Vec<(f32, f32)> = (0..5)
            .map(|i| {
                let angle =
                    std::f32::consts::FRAC_PI_2 + i as f32 * 2.0 * std::f32::consts::PI / 5.0;
                (0.5 * angle.cos(), 0.5 * angle.sin())
            })
            .collect();
        let poly_mat: Material = LambertianMaterial::new(Color3::new(0.9, 0.9, 0.2)).into();
        scene.add_object(Arc::new(polygon(
            Point3::new(c3, 0.1, zo),
            Vec3::new(r, 0., 0.),
            Vec3::new(0., 0., r),
            pentagon_verts,
            poly_mat,
        )));
        // Cross-shaped FunctionRegion patch
        let cross_fn = FunctionRegion::new(
            Arc::new(|a: f32, b: f32| {
                (a.abs() < 0.2 && b.abs() < 0.5) || (a.abs() < 0.5 && b.abs() < 0.2)
            }),
            0.4 * 0.5 * 2.0 - 0.4 * 0.4, // two bars minus overlap
            (-0.5, -0.5, 0.5, 0.5),
        );
        let fn_mat: Material = MetalMaterial::new(Color3::new(0.7, 0.7, 0.9), 0.0).into();
        scene.add_object(Arc::new(function_patch(
            Point3::new(c4, 0.1, zo),
            Vec3::new(r, 0., 0.),
            Vec3::new(0., 0., r),
            cross_fn,
            fn_mat,
        )));
        // Glass box (shape_box3d with dielectric)
        let glass_box = shape_box3d(
            Point3::new(c5 - r * 0.7, 0.1, zo - r * 0.7),
            Point3::new(c5 + r * 0.7, r * 1.4, zo + r * 0.7),
            Material::from(DielectricMaterial::new(1.5)),
        );
        scene.add_object(Arc::new(glass_box));
        // DiffuseLight textured with image (random_world featured)
        scene.add_sphere(
            Point3::new(c1, r, zo + 70.),
            r,
            DiffuseLightMaterial::textured(
                ImageTexture::load_arc("./earthmap.png")
                    .expect("Failed to load earthmap.png for master_stress_test"),
            ),
        );
        // Moving metal sphere (motion blur + Metal::with_ior, random_world type 1 moving)
        scene.add_sphere_moving(
            Point3::new(c2, r, zo + 70.),
            Point3::new(c2 + 30., r, zo + 70.),
            r,
            MetalMaterial::with_ior(Color3::new(0.8, 0.6, 0.5), 0.2, 2.5),
        );

        // ── Camera ───────────────────────────────────────────────────
        // Camera centred vertically to frame all 5 levels (y: 56→460).
        scene.config.aspect_ratio = 16. / 9.;
        scene.config.image_width = 800;
        scene.config.samples_per_pixel = 200;
        scene.config.max_depth = 50;
        scene.config.vfov = 40.0;
        scene.config.look_from = Point3::new(278., 278., -600.);
        scene.config.look_at = Point3::new(278., 278., 278.);
        scene.config.vup = Direction3::new(0., 1., 0.);
        scene.config.defocus_angle = 0.0;
        scene.config.focus_distance = 800.0;
        scene.config.background = Color3::new(0., 0., 0.);

        scene
    }
    /// Diagnostic scene for coated dielectric over textured Lambertian.
    ///
    /// Three spheres side-by-side so you can compare:
    ///
    ///   Left   — bare checker Lambertian (reference: texture always visible)
    ///   Centre — coated checker (tinted dielectric over checker)
    ///            - lit face: Fresnel reflection/refraction at coating
    ///            - grazing rim: mirror-like reflection, checker obscured
    ///            - shadow side: checker visible through transparent coating
    ///   Right  — bare dielectric (reference: glass with no substrate)
    ///
    /// A strong area light from the upper-right creates a clear lit face
    /// and a shadow side on each sphere, making it easy to judge whether
    /// the substrate texture shows through the coating in each region.
    pub fn coated_dielectric_test() -> Self {
        let mut scene = Self::new().with_env_map(Arc::new(EnvironmentMap::new(
            image::open("./kiara_1_dawn_4k.hdr")
                .expect("Failed to load kiara_1_dawn_4k.hdr for coated_dielectric_test")
                .into(),
        )));

        // Ground — plain grey Lambertian so it doesn't compete for attention.
        let ground: Material = LambertianMaterial::new(Color3::new(0.5, 0.5, 0.5)).into();
        scene.add_quad(
            Point3::new(-5., 0., 0.),
            Vec3::new(10., 0., 0.),
            Vec3::new(0., 0., 12.),
            ground,
        );

        // --- 3D world-space checker (reference — current default) ---

        // Left — bare checker Lambertian (3D world-space mapping).
        let ws_checker: Arc<dyn Texture> = Arc::new(WorldSpaceMapping::new(
            CheckerTexture::from_color(Color3::new(0.2, 0.4, 0.1), Color3::new(0.9, 0.9, 0.9)),
            0.32,
        ));
        scene.add_sphere(
            Point3::new(-2.5, 1., 5.),
            1.,
            LambertianMaterial::with_texture(Color3::ZERO, ws_checker.clone()),
        );

        // Centre — coated checker (tinted dielectric over checker Lambertian).
        scene.add_sphere(
            Point3::new(0., 1., 5.),
            1.,
            Material::from(LambertianMaterial::with_texture(Color3::ZERO, ws_checker))
                .coated(DielectricMaterial::tinted(1.5, Color3::new(0.8, 0.8, 1.0)).into()),
        );

        // Right — bare dielectric (reference: glass with no substrate).
        scene.add_sphere(Point3::new(2.5, 1., 5.), 1., DielectricMaterial::new(1.5));

        // --- Spherical UV-mapped checker (uses latitude/longitude from geometry) ---

        // Far-left — bare UV checker Lambertian.
        let uv_checker: Arc<dyn Texture> = Arc::new(SphericalUvMapping::new(
            UvCheckerTexture::new(8.0, Color3::new(0.2, 0.4, 0.1), Color3::new(0.9, 0.9, 0.9)),
        ));
        scene.add_sphere(
            Point3::new(-5., 1., 5.),
            1.,
            LambertianMaterial::with_texture(Color3::ZERO, uv_checker),
        );

        // --- Triplanar-mapped checker (projects checker from 3 axes, blends by normal) ---

        // Far-right — bare triplanar checker Lambertian.
        let tri_checker: Arc<dyn Texture> = Arc::new(TriplanarMapping::new(
            CheckerTexture::from_color(Color3::new(0.2, 0.4, 0.1), Color3::new(0.9, 0.9, 0.9)),
            4.0,
            0.32,
        ));
        scene.add_sphere(
            Point3::new(5., 1., 5.),
            1.,
            LambertianMaterial::with_texture(Color3::ZERO, tri_checker),
        );

        // Strong area light from upper-right to create a clear lit face
        // and a shadow side on each sphere.
        scene.add_quad(
            Point3::new(0., 5., 3.),
            Vec3::new(4., 0., 0.),
            Vec3::new(0., 0., 4.),
            DiffuseLightMaterial::new(Color3::new(12., 12., 12.)),
        );

        scene.config.aspect_ratio = 16. / 9.;
        scene.config.image_width = 800;
        scene.config.samples_per_pixel = 200;
        scene.config.max_depth = 50;
        scene.config.vfov = 30.0;
        scene.config.look_from = Point3::new(0., 2., 20.);
        scene.config.look_at = Point3::new(0., 1., 5.);
        scene.config.vup = Direction3::new(0., 1., 0.);
        scene.config.defocus_angle = 0.0;
        scene.config.focus_distance = 10.0;
        scene.config.background = Color3::new(0.1, 0.1, 0.1);

        scene
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}
