pub mod path_tracer;

use crate::environment::EnvironmentMap;
use crate::vec3::Vec3;
pub use path_tracer::PathTracingIntegrator;

use std::sync::Arc;

use crate::hittable::{Intersectable, Sampleable};
use crate::ray::Ray;
pub use crate::sampler::Sampler;

use crate::vec3::Color3;

pub trait Integrator<S: Sampler>: Send + Sync {
    /// Default background radiance for a ray that missed all geometry.
    fn background(&self, direction: Vec3) -> Color3 {
        match self.env_map() {
            Some(env) => env.le(direction),
            None => self.background_color(),
        }
    }

    /// Returns the environment map used for background lighting, if any.
    fn env_map(&self) -> Option<&Arc<EnvironmentMap>>;

    /// Returns the default background color for rays that miss all geometry, used when no
    /// environment map is provided.
    fn background_color(&self) -> Color3;

    // Computes the radiance along a ray by tracing it through the scene, accounting for light
    // interactions with surfaces and materials.
    fn li(
        &self,
        initial_ray: &mut Ray,
        world: &dyn Intersectable,
        lights: &[Arc<dyn Sampleable>],
        session: &mut S::Session<'_>,
    ) -> Color3;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bvh::BvhNode;
    use crate::film::{Film, RgbFilm};
    use crate::flat_bvh::FlatBvh;
    use crate::material::Material;
    use crate::planar::quad;
    use crate::sampler::{NaiveRandomSampler, Point2i, Sampler, StreamRngPair};
    use crate::vec3::{Point3, Vec3};

    /// Type shortcut for the concrete sampler used in tests.
    type TestSampler = StreamRngPair<NaiveRandomSampler, NaiveRandomSampler>;

    /// Minimal integration test: render a 4×4 image of a lit Cornell-box-like
    /// scene and verify the output is non-zero and finite.
    #[test]
    fn render_4x4_minimal_scene() {
        // Build a tiny scene: a light quad and a floor quad.
        let light_mat = Material::light(Color3::new(8.0, 8.0, 8.0));
        let floor_mat = Material::lambertian_color(0.7, 0.7, 0.7);

        let light: Arc<dyn Intersectable> = Arc::new(quad(
            Point3::new(-1., 2., -2.),
            Vec3::new(2., 0., 0.),
            Vec3::new(0., 0., 2.),
            light_mat,
        ));
        let floor: Arc<dyn Intersectable> = Arc::new(quad(
            Point3::new(-3., -1., -3.),
            Vec3::new(6., 0., 0.),
            Vec3::new(0., 0., 6.),
            floor_mat,
        ));

        let light_sample: Arc<dyn Sampleable> = Arc::new(quad(
            Point3::new(-1., 2., -2.),
            Vec3::new(2., 0., 0.),
            Vec3::new(0., 0., 2.),
            Material::light(Color3::new(8.0, 8.0, 8.0)),
        ));

        let mut objects: Vec<Arc<dyn Intersectable>> = vec![light, floor];
        let world = FlatBvh::from(BvhNode::new(&mut objects));
        let lights: Vec<Arc<dyn Sampleable>> = vec![light_sample];

        let integrator = PathTracingIntegrator::new(8, Color3::new(0.0, 0.0, 0.0), None);
        let mut sampler = StreamRngPair::new(
            NaiveRandomSampler::with_seed(42),
            NaiveRandomSampler::with_seed(43),
            1,
        );
        let mut film = RgbFilm::new((4, 4), 1.0, false);

        // Trace a few rays from a fixed origin.
        for y in 0..4u32 {
            for x in 0..4u32 {
                // Simple camera: origin at (0, 0, 4), rays toward -z.
                let u = (x as f64 + 0.5) / 4.0;
                let v = (y as f64 + 0.5) / 4.0;
                let direction = Vec3::new(u - 0.5, v - 0.5, -1.0).unit_vector();
                let mut ray = Ray::new_with_time(Vec3::new(0., 0., 4.), direction, 0.0);

                let mut session = sampler.begin_pixel(
                    Point2i {
                        x: x as i32,
                        y: y as i32,
                    },
                    0,
                );
                let color = <PathTracingIntegrator as Integrator<TestSampler>>::li(
                    &integrator,
                    &mut ray,
                    &world,
                    &lights,
                    &mut session,
                );
                film.add_sample(x, y, color);
            }
        }

        // Verify: every pixel should have finite, non-negative color.
        let rgb = film.to_rgb8();
        assert_eq!(rgb.len(), 4 * 4 * 3);
        // At least some pixels should be non-zero (the light hits the floor).
        let total: u32 = rgb.iter().map(|&b| b as u32).sum();
        assert!(
            total > 0,
            "rendered image should have non-zero pixel values"
        );
    }

    /// Smoke test: verify the integrator runs without panicking and produces
    /// finite output. The two-stream architecture means materials can consume
    /// a variable number of RNG calls, so the old fixed-dim invariant test
    /// no longer applies.
    #[test]
    fn integrator_smoke_test() {
        // Scene: a Lambertian floor and a light quad.
        let floor_mat = Material::lambertian_color(0.7, 0.7, 0.7);

        let floor: Arc<dyn Intersectable> = Arc::new(quad(
            Point3::new(-3., -1., -3.),
            Vec3::new(6., 0., 0.),
            Vec3::new(0., 0., 6.),
            floor_mat,
        ));

        let light_quad: Arc<dyn Intersectable> = Arc::new(quad(
            Point3::new(-1., 2., -2.),
            Vec3::new(2., 0., 0.),
            Vec3::new(0., 0., 2.),
            Material::light(Vec3::new(8.0, 8.0, 8.0)),
        ));

        let light_sample: Arc<dyn Sampleable> = Arc::new(quad(
            Point3::new(-1., 2., -2.),
            Vec3::new(2., 0., 0.),
            Vec3::new(0., 0., 2.),
            Material::light(Vec3::new(8.0, 8.0, 8.0)),
        ));

        let mut objects: Vec<Arc<dyn Intersectable>> = vec![floor, light_quad];
        let world = FlatBvh::from(BvhNode::new(&mut objects));
        let lights: Vec<Arc<dyn Sampleable>> = vec![light_sample];

        let integrator = PathTracingIntegrator::new(8, Color3::ZERO, None);
        let mut sampler = StreamRngPair::new(
            NaiveRandomSampler::with_seed(42),
            NaiveRandomSampler::with_seed(43),
            1,
        );

        let dir = Vec3::new(0.0, -1.0, -1.0).unit_vector();
        let mut ray = Ray::new_with_time(Vec3::new(0., 1.5, 4.), dir, 0.0);
        let mut session = sampler.begin_pixel(Point2i { x: 0, y: 0 }, 0);
        let color = <PathTracingIntegrator as Integrator<TestSampler>>::li(
            &integrator,
            &mut ray,
            &world,
            &lights,
            &mut session,
        );
        assert!(color.x.is_finite() && color.y.is_finite() && color.z.is_finite());
    }
}
