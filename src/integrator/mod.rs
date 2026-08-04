use std::sync::Arc;

use crate::intersect::Intersectable;
use crate::light::environment::EnvironmentMap;
use crate::math::vec3::{Color3, Direction3};
use crate::primitives::LightPrimitive;
use crate::ray::Ray;
use crate::sampler::{SampleStream, SamplerRng};

pub mod path_tracer;
pub use path_tracer::PathTracingIntegrator;

/// A trait for integrators that compute radiance along rays in a scene.
///
/// Integrators are responsible for tracing rays through the scene, handling light interactions with
/// surfaces and materials, and returning the resulting color. They may also provide background
/// radiance for rays that miss all geometry.
pub trait Integrator: Send + Sync {
    /// Default background radiance for a ray that missed all geometry.
    fn background(&self, direction: Direction3) -> Color3 {
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
    fn li<S: SampleStream, R: SamplerRng>(
        &self,
        initial_ray: &mut Ray,
        world: &dyn Intersectable,
        lights: &[LightPrimitive],
        stream: &mut S,
        rng: &mut R,
    ) -> Color3;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bvh::Bvh;
    use crate::bvh::builder::TreeBuilder;
    use crate::film::{Film, RgbFilm};
    use crate::material::{DiffuseEmitterMaterial, DiffuseReflector, Material};
    use crate::math::vec3::{Color3, Direction3, Point3};
    use crate::primitives::{LightPrimitive, Primitive};
    use crate::sampler::NaiveRandomSampler;
    use crate::shape::quad;
    use glam::Vec3;

    /// Minimal integration test: render a 4×4 image of a lit Cornell-box-like
    /// scene and verify the output is non-zero and finite.
    #[test]
    fn render_4x4_minimal_scene() {
        // Build a tiny scene: a light quad and a floor quad.
        let light_mat: Material = DiffuseEmitterMaterial::new(Color3::new(8.0, 8.0, 8.0)).into();
        let floor_mat: Material = DiffuseReflector::new(Color3::new(0.7, 0.7, 0.7)).into();

        let light: Primitive = quad(
            Point3::new(-1., 2., -2.),
            Vec3::new(2., 0., 0.),
            Vec3::new(0., 0., 2.),
            light_mat,
        )
        .into();
        let floor: Primitive = quad(
            Point3::new(-3., -1., -3.),
            Vec3::new(6., 0., 0.),
            Vec3::new(0., 0., 6.),
            floor_mat,
        )
        .into();

        let light_sample: LightPrimitive = Primitive::from(quad(
            Point3::new(-1., 2., -2.),
            Vec3::new(2., 0., 0.),
            Vec3::new(0., 0., 2.),
            Material::from(DiffuseEmitterMaterial::new(Color3::new(8.0, 8.0, 8.0))),
        ))
        .into();

        let mut objects: Vec<Primitive> = vec![light, floor];
        let world = Bvh::<8, _>::from(TreeBuilder::new(&mut objects));
        let lights: Vec<LightPrimitive> = vec![light_sample];

        let integrator = PathTracingIntegrator::new(8, Color3::ZERO, None);
        let mut stream = NaiveRandomSampler::with_seed(42);
        let mut rng = NaiveRandomSampler::with_seed(43);
        let mut film = RgbFilm::new((4, 4), 1.0, false);

        // Trace a few rays from a fixed origin.
        for y in 0..4u32 {
            for x in 0..4u32 {
                // Simple camera: origin at (0, 0, 4), rays toward -z.
                let u = (x as f32 + 0.5) / 4.0;
                let v = (y as f32 + 0.5) / 4.0;
                let direction = Vec3::new(u - 0.5, v - 0.5, -1.0).normalize();
                let mut ray =
                    Ray::new_with_time(Point3::new(0., 0., 4.), Direction3(direction), 0.0);

                let color = integrator.li(&mut ray, &world, &lights, &mut stream, &mut rng);
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
        let floor_mat: Material = DiffuseReflector::new(Color3::new(0.7, 0.7, 0.7)).into();

        let floor: Primitive = quad(
            Point3::new(-3., -1., -3.),
            Vec3::new(6., 0., 0.),
            Vec3::new(0., 0., 6.),
            floor_mat,
        )
        .into();

        let light_quad: Primitive = quad(
            Point3::new(-1., 2., -2.),
            Vec3::new(2., 0., 0.),
            Vec3::new(0., 0., 2.),
            Material::from(DiffuseEmitterMaterial::new(Color3::splat(8.0))),
        )
        .into();

        let light_sample: LightPrimitive = Primitive::from(quad(
            Point3::new(-1., 2., -2.),
            Vec3::new(2., 0., 0.),
            Vec3::new(0., 0., 2.),
            Material::from(DiffuseEmitterMaterial::new(Color3::splat(8.0))),
        ))
        .into();

        let mut objects: Vec<Primitive> = vec![floor, light_quad];
        let world = Bvh::<8, _>::from(TreeBuilder::new(&mut objects));
        let lights: Vec<LightPrimitive> = vec![light_sample];

        let integrator = PathTracingIntegrator::new(8, Color3::ZERO, None);
        let mut stream = NaiveRandomSampler::with_seed(42);
        let mut rng = NaiveRandomSampler::with_seed(43);

        let dir = Vec3::new(0.0, -1.0, -1.0).normalize();
        let mut ray = Ray::new_with_time(Point3::new(0., 1.5, 4.), Direction3(dir), 0.0);
        let color = integrator.li(&mut ray, &world, &lights, &mut stream, &mut rng);
        assert!(color.x().is_finite() && color.y().is_finite() && color.z().is_finite());
    }
}
