use crate::intersect::Intersectable;
use crate::intersect::interaction::MaterialHit;
use crate::math::interval::Interval;
use crate::math::vec3::{Color3, Direction3};
use crate::primitives::LightPrimitive;
use crate::ray::Ray;
use crate::sampler::{SampleStream, SamplerRng};

pub mod path_tracer;
pub use path_tracer::PathTracingIntegrator;

/// Maximum depth for the split delta path (mirror direction from Mix one-delta).
/// Prevents exponential cascade when `max_depth` is large (e.g. 50) and the
/// mirror direction repeatedly hits delta-Mix surfaces. Matches
/// `MAX_INTERNAL_BOUNCES` in Coated material.
const SPLIT_MAX_DEPTH: u32 = 5;

// Result of a single bounce of path tracing, returned by `process_bounce`.
#[derive(Clone, Copy)]
pub struct BounceResult<S> {
    /// Direct contribution: emission + NEE, already MIS-weighted.
    pub contribution: Color3,
    /// Continuation ray, or None if the path terminates.
    pub next_ray: Option<Ray>,
    /// Delta child ray from Split, or None.
    pub delta_child: Option<(Ray, S)>, // integrator builds the child state
}

/// A trait for integrators that compute radiance along rays in a scene.
///
/// Integrators are responsible for tracing rays through the scene, handling light interactions with
/// surfaces and materials, and returning the resulting color. They may also provide background
/// radiance for rays that miss all geometry.
pub trait Integrator: Send + Sync {
    /// Per-path state. Owned, no borrows — the GAT is deferred until a real
    /// borrowing state exists (one implementor in-crate, so the upgrade is
    /// a contained breaking change).
    type PathState: Clone + Send + Sync + Default;

    /// Returns the maximum depth (number of bounces) for ray tracing.
    fn max_depth(&self) -> u32;

    /// Returns the background radiance for rays that miss all geometry.
    fn init_state(&self) -> Self::PathState {
        Self::PathState::default()
    }

    /// One bounce of path tracing — the universal stage primitive.
    /// Does NOT do: primary intersection (renderer does), background (renderer
    /// calls eval_background on miss).
    fn process_bounce<S: SampleStream, R: SamplerRng>(
        &self,
        ray: &Ray,
        hit: &MaterialHit<'_>,
        world: &impl Intersectable,
        lights: &[LightPrimitive],
        state: &mut Self::PathState,
        bounce: u32,
        stream: &mut S,
        rng: &mut R,
    ) -> BounceResult<Self::PathState>;

    /// Background radiance for a miss ray, MIS-weighted against the path's
    /// previous BSDF PDF.
    fn eval_background(&self, direction: Direction3, state: &Self::PathState) -> Color3;

    /// Reference full-path driver: intersect → process_bounce → eval_background,
    /// with delta-child recursion. Default method — implementors only provide
    /// the per-bounce primitives. The wavefront renderer re-implements this
    /// loop as per-stage kernels instead of calling it.
    fn li<S: SampleStream, R: SamplerRng>(
        &self,
        initial_ray: &mut Ray,
        world: &impl Intersectable,
        lights: &[LightPrimitive],
        stream: &mut S,
        rng: &mut R,
    ) -> Color3
    where
        Self: Sized,
    {
        trace_path(
            self,
            initial_ray,
            world,
            lights,
            stream,
            rng,
            self.init_state(),
            self.max_depth(),
        )
    }
}

/// The naive per-pixel path loop, shared by the default `li()` and the
/// delta-child recursion. This is the old `li_inner`, extracted as a
/// module-level generic function so the default `li()` can drive any
/// integrator.
fn trace_path<I: Integrator, S: SampleStream, R: SamplerRng>(
    integrator: &I,
    initial_ray: &mut Ray,
    world: &impl Intersectable,
    lights: &[LightPrimitive],
    stream: &mut S,
    rng: &mut R,
    mut state: I::PathState,
    remaining_depth: u32,
) -> Color3 {
    let mut accumulated = Color3::ZERO;
    let mut ray = *initial_ray;
    for bounce in 0..remaining_depth {
        if let Some(mat_hit) = world.intersect(&ray, Interval::from(0.001, f32::INFINITY)) {
            let result = integrator.process_bounce(
                &ray, &mat_hit, world, lights, &mut state, bounce, stream, rng,
            );
            accumulated += result.contribution;

            // Trace the split's delta child (mirror direction) with its own
            // fresh state, capped by SPLIT_MAX_DEPTH.
            if let Some((mut child_ray, child_state)) = result.delta_child {
                let depth = remaining_depth
                    .saturating_sub(bounce + 1)
                    .min(SPLIT_MAX_DEPTH);
                accumulated += trace_path(
                    integrator,
                    &mut child_ray,
                    world,
                    lights,
                    stream,
                    rng,
                    child_state,
                    depth,
                );
            }

            match result.next_ray {
                Some(next) => ray = next,
                None => return accumulated,
            }
        } else {
            // Ray missed the world geometry — accumulate background and terminate.
            return accumulated + integrator.eval_background(ray.direction.normalize(), &state);
        }
    }

    // Max bounce count reached — terminate the path. This can still contribute
    // to the final image if the last bounce was a non-delta and the accumulated
    // attenuation is non-zero.
    accumulated
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
