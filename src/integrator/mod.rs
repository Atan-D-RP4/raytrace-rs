pub mod path_tracer;

pub use path_tracer::PathTracingIntegrator;

use crate::hittable::{Intersectable, Sampleable};
use crate::ray::Ray;
use crate::sampler::{DimCursor, Sampler};

pub trait Integrator<S: Sampler>: Send + Sync {
    // Computes the radiance along a ray by tracing it through the scene, accounting for light
    // interactions with surfaces and materials.
    fn li(
        &self,
        initial_ray: &mut Ray,
        world: &dyn Intersectable,
        lights: &dyn Sampleable<S>,
        sampler: &mut DimCursor<S>,
    ) -> crate::vec3::Color3;
}
