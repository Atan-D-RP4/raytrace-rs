pub mod path_tracer;

pub use path_tracer::PathTracingIntegrator;

use std::sync::Arc;

use crate::hittable::{Intersectable, Sampleable};
use crate::ray::Ray;
use crate::sampler::{DimCursor, Sampler};
use crate::vec3::Color3;

pub trait Integrator<S: Sampler>: Send + Sync {
    // Computes the radiance along a ray by tracing it through the scene, accounting for light
    // interactions with surfaces and materials.
    fn li(
        &self,
        initial_ray: &mut Ray,
        world: &dyn Intersectable,
        lights: &[Arc<dyn Sampleable>],
        sampler: &mut DimCursor<S>,
    ) -> Color3;
}
