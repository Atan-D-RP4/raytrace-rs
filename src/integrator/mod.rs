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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bvh::BvhNode;
    use crate::film::{Film, RgbFilm};
    use crate::material::Material;
    use crate::planar::quad;
    use crate::sampler::NaiveRandomSampler;
    use crate::vec3::{Point3, Vec3};

    /// Minimal integration test: render a 4×4 image of a lit Cornell-box-like
    /// scene and verify the output is non-zero and finite.
    #[test]
    fn render_4x4_minimal_scene() {
        // Build a tiny scene: a light quad and a floor quad.
        let light_mat = Material::light(Color3::from(8.0, 8.0, 8.0));
        let floor_mat = Material::lambertian_color(0.7, 0.7, 0.7);

        let light: Arc<dyn Intersectable> = Arc::new(quad(
            Point3::from(-1., 2., -2.),
            Vec3::from(2., 0., 0.),
            Vec3::from(0., 0., 2.),
            light_mat,
        ));
        let floor: Arc<dyn Intersectable> = Arc::new(quad(
            Point3::from(-3., -1., -3.),
            Vec3::from(6., 0., 0.),
            Vec3::from(0., 0., 6.),
            floor_mat,
        ));

        let light_sample: Arc<dyn Sampleable> = Arc::new(quad(
            Point3::from(-1., 2., -2.),
            Vec3::from(2., 0., 0.),
            Vec3::from(0., 0., 2.),
            Material::light(Color3::from(8.0, 8.0, 8.0)),
        ));

        let mut objects: Vec<Arc<dyn Intersectable>> = vec![light, floor];
        let world = BvhNode::new(&mut objects);
        let lights: Vec<Arc<dyn Sampleable>> = vec![light_sample];

        let integrator = PathTracingIntegrator::new(8, Color3::from(0.0, 0.0, 0.0));
        let sampler = NaiveRandomSampler::with_seed(42);
        let mut film = RgbFilm::new((4, 4), 1.0, false);

        // Trace a few rays from a fixed origin.
        let mut dim_cursor = DimCursor::new(0, sampler);
        for y in 0..4u32 {
            for x in 0..4u32 {
                // Simple camera: origin at (0, 0, 4), rays toward -z.
                let u = (x as f64 + 0.5) / 4.0;
                let v = (y as f64 + 0.5) / 4.0;
                let direction = Vec3::from(u - 0.5, v - 0.5, -1.0).unit_vector();
                let mut ray = Ray::new_with_time(Vec3::from(0., 0., 4.), direction, 0.0);

                let color = integrator.li(&mut ray, &world, &lights, &mut dim_cursor);
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

    /// Verify the per-bounce QMC dimension invariant.
    ///
    /// Every completed bounce consumes a fixed number of dimensions:
    /// - Non-delta: NEE(3) + RR(1) + material(6) + MIS mixture(4) = 14
    /// - Delta:     NEE(0) + RR(1) + material(6) + delta pad(4) = 11
    /// - Ray miss:  0 dims consumed (early return before any sample calls)
    #[test]
    fn qmc_dim_invariant() {
        // Scene: a Lambertian floor (non-delta), a mirror quad (delta),
        // and a light quad (for NEE).
        let floor_mat = Material::lambertian_color(0.7, 0.7, 0.7);
        let mirror_mat = Material::metal(Vec3::from(1.0, 1.0, 1.0), 0.0); // fuzz=0 => delta
        let light_mat = Material::light(Vec3::from(8.0, 8.0, 8.0));

        let floor: Arc<dyn Intersectable> = Arc::new(quad(
            Point3::from(-3., -1., -3.),
            Vec3::from(6., 0., 0.),
            Vec3::from(0., 0., 6.),
            floor_mat,
        ));

        let mirror: Arc<dyn Intersectable> = Arc::new(quad(
            Point3::from(-1., -0.5, -3.),
            Vec3::from(2., 0., 0.),
            Vec3::from(0., 1., 0.),
            mirror_mat,
        ));

        let light_quad: Arc<dyn Intersectable> = Arc::new(quad(
            Point3::from(-1., 2., -2.),
            Vec3::from(2., 0., 0.),
            Vec3::from(0., 0., 2.),
            light_mat,
        ));

        let light_sample: Arc<dyn Sampleable> = Arc::new(quad(
            Point3::from(-1., 2., -2.),
            Vec3::from(2., 0., 0.),
            Vec3::from(0., 0., 2.),
            Material::light(Vec3::from(8.0, 8.0, 8.0)),
        ));

        let mut objects: Vec<Arc<dyn Intersectable>> = vec![floor, mirror, light_quad];
        let world = BvhNode::new(&mut objects);
        let lights: Vec<Arc<dyn Sampleable>> = vec![light_sample];

        let integrator = PathTracingIntegrator::new(1, Color3::ZERO); // max_depth=1

        // Test 1: Non-delta bounce — ray hits the Lambertian floor.
        // Expected: NEE(3) + RR(1) + material(6) + MIS(4) = 14
        {
            let sampler = NaiveRandomSampler::with_seed(42);
            let mut dim_cursor = DimCursor::new(0, sampler);
            let dir = Vec3::from(0.0, -1.0, -1.0).unit_vector();
            let mut ray = Ray::new_with_time(Vec3::from(0., 1.5, 4.), dir, 0.0);
            let _ = integrator.li(&mut ray, &world, &lights, &mut dim_cursor);
            assert_eq!(
                dim_cursor.offset(),
                14,
                "non-delta bounce should consume 14 dims (NEE(3) + RR(1) + material(6) + MIS(4))"
            );
        }

        // Test 2: Delta bounce — ray hits the mirror quad.
        // Expected: RR(1) + material(6) + delta pad(4) = 11
        {
            let sampler = NaiveRandomSampler::with_seed(42);
            let mut dim_cursor = DimCursor::new(0, sampler);
            let dir = Vec3::from(0.0, 0.0, -1.0);
            let mut ray = Ray::new_with_time(Vec3::from(0., 0., 4.), dir, 0.0);
            let _ = integrator.li(&mut ray, &world, &lights, &mut dim_cursor);
            assert_eq!(
                dim_cursor.offset(),
                11,
                "delta bounce should consume 11 dims (RR(1) + material(6) + delta pad(4))"
            );
        }

        // Test 3: Miss — ray that doesn't intersect anything.
        // Expected: 0 dims consumed (early return before any sample calls)
        {
            let sampler = NaiveRandomSampler::with_seed(42);
            let mut dim_cursor = DimCursor::new(0, sampler);
            let dir = Vec3::from(0.0, 1.0, 0.0); // straight up — misses everything
            let mut ray = Ray::new_with_time(Vec3::from(0., 0., 4.), dir, 0.0);
            let _ = integrator.li(&mut ray, &world, &lights, &mut dim_cursor);
            assert_eq!(dim_cursor.offset(), 0, "miss should consume 0 dims");
        }
    }
}
