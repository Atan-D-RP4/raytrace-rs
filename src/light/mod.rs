use std::sync::Arc;

use crate::math::vec3::{Color3, Direction3, Point3};

pub mod environment;

/// Result of sampling a direction toward a light source.
///
/// Contains everything needed to evaluate the direct lighting contribution:
/// the (non-normalized) direction from the surface point to the light,
/// the light's surface normal at the sampled point, the distance, and
/// the area PDF of the sample.
pub struct LightSample {
    /// Non-normalized direction from surface point to the sampled point on the light.
    /// `.unit_vector()` gives the unit direction; `.length()` gives the distance.
    pub direction: Direction3,
    /// Light's outward surface normal at the sampled point (unit length).
    pub normal: Direction3,
    /// Distance from the surface point to the sampled point on the light.
    pub distance: f32,
    /// Area PDF of this sample (probability density per unit area on the light surface).
    pub pdf: f32,
    /// Emission color of the light at the sampled point (radiance).
    pub emission: Color3,
}

pub trait Sampleable: Send + Sync {
    /// Returns the PDF value for sampling this hittable from a given origin and direction.
    /// Default returns 0.0 (no contribution to the PDF).
    fn pdf_value(&self, origin: Point3, direction: Direction3, time: f32) -> f32 {
        let _ = (origin, direction, time);
        0.0
    }

    /// Samples a random direction toward this hittable from a given origin.
    /// Takes `(u, v)` in `[0, 1)` for sampling. Default returns Vec3::ZERO.
    fn random_direction(&self, origin: Point3, u: f32, v: f32, time: f32) -> Direction3 {
        let _ = (origin, u, v, time);
        Direction3::ZERO
    }

    /// Samples a point on the light and returns everything needed for direct lighting:
    /// direction, surface normal, distance, and area PDF.
    ///
    /// The returned [`LightSample`] is self-consistent — the direction points from
    /// `origin` to the sampled point, the normal is the outward normal at that point,
    /// and the distance equals `direction.length()`.
    fn sample_light(&self, origin: Point3, u: f32, v: f32, time: f32) -> LightSample;
}

impl<T: Sampleable + ?Sized> Sampleable for Arc<T> {
    fn pdf_value(&self, origin: Point3, direction: Direction3, time: f32) -> f32 {
        (**self).pdf_value(origin, direction, time)
    }

    fn random_direction(&self, origin: Point3, u: f32, v: f32, time: f32) -> Direction3 {
        (**self).random_direction(origin, u, v, time)
    }

    fn sample_light(&self, origin: Point3, u: f32, v: f32, time: f32) -> LightSample {
        (**self).sample_light(origin, u, v, time)
    }
}
