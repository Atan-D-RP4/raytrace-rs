use std::f64::consts::PI;

use crate::hittable::Hittable;
use crate::onb::Onb;
use crate::vec3::{Point3, Vec3, random_cosine_direction};

use rand::RngExt;

pub trait PDF {
    /// Evaluates the PDF value for a given direction.
    fn value(&self, direction: Vec3) -> f64;

    /// Generates a random direction according to the PDF.
    fn generate(&self, rng: &mut dyn rand::Rng) -> Vec3;
}

pub struct UniformSpherePDF;

impl PDF for UniformSpherePDF {
    fn value(&self, _direction: Vec3) -> f64 {
        1.0 / (4.0 * std::f64::consts::PI)
    }

    fn generate(&self, rng: &mut dyn rand::Rng) -> Vec3 {
        // Rejection sampling to generate a random unit vector uniformly distributed on the surface of the unit sphere.
        loop {
            let point = Vec3::from(
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
            );
            let len_squared = point.length_squared();
            if 1e-160 < len_squared && len_squared <= 1.0 {
                return point / len_squared.sqrt();
            }
        }
    }
}

#[allow(clippy::new_without_default)]
impl UniformSpherePDF {
    pub fn new() -> Self {
        Self
    }
}

pub struct CosinePDF {
    /// The normal vector defining the hemisphere for cosine-weighted sampling.
    pub uvw: Onb,
}

impl CosinePDF {
    pub fn new(normal: Vec3) -> Self {
        Self {
            uvw: Onb::build_from_normal(normal),
        }
    }
}

impl PDF for CosinePDF {
    fn value(&self, direction: Vec3) -> f64 {
        let cos_theta = direction.unit_vector().dot(&self.uvw.w);
        (cos_theta / PI).max(0.)
    }

    fn generate(&self, rng: &mut dyn rand::Rng) -> Vec3 {
        self.uvw.local_to_world(random_cosine_direction(rng))
    }
}

pub struct HittablePDF<'a> {
    objects: &'a dyn Hittable,
    origin: Point3,
}

impl<'a> HittablePDF<'a> {
    pub fn new(objects: &'a dyn Hittable, origin: Point3) -> Self {
        HittablePDF { objects, origin }
    }
}

impl<'a> PDF for HittablePDF<'a> {
    fn value(&self, direction: Vec3) -> f64 {
        self.objects.pdf_value(self.origin, direction)
    }

    fn generate(&self, rng: &mut dyn rand::Rng) -> Vec3 {
        self.objects.random(self.origin, rng)
    }
}

pub struct MixturePDF<'a> {
    pdfs: Vec<&'a dyn PDF>,
}

impl<'a> MixturePDF<'a> {
    pub fn new(pdfs: Vec<&'a dyn PDF>) -> Self {
        MixturePDF { pdfs }
    }
}

impl<'a> PDF for MixturePDF<'a> {
    fn value(&self, direction: Vec3) -> f64 {
        let weight = 1.0 / self.pdfs.len() as f64;
        self.pdfs
            .iter()
            .map(|pdf| pdf.value(direction) * weight)
            .sum()
    }

    fn generate(&self, rng: &mut dyn rand::Rng) -> Vec3 {
        let index = rng.random_range(0..self.pdfs.len());
        self.pdfs[index].generate(rng)
    }
}
