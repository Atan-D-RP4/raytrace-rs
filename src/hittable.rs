use std::sync::Arc;

use crate::aabb::Aabb;
use crate::interval::Interval;
use crate::material::Material;
use crate::ray::Ray;
use crate::vec3::{Vec3, dot};

pub struct HitRecord {
    pub time: f64,
    pub point: Vec3,
    pub normal: Vec3,
    pub front_face: bool,
    pub material: Arc<Material>,
}

impl HitRecord {
    pub fn new(t: f64, point: Vec3, normal: Vec3, mat: Arc<Material>) -> Self {
        Self {
            time: t,
            point,
            normal,
            front_face: false,
            material: mat,
        }
    }

    pub fn set_face_normal(&mut self, ray: &Ray, outward_normal: &Vec3) {
        self.front_face = dot(&ray.direction, outward_normal) < 0.;
        self.normal = if self.front_face {
            *outward_normal
        } else {
            -(*outward_normal)
        }
    }
}

pub trait Hittable: Send + Sync {
    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord>;
    fn bounding_box(&self) -> Aabb;
}

impl Hittable for Vec<Box<dyn Hittable>> {
    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord> {
        let mut closest = ray_t.max;
        let mut result = None;

        self.iter().for_each(|object| {
            if let Some(record) = object.hit(ray, Interval::from(ray_t.min, closest)) {
                closest = record.time;
                result = Some(record);
            }
        });

        result
    }

    fn bounding_box(&self) -> Aabb {
        self.iter()
            .fold(Aabb::new(), |acc, obj| Aabb::merge(acc, obj.bounding_box()))
    }
}
