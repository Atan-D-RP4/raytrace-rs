use crate::vec3::{Point3, Vec3};

#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Point3,
    pub direction: Vec3,
    pub time: f64,
    pub inverse_direction: Vec3,
}

impl Ray {
    pub fn new(origin: Point3, direction: Vec3) -> Self {
        Self {
            origin,
            direction,
            time: 0.,
            inverse_direction: Vec3::from(1. / direction.x, 1. / direction.y, 1. / direction.z),
        }
    }

    pub fn new_with_time(origin: Point3, direction: Vec3, time: f64) -> Self {
        Self {
            origin,
            direction,
            time,
            inverse_direction: Vec3::from(1. / direction.x, 1. / direction.y, 1. / direction.z),
        }
    }

    pub fn at(&self, t: f64) -> Point3 {
        let origin: Vec3 = self.origin;
        let direction = self.direction;
        origin + direction * t
    }
}
