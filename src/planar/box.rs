use std::sync::Arc;

use crate::hittable::Hittable;
use crate::material::Material;
use crate::vec3::{Point3, Vec3};

use super::quad;

pub fn box3d(a: Point3, b: Point3, material: Material) -> Vec<Arc<dyn Hittable>> {
    let mut sides: Vec<Arc<dyn Hittable>> = Vec::with_capacity(6);

    let min = Point3::from(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z));
    let max = Point3::from(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z));

    let dx = Vec3::from(max.x - min.x, 0., 0.);
    let dy = Vec3::from(0., max.y - min.y, 0.);
    let dz = Vec3::from(0., 0., max.z - min.z);

    sides.push(Arc::new(quad(
        Point3::from(min.x, min.y, max.z),
        dx,
        dy,
        material.clone(),
    )));
    sides.push(Arc::new(quad(
        Point3::from(max.x, min.y, max.z),
        -dz,
        dy,
        material.clone(),
    )));
    sides.push(Arc::new(quad(
        Point3::from(max.x, min.y, min.z),
        -dx,
        dy,
        material.clone(),
    )));
    sides.push(Arc::new(quad(
        Point3::from(min.x, min.y, min.z),
        dz,
        dy,
        material.clone(),
    )));
    sides.push(Arc::new(quad(
        Point3::from(min.x, max.y, max.z),
        dx,
        -dz,
        material.clone(),
    )));
    sides.push(Arc::new(quad(
        Point3::from(min.x, min.y, min.z),
        dx,
        dz,
        material,
    )));

    sides
}
