use std::sync::Arc;

use crate::hittable::Intersectable;
use crate::material::Material;
use crate::planar::quad;

use glam::Vec3;

use crate::vec3::Point3;

pub fn box3d(a: Point3, b: Point3, material: Material) -> Vec<Arc<dyn Intersectable>> {
    let mut sides: Vec<Arc<dyn Intersectable>> = Vec::with_capacity(6);

    let min = Point3::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z));
    let max = Point3::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z));

    let dx = Vec3::new(max.x - min.x, 0., 0.);
    let dy = Vec3::new(0., max.y - min.y, 0.);
    let dz = Vec3::new(0., 0., max.z - min.z);

    // Front face is CCW from outside, so normal points outwards.
    sides.push(Arc::new(quad(
        Point3::new(min.x, min.y, max.z),
        dx,
        dy,
        material.clone(),
    )));
    // Back face is CCW from outside, so normal points outwards.
    sides.push(Arc::new(quad(
        Point3::new(max.x, min.y, max.z),
        -dz,
        dy,
        material.clone(),
    )));
    // Left face is CCW from outside, so normal points outwards.
    sides.push(Arc::new(quad(
        Point3::new(max.x, min.y, min.z),
        -dx,
        dy,
        material.clone(),
    )));
    // Right face is CCW from outside, so normal points outwards.
    sides.push(Arc::new(quad(
        Point3::new(min.x, min.y, min.z),
        dz,
        dy,
        material.clone(),
    )));
    // Top face is CCW from outside, so normal points outwards.
    sides.push(Arc::new(quad(
        Point3::new(min.x, max.y, max.z),
        dx,
        -dz,
        material.clone(),
    )));
    // Bottom face is CCW from outside, so normal points outwards.
    sides.push(Arc::new(quad(
        Point3::new(min.x, min.y, min.z),
        dx,
        dz,
        material,
    )));

    sides
}
