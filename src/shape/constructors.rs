use std::borrow::Borrow;

use glam::Vec3;

use crate::material::Material;

use super::regions::FunctionRegion;
use super::{PlanarShape, ShapeObject};

use crate::math::vec3::Point3;

/// Construct a parallelogram (quad) from corner `Q` and side vectors `u`, `v`.
///
/// Parameter naming matches *Ray Tracing in One Weekend* (RTIOW) notation.
#[allow(non_snake_case)]
pub fn quad<M: Borrow<Material>>(
    Q: Point3,
    u: Vec3,
    v: Vec3,
    material: M,
) -> ShapeObject<PlanarShape, M> {
    let shape = PlanarShape::quad(Q, u, v);
    ShapeObject::new(shape, material)
}

/// Construct an ellipse (unit-disk in parametric space) from center and side vectors.
pub fn ellipse<M: Borrow<Material>>(
    center: Point3,
    side_a: Vec3,
    side_b: Vec3,
    material: M,
) -> ShapeObject<PlanarShape, M> {
    let shape = PlanarShape::ellipse(center, side_a, side_b);
    ShapeObject::new(shape, material)
}

/// Construct a triangle from corner and two side vectors.
pub fn tri<M: Borrow<Material>>(
    corner: Point3,
    side_a: Vec3,
    side_b: Vec3,
    material: M,
) -> ShapeObject<PlanarShape, M> {
    let shape = PlanarShape::tri(corner, side_a, side_b);
    ShapeObject::new(shape, material)
}

/// Construct an annulus (ring) with configurable inner radius.
pub fn annulus<M: Borrow<Material>>(
    center: Point3,
    side_a: Vec3,
    side_b: Vec3,
    inner: f32,
    material: M,
) -> ShapeObject<PlanarShape, M> {
    let shape = PlanarShape::annulus(center, side_a, side_b, inner);
    ShapeObject::new(shape, material)
}

/// Construct a rounded rectangle with configurable corner radius.
pub fn rounded_rect<M: Borrow<Material>>(
    corner: Point3,
    side_a: Vec3,
    side_b: Vec3,
    radius: f32,
    material: M,
) -> ShapeObject<PlanarShape, M> {
    let shape = PlanarShape::rounded_rect(corner, side_a, side_b, radius);
    ShapeObject::new(shape, material)
}

/// Construct a superellipse `|x|ⁿ + |y|ⁿ ≤ 1` with configurable exponent `n`.
pub fn superellipse<M: Borrow<Material>>(
    corner: Point3,
    side_a: Vec3,
    side_b: Vec3,
    n: f32,
    material: M,
) -> ShapeObject<PlanarShape, M> {
    let shape = PlanarShape::superellipse(corner, side_a, side_b, n);
    ShapeObject::new(shape, material)
}

/// Construct an arbitrary N-gon polygon from a list of `(a, b)` vertices.
///
/// The vertex list must be non-empty and describe a simple, closed, counter-clockwise
/// polygon in the parametric (a, b) space of the planar shape.
pub fn polygon<M: Borrow<Material>>(
    corner: Point3,
    side_a: Vec3,
    side_b: Vec3,
    vertices: Vec<(f32, f32)>,
    material: M,
) -> ShapeObject<PlanarShape, M> {
    let shape = PlanarShape::polygon(corner, side_a, side_b, vertices);
    ShapeObject::new(shape, material)
}

/// Construct a boolean-predicate function patch from a `FunctionRegion`.
///
/// The function defines `contains(a, b)` — any (a, b) that satisfies the predicate
/// is inside the shape. Useful for analytical or procedural shapes that don't fit
/// into a fixed parametric form.
pub fn function_patch<M: Borrow<Material>>(
    corner: Point3,
    side_a: Vec3,
    side_b: Vec3,
    region: FunctionRegion,
    material: M,
) -> ShapeObject<PlanarShape, M> {
    let shape = PlanarShape::function(corner, side_a, side_b, region);
    ShapeObject::new(shape, material)
}
