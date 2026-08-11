use std::borrow::Borrow;

use glam::Vec3;

use crate::bvh::aabb::Aabb;
use crate::intersect::interaction::Hit;
use crate::material::Material;
use crate::math::interval::Interval;
use crate::math::vec3::{Direction3, Point3};
use crate::ray::Ray;
use crate::texture::UVDifferentiable;

use super::{Shape3D, ShapeObject, ShapeSurfaceSampling};

/// An axis-aligned box defined by its min and max corners.
///
/// Six faces tested inline (no internal BVH — brute-forcing 6 slabs is optimal).
/// Uniform material across all faces; per-face materials require 6 separate quads
/// via the [`box3d`] constructor.
///
/// Wrap via `ShapeObject<BoxShape, M>` or use [`shape_box3d`].
#[derive(Clone)]
pub struct BoxShape {
    /// The minimum corner of the box (smallest x, y, z).
    min: Point3,
    /// The maximum corner of the box (largest x, y, z).
    max: Point3,
    /// Face areas for sampling: [±x, ±y, ±z].
    face_areas: [f32; 6],
    /// Cumulative area for weighted face selection in sample().
    total_area: f32,
    /// Precomputed side lengths for UV computation.
    dx: f32,
    dy: f32,
    dz: f32,
}

impl BoxShape {
    /// Creates an axis-aligned box from two corner points.
    /// The corners are sorted internally so `min` ≤ `max` on every axis.
    pub fn new(a: Point3, b: Point3) -> Self {
        // Sort corners so every axis has min ≤ max.
        let min = a.min(b.into_inner());
        let max = a.max(b.into_inner());

        // Compute side lengths for UV mapping and area calculations.
        let dx = max.x() - min.x();
        let dy = max.y() - min.y();
        let dz = max.z() - min.z();

        // Face areas: two of each orientation → total = 2(area_xy + area_yz + area_zx).
        // ±X faces: dy*dz, ±Y faces: dx*dz, ±Z faces: dx*dy.
        let area_x = dy * dz;
        let area_y = dx * dz;
        let area_z = dx * dy;

        let total_area = 2.0 * (area_x + area_y + area_z);

        Self {
            min,
            max,
            face_areas: [area_x, area_x, area_y, area_y, area_z, area_z],
            total_area,
            dx,
            dy,
            dz,
        }
    }

    /// Returns the face hit and parametric `(a, b)` coordinates for the ray intersection.
    ///
    /// Face indices: 0=+x, 1=-x, 2=+y, 3=-y, 4=+z, 5=-z.
    ///
    /// Kay–Kajiya slab method: for each axis, find the ray's entry/exit distances
    /// along that axis (t_near, t_far). The 3D intersection is the overlap of all
    /// three axis intervals. The axis that produced `t_enter` tells us which face
    /// was hit.
    fn intersect_faces(&self, ray: &Ray, ray_t: Interval) -> Option<(usize, f32, f32, f32)> {
        let inv_d = ray.inverse_direction().into_inner();

        let origin = ray.origin().into_inner();

        // Slab intersection for each axis.
        let t1 = (self.min.into_inner() - origin) * inv_d;
        let t2 = (self.max.into_inner() - origin) * inv_d;

        // Compute the near and far intersection distances along each axis.
        let t_near = t1.min(t2);
        let t_far = t1.max(t2);

        // The overall entry and exit distances are the max of the near distances
        // and the min of the far distances. If they don't overlap, there's no hit.
        let t_enter = t_near.max_element();
        let t_exit = t_far.min_element();

        // If the ray misses the box or is outside the valid t range, return None.
        if t_enter >= t_exit || t_exit <= ray_t.min || t_enter >= ray_t.max {
            return None;
        }

        // Determine the actual hit time within the ray's valid interval.
        let t = if t_enter > ray_t.min { t_enter } else { t_exit };
        if t <= ray_t.min || t >= ray_t.max {
            return None;
        }

        // Compute the hit point in world space.
        let hit_point = origin + ray.direction().into_inner() * t;

        // Determine which face was hit by tracking which axis produced t_enter
        // in the max() chain. Direct f32 comparison is correct here because
        // max(a, b, c) guarantees result == one of the inputs.
        //
        // Entry face mapping (Kay–Kajiya):
        //   dir.x > 0 → entry at min.x → -X face (1)
        //   dir.x < 0 → entry at max.x → +X face (0)
        // Same pattern for Y and Z.
        //
        // When the ray starts inside, we use the exit point. XOR 1 flips to
        // the opposite face of the pair (0↔1, 2↔3, 4↔5).
        let is_entry = t == t_enter;

        let (enter, a, b) = if t_enter == t_near.x {
            let a = (hit_point.y - self.min.y()) / self.dy;
            let b = (hit_point.z - self.min.z()) / self.dz;
            (if inv_d.x > 0.0 { 1 } else { 0 }, a, b)
        } else if t_enter == t_near.y {
            let a = (hit_point.x - self.min.x()) / self.dx;
            let b = (hit_point.z - self.min.z()) / self.dz;
            (if inv_d.y > 0.0 { 3 } else { 2 }, a, b)
        } else {
            let a = (hit_point.x - self.min.x()) / self.dx;
            let b = (hit_point.y - self.min.y()) / self.dy;
            (if inv_d.z > 0.0 { 5 } else { 4 }, a, b)
        };

        // XOR 1 flips entry→exit face for inside-box rays.
        let face = if is_entry { enter } else { enter ^ 1 };
        Some((face, t, a, b))
    }
}

impl UVDifferentiable for BoxShape {
    fn uv_gradient(&self, mapping_point: &Point3) -> (Direction3, Direction3) {
        // Box faces are flat quads — UV gradients are constant per face.
        // Determine which face by finding the closest slab boundary.
        // The hit point is on the surface, so one coordinate is always at a
        // boundary. At edges/corners, the closest-boundary approach picks
        // deterministically. The result is only used for texture filtering
        // footprints, so measure-zero ambiguous cases are harmless.
        let p = mapping_point.into_inner();
        let d = (p - self.min.into_inner())
            .abs()
            .min((self.max.into_inner() - p).abs());

        if d.x <= d.y && d.x <= d.z {
            // ±X face: u = y/dy, v = z/dz
            (Direction3::Y / self.dy, Direction3::Z / self.dz)
        } else if d.y <= d.x && d.y <= d.z {
            // ±Y face: u = x/dx, v = z/dz
            (Direction3::X / self.dx, Direction3::Z / self.dz)
        } else {
            // ±Z face: u = x/dx, v = y/dy
            (Direction3::X / self.dx, Direction3::Y / self.dy)
        }
    }
}

impl Shape3D for BoxShape {
    fn intersect_shape(&self, ray: &Ray, ray_t: Interval) -> Option<Hit> {
        let (face, t, a, b) = self.intersect_faces(ray, ray_t)?;
        let point = ray.at(t);

        // Face → outward normal. Face indices match intersect_faces:
        // 0=+x, 1=-x, 2=+y, 3=-y, 4=+z, 5=-z.
        let normal = match face {
            0 => Direction3::X,
            1 => Direction3::NEG_X,
            2 => Direction3::Y,
            3 => Direction3::NEG_Y,
            4 => Direction3::Z,
            _ => Direction3::NEG_Z,
        };

        let hit = Hit::new(t, point, point, normal, Some((a, b)), None);
        Some(hit)
    }

    fn bounding_box(&self) -> Aabb {
        Aabb::from_corners(self.min, self.max)
    }
}

impl ShapeSurfaceSampling for BoxShape {
    fn area(&self) -> f32 {
        self.total_area
    }

    fn sample(&self, u: f32, v: f32, _time: f32) -> (Point3, Direction3) {
        // Select face by area-weighted distribution.
        let r = u * self.total_area;
        let mut cumulative = 0.0;
        let mut face = 0usize;
        for i in 0..6 {
            cumulative += self.face_areas[i];
            if r < cumulative {
                face = i;
                break;
            }
        }

        // Compute face corner and side vectors, then sample within the face.
        let (corner, side_a, side_b, normal) = match face {
            0 => (
                Point3::new(self.max.x(), self.min.y(), self.min.z()),
                Vec3::new(0.0, self.dy, 0.0),
                Vec3::new(0.0, 0.0, self.dz),
                Direction3::X,
            ),
            1 => (
                Point3::new(self.min.x(), self.min.y(), self.min.z()),
                Vec3::new(0.0, self.dy, 0.0),
                Vec3::new(0.0, 0.0, self.dz),
                Direction3::NEG_X,
            ),
            2 => (
                Point3::new(self.min.x(), self.max.y(), self.min.z()),
                Vec3::new(self.dx, 0.0, 0.0),
                Vec3::new(0.0, 0.0, self.dz),
                Direction3::Y,
            ),
            3 => (
                Point3::new(self.min.x(), self.min.y(), self.min.z()),
                Vec3::new(self.dx, 0.0, 0.0),
                Vec3::new(0.0, 0.0, self.dz),
                Direction3::NEG_Y,
            ),
            4 => (
                Point3::new(self.min.x(), self.min.y(), self.max.z()),
                Vec3::new(self.dx, 0.0, 0.0),
                Vec3::new(0.0, self.dy, 0.0),
                Direction3::Z,
            ),
            _ => (
                Point3::new(self.min.x(), self.min.y(), self.min.z()),
                Vec3::new(self.dx, 0.0, 0.0),
                Vec3::new(0.0, self.dy, 0.0),
                Direction3::NEG_Z,
            ),
        };

        // Re-normalize u to within-face coordinate so QMC stratification
        // is preserved: u selects the face, then u_face is uniform in [0,1)
        // on that face.
        let prev_cumulative = cumulative - self.face_areas[face];
        let u_face = if self.total_area > 0.0 {
            (r - prev_cumulative) / self.face_areas[face]
        } else {
            0.0
        };
        let point = corner + side_a * v + side_b * u_face;
        (point, normal)
    }
}

/// A material-wrapped axis-aligned box. Single material on all 6 faces.
///
/// For per-face materials, create 6 independent quads via [`box3d`].
pub type Box3D<M> = ShapeObject<BoxShape, M>;

/// Creates an axis-aligned box from corners `a` and `b` with a single material
/// on all 6 faces.
pub fn shape_box3d<M: Borrow<Material>>(a: Point3, b: Point3, material: M) -> Box3D<M> {
    ShapeObject::new(BoxShape::new(a, b), material)
}
