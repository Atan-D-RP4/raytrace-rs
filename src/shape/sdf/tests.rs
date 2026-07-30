use crate::interval::Interval;
use crate::ray::Ray;
use crate::shape::*;

struct TestSphere {
    radius: f32,
}

impl SdfFn for TestSphere {
    fn eval<T: Scalar>(&self, x: T, y: T, z: T) -> T {
        (x * x + y * y + z * z).sqrt() - T::from_f32(self.radius)
    }
}

#[test]
fn sphere_tracing_hits_sphere_from_outside() {
    let sdf = SdfShape::new(
        TestSphere { radius: 1.0 },
        Aabb::from_corners(Point3::new(-1.1, -1.1, -1.1), Point3::new(1.1, 1.1, 1.1)),
    );
    // Ray from (0, 0, 5) toward origin along -z
    let ray = Ray::new(Point3::new(0.0, 0.0, 5.0), Direction3::new(0.0, 0.0, -1.0));
    let ray_t = Interval::from(0.001, 100.0);
    let hit = sdf.intersect_shape(&ray, ray_t);
    assert!(hit.is_some(), "sphere should be hit");
    let hit = hit.unwrap();
    // Expected: hit at t ≈ 4 (from z=5, sphere surface at z=1)
    assert!((hit.time - 4.0).abs() < 0.01, "hit at t={}", hit.time);
    // Front face at z=1 (sphere center at origin, radius 1)
    assert!(
        (hit.point.z() - 1.0).abs() < 0.01,
        "hit point z={}",
        hit.point.z()
    );
}

#[test]
fn sphere_tracing_misses() {
    let sdf = SdfShape::new(
        TestSphere { radius: 1.0 },
        Aabb::from_corners(Point3::new(-1.1, -1.1, -1.1), Point3::new(1.1, 1.1, 1.1)),
    );
    let ray = Ray::new(
        Point3::new(0.0, 0.0, 5.0),
        Direction3::new(0.0, 2.0, -1.0).normalize(),
    );
    let ray_t = Interval::from(0.001, 100.0);
    let hit = sdf.intersect_shape(&ray, ray_t);
    assert!(
        hit.is_none(),
        "should miss, but got hit at t={:?}",
        hit.map(|h| h.time)
    );
}

#[test]
fn cylinder_sdf_hit_from_camera_angle() {
    struct Cylinder {
        r: f32,
        h: f32,
    }
    impl SdfFn for Cylinder {
        fn eval<T: Scalar>(&self, x: T, y: T, z: T) -> T {
            let d = (x * x + z * z).sqrt() - T::from_f32(self.r);
            let h = y.abs() - T::from_f32(self.h / 2.0);
            d.max(h)
        }
    }
    let sdf = SdfShape::new(
        Cylinder { r: 50.0, h: 100.0 },
        Aabb::from_corners(
            Point3::new(-50.0, -50.0, -50.0),
            Point3::new(50.0, 50.0, 50.0),
        ),
    );
    // Camera at (278, 278, -800), ray toward cylinder center (0, 0, 0)
    let origin = Point3::new(278.0, 278.0, -800.0);
    let dir = Direction3::new(-278.0, -278.0, 800.0).normalize();
    let ray = Ray::new(origin, dir);
    let ray_t = Interval::from(0.001, 1000.0);
    let hit = sdf.intersect_shape(&ray, ray_t);
    assert!(hit.is_some(), "camera ray should hit cylinder");
    let hit = hit.unwrap();
    // Hit should be within the cylinder's geometry
    assert!(
        hit.time > 800.0 && hit.time < 900.0,
        "unexpected t={}",
        hit.time
    );
    assert!(
        hit.point.x() > -50.0 && hit.point.x() < 50.0,
        "x out of range: {}",
        hit.point.x()
    );
    assert!(
        hit.point.z() > -50.0 && hit.point.z() < 50.0,
        "z out of range: {}",
        hit.point.z()
    );
}

#[test]
fn cylinder_sdf_does_not_self_intersect_from_surface() {
    struct Cylinder {
        r: f32,
        h: f32,
    }
    impl SdfFn for Cylinder {
        fn eval<T: Scalar>(&self, x: T, y: T, z: T) -> T {
            let d = (x * x + z * z).sqrt() - T::from_f32(self.r);
            let h = y.abs() - T::from_f32(self.h / 2.0);
            d.max(h)
        }
    }
    let sdf = SdfShape::new(
        Cylinder { r: 50.0, h: 100.0 },
        Aabb::from_corners(
            Point3::new(-50.0, -50.0, -50.0),
            Point3::new(50.0, 50.0, 50.0),
        ),
    );
    // Surface-originating ray: starts at the cylinder surface pointing outward along +x.
    // Without the self-intersection guard, this would return a hit at t ≈ 0.001.
    // With the guard, it should escape the surface and find no hit within a short interval.
    let origin = Point3::new(50.0, 0.0, 0.0);
    let dir = Direction3::new(1.0, 0.0, 0.0); // outward from surface
    let ray = Ray::new(origin, dir);
    // Short interval: just enough to escape the self-intersection zone
    let ray_t = Interval::from(0.001, 0.1);
    let hit = sdf.intersect_shape(&ray, ray_t);
    assert!(
        hit.is_none(),
        "surface-originating ray should not self-intersect, but got hit at t={:?}",
        hit.map(|h| h.time)
    );
}

#[test]
fn cylinder_sdf_non_normalized_camera_ray() {
    // Simulates the camera ray from the Cornell box sdf_test scene:
    //   camera at (278, 278, -800), direction toward cylinder (0, 0, 0)
    //   → direction = (-278, -278, 800) with length ≈ 891.
    // Before the inv_dir_len fix, the sphere-tracing step `t += d` would
    // overshoot the cylinder (first step jumps from t=0.001 to t=≈797),
    // skipping the surface at t≈0.941 entirely.
    struct Cylinder {
        r: f32,
        h: f32,
    }
    impl SdfFn for Cylinder {
        fn eval<T: Scalar>(&self, x: T, y: T, z: T) -> T {
            let d = (x * x + z * z).sqrt() - T::from_f32(self.r);
            let h = y.abs() - T::from_f32(self.h / 2.0);
            d.max(h)
        }
    }
    let sdf = SdfShape::new(
        Cylinder { r: 50.0, h: 100.0 },
        Aabb::from_corners(
            Point3::new(-50.0, -50.0, -50.0),
            Point3::new(50.0, 50.0, 50.0),
        ),
    );
    // Camera ray: non-normalized direction, |dir| ≈ 891
    let origin = Point3::new(278.0, 278.0, -800.0);
    let dir = Point3::new(0.0, 0.0, 0.0) - origin; // (-278, -278, 800)
    let ray = Ray::new(
        Point3::new(278.0, 278.0, -800.0),
        Direction3(dir.into_inner()),
    );
    let ray_t = Interval::from(0.001, 1000.0);
    let hit = sdf.intersect_shape(&ray, ray_t);
    assert!(hit.is_some(), "camera ray should hit cylinder");
    let hit = hit.unwrap();
    // Hit should be at t ≈ 0.941 (ray-parameter units), not t ≈ 797
    // which is what the unfixed sphere tracing would produce.
    assert!(
        hit.time > 0.9 && hit.time < 1.0,
        "expected t≈0.941, got t={}",
        hit.time
    );
    assert!(
        (hit.point.x() - (-50.0)).abs() > 40.0,
        "hit should be on front face (not opposite side), got x={}",
        hit.point.x()
    );
}

#[test]
fn cylinder_sdf_self_intersection_guard_still_hits_real_surface() {
    struct Cylinder {
        r: f32,
        h: f32,
    }
    impl SdfFn for Cylinder {
        fn eval<T: Scalar>(&self, x: T, y: T, z: T) -> T {
            let d = (x * x + z * z).sqrt() - T::from_f32(self.r);
            let h = y.abs() - T::from_f32(self.h / 2.0);
            d.max(h)
        }
    }
    let sdf = SdfShape::new(
        Cylinder { r: 50.0, h: 100.0 },
        Aabb::from_corners(
            Point3::new(-50.0, -50.0, -50.0),
            Point3::new(50.0, 50.0, 50.0),
        ),
    );
    // Surface-originating ray with a LONG interval should still hit the
    // OPPOSITE side of the cylinder after escaping the start surface.
    let origin = Point3::new(50.0, 0.0, 0.0);
    let dir = Direction3::new(1.0, 0.0, 0.0); // through +x, should exit AABB
    let ray = Ray::new(origin, dir);
    // Long enough to pass through the cylinder and exit AABB
    let ray_t = Interval::from(0.001, 200.0);
    let hit = sdf.intersect_shape(&ray, ray_t);
    // No hit because the ray goes away from the cylinder (outward +x)
    // Actually, the cylinder extends from -50 to 50 in x, so at x=50 the
    // outward+1 ray goes away from the cylinder and never hits it again.
    // This is the correct behavior for a surface-originating outward ray.
    // For a ray going inward (through the cylinder), we should get a hit
    // on the opposite side.
    assert!(
        hit.is_none(),
        "outward surface ray should not re-hit cylinder, but got hit at t={:?}",
        hit.map(|h| h.time)
    );

    // Now test an inward ray: from the surface going INTO the cylinder.
    // With the was_outside=false forward-step strategy, the sphere tracing
    // traverses the interior and converges to the opposite surface.
    let dir = Direction3::new(-1.0, 0.0, 0.0); // inward through cylinder
    let ray = Ray::new(origin, dir);
    let hit = sdf.intersect_shape(&ray, ray_t);
    assert!(
        hit.is_some(),
        "inward surface ray should hit opposite side of cylinder"
    );
    let hit = hit.unwrap();
    // Hit should be at x ≈ -50 (the opposite side), t ≈ 100
    assert!(
        (hit.point.x() - (-50.0)).abs() < 1.0,
        "expected hit at x≈-50, got x={} at t={}",
        hit.point.x(),
        hit.time
    );
}
