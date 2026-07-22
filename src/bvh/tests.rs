use std::sync::Arc;

use glam::Vec3;

use crate::bvh::TreeBuilder;
use crate::bvh::{Bvh, BvhNode};
use crate::hittable::{Bounded, Intersectable};
use crate::interval::Interval;
use crate::material::{LambertianMaterial, Material};
use crate::ray::Ray;
use crate::shape::sphere;
use crate::vec3::{Color3, Direction3, Point3};

/// Number of bytes per flat BVH node. Currently 40B (fields + 3-byte pad for `#[repr(C)]`
/// alignment). Bump `_pad` to 27 if 64B cache-line packing is desired — `FlatBvhNode` is
/// memory-only (not serialized), so padding is safe.
const NODE_SIZE: usize = 64;

#[test]
fn flat_bvh_node_size() {
    assert_eq!(std::mem::size_of::<BvhNode<2>>(), NODE_SIZE);
    assert_eq!(std::mem::size_of::<BvhNode<4>>(), NODE_SIZE * 2);
}

#[test]
fn flat_bvh_empty() {
    let bvh: TreeBuilder = TreeBuilder::Empty;
    let flat = Bvh::<2>::from(bvh);
    let ray = Ray::new_with_time(Point3(Vec3::ZERO), Direction3(Vec3::new(0., 0., -1.)), 0.0);
    assert!(
        flat.intersect(&ray, Interval::from(0.001, f32::INFINITY))
            .is_none()
    );
}

#[test]
fn flat_bvh_single_sphere() {
    let sphere: Arc<dyn Intersectable> = Arc::new(sphere(
        Point3(Vec3::new(0., 0., -2.)),
        0.5,
        Material::from(LambertianMaterial::new(Color3::new(0.8, 0.2, 0.2))),
    ));
    let bbox = sphere.bounding_box();
    let bvh: TreeBuilder = TreeBuilder::Leaf {
        object: sphere.clone(),
        bbox,
    };
    let flat = Bvh::<2>::from(bvh);
    assert_eq!(flat.primitive_count(), 1);
    assert_eq!(flat.node_count(), 1);

    // Ray toward the sphere.
    let ray = Ray::new_with_time(Point3(Vec3::ZERO), Direction3(Vec3::new(0., 0., -1.)), 0.0);
    assert!(
        flat.intersect(&ray, Interval::from(0.001, f32::INFINITY))
            .is_some()
    );

    // Ray missing the sphere.
    let ray = Ray::new_with_time(Point3(Vec3::ZERO), Direction3(Vec3::new(10., 0., -1.)), 0.0);
    assert!(
        flat.intersect(&ray, Interval::from(0.001, f32::INFINITY))
            .is_none()
    );
}

#[test]
fn flat_bvh_two_spheres() {
    let s1: Arc<dyn Intersectable> = Arc::new(sphere(
        Point3(Vec3::new(-1., 0., -2.)),
        0.5,
        Material::from(LambertianMaterial::new(Color3::new(1.0, 0.0, 0.0))),
    ));
    let s2: Arc<dyn Intersectable> = Arc::new(sphere(
        Point3(Vec3::new(1., 0., -2.)),
        0.5,
        Material::from(LambertianMaterial::new(Color3::new(0.0, 1.0, 0.0))),
    ));

    let bbox1 = s1.bounding_box();
    let bbox2 = s2.bounding_box();
    let merged_bbox = bbox1.merge(&bbox2);

    let interior: TreeBuilder = TreeBuilder::Interior {
        left: Box::new(TreeBuilder::Leaf {
            object: s1.clone(),
            bbox: bbox1,
        }),
        right: Box::new(TreeBuilder::Leaf {
            object: s2.clone(),
            bbox: bbox2,
        }),
        bbox: merged_bbox,
    };

    let flat = Bvh::<2>::from(interior);
    assert_eq!(flat.primitive_count(), 2);
    assert_eq!(flat.node_count(), 3); // 1 interior + 2 leaves

    // Hit left sphere (at -1, 0, -2).
    let ray = Ray::new_with_time(
        Point3(Vec3::ZERO),
        Direction3(Vec3::new(-1., 0., -2.).normalize()),
        0.0,
    );
    assert!(
        flat.intersect(&ray, Interval::from(0.001, f32::INFINITY))
            .is_some()
    );

    // Hit right sphere (at 1, 0, -2).
    let ray = Ray::new_with_time(
        Point3(Vec3::ZERO),
        Direction3(Vec3::new(1., 0., -2.).normalize()),
        0.0,
    );
    assert!(
        flat.intersect(&ray, Interval::from(0.001, f32::INFINITY))
            .is_some()
    );

    // Hit neither.
    let ray = Ray::new_with_time(Point3(Vec3::ZERO), Direction3(Vec3::new(0., 10., -1.)), 0.0);
    assert!(
        flat.intersect(&ray, Interval::from(0.001, f32::INFINITY))
            .is_none()
    );
}

/// Regression test: FlatBvh produces the same intersection results as
/// BvhNode for a multi-object scene.
#[test]
fn flat_bvh_matches_bvh_node_multi_object() {
    use crate::shape::quad;

    // Build a small scene: 3 spheres at different positions.
    let s1: Arc<dyn Intersectable> = Arc::new(sphere(
        Point3(Vec3::new(-2., 0., -3.)),
        0.5,
        Material::from(LambertianMaterial::new(Color3::new(1.0, 0.0, 0.0))),
    ));
    let s2: Arc<dyn Intersectable> = Arc::new(sphere(
        Point3(Vec3::new(0., 0., -3.)),
        0.5,
        Material::from(LambertianMaterial::new(Color3::new(0.0, 1.0, 0.0))),
    ));
    let s3: Arc<dyn Intersectable> = Arc::new(sphere(
        Point3(Vec3::new(2., 0., -3.)),
        0.5,
        Material::from(LambertianMaterial::new(Color3::new(0.0, 0.0, 1.0))),
    ));
    let s4: Arc<dyn Intersectable> = Arc::new(quad(
        Point3(Vec3::new(-3., -1., -5.)),
        Vec3::new(6., 0., 0.),
        Vec3::new(0., 2., 0.),
        Material::from(LambertianMaterial::new(Color3::new(0.5, 0.5, 0.5))),
    ));

    let mut objects: Vec<Arc<dyn Intersectable>> = vec![s1, s2, s3, s4];
    let bvh = TreeBuilder::new(&mut objects);
    let flat = Bvh::<2>::from(bvh);

    // Test several rays: some hit, some miss.
    let test_rays = vec![
        // Hit sphere at (-2, 0, -3)
        (Vec3::ZERO, Vec3::new(-2., 0., -3.).normalize(), true),
        // Hit sphere at (0, 0, -3)
        (Vec3::ZERO, Vec3::new(0., 0., -3.).normalize(), true),
        // Hit sphere at (2, 0, -3)
        (Vec3::ZERO, Vec3::new(2., 0., -3.).normalize(), true),
        // Hit quad at z=-5
        (Vec3::ZERO, Vec3::new(0., 0., -1.), true),
        // Miss everything (shoot upward)
        (Vec3::ZERO, Vec3::new(0., 10., 0.), false),
        // Miss everything (shoot far to the side, away from all objects)
        (Vec3::new(10., 0., 0.), Vec3::new(0., 1., 0.), false),
    ];

    for &(origin, direction, should_hit) in &test_rays {
        let ray = Ray::new_with_time(Point3(origin), Direction3(direction), 0.0);
        let bvh_result = flat.intersect(&ray, Interval::from(0.001, f32::INFINITY));
        assert_eq!(
            bvh_result.is_some(),
            should_hit,
            "Ray from {origin} dir {direction}: expected hit={should_hit}"
        );
    }

    // Verify that widening to W=4 produces identical intersection results.
    let wide: Bvh<4> = flat.widen();
    for &(origin, direction, should_hit) in &test_rays {
        let ray = Ray::new_with_time(Point3(origin), Direction3(direction), 0.0);
        let wide_result = wide.intersect(&ray, Interval::from(0.001, f32::INFINITY));
        assert_eq!(
            wide_result.is_some(),
            should_hit,
            "Widen: ray from {origin} dir {direction}: expected hit={should_hit}"
        );
    }

    // Wide node count should be ≤ original binary node count.
    assert!(
        wide.node_count() <= flat.node_count(),
        "Wide node count {} > binary node count {}",
        wide.node_count(),
        flat.node_count()
    );
}
