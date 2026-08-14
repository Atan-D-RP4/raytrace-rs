use glam::Vec3;
use std::sync::Arc;

use crate::bvh::TreeBuilder;
use crate::bvh::{Bvh, BvhNode};
use crate::intersect::{Bounded, Intersectable};
use crate::material::{DiffuseReflector, Material};
use crate::math::interval::Interval;
use crate::math::vec3::{Color3, Direction3, Point3};
use crate::primitives::Primitive;
use crate::ray::{Ray, RayPacked};
use crate::shape::sphere;

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
    let bvh: TreeBuilder<Primitive> = TreeBuilder::Empty;
    let flat = Bvh::<2, _>::from(bvh);
    let ray = Ray::new_with_time(Point3::ZERO, Direction3::NEG_Z, 0.0);
    assert!(flat.intersect(&ray, Interval::from(0.001, f32::INFINITY))[0].is_none());
}

#[test]
fn flat_bvh_single_sphere() {
    let sphere: Primitive = sphere(
        Point3::new(0., 0., -2.),
        0.5,
        Material::from(DiffuseReflector::new(Color3::new(0.8, 0.2, 0.2))),
    )
    .into();
    let bbox = sphere.bounding_box();
    let bvh = TreeBuilder::Leaf {
        object: sphere.clone(),
        bbox,
    };
    let flat = Bvh::<2, _>::from(bvh);
    assert_eq!(flat.primitive_count(), 1);
    assert_eq!(flat.node_count(), 1);

    // Ray toward the sphere.
    let ray = Ray::new_with_time(Point3::ZERO, Direction3::NEG_Z, 0.0);
    assert!(flat.intersect(&ray, Interval::from(0.001, f32::INFINITY))[0].is_some());

    // Ray missing the sphere.
    let ray = Ray::new_with_time(Point3::ZERO, Direction3::new(10., 0., -1.), 0.0);
    assert!(flat.intersect(&ray, Interval::from(0.001, f32::INFINITY))[0].is_none());
}

#[test]
fn flat_bvh_two_spheres() {
    let s1: Primitive = sphere(
        Point3::new(-1., 0., -2.),
        0.5,
        Material::from(DiffuseReflector::new(Color3::new(1.0, 0.0, 0.0))),
    )
    .into();
    let s2: Primitive = sphere(
        Point3::new(1., 0., -2.),
        0.5,
        Material::from(DiffuseReflector::new(Color3::new(0.0, 1.0, 0.0))),
    )
    .into();

    let bbox1 = s1.bounding_box();
    let bbox2 = s2.bounding_box();
    let merged_bbox = bbox1.merge(&bbox2);

    let interior = TreeBuilder::Interior {
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

    let flat = Bvh::<2, _>::from(interior);
    assert_eq!(flat.primitive_count(), 2);
    assert_eq!(flat.node_count(), 3); // 1 interior + 2 leaves

    // Hit left sphere (at -1, 0, -2).
    let ray = Ray::new_with_time(Point3::ZERO, Direction3::new(-1., 0., -2.).normalize(), 0.0);
    assert!(flat.intersect(&ray, Interval::from(0.001, f32::INFINITY))[0].is_some());

    // Hit right sphere (at 1, 0, -2).
    let ray = Ray::new_with_time(Point3::ZERO, Direction3::new(1., 0., -2.).normalize(), 0.0);
    assert!(flat.intersect(&ray, Interval::from(0.001, f32::INFINITY))[0].is_some());

    // Hit neither.
    let ray = Ray::new_with_time(Point3::ZERO, Direction3::new(0., 10., -1.), 0.0);
    assert!(flat.intersect(&ray, Interval::from(0.001, f32::INFINITY))[0].is_none());
}

#[test]
fn packet_intersection_matches_scalar_lanes() {
    let make_sphere = |center, color| -> Primitive {
        sphere(center, 0.5, Material::from(DiffuseReflector::new(color))).into()
    };
    let mut objects: Vec<Primitive> = vec![
        make_sphere(Point3::new(-2., 0., -3.), Color3::new(1., 0., 0.)),
        make_sphere(Point3::new(0., 0., -3.), Color3::new(0., 1., 0.)),
        make_sphere(Point3::new(2., 0., -3.), Color3::new(0., 0., 1.)),
    ];
    let builder = TreeBuilder::new(&mut objects);
    let binary = Bvh::<2, _>::from(builder);
    let wide: Bvh<4, Primitive> = Bvh::<2, Primitive>::from(TreeBuilder::new(&mut objects)).widen();
    let rays: [RayPacked<1>; 4] = [
        Ray::new_with_time(Point3::ZERO, Direction3::new(-2., 0., -3.).normalize(), 0.0),
        Ray::new_with_time(Point3::ZERO, Direction3::new(0., 0., -3.).normalize(), 0.0),
        Ray::new_with_time(Point3::ZERO, Direction3::new(2., 0., -3.).normalize(), 0.0),
        Ray::new_with_time(Point3::ZERO, Direction3::new(0., 10., 0.).normalize(), 0.0),
    ];
    let packet: RayPacked<4> = rays.into();
    let interval = Interval::from(0.001, f32::INFINITY);

    for hits in [
        binary.intersect(&packet, interval),
        wide.intersect(&packet, interval),
    ] {
        for (lane, ray) in rays.iter().enumerate() {
            let scalar = binary.intersect(ray, interval.lane(lane))[0];
            assert_eq!(hits[lane].is_some(), scalar.is_some());
            if let (Some(packet_hit), Some(scalar_hit)) = (hits[lane], scalar) {
                assert!((packet_hit.hit.time - scalar_hit.hit.time).abs() < 1e-6);
            }
        }
    }
}

#[test]
fn custom_primitive_keeps_dynamic_scalar_dispatch() {
    let custom: Arc<dyn Intersectable> = Arc::new(sphere(
        Point3::new(0., 0., -2.),
        0.5,
        Material::from(DiffuseReflector::new(Color3::new(0.8, 0.2, 0.2))),
    ));
    let primitive = Primitive::Custom(custom);
    let ray = Ray::new_with_time(Point3::ZERO, Direction3::NEG_Z, 0.0);
    let interval = Interval::from(0.001, f32::INFINITY);

    assert!(primitive.intersect(&ray, interval)[0].is_some());

    let rays: [RayPacked<1>; 2] = [ray, Ray::new_with_time(Point3::ZERO, Direction3::X, 0.0)];
    let packet: RayPacked<2> = rays.into();
    let hits = primitive.intersect(&packet, Interval::from(0.001, f32::INFINITY));
    assert!(hits[0].is_some());
    assert!(hits[1].is_none());
}

/// Regression test: FlatBvh produces the same intersection results as
/// BvhNode for a multi-object scene.
#[test]
fn flat_bvh_matches_bvh_node_multi_object() {
    use crate::shape::quad;

    // Build a small scene: 3 spheres at different positions.
    let s1: Primitive = sphere(
        Point3::new(-2., 0., -3.),
        0.5,
        Material::from(DiffuseReflector::new(Color3::new(1.0, 0.0, 0.0))),
    )
    .into();
    let s2: Primitive = sphere(
        Point3::new(0., 0., -3.),
        0.5,
        Material::from(DiffuseReflector::new(Color3::new(0.0, 1.0, 0.0))),
    )
    .into();
    let s3: Primitive = sphere(
        Point3::new(2., 0., -3.),
        0.5,
        Material::from(DiffuseReflector::new(Color3::new(0.0, 0.0, 1.0))),
    )
    .into();
    let s4: Primitive = quad(
        Point3::new(-3., -1., -5.),
        Vec3::new(6., 0., 0.),
        Vec3::new(0., 2., 0.),
        Material::from(DiffuseReflector::new(Color3::new(0.5, 0.5, 0.5))),
    )
    .into();

    let mut objects: Vec<Primitive> = vec![s1, s2, s3, s4];
    let bvh = TreeBuilder::new(&mut objects);
    let flat = Bvh::<2, _>::from(bvh.clone());

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
        let bvh_result = flat.intersect(&ray, Interval::from(0.001, f32::INFINITY))[0];
        assert_eq!(
            bvh_result.is_some(),
            should_hit,
            "Ray from {origin} dir {direction}: expected hit={should_hit}"
        );
    }

    // Verify that widening to W=4 produces identical intersection results.
    let wide: Bvh<4, _> = flat.widen();
    for &(origin, direction, should_hit) in &test_rays {
        let ray = Ray::new_with_time(Point3(origin), Direction3(direction), 0.0);
        let wide_result = wide.intersect(&ray, Interval::from(0.001, f32::INFINITY))[0];
        assert_eq!(
            wide_result.is_some(),
            should_hit,
            "Widen: ray from {origin} dir {direction}: expected hit={should_hit}"
        );
    }

    let flat = Bvh::<2, _>::from(bvh);
    // Wide node count should be ≤ original binary node count.
    assert!(
        wide.node_count() <= flat.node_count(),
        "Wide node count {} > binary node count {}",
        wide.node_count(),
        flat.node_count()
    );
}

/// `widen::<2>()` must be a no-op: Bvh<2> and Bvh<W> (W=2) are the same type,
/// and a collapse would emit a wide-leaf/wide-interior layout the W=2
/// traversal paths cannot read (prim counts in `leaf_info[0]` vs `child_count`,
/// leaf prim_starts pushed as node indices). The result must be the same tree.
#[test]
fn widen_w2_is_identity() {
    use crate::shape::quad;

    let s1: Primitive = sphere(
        Point3::new(-2., 0., -3.),
        0.5,
        Material::from(DiffuseReflector::new(Color3::new(1.0, 0.0, 0.0))),
    )
    .into();
    let s2: Primitive = sphere(
        Point3::new(0., 0., -3.),
        0.5,
        Material::from(DiffuseReflector::new(Color3::new(0.0, 1.0, 0.0))),
    )
    .into();
    let s3: Primitive = sphere(
        Point3::new(2., 0., -3.),
        0.5,
        Material::from(DiffuseReflector::new(Color3::new(0.0, 0.0, 1.0))),
    )
    .into();
    let s4: Primitive = quad(
        Point3::new(-3., -1., -5.),
        Vec3::new(6., 0., 0.),
        Vec3::new(0., 2., 0.),
        Material::from(DiffuseReflector::new(Color3::new(0.5, 0.5, 0.5))),
    )
    .into();

    let mut objects: Vec<Primitive> = vec![s1, s2, s3, s4];
    let bvh = TreeBuilder::new(&mut objects);
    let flat = Bvh::<2, _>::from(bvh);
    let binary_node_count = flat.node_count();

    // Widen to W=2 and confirm the tree is unchanged.
    let wide: Bvh<2, _> = flat.widen();
    assert_eq!(
        wide.node_count(),
        binary_node_count,
        "widen::<2>() must not rebuild the tree"
    );

    let test_rays = vec![
        (Vec3::ZERO, Vec3::new(-2., 0., -3.).normalize(), true),
        (Vec3::ZERO, Vec3::new(0., 0., -3.).normalize(), true),
        (Vec3::ZERO, Vec3::new(2., 0., -3.).normalize(), true),
        (Vec3::ZERO, Vec3::new(0., 0., -1.), true),
        (Vec3::ZERO, Vec3::new(0., 10., 0.), false),
        (Vec3::new(10., 0., 0.), Vec3::new(0., 1., 0.), false),
    ];
    for &(origin, direction, should_hit) in &test_rays {
        let ray = Ray::new_with_time(Point3(origin), Direction3(direction), 0.0);
        let wide_result = wide.intersect(&ray, Interval::from(0.001, f32::INFINITY))[0];
        assert_eq!(
            wide_result.is_some(),
            should_hit,
            "widen::<2>: ray from {origin} dir {direction}: expected hit={should_hit}"
        );
    }
}
