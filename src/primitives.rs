use std::borrow::Borrow;
use std::sync::Arc;

use crate::bvh::Bvh;
use crate::bvh::aabb::Aabb;
use crate::const_medium::ConstantMedium;
use crate::intersect::interaction::MaterialHit;
use crate::intersect::{Bounded, Intersectable};
use crate::light::environment::EnvironmentLight;
use crate::light::{LightSample, Sampleable};
use crate::material::Material;
use crate::math::interval::Interval;
use crate::math::vec3::{Color3, Direction3, Point3};
use crate::ray::Ray;
use crate::sampling::pdf::AreaPdf;
use crate::shape::{BoxShape, PlanarShape, SdfExpr, SdfShape, ShapeObject, SphereShape};
use crate::transform::{AnimatedTransform, StaticTransform, TransformObject};

#[derive(Clone)]
pub enum Primitive {
    Empty,                                                         // 0 — BVH leaf-packing sentinel
    Sphere(ShapeObject<SphereShape, Arc<Material>>),               // 1
    MovingSphere(ShapeObject<SphereShape, Arc<Material>>),         // 2 — SphereShape::new_moving
    Planar(ShapeObject<PlanarShape, Arc<Material>>),               // 3
    Box(ShapeObject<BoxShape, Arc<Material>>),                     // 4
    Sdf(ShapeObject<SdfShape<SdfExpr>, Arc<Material>>),            // 5 — F: data-only SdfFn struct
    Transformed(TransformObject<Box<Primitive>, StaticTransform>), // 6
    Animated(TransformObject<Box<Primitive>, AnimatedTransform>),  // 7
    Volume(ConstantMedium<Arc<Primitive>, true>),                  // 8
    Aggregate(Arc<Bvh<2, Box<Primitive>>>),                        // 9
    Custom(Arc<dyn Intersectable>), // 10 — CPU-only; must be empty in shipped scenes
}

impl Intersectable for Primitive {
    fn intersect<'a>(&'a self, ray: &Ray, ray_t: Interval) -> Option<MaterialHit<'a>> {
        match self {
            Primitive::Empty => None,
            Primitive::Sphere(s) => s.intersect(ray, ray_t),
            Primitive::MovingSphere(s) => s.intersect(ray, ray_t),
            Primitive::Planar(p) => p.intersect(ray, ray_t),
            Primitive::Box(b) => b.intersect(ray, ray_t),
            Primitive::Sdf(s) => s.intersect(ray, ray_t),
            Primitive::Transformed(t) => t.intersect(ray, ray_t),
            Primitive::Animated(a) => a.intersect(ray, ray_t),
            Primitive::Volume(v) => v.intersect(ray, ray_t),
            Primitive::Aggregate(bvh) => bvh.intersect(ray, ray_t),
            Primitive::Custom(c) => c.intersect(ray, ray_t),
        }
    }
}

impl Bounded for Primitive {
    fn bounding_box(&self) -> Aabb {
        match self {
            Primitive::Empty => Aabb::empty(),
            Primitive::Sphere(s) => s.bounding_box(),
            Primitive::MovingSphere(s) => s.bounding_box(),
            Primitive::Planar(p) => p.bounding_box(),
            Primitive::Box(b) => b.bounding_box(),
            Primitive::Sdf(s) => s.bounding_box(),
            Primitive::Transformed(t) => t.bounding_box(),
            Primitive::Animated(a) => a.bounding_box(),
            Primitive::Volume(v) => v.bounding_box(),
            Primitive::Aggregate(bvh) => bvh.bounding_box(),
            Primitive::Custom(c) => c.bounding_box(),
        }
    }
}

impl Sampleable for Primitive {
    fn pdf_value(&self, origin: Point3, direction: Direction3, time: f32) -> f32 {
        match self {
            Primitive::Empty
            | Primitive::Volume(_)
            | Primitive::Aggregate(_)
            | Primitive::Custom(_) => {
                // Not a light: no surface to sample, no emission. Zero contribution.
                // (debug_assert: the scene build only pushes emissive primitives into lights)
                0.0
            }

            Primitive::Sphere(s) => s.pdf_value(origin, direction, time),
            Primitive::MovingSphere(s) => s.pdf_value(origin, direction, time),
            Primitive::Planar(p) => p.pdf_value(origin, direction, time),
            Primitive::Box(b) => b.pdf_value(origin, direction, time),
            Primitive::Sdf(s) => s.pdf_value(origin, direction, time),
            Primitive::Transformed(t) => t.pdf_value(origin, direction, time),
            Primitive::Animated(a) => a.pdf_value(origin, direction, time),
        }
    }

    fn random_direction(&self, origin: Point3, u: f32, v: f32, time: f32) -> Direction3 {
        match self {
            Primitive::Empty
            | Primitive::Volume(_)
            | Primitive::Aggregate(_)
            | Primitive::Custom(_) => {
                // Not a light: no surface to sample, no emission. Zero contribution.
                // (debug_assert: the scene build only pushes emissive primitives into lights)
                Direction3::ZERO
            }
            Primitive::Sphere(s) => s.random_direction(origin, u, v, time),
            Primitive::MovingSphere(s) => s.random_direction(origin, u, v, time),
            Primitive::Planar(p) => p.random_direction(origin, u, v, time),
            Primitive::Box(b) => b.random_direction(origin, u, v, time),
            Primitive::Sdf(s) => s.random_direction(origin, u, v, time),
            Primitive::Transformed(t) => t.random_direction(origin, u, v, time),
            Primitive::Animated(a) => a.random_direction(origin, u, v, time),
        }
    }

    fn sample_light(&self, origin: Point3, u: f32, v: f32, time: f32) -> LightSample {
        match self {
            Primitive::Empty
            | Primitive::Volume(_)
            | Primitive::Aggregate(_)
            | Primitive::Custom(_) => {
                // Not a light: no surface to sample, no emission. Zero contribution.
                // (debug_assert: the scene build only pushes emissive primitives into lights)
                LightSample {
                    direction: Direction3::ZERO,
                    normal: Direction3::ZERO,
                    distance: 0.0,
                    pdf: AreaPdf(0.0),
                    emission: Color3::ZERO,
                }
            }
            Primitive::Sphere(s) => s.sample_light(origin, u, v, time),
            Primitive::MovingSphere(s) => s.sample_light(origin, u, v, time),
            Primitive::Planar(p) => p.sample_light(origin, u, v, time),
            Primitive::Box(b) => b.sample_light(origin, u, v, time),
            Primitive::Sdf(s) => s.sample_light(origin, u, v, time),
            Primitive::Transformed(t) => t.sample_light(origin, u, v, time),
            Primitive::Animated(a) => a.sample_light(origin, u, v, time),
        }
    }
}

pub enum LightPrimitive {
    Primitive(Box<Primitive>),
    EnvLight(Arc<EnvironmentLight>), // environment light (infinite)
}

impl Sampleable for LightPrimitive {
    fn pdf_value(&self, origin: Point3, direction: Direction3, time: f32) -> f32 {
        match self {
            LightPrimitive::Primitive(p) => p.pdf_value(origin, direction, time),
            LightPrimitive::EnvLight(e) => e.pdf_value(origin, direction, time),
        }
    }

    fn random_direction(&self, origin: Point3, u: f32, v: f32, time: f32) -> Direction3 {
        match self {
            LightPrimitive::Primitive(p) => p.random_direction(origin, u, v, time),
            LightPrimitive::EnvLight(e) => e.random_direction(origin, u, v, time),
        }
    }

    fn sample_light(&self, origin: Point3, u: f32, v: f32, time: f32) -> LightSample {
        match self {
            LightPrimitive::Primitive(p) => p.sample_light(origin, u, v, time),

            LightPrimitive::EnvLight(e) => e.sample_light(origin, u, v, time),
        }
    }
}

impl From<Primitive> for LightPrimitive {
    fn from(p: Primitive) -> Self {
        LightPrimitive::Primitive(Box::new(p))
    }
}

/// Ergonomic conversions from material-wrapped shapes into [`Primitive`].
///
/// The material is normalized to `Arc<Material>` at this boundary (shared ownership for
/// multi-object materials), matching the enum's storage.
macro_rules! impl_primitive_from_shape {
    ($variant:ident, $shape:ty) => {
        impl<M: Borrow<Material> + Send + Sync> From<ShapeObject<$shape, M>> for Primitive {
            fn from(so: ShapeObject<$shape, M>) -> Self {
                Primitive::$variant(ShapeObject::new(
                    so.shape().clone(),
                    Arc::new(so.material().clone()),
                ))
            }
        }
    };
}

impl_primitive_from_shape!(Sphere, SphereShape);
impl_primitive_from_shape!(Planar, PlanarShape);
impl_primitive_from_shape!(Box, BoxShape);
impl_primitive_from_shape!(Sdf, SdfShape<SdfExpr>);

impl From<TransformObject<Box<Primitive>, StaticTransform>> for Primitive {
    fn from(t: TransformObject<Box<Primitive>, StaticTransform>) -> Self {
        Primitive::Transformed(t)
    }
}

impl From<TransformObject<Box<Primitive>, AnimatedTransform>> for Primitive {
    fn from(t: TransformObject<Box<Primitive>, AnimatedTransform>) -> Self {
        Primitive::Animated(t)
    }
}

impl From<ConstantMedium<Arc<Primitive>>> for Primitive {
    fn from(cm: ConstantMedium<Arc<Primitive>>) -> Self {
        Primitive::Volume(cm)
    }
}

impl From<Arc<Bvh<2, Box<Primitive>>>> for Primitive {
    fn from(bvh: Arc<Bvh<2, Box<Primitive>>>) -> Self {
        Primitive::Aggregate(bvh)
    }
}
