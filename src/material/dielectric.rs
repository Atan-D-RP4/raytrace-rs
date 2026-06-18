//! Dielectric (glass/water) material with refraction and reflection.
//!
//! Transparent materials like glass or water. Light hitting the surface
//! either reflects or refracts, with the ratio governed by Fresnel's equations
//! and the refractive indices of the two media.
//!
//! **Snell's Law**: `η₁ sin(θ₁) = η₂ sin(θ₂)` determines the refraction angle.
//!
//! **Total Internal Reflection**: when light in a denser medium hits the
//! boundary at a steep angle, no refraction is possible — all light reflects.
//!
//! **Fresnel**: at normal incidence, `((η₁ - η₂) / (η₁ + η₂))²` reflects.
//! At grazing angles, nearly all light reflects regardless of IOR.
//!
//! This is a **delta material** — it scatters in a single determined direction,
//! not over a distribution. The integrator must skip MIS weighting and use the
//! sampled direction directly.

use crate::hittable::SurfaceInteraction;
use crate::vec3::{Color3, Vec3, reflect, refract};

use super::GPU_NONE;
use super::fresnel_schlick;
use super::{Bsdf, BsdfSample, GpuMaterialBuffer, GpuMaterialNode, GpuMaterialType};

/// Dielectric transmission/reflection controlled by refractive index.
#[derive(Clone)]
pub struct DielectricMaterial {
    /// Index of refraction (1.0 = air, 1.33 = water, 1.5 = glass, 2.42 = diamond).
    pub refractive_idx: f64,
    /// Optional tint color for colored glass. Pure white means no tint.
    pub tint: Color3,
}

impl Bsdf for DielectricMaterial {
    /// Compute refraction ratio from the two media, then use Fresnel to decide
    /// between reflection and refraction. Returns the chosen direction with
    /// unit attenuation (delta material — all energy goes one way).
    fn sample(
        &self,
        wo: Vec3,
        si: &SurfaceInteraction,
        u: f64,
        _v: f64,
        _w: f64,
        _x: f64,
        _y: f64,
        _z: f64,
    ) -> Option<BsdfSample> {
        let ri = if si.front_face() {
            1.0 / self.refractive_idx
        } else {
            self.refractive_idx
        };
        let cos_theta = wo.dot(&si.shading_normal()).min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        let direction =
            if ri * sin_theta > 1.0 || fresnel_schlick(cos_theta, self.refractive_idx) > u {
                reflect(&-wo, &si.shading_normal())
            } else {
                refract(&-wo, &si.shading_normal(), ri)
            };
        Some(BsdfSample::Delta {
            wi: direction,
            f_cos: self.tint,
        })
    }

    /// Delta material — cannot evaluate at arbitrary directions. Returns zero.
    fn eval(&self, _wo: Vec3, _wi: Vec3, _si: &SurfaceInteraction) -> Color3 {
        Color3::from(0., 0., 0.)
    }

    /// Delta material — probability of any specific direction is zero.
    fn pdf(&self, _wo: Vec3, _wi: Vec3, _si: &SurfaceInteraction) -> f64 {
        0.0
    }

    fn clone_box(&self) -> Box<dyn Bsdf> {
        Box::new(self.clone())
    }

    fn serialize_gpu(&self, buf: &mut GpuMaterialBuffer) -> u32 {
        let params = vec![self.refractive_idx];
        let param_offset = buf.params.len() as u32;
        buf.push_params(&params);
        buf.nodes.push(GpuMaterialNode {
            material_type: GpuMaterialType::Dielectric as u32,
            param_offset,
            child_a: GPU_NONE,
            child_b: GPU_NONE,
            texture_index: GPU_NONE,
        });
        buf.nodes.len() as u32 - 1
    }
}
