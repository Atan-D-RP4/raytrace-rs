/// Reference CPU Monte-Carlo path-tracing integrator.
///
/// Iteratively traces/scatters up to `depth` bounces and multiplies
/// attenuation along the path.
///
/// TODO(gpu): mirror this boundary in a separate path-trace kernel / WGSL entrypoint.
/// The `li()` method is the natural split point: it takes a ray and returns radiance.
///
use std::sync::Arc;

use crate::film::rgb::LUMINANCE;
use crate::hittable::{Intersectable, Sampleable, SurfaceInteraction};
use crate::integrator::Integrator;
use crate::interval::Interval;
use crate::material::{BsdfSample, PdfKind};
use crate::pdf::{
    CosinePDF, GgxSamplePDF, HittablePDF, MixturePDF, PDF, UniformHemispherePDF, UniformSpherePDF,
};
use crate::ray::Ray;
use crate::sampler::{DimCursor, Sampler};
use crate::vec3::Color3;

pub struct PathTracingIntegrator {
    max_depth: u32,
    background: Color3,
}

impl PathTracingIntegrator {
    pub fn new(max_depth: u32, background: Color3) -> Self {
        Self {
            max_depth,
            background,
        }
    }
}

impl<S: Sampler> Integrator<S> for PathTracingIntegrator {
    fn li(
        &self,
        initial_ray: &mut Ray,
        world: &dyn Intersectable,
        lights: &[Arc<dyn Sampleable>],
        dim_cursor: &mut DimCursor<S>,
    ) -> Color3 {
        let mut accumulated_attenuation = Color3::ONE;
        let mut accumulated_color = Color3::ZERO;
        let mut ray = *initial_ray;

        for bounce in 0..self.max_depth {
            let bounce_start = dim_cursor.checkpoint();
            if let Some(mat_hit) = world.intersect(&ray, Interval::from(0.001, f64::INFINITY)) {
                // Create a SurfaceInteraction from the material hit and the ray
                let si = SurfaceInteraction::from_material_hit(mat_hit, &ray);
                let material = si.material();

                // Compute the emitted light from the material at the intersection point
                let emission = material.emitted(&si);
                // Accumulate the emitted light, scaled by the current attenuation
                accumulated_color += accumulated_attenuation * emission;

                // Sample the material to get the next ray and attenuation
                let max_attenuation = accumulated_attenuation
                    .x
                    .max(accumulated_attenuation.y)
                    .max(accumulated_attenuation.z);

                // If the maximum attenuation is very small, terminate the path early to avoid unnecessary computation
                if max_attenuation < 1e-6 {
                    return accumulated_color;
                }

                // Sample a random number for Russian Roulette
                let rr = dim_cursor.next_sample();

                // Russian Roulette: survival probability proportional to current
                // path throughput.  The 0.05 floor bounds variance from low-throughput paths.
                if bounce >= 5 {
                    let survival = max_attenuation.clamp(0.05, 1.0);
                    if rr > survival {
                        return accumulated_color;
                    }
                    accumulated_attenuation /= survival;
                }

                // Outgoing direction (away from the surface) is the negative of the ray direction
                let wo = -ray.direction.unit_vector();

                // Sample the material to get the next ray and attenuation
                let u_mat = dim_cursor.next_sample();
                let v_mat = dim_cursor.next_sample();
                let w_mat = dim_cursor.next_sample();
                let x_mat = dim_cursor.next_sample();
                let y_mat = dim_cursor.next_sample();
                let z_mat = dim_cursor.next_sample();

                if let Some(sample) =
                    material.sample(wo, &si, u_mat, v_mat, w_mat, x_mat, y_mat, z_mat)
                {
                    match sample {
                        BsdfSample::Delta { wi, f_cos } => {
                            accumulated_attenuation = accumulated_attenuation * f_cos;
                            ray = Ray::new_with_time(si.point(), wi, ray.time);
                            // Pad to fixed 9-dim stride so subsequent bounces use consistent
                            // Sobol dimensions regardless of this bounce's path structure.
                            for _ in 0..4 {
                                let _ = dim_cursor.next_dim();
                            }
                        }
                        BsdfSample::NonDelta { pdf_kind } => {
                            // Build surface-PDF matching the material's sampling distribution.
                            let surface_pdf: &dyn PDF<_> = match pdf_kind {
                                PdfKind::Cosine { normal } => &CosinePDF::new(normal),
                                PdfKind::Ggx { wo, normal, alpha } => {
                                    &GgxSamplePDF::new(wo, normal, alpha)
                                }
                                PdfKind::UniformSphere => &UniformSpherePDF::new(),
                                PdfKind::Delta => unreachable!(),
                            };

                            // Mixture: env + light + surface PDFs. Multiple sampling strategies
                            // per bounce reduces noise for difficult paths.
                            let env_pdf = UniformHemispherePDF::new(si.shading_normal());
                            let light_pdf = HittablePDF::new(lights, si.point());

                            // Mixture size adapts to background luminance — dark scenes
                            // skip env slots to avoid wasting samples on hemisphere
                            // directions that carry no energy. Scene-level heuristic
                            // (bright sky illuminates dark interiors just fine).
                            // Thresholds: 0.3=bright sky, 0.1=overcast, 0.01=near-black.
                            let bg_lum = LUMINANCE * self.background;
                            let bg_lum = bg_lum.x + bg_lum.y + bg_lum.z;

                            // Mixture components are duplicated to give them more weight in the sampling distribution.
                            let pdfs_8: &[&dyn PDF<S>; 8] = &[
                                &env_pdf,
                                &light_pdf,
                                surface_pdf,
                                surface_pdf,
                                surface_pdf,
                                surface_pdf,
                                &env_pdf,
                                &env_pdf,
                            ];
                            let pdfs_6: &[&dyn PDF<S>; 6] = &[
                                &env_pdf,
                                &light_pdf,
                                surface_pdf,
                                surface_pdf,
                                surface_pdf,
                                &env_pdf,
                            ];
                            let pdfs_5: &[&dyn PDF<S>; 5] =
                                &[&env_pdf, &light_pdf, surface_pdf, surface_pdf, surface_pdf];
                            let pdfs_3: &[&dyn PDF<S>; 3] =
                                &[&light_pdf as &dyn PDF<S>, surface_pdf, surface_pdf];

                            let mix_6 = MixturePDF::new(pdfs_6);
                            let mix_5 = MixturePDF::new(pdfs_5);
                            let mix_3 = MixturePDF::new(pdfs_3);
                            let mix_8 = MixturePDF::new(pdfs_8);

                            // More env slots for bright backgrounds, fewer for dark — avoids
                            // wasting samples on hemisphere directions that don't carry energy.
                            let sampling_pdf: &dyn PDF<S> = if bg_lum > 0.3 {
                                &mix_8
                            } else if bg_lum > 0.1 {
                                &mix_6
                            } else if bg_lum > 0.01 {
                                &mix_5
                            } else {
                                &mix_3
                            };

                            // Track mixture consumption for fixed-dim stride padding.
                            let mix_start = dim_cursor.offset();

                            // PlanarPatch::random() returns a non-unit vector (distance to light);
                            // BRDFs expect unit length so call .unit_vector().
                            let direction = sampling_pdf.generate(dim_cursor).unit_vector();
                            let mix_consumed = dim_cursor.offset() - mix_start;

                            let scattered_ray = Ray::new_with_time(si.point(), direction, ray.time);
                            let pdf_val = sampling_pdf.value(direction);

                            let f_cos = material.eval(wo, direction, &si);

                            // Single-sample MIS: f * cos / p_mix(x)
                            let weight = 1. / pdf_val.max(1e-6);
                            accumulated_attenuation = accumulated_attenuation * weight * f_cos;
                            ray = scattered_ray;

                            // Pad mixture dims to exactly 4 (1 selection + 3 direction).
                            for _ in mix_consumed..4 {
                                let _ = dim_cursor.next_dim();
                            }
                        }
                    }
                    // QMC invariant: every completed bounce must consume exactly 11 dims.
                    debug_assert_eq!(dim_cursor.offset() - bounce_start, 11);
                } else {
                    // Emissive materials return None — no scattering. Emission already added
                    // to accumulated_color via emitted() above.
                    return accumulated_color;
                }
            } else {
                // Ray missed the world geometry — accumulate background and terminate.
                return accumulated_color + accumulated_attenuation * self.background;
            }
        }

        // Max bounce count reached — terminate the path. This can still contribute to the final
        // image if the last bounce was a non-delta and the accumulated attenuation is non-zero.
        accumulated_color
    }
}
