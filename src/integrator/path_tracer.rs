/// Reference CPU Monte-Carlo path-tracing integrator.
///
/// Iteratively traces/scatters up to `depth` bounces and multiplies
/// attenuation along the path.
///
/// TODO(gpu): mirror this boundary in a separate path-trace kernel / WGSL entrypoint.
/// The `li()` method is the natural split point: it takes a ray and returns radiance.
///
use crate::hittable::{Intersectable, Sampleable, SurfaceInteraction};
use crate::integrator::Integrator;
use crate::interval::Interval;
use crate::material::PdfKind;
use crate::pdf::{CosinePDF, GgxSamplePDF, HittablePDF, MixturePDF, PDF, UniformSpherePDF};
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
        lights: &dyn Sampleable<S>,
        sampler: &mut DimCursor<S>,
    ) -> Color3 {
        let mut accumulated_attenuation = Color3::ONE;
        let mut accumulated_color = Color3::ZERO;
        let mut ray = *initial_ray;

        for bounce in 0..self.max_depth {
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
                let rr = sampler.next_sample();

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
                let u_mat = sampler.next_sample();
                let v_mat = sampler.next_sample();
                let w_mat = sampler.next_sample();
                let x_mat = sampler.next_sample();

                if let Some(sample) = material.sample(wo, &si, u_mat, v_mat, w_mat, x_mat) {
                    // Per-sample delta routing: composed materials (Coated/Mix) decide via PDFKind
                    // whether to continue the path or not. If the sample is a delta, continue path.
                    if matches!(sample.pdf_kind, crate::material::PdfKind::Delta) {
                        accumulated_attenuation = accumulated_attenuation * sample.f_cos;
                        ray = Ray::new_with_time(si.point(), sample.wi, ray.time);
                        // Fixed stride per bounce: pad to consume 4 mixture+direction dims,
                        // ensuring the next bounce reads from the same Sobol dimensions
                        // regardless of this bounce's path decisions (QMC consistency).
                        for _ in 0..4 {
                            let _ = sampler.next_sample();
                        }
                    } else {
                        // Non-delta materials: mixture PDF(light + material surface) sampling.
                        // If the material sample returns None, we terminate the path.
                        let surface_pdf: &dyn PDF<_> = match sample.pdf_kind {
                            PdfKind::Cosine { normal } => &CosinePDF::new(normal),
                            PdfKind::Ggx { wo, normal, alpha } => {
                                &GgxSamplePDF::new(wo, normal, alpha)
                            }
                            PdfKind::UniformSphere => &UniformSpherePDF::new(),
                            PdfKind::Delta => unreachable!(),
                        };

                        let light_pdf = HittablePDF::new(lights, si.point());
                        let pdfs = &[&light_pdf, surface_pdf, surface_pdf];
                        let sampling_pdf = MixturePDF::new(pdfs);

                        // Track mixture-dimension consumption so we can pad to a fixed
                        // 4-dim stride (1 selection + 3 direction) regardless of which
                        // mixture component was selected.
                        let mix_start = sampler.offset();

                        // Sample the mixture PDF to get the next direction Unitize for BRDF eval -
                        // PlanarPatch::random() returns a non-unit vector (distance to light), but
                        // BRDFs expect unit length.
                        let direction = sampling_pdf.generate(sampler).unit_vector();
                        let mix_consumed = sampler.offset() - mix_start;

                        let scattered_ray = Ray::new_with_time(si.point(), direction, ray.time);
                        let pdf_val = sampling_pdf.value(direction);

                        let f_cos = material.eval(wo, direction, &si);

                        // Standard single-sample MIS unbiased estimator: f * cos / p_mix(x)
                        let weight = 1. / pdf_val.max(1e-6); // Avoid division by zero
                        accumulated_attenuation = accumulated_attenuation * weight * f_cos;
                        ray = scattered_ray;

                        // Pad to fixed mixture stride: the mixture PDF consumes 3-4 dims
                        // (1 selection + 2-3 direction). Pad to ensure exactly 4, keeping
                        // subsequent bounces at consistent Sobol dimensions.
                        for _ in mix_consumed..4 {
                            let _ = sampler.next_sample();
                        }
                    }
                } else {
                    // Material sample returned None, terminate the path
                    return accumulated_color;
                }
            } else {
                // If the ray hits nothing, return the background color

                // let unit_direction = ray.direction.unit_vector();
                // let t = 0.5 * (unit_direction.y + 1.0);
                // The background gradient
                // let background = ((1.0 - t) * Vec3::from(1.0, 1.0, 1.0)) + (t * Vec3::from(0.5, 0.7, 1.0));

                return accumulated_color + accumulated_attenuation * self.background;
            }
        }

        accumulated_color
    }
}
