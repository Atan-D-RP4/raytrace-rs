/// Reference CPU Monte-Carlo path-tracing integrator.
///
/// Iteratively traces/scatters up to `depth` bounces and multiplies
/// attenuation along the path.
///
/// TODO(gpu): mirror this boundary in a separate path-trace kernel / WGSL entrypoint.
/// The `li()` method is the natural split point: it takes a ray and returns radiance.
///
use std::sync::Arc;

use crate::hittable::{Intersectable, Sampleable, SurfaceInteraction};
use crate::integrator::Integrator;
use crate::interval::Interval;
use crate::material::BsdfSample;
use crate::pdf::{HittablePDF, PDF, PdfEnum, UniformHemispherePDF, power_heuristic};
use crate::ray::Ray;
use crate::sampler::{DimCursor, Sampler};
use crate::vec3::{Color3, Vec3};

/// One-sample MIS estimator with power heuristic (β=2).
///
/// Selects one strategy uniformly at random (probability 1/N), generates a
/// direction from that strategy's PDF, then evaluates ALL N PDFs at the
/// sampled direction. Returns the direction and the MIS-weighted estimator:
///
///   contribution = N * (p_sel² / Σp_j²) * f / p_sel
///
/// where p_sel is the selected strategy's PDF value and p_j are all PDF values
/// at the sampled direction. This is provably unbiased and has lower variance
/// than the mixture-based f/p_mix estimator.
/// NOTE: If we ever want to support more MIS strategies, we can make this a const-generic function
/// with a slice of PDFs instead of a fixed-size array. Or if we need to swap out the MIS strategy,
/// we can make this generic over a MisWeightingStrategy trait that takes the selected PDF and all
/// PDFs and returns a weight.
#[inline(always)]
fn mis_sample<S: Sampler, const N: usize>(
    pdfs: [&dyn PDF<S>; N],
    eval_fn: impl FnOnce(Vec3) -> crate::vec3::Color3,
    dim_cursor: &mut DimCursor<S>,
) -> (Vec3, crate::vec3::Color3) {
    // 1. Select strategy uniformly: 1 QMC dim for selection
    let u_select = dim_cursor.next_sample();
    let sel_idx = (u_select * N as f64).min(N as f64 - 1e-15) as usize;

    // 2. Generate direction from selected strategy
    let direction = pdfs[sel_idx].generate(dim_cursor).unit_vector();

    // 3. Evaluate ALL PDFs at the sampled direction, compute sum of squares
    let mut pdf_sum_sq = 0.0;
    let mut pdf_vals = [0.0f64; N];
    for (i, pdf) in pdfs.iter().enumerate() {
        let v = pdf.value(direction);
        pdf_vals[i] = v;
        pdf_sum_sq += v * v;
    }

    // 4. Compute MIS weight: w_sel = p_sel² / Σ(p_j²)
    let p_sel = pdf_vals[sel_idx];
    let mis_weight = power_heuristic(p_sel, pdf_sum_sq);

    // 5. Compute contribution: N * w_sel * f / p_sel
    let f = eval_fn(direction);
    let contribution = if p_sel > 1e-10 {
        f * (N as f64 * mis_weight / p_sel)
    } else {
        crate::vec3::Color3::ZERO
    };

    (direction, contribution)
}

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
                    let (direction, bias) = match sample {
                        BsdfSample::Delta { wi, f_cos } => {
                            // Pad to fixed 4-dim stride so subsequent bounces use consistent
                            // Sobol dimensions regardless of this bounce's path structure.
                            for _ in 0..4 {
                                let _ = dim_cursor.next_dim();
                            }
                            (wi, f_cos)
                        }
                        BsdfSample::NonDelta { pdf_kinds, count } => {
                            // One-sample MIS with power heuristic (β=2).
                            // Selects one strategy uniformly, generates direction from it,
                            // evaluates ALL PDFs, computes weighted estimator:
                            //   N * (p_sel² / Σp_j²) * f / p_sel
                            let env_pdf = UniformHemispherePDF::new(si.shading_normal());
                            let light_pdf = HittablePDF::new(lights, si.point(), ray.time);

                            // Track consumption for fixed-dim stride padding.
                            let mix_start = dim_cursor.offset();

                            // Build surface PDFs only for valid slots.
                            let eval = |d: Vec3| material.eval(wo, d, &si);
                            let (direction, contribution) = match count {
                                0 => {
                                    let pdfs: [&dyn PDF<S>; 2] = [&env_pdf, &light_pdf];
                                    mis_sample(pdfs, eval, dim_cursor)
                                }
                                1 => {
                                    let s0 = PdfEnum::new(&pdf_kinds[0]);
                                    let pdfs: [&dyn PDF<S>; 3] = [&env_pdf, &light_pdf, &s0];
                                    mis_sample(pdfs, eval, dim_cursor)
                                }
                                2 => {
                                    let s0 = PdfEnum::new(&pdf_kinds[0]);
                                    let s1 = PdfEnum::new(&pdf_kinds[1]);
                                    let pdfs: [&dyn PDF<S>; 4] = [&env_pdf, &light_pdf, &s0, &s1];
                                    mis_sample(pdfs, eval, dim_cursor)
                                }
                                _ => unreachable!("at most 2 surface PDFs"),
                            };

                            // Pad mixture dims to exactly 4 (1 selection + 3 direction).
                            let mix_consumed = dim_cursor.offset() - mix_start;
                            for _ in mix_consumed..4 {
                                let _ = dim_cursor.next_dim();
                            }
                            (direction, contribution)
                        }
                    };

                    accumulated_attenuation = accumulated_attenuation * bias;
                    ray = Ray::new_with_time(si.point(), direction, ray.time);

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
