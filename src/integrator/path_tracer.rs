/// Reference CPU Monte-Carlo path-tracing integrator.
///
/// Iteratively traces/scatters up to `depth` bounces and multiplies
/// attenuation along the path.
///
/// TODO(gpu): mirror this boundary in a separate path-trace kernel / WGSL entrypoint.
/// The `li()` method is the natural split point: it takes a ray and returns radiance.
///
use std::f64::consts::PI;
use std::sync::Arc;

use crate::hittable::{Intersectable, Sampleable, SurfaceInteraction};
use crate::integrator::Integrator;
use crate::interval::Interval;
use crate::material::{BsdfSample, Material, PdfKind};
use crate::pdf::{Emitter, PDF, PdfEnum, power_heuristic};
use crate::ray::Ray;
use crate::sampler::{DimCursor, SampleDims, Sampler};
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
fn mis_sample<S: Sampler>(
    pdfs: &[&dyn PDF<S>],
    eval_fn: impl FnOnce(Vec3) -> crate::vec3::Color3,
    dim_cursor: &mut DimCursor<S>,
) -> (Vec3, Color3, f64) {
    let n = pdfs.len();
    debug_assert!(n > 0, "mis_sample requires at least one PDF strategy");

    // 1. Select strategy uniformly: 1 QMC dim for selection
    let u_select = dim_cursor.next_sample();
    let sel_idx = (u_select * n as f64).min(n as f64 - 1e-15) as usize;

    // 2. Generate direction from selected strategy
    let direction = pdfs[sel_idx].generate(dim_cursor).unit_vector();

    // 3. Evaluate ALL PDFs at the sampled direction, compute sum of squares
    let mut pdf_sum_sq = 0.0;
    let mut pdf_sum = 0.0;
    // Stack-allocate for the small strategy counts we support (max ~4).
    let mut pdf_vals = [0.0f64; 8];
    for (i, pdf) in pdfs.iter().enumerate() {
        let v = pdf.value(direction);
        pdf_vals[i] = v;
        pdf_sum_sq += v * v;
        pdf_sum += v;
    }

    // 4. Compute MIS weight: w_sel = p_sel² / Σ(p_j²)
    let p_sel = pdf_vals[sel_idx];
    let mis_weight = power_heuristic(p_sel, pdf_sum_sq);

    // 5. Compute contribution: N * w_sel * f / p_sel
    let f = eval_fn(direction);
    let contribution = if p_sel > 1e-10 {
        f * (n as f64 * mis_weight / p_sel)
    } else {
        crate::vec3::Color3::ZERO
    };

    (direction, contribution, pdf_sum / n as f64)
}

/// Compute the BSDF mixture PDF value for a direction.
///
/// Matches the mixture structure used in the scatter step (env + material PDFs, without light_pdf,
/// since NEE handles light sampling separately).
#[inline]
fn bsdf_mixture_pdf(
    wo: Vec3,
    wi: Vec3,
    si: &SurfaceInteraction,
    material: &crate::material::Material,
    is_volume: bool,
) -> f64 {
    // env_pdf value: UniformHemisphere (surfaces) or UniformSphere (volumes)
    let env_value = if is_volume {
        1.0 / (4.0 * PI)
    } else {
        let cos_theta = wi.dot(&si.shading_normal());
        if cos_theta > 0.0 {
            1.0 / (2.0 * PI)
        } else {
            0.0
        }
    };

    // Material PDF values — matches the strategy structure in the scatter step
    let (mat_sum, n_mat) = match material {
        Material::Mix { a, b, .. } => {
            let pa = a.pdf(wo, wi, si);
            let pb = b.pdf(wo, wi, si);
            let has_a = a.pdf_kind(wo, si).is_some();
            let has_b = b.pdf_kind(wo, si).is_some();
            (pa + pb, has_a as usize + has_b as usize)
        }
        _ => {
            let p = material.pdf(wo, wi, si);
            let has_p = material.pdf_kind(wo, si).is_some();
            (p, has_p as usize)
        }
    };

    let n = 1 + n_mat; // env + material strategies
    (env_value + mat_sum) / n as f64
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
        let mut prev_bsdf_pdf: f64 = 0.0;
        let mut prev_was_delta: bool = true;

        for bounce in 0..self.max_depth {
            if let Some(mat_hit) = world.intersect(&ray, Interval::from(0.001, f64::INFINITY)) {
                let si = SurfaceInteraction::from_material_hit(mat_hit, &ray);
                let material = si.material();

                // Accumulate emission with MIS weight to avoid double-counting with NEE.
                // At bounce 0 or after a delta bounce, no previous scatter exists that could
                // overlap with NEE, so emission is added at full weight (PBRT convention).
                let emission = si.emission();
                if bounce == 0 || prev_was_delta {
                    accumulated_color += accumulated_attenuation * emission;
                } else {
                    // Compute the light's solid-angle PDF for the continuation direction.
                    let light_pdf_emit = <Emitter as PDF<S>>::value(
                        &Emitter::new(lights, ray.origin, ray.time),
                        ray.direction.unit_vector(),
                    );
                    let sum_sq = prev_bsdf_pdf * prev_bsdf_pdf + light_pdf_emit * light_pdf_emit;
                    let w_emit = power_heuristic(prev_bsdf_pdf, sum_sq);
                    accumulated_color += w_emit * accumulated_attenuation * emission;
                }
                // Outgoing direction (away from the surface) is the negative of the ray direction
                let wo = -ray.direction.unit_vector();

                // Sample the material to get the next ray and attenuation
                let max_attenuation = accumulated_attenuation
                    .x
                    .max(accumulated_attenuation.y)
                    .max(accumulated_attenuation.z);

                // If the maximum attenuation is very small, terminate the path early to avoid unnecessary computation
                if max_attenuation < 1e-6 {
                    return accumulated_color;
                }

                let is_volume = si.shading_normal().near_zero();

                // Shadow ray based Next Event Estimation (NEE) for direct lighting.
                // Skip for delta materials (mirrors, glass) — BSDF is zero for any
                // direction that doesn't match the single specular direction.
                if !lights.is_empty() && !material.is_delta() {
                    // Pick a random light source to sample from the list
                    let light_idx = (dim_cursor.next_sample() * lights.len() as f64) as usize;
                    let light = &lights[light_idx % lights.len()];

                    // Sample a point on the light source — returns direction, normal, distance, and area PDF
                    let (u, v) = (dim_cursor.next_sample(), dim_cursor.next_sample());
                    let sample = light.sample_light(si.point(), u, v, ray.time);
                    let light_unit = sample.direction.unit_vector();
                    let light_emission = sample.emission;

                    // Shadow ray: test visibility/occlusion between the surface point and the light source
                    let shadow_ray = Ray::new_with_time(si.point(), light_unit, ray.time);
                    let far = (sample.distance - 0.001).max(0.001);
                    let shadow_ray_interval = Interval::from(0.001, far);
                    let occluded = world.intersect(&shadow_ray, shadow_ray_interval).is_some();
                    if !occluded {
                        // Unoccluded — compute direct lighting contribution.
                        // Area-sampling form: L ≈ f_r · L_e · |cos θ_s| · |cos θ_l| · V / (p_A · d²)

                        // MIS weight: compare the light sampler's PDF against the BSDF mixture PDF
                        // at the NEE direction. This weights NEE proportionally to how much better
                        // it is than the continuation ray for this particular direction.
                        let light_pdf_at_nee = <Emitter as PDF<S>>::value(
                            &Emitter::new(lights, si.point(), ray.time),
                            light_unit,
                        );
                        let bsdf_pdf_at_nee =
                            bsdf_mixture_pdf(wo, light_unit, &si, material, is_volume);
                        let sum_sq_nee =
                            light_pdf_at_nee * light_pdf_at_nee + bsdf_pdf_at_nee * bsdf_pdf_at_nee;
                        let w_nee = power_heuristic(light_pdf_at_nee, sum_sq_nee);

                        let f = material.eval(wo, light_unit, &si);
                        let cos_light = sample.normal.dot(&(-light_unit)).abs();

                        // N factor: uniform selection over N lights, estimator = N * contribution.
                        // material.eval() already includes the surface cosine factor (|cos θ_s|)
                        // as required by the rendering equation — no additional cos_surface here.
                        let n_lights = lights.len() as f64;
                        let direct = w_nee
                            * n_lights
                            * accumulated_attenuation
                            * light_emission
                            * f
                            * cos_light
                            / (sample.pdf * sample.distance * sample.distance);
                        accumulated_color += direct;
                    }
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

                // Sample the material to get the next ray and attenuation
                let u_mat = dim_cursor.next_sample();
                let v_mat = dim_cursor.next_sample();
                let w_mat = dim_cursor.next_sample();
                let x_mat = dim_cursor.next_sample();
                let y_mat = dim_cursor.next_sample();
                let z_mat = dim_cursor.next_sample();

                if let Some(sample) = material.sample(
                    wo,
                    &si,
                    SampleDims {
                        u: u_mat,
                        v: v_mat,
                        w: w_mat,
                        x: x_mat,
                        y: y_mat,
                        z: z_mat,
                    },
                ) {
                    let mut new_prev_was_delta = false;
                    let mut new_prev_bsdf_pdf = 0.0;
                    let (direction, bias) = match sample {
                        BsdfSample::Delta { wi, f_cos } => {
                            // Pad to fixed 4-dim stride so subsequent bounces use consistent
                            // Sobol dimensions regardless of this bounce's path structure.
                            for _ in 0..4 {
                                let _ = dim_cursor.next_dim();
                            }
                            new_prev_was_delta = true;
                            (wi, f_cos)
                        }
                        BsdfSample::NonDelta { pdf_kinds, count } => {
                            // One-sample MIS with power heuristic (β=2).
                            // Selects one strategy uniformly, generates direction from it,
                            // evaluates ALL PDFs, computes weighted estimator:
                            //   N * (p_sel² / Σp_j²) * f / p_sel
                            //
                            // Strategy selection:
                            // - Always include env_pdf: provides hemisphere/sphere fallback
                            //   for indirect illumination and escape directions. Without it,
                            //   MIS weights become suboptimal for multi-bounce paths and
                            //   caustic-adjacent geometry (e.g. glass sphere in Cornell box).
                            // - Surfaces: env_pdf = UniformHemisphere (general fallback, never
                            //   duplicates material PDFs). Volumes: env_pdf = UniformSphere.
                            // - light_pdf is excluded — NEE handles light sampling separately.

                            // PdfEnum dispatches via match on a hand-rolled enum (zero-cost).
                            let env_pdf: PdfEnum<S> = if is_volume {
                                PdfEnum::new(&crate::material::PdfKind::UniformSphere)
                            } else {
                                PdfEnum::new(&PdfKind::UniformHemisphere {
                                    normal: si.shading_normal(),
                                })
                            };

                            // Track consumption for fixed-dim stride padding.
                            let mix_start = dim_cursor.offset();

                            // Build surface PDFs only for valid slots.
                            let eval = |d: Vec3| material.eval(wo, d, &si);

                            // Build the strategy list in a fixed-size array.
                            // light_pdf is excluded — NEE handles light sampling separately.
                            // Max capacity: env(1) + s0(0/1) + s1(0/1) = 1..3.
                            // env_pdf is UniformHemisphere (surfaces) or UniformSphere (volumes) —
                            // never duplicates a material PDF (Cosine, Ggx, UniformSphere).
                            let make_pdf = |idx: usize| -> Option<PdfEnum<S>> {
                                if count > idx as u8 {
                                    Some(PdfEnum::new(&pdf_kinds[idx]))
                                } else {
                                    None
                                }
                            };
                            let s0 = make_pdf(0);
                            let s1 = make_pdf(1);

                            // Fixed-size array replaces Vec — zero heap allocation.
                            // Indices 1..3 are overwritten with material PDFs before use.
                            let mut pdf_refs: [&dyn PDF<S>; 4] =
                                [&env_pdf, &env_pdf, &env_pdf, &env_pdf];
                            let mut n = 1usize;
                            if let Some(ref s) = s0 {
                                pdf_refs[n] = s;
                                n += 1;
                            }
                            if let Some(ref s) = s1 {
                                pdf_refs[n] = s;
                                n += 1;
                            }

                            let (direction, contribution, p_mix) =
                                mis_sample(&pdf_refs[..n], eval, dim_cursor);

                            // Pad mixture dims to exactly 4 (1 selection + 3 direction).
                            let mix_consumed = dim_cursor.offset() - mix_start;
                            for _ in mix_consumed..4 {
                                let _ = dim_cursor.next_dim();
                            }
                            new_prev_bsdf_pdf = p_mix;
                            (direction, contribution)
                        }
                    };

                    prev_was_delta = new_prev_was_delta;
                    prev_bsdf_pdf = new_prev_bsdf_pdf;
                    accumulated_attenuation = accumulated_attenuation * bias;
                    ray = Ray::new_with_time(si.point(), direction, ray.time);
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
