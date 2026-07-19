/// Reference CPU Monte-Carlo path-tracing integrator.
///
/// Iteratively traces/scatters up to `depth` bounces and multiplies
/// attenuation along the path.
///
/// TODO(gpu): mirror this boundary in a separate path-trace kernel / WGSL entrypoint.
/// The `li()` method is the natural split point: it takes a ray and returns radiance.
///
use std::f32::consts::PI;
use std::sync::Arc;

use crate::environment::EnvironmentMap;
use crate::hittable::{Intersectable, Sampleable, SurfaceInteraction};
use crate::integrator::Integrator;
use crate::interval::Interval;
use crate::material::{BsdfScatter, MAX_BSDF_STRATS, Material};
use crate::pdf::{EmitterPDF, EnvPdf, MisHeuristic, PDF, PdfKind};
use crate::ray::Ray;
use crate::sampler::{SampleStream, SamplerRng};

use crate::vec3::{Color3, Direction3};

/// Maximum depth for the split delta path (mirror direction from Mix one-delta).
/// Prevents exponential cascade when `max_depth` is large (e.g. 50) and the
/// mirror direction repeatedly hits delta-Mix surfaces.  Matches
/// `MAX_INTERNAL_BOUNCES` in Coated material.
const SPLIT_MAX_DEPTH: u32 = 5;

/// Per-bounce throughput clamp to prevent fireflies from high-variance paths.
/// MIS-weighted contributions (e.g. from coated material NonDelta fallback) can
/// amplify `accumulated_attenuation` well beyond physical range; capping it here
/// stops the amplification from propagating to downstream bounces.
/// Currently set to `f32::MAX` to avoid clamping, but can be reduced if fireflies are observed.
const PATH_THROUGHPUT_LIMIT: f32 = f32::MAX - f32::EPSILON;

/// One-sample MIS estimator with power heuristic (β=2).
///
/// Selects one strategy uniformly at random (probability 1/count), generates a
/// direction from that strategy's PDF, then evaluates ALL `count` PDFs at the
/// sampled direction. Returns the direction and the MIS-weighted estimator:
///
///   contribution = count * (p_sel² / Σp_j²) * f / p_sel
///
/// where p_sel is the selected strategy's PDF value and p_j are all PDF values
/// at the sampled direction. This is provably unbiased and has lower variance
/// than the mixture-based f/p_mix estimator.
///
/// `pdfs` is a fixed-size array (`N` ≥ `count`). Only the first `count`
/// entries are populated; the remaining `N-count` slots are ignored.
/// `sel_idx` is the pre-selected strategy index (from `SamplerRng`).
/// `(pdf_u, pdf_v)` are the correlated 2D samples for direction generation.
#[inline(always)]
fn mis_sample<const N: usize>(
    pdfs: [&dyn PDF; N],
    count: usize,
    eval_fn: impl FnOnce(Direction3) -> Color3,
    sel_idx: usize,
    pdf_u: f32,
    pdf_v: f32,
) -> (Direction3, Color3, f32) {
    debug_assert!(count > 0, "mis_sample requires at least one PDF strategy");
    debug_assert!(count <= N, "mis_sample count exceeds array capacity");

    // 1. Generate direction from selected strategy
    let direction = pdfs[sel_idx].generate(pdf_u, pdf_v).normalize();

    // 2. Evaluate ALL PDFs at the sampled direction, compute sum of squares.
    //    Only count entries are populated — remaining N-count are stale.
    let mut pdf_sum = 0.0;
    let mut pdf_vals = [0.0f32; N];
    for (i, pdf) in pdfs.iter().enumerate().take(count) {
        let v = pdf.value(direction);
        pdf_vals[i] = v;
        pdf_sum += v;
    }

    // 3. Compute MIS weight: w_sel = p_sel² / Σ(p_j²)
    let mis_weight = MisHeuristic::Power.weight::<N>(sel_idx, &pdf_vals);

    // 4. Compute contribution: N * w_sel * f / p_sel
    let p_sel = pdf_vals[sel_idx];
    let f = eval_fn(direction);
    let contribution = if p_sel > 1e-10 {
        f * (count as f32 * mis_weight / p_sel)
    } else {
        Color3::ZERO
    };
    (direction, contribution, pdf_sum / count as f32)
}

/// Compute the BSDF mixture PDF value for a direction.
///
/// Matches the mixture structure used in the scatter step (env + material PDFs, without light_pdf,
/// since NEE handles light sampling separately).
#[inline]
fn bsdf_mixture_pdf(
    wo: Direction3,
    wi: Direction3,
    si: &SurfaceInteraction,
    material: &Material,
    env_map: Option<&Arc<EnvironmentMap>>,
    is_volume: bool,
) -> f32 {
    // env_pdf value: UniformHemisphere (surfaces) or UniformSphere (volumes)
    let env_value = match env_map {
        Some(env_map) => env_map.to_solid_angle_pdf(wi),
        None => {
            if is_volume {
                1.0 / (4.0 * PI)
            } else {
                let cos_theta = wi.dot(si.shading_normal().into_inner());
                if cos_theta > 0.0 {
                    1.0 / (2.0 * PI)
                } else {
                    0.0
                }
            }
        }
    };

    // Material PDF values — matches the strategy structure in the scatter step
    let (mat_sum, n_mat) = match material {
        Material::Mix(inner) => {
            let (a, b) = (inner.a.as_ref(), inner.b.as_ref());
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
    (env_value + mat_sum) / n as f32
}

pub struct PathTracingIntegrator {
    max_depth: u32,
    background: Color3,
    env_map: Option<Arc<EnvironmentMap>>,
}

impl PathTracingIntegrator {
    pub fn new(max_depth: u32, background: Color3, env_map: Option<Arc<EnvironmentMap>>) -> Self {
        Self {
            max_depth,
            background,
            env_map,
        }
    }
}

impl Integrator for PathTracingIntegrator {
    fn env_map(&self) -> Option<&Arc<EnvironmentMap>> {
        self.env_map.as_ref()
    }

    fn background_color(&self) -> Color3 {
        self.background
    }

    fn li<S: SampleStream, R: SamplerRng>(
        &self,
        initial_ray: &mut Ray,
        world: &dyn Intersectable,
        lights: &[Arc<dyn Sampleable>],
        stream: &mut S,
        rng: &mut R,
    ) -> Color3 {
        self.li_inner::<S, R>(initial_ray, world, lights, stream, rng, self.max_depth)
    }
}

impl PathTracingIntegrator {
    fn li_inner<S: SampleStream, R: SamplerRng>(
        &self,
        initial_ray: &mut Ray,
        world: &dyn Intersectable,
        lights: &[Arc<dyn Sampleable>],
        stream: &mut S,
        rng: &mut R,
        remaining_depth: u32,
    ) -> Color3 {
        let mut accumulated_attenuation = Color3::ONE;
        let mut accumulated_color = Color3::ZERO;
        let mut ray = *initial_ray;
        let mut prev_bsdf_pdf: f32 = 0.0;
        let mut prev_was_delta: bool = true;

        for bounce in 0..remaining_depth {
            if let Some(mat_hit) = world.intersect(&ray, Interval::from(0.001, f32::INFINITY)) {
                let si = SurfaceInteraction::from_material_hit(mat_hit, &ray);
                let material = si.material();
                let normal = si.shading_normal();
                let hit_time = si.time();
                let hit_point = si.point();

                let light_pdf = if lights.is_empty() {
                    0.0
                } else {
                    EmitterPDF::new(lights, si.point(), ray.time).value(ray.direction.normalize())
                };

                // Accumulate emission with MIS weight to avoid double-counting with NEE.
                // At bounce 0 or after a delta bounce, no previous scatter exists that could
                // overlap with NEE, so emission is added at full weight (PBRT convention).
                // Outgoing direction (away from the surface) is the negative of the ray direction.
                let wo = -ray.direction.normalize();
                // NEE uses wo-aware emission (e.g., Beer's law for coated)
                let emission = material.emitted(wo, &si);
                if bounce == 0 || prev_was_delta {
                    accumulated_color += accumulated_attenuation * emission;
                } else {
                    // Compute the light's solid-angle PDF for the continuation direction.
                    let light_pdf_emit = light_pdf;
                    let w_emit =
                        MisHeuristic::Power.weight::<2>(0, &[prev_bsdf_pdf, light_pdf_emit]);
                    accumulated_color += w_emit * accumulated_attenuation * emission;
                }

                // Sample the material to get the next ray and attenuation
                let max_attenuation = accumulated_attenuation
                    .x()
                    .max(accumulated_attenuation.y())
                    .max(accumulated_attenuation.z());

                // If the maximum attenuation is very small, terminate the path early to avoid unnecessary computation
                if max_attenuation < 1e-6 {
                    return accumulated_color;
                }

                let is_volume = normal.length_squared() < 1e-10;

                // Shadow ray based Next Event Estimation (NEE) for direct lighting.
                // Skip for delta materials (mirrors, glass) — BSDF is zero for any
                // direction that doesn't match the single specular direction.
                if !lights.is_empty() && !material.is_delta() {
                    // Pick a random light source to sample from the list
                    let light_idx = (rng.next() * lights.len() as f32) as usize;
                    let light = &lights[light_idx % lights.len()];

                    // Sample a point on the light source — returns direction, normal, distance, and area PDF
                    let (u, v) = stream.next_2d();
                    let sample = light.sample_light(si.point(), u, v, ray.time);
                    let light_unit = sample.direction.normalize();
                    let light_emission = sample.emission;

                    // Shadow ray: test visibility/occlusion between the surface point and the light source
                    let shadow_ray = Ray::new_with_time(si.point(), light_unit, ray.time);
                    let far = (sample.distance - 0.001).max(0.001);
                    let shadow_ray_interval = Interval::from(0.001, far);
                    let occluded = world.occluded(&shadow_ray, shadow_ray_interval);
                    if !occluded {
                        // Unoccluded — compute direct lighting contribution.
                        // Area-sampling form: L ≈ f_r · L_e · |cos θ_s| · |cos θ_l| · V / (p_A · d²)

                        // MIS weight: compare the light sampler's PDF against the BSDF mixture PDF
                        // at the NEE direction. This weights NEE proportionally to how much better
                        // it is than the continuation ray for this particular direction.
                        // Re-evaluate light PDF at the NEE direction (not the incoming ray's
                        // direction): the MIS weight compares how well the light-sampling
                        // strategy and the BSDF continuation strategy explain this sampled
                        // NEE direction.
                        let light_pdf_at_nee =
                            EmitterPDF::new(lights, si.point(), ray.time).value(light_unit);
                        let bsdf_pdf_at_nee = bsdf_mixture_pdf(
                            wo,
                            light_unit,
                            &si,
                            material,
                            self.env_map.as_ref(),
                            is_volume,
                        );
                        let w_nee = MisHeuristic::Power
                            .weight::<2>(0, &[light_pdf_at_nee, bsdf_pdf_at_nee]);

                        let f = material.eval(wo, light_unit, &si);
                        let cos_light = sample.normal.dot((-light_unit).into_inner()).abs();

                        // N factor: uniform selection over N lights, estimator = N * contribution.
                        // material.eval() already includes the surface cosine factor (|cos θ_s|)
                        // as required by the rendering equation — no additional cos_surface here.
                        let n_lights = lights.len() as f32;
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
                let rr = rng.next();

                // Russian Roulette: survival probability proportional to current
                // path throughput.  The 0.05 floor bounds variance from low-throughput paths.
                if bounce >= 5 {
                    let survival = max_attenuation.clamp(0.05, 1.0);
                    if rr > survival {
                        return accumulated_color;
                    }
                    accumulated_attenuation /= survival;
                }

                // Scope-limited next_dim to release session borrow before Split recursion.
                let scatter_result = {
                    let mut next_mat_dim = || -> f32 { rng.next() };
                    material.scatter(wo, &si, &mut next_mat_dim)
                };

                if let Some(scatter) = scatter_result {
                    let mut new_prev_was_delta = false;
                    let mut new_prev_bsdf_pdf = 0.0;

                    let (direction, bias, eta) = match scatter {
                        BsdfScatter::Delta { wi, f_cos, eta } => {
                            new_prev_was_delta = true;
                            (wi, f_cos, eta)
                        }
                        BsdfScatter::NonDelta { pdf_kinds } => {
                            let (d, c, p) = self.mis_sample_continuation::<S, R>(
                                pdf_kinds,
                                wo,
                                &si,
                                material,
                                normal,
                                (stream, rng),
                            );
                            new_prev_bsdf_pdf = p;
                            (d, c, None)
                        }
                        BsdfScatter::Split {
                            delta_wi,
                            delta_f_cos,
                            delta_eta,
                            non_delta_pdf_kinds,
                        } => {
                            // Trace the delta child's deterministic path (mirror direction).
                            let delta_ray = Ray::new_with_differentials(
                                si.point(),
                                delta_wi,
                                ray.time,
                                ray.propagate_differentials(
                                    normal,
                                    hit_time,
                                    delta_eta,
                                    hit_point,
                                    si.hit().curvature,
                                ),
                            );
                            let mut delta_ray_mut = delta_ray;
                            let delta_li = self.li_inner::<S, R>(
                                &mut delta_ray_mut,
                                world,
                                lights,
                                stream,
                                rng,
                                remaining_depth
                                    .saturating_sub(bounce + 1)
                                    .min(SPLIT_MAX_DEPTH),
                            );
                            accumulated_color += accumulated_attenuation * delta_f_cos * delta_li;

                            // Continue with the non-delta child's MIS continuation.
                            let (d, c, p) = self.mis_sample_continuation::<S, R>(
                                non_delta_pdf_kinds,
                                wo,
                                &si,
                                material,
                                normal,
                                (stream, rng),
                            );
                            new_prev_bsdf_pdf = p;
                            (d, c, None)
                        }
                    };

                    prev_was_delta = new_prev_was_delta;
                    prev_bsdf_pdf = new_prev_bsdf_pdf;
                    accumulated_attenuation *= bias;
                    // Clamp per-bounce throughput to prevent fireflies from
                    // high-variance paths (e.g. coated material NonDelta fallback
                    // where the global-frame PDF and internal-frame eval mismatch
                    // produces large MIS-weighted contributions that would
                    // otherwise amplify all subsequent bounces).
                    let max_att = accumulated_attenuation
                        .x()
                        .max(accumulated_attenuation.y())
                        .max(accumulated_attenuation.z());
                    if max_att > PATH_THROUGHPUT_LIMIT {
                        accumulated_attenuation *= PATH_THROUGHPUT_LIMIT / max_att;
                    }

                    // Update the ray for the next bounce, preserving and regenerating
                    // ray differentials so texture filtering survives indirect bounces.
                    let new_ray = Ray::new_with_differentials(
                        si.point(),
                        direction,
                        ray.time,
                        ray.propagate_differentials(
                            normal,
                            hit_time,
                            eta,
                            hit_point,
                            si.hit().curvature,
                        ),
                    );
                    ray = new_ray;
                } else {
                    // Emissive materials return None — no scattering. Emission already added
                    // to accumulated_color via emitted() above.
                    return accumulated_color;
                }
            } else {
                let direction = ray.direction.normalize();
                let background_color = if let Some(env_map) = &self.env_map {
                    env_map.le(direction)
                } else {
                    self.background
                };
                // Ray missed the world geometry — accumulate background and terminate.
                // When an environment map is present, indirect bounces need MIS weighting:
                // the bounce direction was sampled by the BSDF (not the env map), so the
                // env map contribution is weighted by how likely the BSDF would have chosen
                // that direction vs the env map's own distribution. Without this, a narrow
                // BSDF lobe pointing at a bright environment pixel produces fireflies.
                if bounce == 0 || prev_was_delta {
                    // First bounce or delta path: the direction was determined by the camera
                    // or a deterministic scatter — no MIS weight needed.
                    return accumulated_color + accumulated_attenuation * background_color;
                }
                let env_pdf = match &self.env_map {
                    Some(env_map) => env_map.to_solid_angle_pdf(direction),
                    None => 1.0 / (4.0 * PI),
                };
                let w_miss = MisHeuristic::Power.weight::<2>(0, &[prev_bsdf_pdf, env_pdf]);
                return accumulated_color + w_miss * accumulated_attenuation * background_color;
            }
        }

        // Max bounce count reached — terminate the path. This can still contribute to the final
        // image if the last bounce was a non-delta and the accumulated attenuation is non-zero.
        accumulated_color
    }

    /// One-sample MIS with power heuristic (β=2).
    ///
    /// Selects one strategy uniformly, generates a direction from it,
    /// evaluates ALL PDFs, and returns the weighted estimator:
    /// `N · (p_sel² / Σp_j²) · f / p_sel`
    ///
    /// Strategy layout:
    /// - Index 0 always the environment PDF.  Always including the env
    ///   fallback gives indirect illumination an escape direction and
    ///   prevents degenerate MIS when no material strategy reaches the
    ///   sampled light direction (e.g. light behind a glass sphere).
    /// - Indices 1..N are the material strategies from `pdf_kinds`.
    /// - The light PDF (`light_pdf`) is excluded — NEE already handles
    ///   light sampling with its own dedicated MIS weight.
    fn mis_sample_continuation<S: SampleStream, R: SamplerRng>(
        &self,
        pdf_kinds: [Option<PdfKind>; MAX_BSDF_STRATS],
        wo: Direction3,
        si: &SurfaceInteraction,
        material: &Material,
        normal: Direction3,
        (stream, rng): (&mut S, &mut R),
    ) -> (Direction3, Color3, f32) {
        let eval = |d: Direction3| material.eval(wo, d, si);

        // Environment strategies
        let env_fallback = PdfKind::UniformHemisphere { normal };
        let vol_fallback = PdfKind::UniformSphere;
        let is_volume = normal.length_squared() < 1e-10;
        let env_holder = self.env_map.as_ref().map(EnvPdf::new);

        // Material strategies — PdfKind is Copy, so we can store copies directly.
        let mut mat_strats: [PdfKind; MAX_BSDF_STRATS] = [PdfKind::UniformSphere; MAX_BSDF_STRATS];
        let mut mat_count = 0usize;
        for pk in pdf_kinds.iter().flatten() {
            mat_strats[mat_count] = *pk;
            mat_count += 1;
        }

        // Build reference array: env at index 0, materials follow.
        let mut pdf_refs = [&env_fallback as &dyn PDF; MAX_BSDF_STRATS + 1];

        // Index 0: environment strategy
        if let Some(ref env_pdf) = env_holder {
            pdf_refs[0] = env_pdf;
        } else if is_volume {
            pdf_refs[0] = &vol_fallback;
        } else {
            pdf_refs[0] = &env_fallback;
        }
        let mut n = 1usize;

        // Material strategies at indices 1..n
        mat_strats.iter().take(mat_count).for_each(|mat| {
            pdf_refs[n] = mat;
            n += 1;
        });

        // Selection: independent random from RNG
        let sel_idx_raw = rng.next();
        let sel_idx = (sel_idx_raw * n as f32).min(n as f32 - 1e-15) as usize;
        // Direction: correlated 2D from stream
        let (pdf_u, pdf_v) = stream.next_2d();

        let (direction, contribution, p_mix) = mis_sample(pdf_refs, n, eval, sel_idx, pdf_u, pdf_v);
        (direction, contribution, p_mix)
    }
}
