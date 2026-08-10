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

use crate::integrator::Integrator;
use crate::intersect::Intersectable;
use crate::intersect::interaction::{MaterialHit, SurfaceInteraction};
use crate::light::Sampleable;
use crate::light::environment::EnvironmentMap;
use crate::material::{BsdfScatter, MAX_BSDF_STRATS, Material};
use crate::math::interval::Interval;
use crate::primitives::LightPrimitive;
use crate::ray::Ray;
use crate::sampler::{SampleStream, SamplerRng};
use crate::sampling::pdf::{
    EmitterPDF, MisHeuristic, PdfConvCtx, PdfKind, PdfStrategy, SolidAnglePdf,
};

use crate::math::vec3::{Color3, Direction3};

use super::{BounceResult, PathState, SPLIT_MAX_DEPTH};

/// Per-bounce throughput clamp to prevent fireflies from high-variance paths.
/// MIS-weighted contributions (e.g. from coated material NonDelta fallback) can
/// amplify `accumulated_attenuation` well beyond physical range; capping it here
/// stops the amplification from propagating to downstream bounces.
/// Currently set to `f32::MAX` to avoid clamping, but can be reduced if fireflies are observed.
const PATH_THROUGHPUT_LIMIT: f32 = f32::MAX - 1.;

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
fn mis_sample<'a, const N: usize>(
    pdfs: [PdfStrategy<'a>; N],
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
        Some(env_map) => env_map.to_solid_angle_pdf(wi).0,
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

/// Per-path state for the integrator. This is a small struct that is copied around and mutated in
/// place during path tracing. It contains information about the accumulated throughput, the
/// previous BSDF PDF, and whether the previous scatter was delta (specular). This state is used to
/// compute the contribution of each bounce and to determine when to terminate the path.
#[derive(Clone, Copy)]
pub struct TracingState {
    /// Accumulated throughput (product of f_cos/pdf). Mutated in place by
    /// process_bounce: RR division, bias multiply, throughput clamp.
    pub throughput: Color3,
    /// BSDF mixture PDF of the previous bounce's continuation direction —
    /// feeds emission MIS and background MIS.
    pub prev_bsdf_pdf: f32,
    /// Whether the previous scatter was delta (specular).
    pub prev_was_delta: bool,
    /// The current bounce count (0 for the first bounce).
    bounce: u32,
    /// The remaining depth for the path.
    remaining_depth: u32,
}

impl PathState for TracingState {
    fn bounce(&self) -> u32 {
        self.bounce
    }

    fn remaining_depth(&self) -> u32 {
        // Return the remaining depth
        self.remaining_depth
    }

    fn advance(&mut self) {
        self.bounce += 1;
        self.remaining_depth = self.remaining_depth.saturating_sub(1);
    }

    fn set_remaining_depth(&mut self, depth: u32) {
        self.remaining_depth = depth;
    }
}

impl Default for TracingState {
    /// Fresh path: unit throughput, delta-like flag (no MIS on bounce 0).
    /// NOTE: a derived `Default` would zero the throughput and clear the
    /// delta flag, silently blacking out every path — this must stay manual.
    fn default() -> Self {
        Self {
            throughput: Color3::ONE,
            prev_bsdf_pdf: 0.0,
            prev_was_delta: true,
            bounce: 0,
            remaining_depth: 0,
        }
    }
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
    type PathState = TracingState;

    fn max_depth(&self) -> u32 {
        self.max_depth
    }

    fn process_bounce<S: SampleStream, R: SamplerRng>(
        &self,
        ray: &Ray,
        hit: &MaterialHit<'_>,
        world: &impl Intersectable,
        lights: &[LightPrimitive],
        state: &mut Self::PathState,
        stream: &mut S,
        rng: &mut R,
    ) -> BounceResult<Self::PathState> {
        let si = SurfaceInteraction::from_material_hit(hit, ray);
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
        let mut contribution = if state.bounce() == 0 || state.prev_was_delta {
            state.throughput * emission
        } else {
            // Compute the light's solid-angle PDF for the continuation direction.
            let w_emit = MisHeuristic::Power.weight::<2>(0, &[state.prev_bsdf_pdf, light_pdf]);
            w_emit * state.throughput * emission
        };

        // If the maximum attenuation is very small, terminate the path early to
        // avoid unnecessary computation.
        let max_attenuation = state
            .throughput
            .x()
            .max(state.throughput.y())
            .max(state.throughput.z());
        if max_attenuation < 1e-6 {
            return BounceResult {
                contribution,
                next_ray: None,
                delta_child: None,
            };
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

            // Lights at infinity (environment) have no finite area to sample —
            // their contribution is handled by the background/emission paths,
            // not NEE. Skipping them also avoids a degenerate area→solid-angle
            // conversion (0 · ∞²).
            if sample.distance.is_finite() {
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
                    let w_nee =
                        MisHeuristic::Power.weight::<2>(0, &[light_pdf_at_nee, bsdf_pdf_at_nee]);

                    let f = material.eval(wo, light_unit, &si);
                    let cos_light = sample.normal.dot((-light_unit).into_inner()).abs();

                    // Convert the light's area PDF to solid angle: p_ω = p_A · d² / |cosθ_l|.
                    // The domain newtypes make the conversion explicit — the sample is
                    // measured per unit area on the light surface, but the estimator
                    // needs a solid-angle density.
                    let pdf_solid = SolidAnglePdf::from((
                        sample.pdf,
                        PdfConvCtx {
                            dist: sample.distance,
                            cos_there: cos_light,
                        },
                    ));

                    // N factor: uniform selection over N lights, estimator = N * contribution.
                    // material.eval() already includes the surface cosine factor (|cos θ_s|)
                    // as required by the rendering equation — no additional cos_surface here.
                    let n_lights = lights.len() as f32;
                    let direct =
                        w_nee * n_lights * state.throughput * light_emission * f / pdf_solid.0;
                    contribution += direct;
                }
            }
        }

        // Sample a random number for Russian Roulette
        let rr = rng.next();

        // Russian Roulette: survival probability proportional to current
        // path throughput. The 0.05 floor bounds variance from low-throughput paths.
        if state.bounce() >= 5 {
            let survival = max_attenuation.clamp(0.05, 1.0);
            if rr > survival {
                return BounceResult {
                    contribution,
                    next_ray: None,
                    delta_child: None,
                };
            }
            state.throughput /= survival;
        }

        // Scope-limited next_dim to release session borrow before Split recursion.
        let scatter_result = {
            let mut next_mat_dim = || -> f32 { rng.next() };
            material.scatter(wo, &si, &mut next_mat_dim)
        };

        let mut delta_child = None;
        let mut next_ray = None;

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
                    // Build the delta child's ray and state. The child is a fresh
                    // delta path whose throughput is the PRE-bias throughput scaled
                    // by delta_f_cos (current code computes the child contribution
                    // before the bias multiply). The child's path is traced inline
                    // by the driver (trace_path) with its own remaining_depth.
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

                    let child_remaining_depth = state
                        .remaining_depth()
                        .saturating_sub(1)
                        .min(SPLIT_MAX_DEPTH);

                    let child_state = TracingState {
                        throughput: state.throughput * delta_f_cos,
                        prev_bsdf_pdf: 0.0,
                        prev_was_delta: true,
                        bounce: 0,
                        remaining_depth: child_remaining_depth,
                    };
                    delta_child = Some((delta_ray, child_state));

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

            state.prev_was_delta = new_prev_was_delta;
            state.prev_bsdf_pdf = new_prev_bsdf_pdf;
            state.throughput *= bias;
            // Clamp per-bounce throughput to prevent fireflies from
            // high-variance paths (e.g. coated material NonDelta fallback
            // where the global-frame PDF and internal-frame eval mismatch
            // produces large MIS-weighted contributions that would
            // otherwise amplify all subsequent bounces).
            let max_att = state
                .throughput
                .x()
                .max(state.throughput.y())
                .max(state.throughput.z());
            if max_att > PATH_THROUGHPUT_LIMIT {
                state.throughput *= PATH_THROUGHPUT_LIMIT / max_att;
            }

            // Update the ray for the next bounce, preserving and regenerating
            // ray differentials so texture filtering survives indirect bounces.
            let new_ray = Ray::new_with_differentials(
                si.point(),
                direction,
                ray.time,
                ray.propagate_differentials(normal, hit_time, eta, hit_point, si.hit().curvature),
            );
            next_ray = Some(new_ray);
        }

        BounceResult {
            contribution,
            next_ray,
            delta_child,
        }
    }

    fn eval_background(&self, direction: Direction3, state: &Self::PathState) -> Color3 {
        let background_color = if let Some(env_map) = &self.env_map {
            env_map.le(direction)
        } else {
            self.background
        };

        // Ray missed the world geometry — accumulate background and terminate. When an
        // environment map is present, indirect bounces need MIS weighting: the bounce
        // direction was sampled by the BSDF (not the env map), so the env map contribution
        // is weighted by how likely the BSDF would have chosen that direction vs the env
        // map's own distribution. Without this, a narrow BSDF lobe pointing at a bright
        // environment pixel produces fireflies.
        if state.prev_was_delta {
            // First bounce or delta path: the direction was determined by the camera
            // or a deterministic scatter — no MIS weight needed.
            return state.throughput * background_color;
        }

        // MIS weight for background contribution based on previous BSDF PDF
        let env_pdf = match &self.env_map {
            Some(env_map) => env_map.to_solid_angle_pdf(direction).0,
            None => 1.0 / (4.0 * PI),
        };
        let w_miss = MisHeuristic::Power.weight::<2>(0, &[state.prev_bsdf_pdf, env_pdf]);

        w_miss * state.throughput * background_color
    }
}

impl PathTracingIntegrator {
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

        // Material strategies — PdfKind is Copy, so we can store copies directly.
        let mut mat_strats: [PdfKind; MAX_BSDF_STRATS] = [PdfKind::UniformSphere; MAX_BSDF_STRATS];
        let mut mat_count = 0usize;
        for pk in pdf_kinds.iter().flatten() {
            mat_strats[mat_count] = *pk;
            mat_count += 1;
        }

        // Build strategy array: env at index 0, materials follow.
        let mut strategies = [PdfStrategy::Kind(PdfKind::UniformSphere); MAX_BSDF_STRATS + 1];
        strategies[0] = match self.env_map.as_deref() {
            Some(env) => PdfStrategy::Env(env),
            None if is_volume => PdfStrategy::Kind(vol_fallback),
            None => PdfStrategy::Kind(env_fallback),
        };
        let mut n = 1usize;
        mat_strats.iter().take(mat_count).for_each(|mat| {
            strategies[n] = PdfStrategy::Kind(*mat);
            n += 1;
        });

        // Selection: independent random from RNG
        let sel_idx_raw = rng.next();
        let sel_idx = (sel_idx_raw * n as f32).min(n as f32 - 1e-15) as usize;
        // Direction: correlated 2D from stream
        let (pdf_u, pdf_v) = stream.next_2d();

        let (direction, contribution, p_mix) =
            mis_sample(strategies, n, eval, sel_idx, pdf_u, pdf_v);
        (direction, contribution, p_mix)
    }
}
