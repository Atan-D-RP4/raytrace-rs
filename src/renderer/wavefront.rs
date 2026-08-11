use crate::film::FilmTile;
use crate::integrator::{BounceResult, Integrator, PathState};
use crate::intersect::interaction::MaterialHit;
use crate::intersect::Intersectable;
use crate::math::interval::Interval;
use crate::math::vec3::Color3;
use crate::primitives::LightPrimitive;
use crate::ray::Ray;
use crate::sampler::{SampleStream, SamplerRng};

/// The wavefront batch: parallel per-slot buffers. Grows as delta children are queued. No lifetime
/// — hits are per-stage locals, not stored.
pub struct WavefrontBatch<I: Integrator, S: SampleStream, R: SamplerRng> {
    /// The rays and states are used to track the current state of each path being traced.
    rays: Vec<Option<Ray>>,
    states: Vec<I::PathState>,

    /// The sample streams and RNGs are used to generate new rays and sample the scene.
    streams: Vec<S>,
    rngs: Vec<R>,

    /// Accumulated radiance for each path. Flows up the parent chain when a path is done.
    accumulator: Vec<Color3>,
    /// Camera-ray weight applied to the total radiance at the film add. `None` for delta children
    /// (their radiance flows to the parent), and for rays that have not yet been traced.
    weights: Vec<Option<Color3>>,
    /// Film coordinates for each ray, used to accumulate contributions to the correct pixel. `None`
    /// for delta children and untraced rays.
    pixels: Vec<Option<(u32, u32)>>,

    /// Index of the slot this path's radiance flows into (None = initial path).
    parents: Vec<Option<usize>>,
    /// Outstanding delta children that are yet to be traced.
    pending: Vec<u32>,
    /// Tracks which rays have completed their path tracing (either by reaching max depth or by not
    /// generating a new ray).
    done: Vec<bool>,
}

impl<I: Integrator, S: SampleStream, R: SamplerRng> WavefrontBatch<I, S, R> {
    pub fn new(
        rays: Vec<Option<Ray>>,
        states: Vec<I::PathState>,
        streams: Vec<S>,
        rngs: Vec<R>,
        weights: Vec<Option<Color3>>,
        pixels: Vec<Option<(u32, u32)>>,
    ) -> Self {
        let len = rays.len();
        assert_eq!(len, states.len());
        assert_eq!(len, streams.len());
        assert_eq!(len, rngs.len());
        assert_eq!(len, weights.len());
        assert_eq!(len, pixels.len());
        Self {
            rays,
            states,
            streams,
            rngs,
            accumulator: vec![Color3::ZERO; len],
            weights,
            pixels,
            parents: vec![None; len],
            pending: vec![0; len],
            done: vec![false; len],
        }
    }

    /// Flow a finished path's radiance up the parent chain. Cascades when a
    /// parent was already done and this was its last child.
    pub fn flow_up(&mut self, mut idx: usize, tile: &mut FilmTile) {
        loop {
            let radiance = self.accumulator[idx];
            match self.parents[idx] {
                Some(parent) => {
                    self.accumulator[parent] += radiance;
                    self.pending[parent] -= 1;
                    if self.done[parent] && self.pending[parent] == 0 {
                        idx = parent; // cascade
                    } else {
                        return;
                    }
                }
                None => {
                    let sample = radiance * self.weights[idx].unwrap();
                    if sample.into_inner().is_finite() {
                        let (x, y) = self.pixels[idx].unwrap();
                        tile.add_sample(x, y, sample);
                    }
                    return;
                }
            }
        }
    }

    /// Remove finished slots (done && pending == 0) and remap parent links.
    pub fn compact(&mut self) {
        let mut remap = vec![usize::MAX; self.rays.len()];
        let mut write = 0;
        for (i, remapped) in remap.iter_mut().enumerate() {
            if self.done[i] && self.pending[i] == 0 {
                continue; // finished — radiance already flowed up
            }
            *remapped = write;
            if write != i {
                self.rays[write] = self.rays[i];
                self.states[write] = self.states[i].clone();
                self.streams[write] = self.streams[i];
                self.rngs[write] = self.rngs[i];
                self.accumulator[write] = self.accumulator[i];
                self.weights[write] = self.weights[i];
                self.pixels[write] = self.pixels[i];
                self.pending[write] = self.pending[i];
                self.done[write] = self.done[i];
            }
            write += 1;
        }
        self.rays.truncate(write);
        self.states.truncate(write);
        self.streams.truncate(write);
        self.rngs.truncate(write);
        self.accumulator.truncate(write);
        self.weights.truncate(write);
        self.pixels.truncate(write);
        self.pending.truncate(write);
        self.done.truncate(write);
        for i in 0..write {
            if let Some(p) = self.parents[i] {
                self.parents[i] = Some(remap[p]); // parents are never removed — remap is valid
            }
        }
        self.parents.truncate(write);
    }

    /// Stage 1: rays -> hits. One BVH traversal per active ray.
    pub fn trace_rays<'a, W: Intersectable>(&self, world: &'a W) -> Vec<Option<MaterialHit<'a>>> {
        let rays = &self.rays;
        let mut hits = Vec::with_capacity(rays.len());
        for ray in rays {
            hits.push(match ray {
                Some(r) => world.intersect(r, Interval::from(0.001, f32::INFINITY)),
                None => None,
            });
        }
        hits
    }

    /// Stage 2: hits -> bounces. Pure per-slot shading. Waiting slots and the
    /// max_depth == 0 edge produce zero-continuation bounces.
    pub fn shade_hits(
        &mut self,
        integrator: &I,
        hits: &[Option<MaterialHit<'_>>],
        world: &impl Intersectable,
        lights: &[LightPrimitive],
    ) -> Vec<BounceResult<I::PathState>> {
        (0..self.rays.len())
            .map(|i| {
                if self.rays[i].is_none() {
                    return BounceResult {
                        contribution: Color3::ZERO,
                        next_ray: None,
                        delta_child: None,
                    };
                }
                match &hits[i] {
                    Some(hit) => {
                        if self.states[i].remaining_depth() == 0 {
                            BounceResult {
                                contribution: Color3::ZERO,
                                next_ray: None,
                                delta_child: None,
                            }
                        } else {
                            integrator.process_bounce(
                                self.rays[i].as_ref().unwrap(),
                                hit,
                                world,
                                lights,
                                &mut self.states[i],
                                &mut self.streams[i],
                                &mut self.rngs[i],
                            )
                        }
                    }
                    None => BounceResult {
                        contribution: integrator.eval_background(
                            self.rays[i].as_ref().unwrap().direction.normalize(),
                            &self.states[i],
                        ),
                        next_ray: None,
                        delta_child: None,
                    },
                }
            })
            .collect()
    }

    /// Stage 3: bounces -> radiances. Accumulates contributions, routes
    /// continuations, queues delta children with parent links, and flows finished
    /// paths' radiances up the chain (deferred film adds).
    pub fn resolve(&mut self, bounces: &[BounceResult<I::PathState>], tile: &mut FilmTile) {
        for (i, bounce) in bounces.iter().enumerate() {
            if self.rays[i].is_none() {
                continue; // waiting slot — only its children touch it
            }
            self.accumulator[i] += bounce.contribution;

            // Delta child: queue it with a parent link and a stream/rng snapshot.
            // It is shaded in the NEXT iteration — after the parent's later
            // bounces — which is exactly the queued accumulation order.
            if let Some((child_ray, child_state)) = &bounce.delta_child {
                self.rays.push(Some(*child_ray));
                self.states.push(child_state.clone());
                self.streams.push(self.streams[i]);
                self.rngs.push(self.rngs[i]);
                self.accumulator.push(Color3::ZERO);
                self.weights.push(None);
                self.pixels.push(None);
                self.parents.push(Some(i));
                self.pending.push(0);
                self.done.push(false);
                self.pending[i] += 1;
            }

            match &bounce.next_ray {
                Some(next) => {
                    self.states[i].advance();
                    if self.states[i].remaining_depth() == 0 {
                        self.done[i] = true;
                        self.rays[i] = None;
                    } else {
                        self.rays[i] = Some(*next);
                    }
                }
                None => {
                    self.done[i] = true;
                    self.rays[i] = None;
                }
            }

            if self.done[i] && self.pending[i] == 0 {
                self.flow_up(i, tile);
            }
        }
    }

    fn process_bounces(&mut self, bounces: &[BounceResult<I::PathState>], tile: &mut FilmTile) {
        self.resolve(bounces, tile);
        self.compact();
    }

    pub fn process(
        &mut self,
        integrator: &I,
        world: &impl Intersectable,
        lights: &[LightPrimitive],
        tile: &mut FilmTile,
    ) {
        loop {
            let hits = self.trace_rays(world);
            let bounces = self.shade_hits(integrator, &hits, world, lights);
            self.process_bounces(&bounces, tile);
            if self.rays.is_empty() {
                break;
            }
        }
    }
}
