use crate::camera::{Camera, get_camera_sample};
use crate::film::{Film, FilmTile, SharedFramebuffer};
use crate::integrator::{BounceResult, Integrator, PathState};
use crate::intersect::Intersectable;
use crate::intersect::interaction::MaterialHit;
use crate::math::interval::Interval;
use crate::math::vec3::Color3;
use crate::primitives::LightPrimitive;
use crate::ray::{Ray, RayPacked};
use crate::renderer::{Renderer, TiledRenderer};
use crate::sampler::{HashRng, SampleStream, SampleStreamWriter, SamplerRng, pixel_sample_state};

/// Stable identity for a path record in a batch's path arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PathId(usize);

/// Persistent metadata for one path tree node.
///
/// Path records survive while a parent waits for queued delta children. They
/// are deliberately separate from the dense active-ray work list so waiting
/// paths never appear as fake stage slots.
pub(crate) struct PathRecord<S, R, State> {
    pub state: State,
    pub stream: S,
    pub rng: R,
    pub accumulator: Color3,
    pub parent: Option<PathId>,
    pub pending_children: u32,
    pub pixel: Option<(u32, u32)>,
    pub camera_weight: Option<Color3>,
    finished: bool,
}

/// One dense active work item entering the next wavefront stage.
#[derive(Clone, Copy)]
pub(crate) struct ActiveRay {
    pub path: PathId,
    pub ray: Ray,
}

/// The complete visibility result for one active ray.
pub(crate) enum Visibility<'a> {
    Hit(MaterialHit<'a>),
    Miss,
}

/// Capability contract for a staged wavefront buffer.
///
/// Each stage consumes/produces dense work. Inactive paths are represented only
/// by their absence from `ActiveRay`; waiting parents remain in the path arena.
pub(crate) trait WavefrontStages<I: Integrator, S: SampleStream, R: SamplerRng, const B: usize> {
    fn trace_rays<'a, W: Intersectable>(&self, world: &'a W) -> Vec<Visibility<'a>>;

    fn shade_hits(
        &mut self,
        integrator: &I,
        visibility: &[Visibility<'_>],
        world: &impl Intersectable,
        lights: &[LightPrimitive],
    ) -> Vec<BounceResult<I::PathState>>;

    /// Resolve the current wave and emit only the dense continuation work for
    /// the next wave. Terminated paths flow their radiance through the arena.
    fn resolve(
        &mut self,
        bounces: &[BounceResult<I::PathState>],
        tile: &mut FilmTile,
    ) -> Vec<ActiveRay>;

    fn run(
        &mut self,
        integrator: &I,
        world: &impl Intersectable,
        lights: &[LightPrimitive],
        tile: &mut FilmTile,
    ) {
        loop {
            let visibility = self.trace_rays(world);
            let bounces = self.shade_hits(integrator, &visibility, world, lights);
            let next_active = self.resolve(&bounces, tile);
            self.set_active_rays(next_active);
            if self.active_is_empty() {
                break;
            }
        }
    }

    fn set_active_rays(&mut self, active: Vec<ActiveRay>);
    fn active_is_empty(&self) -> bool;
}

/// Runtime-sized wavefront storage whose stage inputs and outputs are dense.
pub(crate) struct WavefrontBatch<I: Integrator, S: SampleStream, R: SamplerRng, const B: usize> {
    /// Persistent path metadata, including parents waiting on delta children.
    paths: Vec<PathRecord<S, R, I::PathState>>,
    /// Dense active work list. Every entry has a valid ray and path identity.
    active_rays: Vec<ActiveRay>,
}

impl<I: Integrator, S: SampleStream, R: SamplerRng, const B: usize> WavefrontBatch<I, S, R, B> {
    pub(crate) fn new(
        active_rays: Vec<ActiveRay>,
        paths: Vec<PathRecord<S, R, I::PathState>>,
    ) -> Self {
        assert!(
            active_rays.iter().all(|active| active.path.0 < paths.len()),
            "active ray references an unknown path record"
        );
        Self { paths, active_rays }
    }

    fn flow_up(&mut self, mut path_id: PathId, tile: &mut FilmTile) {
        loop {
            let radiance = self.paths[path_id.0].accumulator;
            match self.paths[path_id.0].parent {
                Some(parent) => {
                    self.paths[parent.0].accumulator += radiance;
                    self.paths[parent.0].pending_children -= 1;
                    if self.paths[parent.0].finished && self.paths[parent.0].pending_children == 0 {
                        path_id = parent;
                    } else {
                        return;
                    }
                }
                None => {
                    let record = &self.paths[path_id.0];
                    let sample = record.accumulator * record.camera_weight.unwrap();
                    if sample.into_inner().is_finite() {
                        let (x, y) = record.pixel.unwrap();
                        tile.add_sample(x, y, sample);
                    }
                    return;
                }
            }
        }
    }

    fn resolve_bounces(
        &mut self,
        bounces: &[BounceResult<I::PathState>],
        tile: &mut FilmTile,
    ) -> Vec<ActiveRay> {
        let active_rays = std::mem::take(&mut self.active_rays);
        let mut next_active = Vec::with_capacity(active_rays.len());

        for (active, bounce) in active_rays.iter().zip(bounces) {
            let path_id = active.path;
            self.paths[path_id.0].accumulator += bounce.contribution;

            if let Some((child_ray, child_state)) = &bounce.delta_child {
                let parent = &self.paths[path_id.0];
                let child_id = PathId(self.paths.len());
                self.paths.push(PathRecord {
                    state: child_state.clone(),
                    stream: parent.stream,
                    rng: parent.rng,
                    accumulator: Color3::ZERO,
                    parent: Some(path_id),
                    pending_children: 0,
                    pixel: None,
                    camera_weight: None,
                    finished: false,
                });
                self.paths[path_id.0].pending_children += 1;
                next_active.push(ActiveRay {
                    path: child_id,
                    ray: *child_ray,
                });
            }

            match bounce.next_ray {
                Some(next_ray) => {
                    let record = &mut self.paths[path_id.0];
                    record.state.advance();
                    if record.state.remaining_depth() > 0 {
                        next_active.push(ActiveRay {
                            path: path_id,
                            ray: next_ray,
                        });
                    } else {
                        record.finished = true;
                    }
                }
                None => {
                    self.paths[path_id.0].finished = true;
                }
            }

            if self.paths[path_id.0].finished && self.paths[path_id.0].pending_children == 0 {
                self.flow_up(path_id, tile);
            }
        }

        next_active
    }
}

impl<I: Integrator, S: SampleStream, R: SamplerRng, const B: usize> WavefrontStages<I, S, R, B>
    for WavefrontBatch<I, S, R, B>
{
    fn trace_rays<'a, W: Intersectable>(&self, world: &'a W) -> Vec<Visibility<'a>> {
        let mut visibility = Vec::with_capacity(self.active_rays.len());
        for chunk in self.active_rays.chunks(B) {
            if chunk.len() == B {
                let rays: [RayPacked<1>; B] = core::array::from_fn(|i| chunk[i].ray);
                let packet: RayPacked<B> = rays.into();
                let hits = world.intersect(&packet, Interval::from(0.001, f32::INFINITY));
                visibility.extend(hits.into_iter().map(|hit| match hit {
                    Some(hit) => Visibility::Hit(hit),
                    None => Visibility::Miss,
                }));
            } else {
                for active in chunk {
                    let hit = world.intersect(&active.ray, Interval::from(0.001, f32::INFINITY))[0];
                    visibility.push(match hit {
                        Some(hit) => Visibility::Hit(hit),
                        None => Visibility::Miss,
                    });
                }
            }
        }
        visibility
    }

    fn shade_hits(
        &mut self,
        integrator: &I,
        visibility: &[Visibility<'_>],
        world: &impl Intersectable,
        lights: &[LightPrimitive],
    ) -> Vec<BounceResult<I::PathState>> {
        assert_eq!(visibility.len(), self.active_rays.len());
        self.active_rays
            .iter()
            .zip(visibility)
            .map(|(active, visibility)| {
                let path = &mut self.paths[active.path.0];
                match visibility {
                    Visibility::Hit(hit) if path.state.remaining_depth() > 0 => integrator
                        .process_bounce(
                            &active.ray,
                            hit,
                            world,
                            lights,
                            &mut path.state,
                            &mut path.stream,
                            &mut path.rng,
                        ),
                    Visibility::Hit(_) => BounceResult {
                        contribution: Color3::ZERO,
                        next_ray: None,
                        delta_child: None,
                    },
                    Visibility::Miss => BounceResult {
                        contribution: integrator
                            .eval_background(active.ray.direction().normalize(), &path.state),
                        next_ray: None,
                        delta_child: None,
                    },
                }
            })
            .collect()
    }

    fn resolve(
        &mut self,
        bounces: &[BounceResult<I::PathState>],
        tile: &mut FilmTile,
    ) -> Vec<ActiveRay> {
        assert_eq!(bounces.len(), self.active_rays.len());
        self.resolve_bounces(bounces, tile)
    }

    fn set_active_rays(&mut self, active: Vec<ActiveRay>) {
        self.active_rays = active;
    }

    fn active_is_empty(&self) -> bool {
        self.active_rays.is_empty()
    }
}

/// A const-generic renderer whose B=1 specialization is the scalar CPU path.
pub struct WavefrontRenderer<I, const B: usize>
where
    I: Integrator,
{
    pub(crate) samples_per_pixel: u32,
    pub(crate) threshold_abs: f32,
    pub(crate) threshold_rel: f32,
    pub(crate) min_samples_before_adapt: u32,
    pub(crate) tile_size: u32,
    pub(crate) integrator: I,
}

impl<I, const B: usize> WavefrontRenderer<I, B>
where
    I: Integrator,
{
    pub fn new(samples_per_pixel: u32, integrator: I) -> Self {
        assert!(B > 0, "wavefront batch size must be greater than zero");
        Self {
            samples_per_pixel,
            threshold_abs: 1e-4,
            threshold_rel: 0.02,
            min_samples_before_adapt: 64,
            tile_size: 32,
            integrator,
        }
    }

    pub fn set_threshold_abs(&mut self, threshold: f32) {
        self.threshold_abs = threshold;
    }

    pub fn set_threshold_rel(&mut self, threshold: f32) {
        self.threshold_rel = threshold;
    }

    pub fn set_min_samples_before_adapt(&mut self, min_samples: u32) {
        self.min_samples_before_adapt = min_samples;
    }

    pub fn set_tile_size(&mut self, tile_size: u32) {
        assert!(tile_size > 0, "tile size must be greater than zero");
        self.tile_size = tile_size;
    }

    #[allow(clippy::too_many_arguments)]
    fn render_wavefront_tile<C, W>(
        &self,
        camera: &C,
        world: &W,
        lights: &[LightPrimitive],
        converged: &[bool],
        pixel_bases: &[u32],
        tile: &mut FilmTile,
        sample_idx: u32,
    ) where
        C: Camera,
        W: Intersectable,
    {
        let [x_start, x_end, y_start, y_end] = tile.bounds;
        let (width, _) = camera.image_resolution();
        if !self.prepare_tile(tile, converged, width) {
            return;
        }

        let mut active_rays = Vec::with_capacity(B);
        let mut paths = Vec::with_capacity(B);

        for (y, x) in (y_start..y_end).flat_map(|y| (x_start..x_end).map(move |x| (y, x))) {
            let pixel_idx = y as usize * width as usize + x as usize;
            if converged[pixel_idx] {
                continue;
            }

            let (mut stream, mut rng): (SampleStreamWriter, HashRng) =
                pixel_sample_state(pixel_bases, pixel_idx, sample_idx, x as i32, y as i32);
            let camera_sampler = get_camera_sample((x, y), &mut stream, &mut rng);
            let cam_ray = camera
                .generate_ray_differential(&camera_sampler)
                .or_else(|| camera.generate_ray(&camera_sampler));

            if let Some(cam_ray) = cam_ray {
                let path_id = PathId(paths.len());
                paths.push(PathRecord {
                    state: self.integrator.init_state(),
                    stream,
                    rng,
                    accumulator: Color3::ZERO,
                    parent: None,
                    pending_children: 0,
                    pixel: Some((x, y)),
                    camera_weight: Some(cam_ray.weight),
                    finished: false,
                });
                active_rays.push(ActiveRay {
                    path: path_id,
                    ray: cam_ray.ray,
                });
            }

            if active_rays.len() == B {
                self.process_batch(world, lights, tile, &mut active_rays, &mut paths);
            }
        }

        if !active_rays.is_empty() {
            self.process_batch(world, lights, tile, &mut active_rays, &mut paths);
        }
    }

    fn process_batch<W>(
        &self,
        world: &W,
        lights: &[LightPrimitive],
        tile: &mut FilmTile,
        active_rays: &mut Vec<ActiveRay>,
        paths: &mut Vec<PathRecord<SampleStreamWriter, HashRng, I::PathState>>,
    ) where
        W: Intersectable,
    {
        let batch = WavefrontBatch::<I, SampleStreamWriter, HashRng, B>::new(
            std::mem::replace(active_rays, Vec::with_capacity(B)),
            std::mem::replace(paths, Vec::with_capacity(B)),
        );
        let mut batch = batch;
        WavefrontStages::run(&mut batch, &self.integrator, world, lights, tile);
    }
}

impl<I, const B: usize> TiledRenderer for WavefrontRenderer<I, B>
where
    I: Integrator,
{
    type Integrator = I;

    fn integrator(&self) -> &Self::Integrator {
        &self.integrator
    }

    fn samples_per_pixel(&self) -> u32 {
        self.samples_per_pixel
    }

    fn threshold_abs(&self) -> f32 {
        self.threshold_abs
    }

    fn threshold_rel(&self) -> f32 {
        self.threshold_rel
    }

    fn min_samples_before_adapt(&self) -> u32 {
        self.min_samples_before_adapt
    }

    fn tile_size(&self) -> u32 {
        self.tile_size
    }

    fn render_tile<C, W>(
        &self,
        camera: &C,
        world: &W,
        lights: &[LightPrimitive],
        converged: &[bool],
        pixel_bases: &[u32],
        tile: &mut FilmTile,
        sample_idx: u32,
    ) where
        C: Camera,
        W: Intersectable,
    {
        if B == 1 {
            self.render_scalar_tile(
                camera,
                world,
                lights,
                converged,
                pixel_bases,
                tile,
                sample_idx,
            );
        } else {
            self.render_wavefront_tile(
                camera,
                world,
                lights,
                converged,
                pixel_bases,
                tile,
                sample_idx,
            );
        }
    }
}

impl<I, C, F, const B: usize> Renderer<C, F> for WavefrontRenderer<I, B>
where
    I: Integrator,
    C: Camera,
    F: Film,
{
    fn render(
        &self,
        camera: &C,
        film: &mut F,
        scene: (&impl Intersectable, &[LightPrimitive]),
        framebuffer: Option<SharedFramebuffer>,
    ) {
        let (world, lights) = scene;
        self.render_tiled(camera, film, world, lights, framebuffer);
    }
}
