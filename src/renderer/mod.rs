pub mod cpu;

pub use cpu::CpuRenderer;

use crate::camera::Camera;
use crate::film::{Film, SharedFramebuffer};
use crate::hittable::{Intersectable, Sampleable};
use crate::sampler::Sampler;

pub trait Renderer<S, W, L, C, F>: Send + Sync
where
    S: Sampler,
    W: Intersectable,
    L: Sampleable<S>,
    C: Camera,
    F: Film,
{
    /// Renders the scene and returns (width, height, RGB pixel data).
    ///
    /// When `framebuffer` is `Some`, publishes progressive intermediate frames
    /// to the shared framebuffer during rendering (live preview mode).
    /// When `None`, renders all samples and returns the final image only.
    fn render(
        &self,
        camera: &C,
        film: &mut F,
        scene: (&W, &L),
        framebuffer: Option<SharedFramebuffer>,
        make_sampler: impl Fn(i32, i32) -> S + Sync,
    );
}
