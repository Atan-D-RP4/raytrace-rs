use std::{collections::HashMap, num::NonZeroU32, sync::Arc};

use image::{ImageBuffer, Rgb, RgbImage};
use softbuffer::{Context, Surface};
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

use winit::{
    application::ApplicationHandler,
    dpi::{LogicalSize, PhysicalSize},
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::{Theme, Window, WindowId},
};

use raytrace_rs::bvh::BvhNode;
use raytrace_rs::camera::{Camera, Framebuffer, SharedFramebuffer};
use raytrace_rs::flat_bvh::FlatBvh;
use raytrace_rs::hittable::Sampleable;
use raytrace_rs::sampler::SobolQmcSampler;
use raytrace_rs::scene::Scene;

const WIDTH: u32 = 800;

struct WindowState {
    /// IMPORTANT:
    /// surface must be dropped before window
    surface: Surface<Arc<Window>, Arc<Window>>,
    /// Winit OS window handle used for sizing and redraw requests.
    window: Arc<Window>,
    /// Last reported platform theme (currently used only to trigger redraw).
    theme: Theme,

    /// Shared render output consumed by draw loop.
    framebuffer: SharedFramebuffer,
}

impl WindowState {
    /// Creates `WindowState`, initializes softbuffer context/surface,
    /// and sizes surface to current inner window dimensions.
    fn new(window: Window, framebuffer: SharedFramebuffer) -> Self {
        let window = Arc::new(window);

        let context = Context::new(window.clone()).expect("failed to create softbuffer context");

        let mut surface =
            Surface::new(&context, window.clone()).expect("failed to create softbuffer surface");

        let size = window.inner_size();
        resize_surface(&mut surface, size);

        let theme = window.theme().unwrap_or(Theme::Dark);

        Self {
            surface,
            window,
            theme,
            framebuffer,
        }
    }

    /// Handles explicit resize events from winit.
    ///
    /// Resizes backing surface and schedules redraw for fresh present.
    fn resize(&mut self, size: PhysicalSize<u32>) {
        resize_surface(&mut self.surface, size);
        self.window.request_redraw();
    }

    /// Draws current framebuffer snapshot into softbuffer surface.
    ///
    /// Pipeline:
    /// 1. Query current window size.
    /// 2. Ensure softbuffer surface matches that size.
    /// 3. Read shared framebuffer snapshot.
    /// 4. Scale framebuffer to window dimensions (nearest-neighbor).
    /// 5. Present surface.
    /// 6. If render still in progress, request next redraw.
    ///
    /// Current mapping is stretch-to-fill.
    /// Aspect-fit letterboxing can be added by changing src coordinate mapping.
    fn draw(&mut self) {
        let _span = tracing::trace_span!("window_draw").entered();
        let size = self.window.inner_size();
        if size.width == 0 || size.height == 0 {
            tracing::trace!("skip draw: zero-sized window");
            return;
        }

        // TODO(opt-preview): cache last surface size and resize only on dimension change.
        // Current path resizes every redraw to defend against compositor/scale drift.
        profiling::scope!("ui_surface_resize");
        resize_surface(&mut self.surface, size);

        let fb = self.framebuffer.read().unwrap();
        if fb.width == 0 || fb.height == 0 {
            tracing::trace!("skip draw: framebuffer not initialized");
            return;
        }

        tracing::trace!(
            window_width = size.width,
            window_height = size.height,
            fb_width = fb.width,
            fb_height = fb.height,
            finished = fb.finished,
            "blitting framebuffer to surface"
        );

        // TODO(opt-preview): replace full-frame blit with tile/dirty-rect blits.
        // This lowers memory bandwidth and improves interactivity on large windows.
        profiling::scope!("ui_blit");
        let mut buffer = self.surface.buffer_mut().unwrap();
        buffer.fill(0);

        // TODO(viewport): implement aspect-fit viewport (letterbox/pillarbox) instead of stretch.
        for y in 0..size.height {
            let src_y = ((y as u64 * fb.height as u64) / size.height as u64) as u32;
            for x in 0..size.width {
                let src_x = ((x as u64 * fb.width as u64) / size.width as u64) as u32;
                let dst = y as usize * size.width as usize + x as usize;
                let src = (src_y as usize * fb.width as usize + src_x as usize) * 3;

                let r = fb.rgb[src] as u32;
                let g = fb.rgb[src + 1] as u32;
                let b = fb.rgb[src + 2] as u32;

                buffer[dst] = (r << 16) | (g << 8) | b;
            }
        }

        self.window.pre_present_notify();

        profiling::scope!("ui_present");
        buffer.present().unwrap();

        if !fb.finished {
            self.window.request_redraw();
        }
    }
}

/// Resizes softbuffer surface to match current physical window size.
///
/// No-op for zero dimensions (minimized/hidden window).
fn resize_surface<W1, W2>(surface: &mut Surface<W1, W2>, size: PhysicalSize<u32>)
where
    W1: HasDisplayHandle,
    W2: HasWindowHandle,
{
    let Some(width) = NonZeroU32::new(size.width) else {
        tracing::trace!("skip surface resize: width=0");
        return;
    };

    let Some(height) = NonZeroU32::new(size.height) else {
        tracing::trace!("skip surface resize: height=0");
        return;
    };

    tracing::trace!(width, height, "resizing softbuffer surface");
    surface
        .resize(width, height)
        .expect("failed to resize surface");
}

struct App {
    /// All active windows keyed by winit id.
    windows: HashMap<WindowId, WindowState>,
    /// Initial window width used when creating first window.
    width: u32,
    /// Initial window height used when creating first window.
    height: u32,

    /// Shared live-preview framebuffer used by all windows.
    framebuffer: SharedFramebuffer,
}

impl App {
    /// Constructs application state for event loop lifecycle.
    fn new(framebuffer: SharedFramebuffer, width: u32, height: u32) -> Self {
        Self {
            windows: HashMap::new(),
            framebuffer,
            width,
            height,
        }
    }
}

impl ApplicationHandler for App {
    /// Called when app becomes active/resumed.
    ///
    /// Creates initial preview window once, initializes state,
    /// and triggers first redraw request.
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // resumed() can happen multiple times
        if !self.windows.is_empty() {
            return;
        }

        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("CPU Raytracer Preview")
                    .with_inner_size(LogicalSize::new(self.width as f64, self.height as f64)),
            )
            .unwrap();

        let id = window.id();

        let state = WindowState::new(window, self.framebuffer.clone());
        state.window.request_redraw();

        self.windows.insert(id, state);
    }

    /// Handles per-window events and routes redraw/resize/close behavior.
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = self.windows.get_mut(&window_id) else {
            return;
        };

        match event {
            WindowEvent::Resized(size) => {
                window.resize(size);
            }

            WindowEvent::ThemeChanged(theme) => {
                window.theme = theme;
                window.window.request_redraw();
            }

            WindowEvent::RedrawRequested => {
                window.draw();
            }

            WindowEvent::CloseRequested => {
                self.windows.remove(&window_id);

                if self.windows.is_empty() {
                    event_loop.exit();
                }
            }

            _ => {}
        }
    }
}

/// Saves final framebuffer snapshot to PNG.
///
/// This function is intended for post-render output. It exits early
/// unless `framebuffer.finished` is true.
fn save_framebuffer(framebuffer: &SharedFramebuffer, filename: &str) {
    let _span = tracing::info_span!("save_framebuffer", %filename).entered();
    let filename = format!("{}.png", filename);

    let Ok(fb) = framebuffer.read() else {
        error!("failed to lock framebuffer for saving");
        return;
    };

    if !fb.finished {
        return;
    }

    let mut img: RgbImage = ImageBuffer::new(fb.width, fb.height);
    for (i, pixel) in fb.rgb.chunks(3).enumerate() {
        let x = (i as u32) % fb.width;
        let y = (i as u32) / fb.width;
        img.put_pixel(x, y, Rgb([pixel[0], pixel[1], pixel[2]]));
    }

    info!(%filename, "saving image");
    profiling::scope!("image_save");
    if let Err(error) = img.save(&filename) {
        error!(%filename, ?error, "failed to save image");
    } else {
        info!(%filename, "image saved");
    }
}

/// Configures the global rayon thread pool from the `RT_THREADS` environment variable.
///
/// Must be called **before** any rayon usage (the global pool is created once, on first use).
/// Without `RT_THREADS`, rayon defaults to the number of available CPU cores.
fn configure_rayon() {
    let mut builder = rayon::ThreadPoolBuilder::new().num_threads(4);

    if let Ok(n) = std::env::var("RT_THREADS")
        && let Ok(n) = n.parse::<usize>()
    {
        builder = builder.num_threads(n);
    }

    // Silently ignores error if pool was already initialized (shouldn't happen here).
    let _ = builder.build_global();
}

/// Initializes global tracing subscriber for app/process lifetime.
///
/// Uses `RUST_LOG` when available, falls back to `info` level.
fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(false)
        .with_thread_ids(true)
        .with_line_number(true)
        .init();

    info!(threads = rayon::current_num_threads(), "startup");
}

/// Entry point for live-preview application mode.
///
/// Flow:
/// 1. Initialize logging.
/// 2. Build scene config to derive output dimensions.
/// 3. Allocate shared framebuffer.
/// 4. Spawn render worker thread.
/// 5. Run winit event loop + softbuffer presentation on main thread.
fn main() -> Result<(), winit::error::EventLoopError> {
    configure_rayon();
    init_tracing();

    // Parse scene name from CLI args (optional first positional arg).
    // Default is "cornell_box" to preserve original behavior.
    let scene_name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "cornell_box".to_string());
    let scene = match scene_name.as_str() {
        "cornell_box" => Scene::cornell_box(),
        "cornell_box_const_meds" => Scene::cornell_box_const_meds(),
        "complex_scene" => Scene::complex_scene(),
        "simple_light" => Scene::simple_light(),
        "simple_world" => Scene::simple_world(),
        "earth_sphere" => Scene::earth_sphere(),
        "noisy_spheres" => Scene::noisy_spheres(),
        "checkered_spheres" => Scene::checkered_spheres(),
        "quads" => Scene::quads(),
        "random_world" => Scene::random_world(),
        "composition_demo" => Scene::composition_demo(),
        other => {
            error!(scene = %other, "unknown scene");
            eprintln!("Available scenes:");
            eprintln!("  cornell_box, cornell_box_const_meds, complex_scene,");
            eprintln!("  simple_light, simple_world, earth_sphere, noisy_spheres,");
            eprintln!("  checkered_spheres, quads, random_world, composition_demo");
            std::process::exit(1);
        }
    };
    // Re-emit into a headless or live render using the selected scene.
    // Optional env-var overrides: RT_SAMPLES, RT_WIDTH, RT_DEPTH (for fast debugging).
    // Set RT_LIVE=1 for live preview window, otherwise defaults to headless.
    if std::env::var("RT_LIVE").ok().as_deref() == Some("1") {
        live_render(scene, &scene_name)?;
    } else {
        headless_render(scene, &scene_name);
    }

    Ok(())
}

#[profiling::function]
fn live_render(
    scene: Scene<SobolQmcSampler>,
    scene_name: &str,
) -> Result<(), winit::error::EventLoopError> {
    // TODO(gpu): keep this scene-construction boundary mirrored in future GPU pipeline.
    profiling::scope!("scene_build");

    let mut config = *scene.config();

    // Allow env-var overrides for fast iteration during debugging.
    config.image_width = std::env::var("RT_WIDTH")
        .ok()
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(WIDTH as i32);
    config.samples_per_pixel = std::env::var("RT_SAMPLES")
        .ok()
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(1000);
    config.max_depth = std::env::var("RT_DEPTH")
        .ok()
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(50);

    let camera = Camera::from_config(&config);
    let (width, height) = camera.image_dimensions();
    let framebuffer = Arc::new(std::sync::RwLock::new(Framebuffer::new(width, height)));

    let event_loop = EventLoop::new()?;
    let mut app = App::new(framebuffer.clone(), width, height);
    let scene_name = scene_name.to_string();

    // Spawns dedicated CPU render worker thread.
    //
    // Worker thread responsibilities:
    // - Build scene and root BVH.
    // - Run progressive renderer writing into shared framebuffer.
    // - Save completed frame to disk.
    //
    // UI thread remains free to process events and draw continuously.
    std::thread::spawn(move || {
        let _span = tracing::info_span!("render_thread").entered();
        info!("starting render thread");

        profiling::scope!("scene_build");
        let (mut objects, light_objects) = scene.into_objects();
        let world_len = objects.len();
        let light_len = light_objects.len();
        info!(
            object_count = world_len,
            light_count = light_len,
            "building world BVH"
        );
        profiling::scope!("root_bvh_build");
        let world_bvh = BvhNode::new(&mut objects);
        let world = FlatBvh::from_bvh(world_bvh);
        let lights: Arc<dyn Sampleable<SobolQmcSampler>> = Arc::new(light_objects);

        let mut camera = Camera::from_config(&config);
        // TODO(opt-preview): propagate cancellation signal so worker can stop on app exit.
        // TODO(opt-preview): move to tile scheduler with periodic publish for faster perceived convergence.
        profiling::scope!("render");
        camera.render(
            &world,
            &*lights,
            SobolQmcSampler::for_pixel,
            Some(framebuffer.clone()),
        );

        save_framebuffer(&framebuffer, &scene_name);
        info!("render thread complete");
    });

    event_loop.run_app(&mut app)?;
    Ok(())
}

#[profiling::function]
fn headless_render(scene: Scene<SobolQmcSampler>, scene_name: &str) {
    // TODO(gpu): keep this scene-construction boundary mirrored in future GPU pipeline.
    profiling::scope!("scene_build");

    let filename = format!("{}.png", scene_name);

    let mut config = *scene.config();

    // Allow env-var overrides for fast iteration during debugging.
    config.image_width = std::env::var("RT_WIDTH")
        .ok()
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(WIDTH as i32);
    config.samples_per_pixel = std::env::var("RT_SAMPLES")
        .ok()
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(1000);
    config.max_depth = std::env::var("RT_DEPTH")
        .ok()
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(50);

    let mut camera = Camera::from_config(&config);

    let (mut objects, light_objects) = scene.into_objects();

    let world_len = objects.len();
    let light_len = light_objects.len();
    info!(
        object_count = world_len,
        light_count = light_len,
        "building world BVH"
    );
    // TODO(gpu): split accel build from upload/flatten so CPU and GPU can profile same phases.
    profiling::scope!("root_bvh_build");
    let world_bvh = BvhNode::new(&mut objects);
    let world = FlatBvh::from_bvh(world_bvh);

    info!(
        threads = rayon::current_num_threads(),
        "rayon threads in use"
    );

    let start = std::time::Instant::now();
    profiling::scope!("render_cpu");
    let (width, height, rgb_data) =
        camera.render(&world, &light_objects, SobolQmcSampler::for_pixel, None);
    let end = std::time::Instant::now();
    info!(elapsed = ?(end - start), width, height, "render complete");

    let mut img: RgbImage = ImageBuffer::new(width, height);
    for (i, pixel) in rgb_data.chunks(3).enumerate() {
        let x = (i as u32) % width;
        let y = (i as u32) / width;
        img.put_pixel(x, y, Rgb([pixel[0], pixel[1], pixel[2]]));
    }

    info!(%filename, "saving image");
    profiling::scope!("image_save");
    if let Err(error) = img.save(&filename) {
        error!(%filename, ?error, "failed to save image");
        return;
    }
    info!(%filename, "image saved");

    display_image(&filename);
}

fn display_image(filename: &str) {
    match std::process::Command::new("satty")
        .args(["--filename", filename])
        .spawn()
    {
        Ok(mut child) => child.wait().map_or_else(
            |e| panic!("Failed to close satty with error: {e:?}"),
            |_| (),
        ),
        Err(e) => panic!("Failed to spawn `satty` with error: {e:?}"),
    }
}
