use crate::app_session::AppSession;
use crate::backends::BackendKind;
use crate::session_factory::create_session;
use crate::world::World;
use crate::world::particles::ParticleInitPreset;
use anyhow::Context;
use winit::application::ApplicationHandler;
use winit::dpi;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowAttributes, WindowId};

pub struct App {
    session: Option<Box<dyn AppSession>>,
    backend: BackendKind,
    window: Option<Window>,
}

impl App {
    pub fn new() -> Self {
        Self {
            window: None,
            session: None,
            backend: BackendKind::Vulkan,
        }
    }
}

impl ApplicationHandler for App {
    /// This creates the window and the engine before the event loop starts.
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = {
            let window_attributes = WindowAttributes::default()
                .with_title("Vulkan")
                .with_inner_size(dpi::LogicalSize::new(1280.0, 720.0));
            event_loop
                .create_window(window_attributes)
                .expect("Failed to create window")
        };

        let world = World::new(
            glam::Vec3::splat(256.0),
            1_000_000,
            ParticleInitPreset::CollidingBlocks,
            1.7,
        );

        let session = create_session(&window, self.backend, world).expect("Failed to initialize session");

        self.window = Some(window);
        self.session = Some(session);
    }

    /// Infinite loop
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let (Some(session), Some(window)) = (&mut self.session, &self.window) else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => {
                if let Err(err) = session.shutdown() {
                    eprintln!("Error during engine shutdown: {:?}", err);
                }
                event_loop.exit();
                return;
            }
            WindowEvent::Resized(_logical_size) => {
                if let Err(err) = session.resize(window).context("Couldn't resize the window") {
                    eprintln!("Error while resizing the window: {:?}", err);
                }
            }

            WindowEvent::RedrawRequested => {
                if let Err(err) = session.frame(window) {
                    eprintln!("Error during update_and_render: {:?}", err);
                }
                window.request_redraw();
            }
            _ => (),
        }

        if let WindowEvent::KeyboardInput { ref event, .. } = event {
            if event.physical_key == PhysicalKey::Code(KeyCode::Escape) && event.state.is_pressed()
            {
                event_loop.exit();
                return;
            }
        }

        session.handle_input(&event);
    }
}
