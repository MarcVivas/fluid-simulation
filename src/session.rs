use anyhow::{Ok, Result};
use glam::Vec2;
use winit::{
    dpi::PhysicalPosition,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

use crate::app_session::AppSession;
use crate::backends::gpu_backend::GpuBackend;
use crate::viewer::camera::Camera;
use crate::viewer::camera_controller::CameraController;
use crate::world::World;

pub struct Session {
    backend: Box<dyn GpuBackend>,
    camera: Camera,
    camera_controller: CameraController,
    paused: bool,
    #[allow(unused)]
    world: World,
}

impl Session {
    pub fn new(backend: Box<dyn GpuBackend>, window: &Window, world: World) -> Result<Self> {
        let camera = Camera::new(&world.dimensions(), &window.inner_size());
        Ok(Self {
            backend,
            camera,
            camera_controller: CameraController::new(1.0, 0.005, 0.002),
            paused: true,
            world,
        })
    }

    /// Updates and renders the simulation
    pub fn update_and_render(&mut self, window: &Window) -> Result<()> {
        let Some(viewport_size) = self.backend.begin_frame(window)? else {
            return Ok(());
        };

        self.camera_controller.update(&mut self.camera);
        self.camera.build_view_projection_matrix(&viewport_size);

        self.backend
            .execute_frame(self.camera.get_uniform(), self.paused)
    }

    pub fn resize_window(&mut self, window: &Window) -> Result<()> {
        self.backend.resize(window)
    }

    pub fn close(&self) -> Result<()> {
        self.backend.shutdown()
    }

    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }

    pub fn handle_key_input(&mut self, code: KeyCode, is_pressed: bool) {
        self.camera_controller.handle_keyboard(code, is_pressed);
    }

    pub fn set_mouse_position(&mut self, position: Option<PhysicalPosition<f64>>) {
        if let Some(pos) = position {
            self.camera_controller
                .update_mouse_position(Vec2::new(pos.x as f32, pos.y as f32));
        }
    }

    pub fn handle_mouse_button(&mut self, button: MouseButton, state: ElementState) {
        self.camera_controller.handle_mouse_click(&button, &state);
    }

    pub fn handle_zoom(&mut self, delta: MouseScrollDelta) {
        self.camera_controller.handle_scroll(delta);
    }
}

impl AppSession for Session {
    fn frame(&mut self, window: &Window) -> Result<()> {
        self.update_and_render(window)
    }

    fn resize(&mut self, window: &Window) -> Result<()> {
        self.resize_window(window)
    }

    fn shutdown(&mut self) -> Result<()> {
        self.close()
    }

    fn handle_input(&mut self, event: &winit::event::WindowEvent) {
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    let is_pressed = event.state.is_pressed();

                    if code == KeyCode::Space && is_pressed {
                        self.toggle_pause();
                        return;
                    }
                    self.handle_key_input(code, is_pressed);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.set_mouse_position(Some(*position));
            }
            WindowEvent::MouseInput { state, button, .. } => {
                self.handle_mouse_button(*button, *state);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.handle_zoom(*delta);
            }
            _ => {}
        }
    }
}
