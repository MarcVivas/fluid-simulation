use winit::{dpi, event::{ElementState, MouseButton, MouseScrollDelta}, keyboard::KeyCode};

use crate::{rendering::camera::{Camera, CameraUniform}, vulkan::frame::frame_pacer::FramePacer};

pub struct CameraController {
    camera: Camera,
    mouse_position: dpi::PhysicalPosition<f64>,
}

impl CameraController {
    pub fn new(camera: Camera) -> Self {
        Self { camera, mouse_position: dpi::PhysicalPosition::default() }
    }

    pub fn set_mouse_position(&mut self, position: Option<dpi::PhysicalPosition<f64>>) {
        if let Some(pos) = position {
            self.mouse_position = pos;
            self.camera.set_camera_zoom_position(Some(pos));
        }
    }
    pub fn handle_key_input(&mut self, key: KeyCode, is_pressed: bool) {
        self.camera.move_camera(key, is_pressed);
    }

    pub fn handle_zoom(&mut self, delta: MouseScrollDelta) {
        self.camera.zoom_camera(delta);
    }

    pub fn handle_rotation(&mut self, button: &MouseButton, state: &ElementState) {
        self.camera.handle_mouse_input(button, state);
    }

    pub fn update(&mut self, screen_size: glam::Vec2, frame_pacer: &FramePacer) -> anyhow::Result<()> {
        // Delta time could be passed in if variable, or fixed like before
        self.camera.update(0.0016, &screen_size);
        self.camera.build_view_projection_matrix(&screen_size);
        
        let uniform_data = self.camera.get_uniform();
        self.camera.buffer(frame_pacer.ring_index()).update(uniform_data)?;
        Ok(())
    }

    pub fn uniform_data(&self) -> &CameraUniform {
        self.camera.get_uniform()
    }
}
