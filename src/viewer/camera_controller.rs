use winit::{
    event::{ElementState, MouseButton, MouseScrollDelta},
    keyboard::KeyCode,
};

use glam::{Quat, Vec2, Vec3};

use crate::viewer::camera::Camera;

/// Remembers input and uses it to change the camera
#[derive(Debug)]
pub struct CameraController {
    // Sensitivities
    rotate_speed: f32,
    scroll_speed: f32,
    pan_speed: f32,

    // State
    is_left_pressed: bool,
    is_right_pressed: bool,
    is_middle_pressed: bool,

    mouse_position: Vec2,
    last_mouse_position: Vec2,
    scroll_accum: f32,
}

impl CameraController {
    pub fn new(scroll_speed: f32, rotate_speed: f32, pan_speed: f32) -> Self {
        Self {
            rotate_speed,
            scroll_speed,
            pan_speed,

            is_left_pressed: false,
            is_right_pressed: false,
            is_middle_pressed: false,

            mouse_position: Vec2::ZERO,
            last_mouse_position: Vec2::ZERO,
            scroll_accum: 0.0,
        }
    }

    pub fn update_mouse_position(&mut self, pos: Vec2) {
        self.mouse_position = pos;
        if !self.is_left_pressed && !self.is_right_pressed && !self.is_middle_pressed {
            self.last_mouse_position = pos;
        }
    }

    pub fn handle_mouse_click(&mut self, button: &MouseButton, state: &ElementState) {
        let is_pressed = *state == ElementState::Pressed;
        match button {
            MouseButton::Left => self.is_left_pressed = is_pressed,
            MouseButton::Right => self.is_right_pressed = is_pressed,
            MouseButton::Middle => self.is_middle_pressed = is_pressed,
            _ => {}
        }
        if is_pressed {
            self.last_mouse_position = self.mouse_position;
        }
    }

    pub fn handle_scroll(&mut self, mouse_scroll_delta: MouseScrollDelta) {
        self.scroll_accum += match mouse_scroll_delta {
            MouseScrollDelta::LineDelta(_, y) => y,
            MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
        };
    }

    pub fn handle_keyboard(&mut self, _key: KeyCode, _is_pressed: bool) {}

    // Returns: (delta_yaw, delta_pitch, zoom_factor, delta_pan_screen)
    pub fn process_frame_deltas(&mut self) -> (f32, f32, f32, Vec2) {
        let mut d_yaw = 0.0;
        let mut d_pitch = 0.0;
        let mut d_pan = Vec2::ZERO;

        let delta = self.mouse_position - self.last_mouse_position;

        // 1. Orbit (Left Click)
        if self.is_left_pressed {
            d_yaw = -delta.x * self.rotate_speed;
            d_pitch = delta.y * self.rotate_speed;
        }

        // 2. Pan (Right Click or Middle Click)
        if self.is_right_pressed || self.is_middle_pressed {
            d_pan = Vec2::new(delta.x, delta.y) * self.pan_speed;
        }

        // 3. Zoom
        // Return the raw accumulation scaled by speed, handled exponentially in update
        let d_zoom = self.scroll_accum * self.scroll_speed;

        self.last_mouse_position = self.mouse_position;
        self.scroll_accum = 0.0;

        (d_yaw, d_pitch, d_zoom, d_pan)
    }

    pub fn update(&mut self, camera: &mut Camera) {
        let (delta_yaw, delta_pitch, zoom_amount, pan_delta_screen) = self.process_frame_deltas();

        // 1. Orbit
        camera.yaw += delta_yaw;
        camera.pitch += delta_pitch;

        // Clamp pitch to avoid flipping
        const PITCH_LIMIT: f32 = 89.0f32 * (std::f32::consts::PI / 180.0);
        camera.pitch = camera.pitch.clamp(0.1, PITCH_LIMIT);

        // 2. Zoom (Exponential / Multiplicative)
        if zoom_amount != 0.0 {
            // 0.90 base means ~10% zoom per scroll "tick"
            // If zoom_amount is positive (scrolling up), we get closer (multiply by < 1.0)
            let zoom_factor = 0.95f32.powf(zoom_amount);
            camera.distance = (camera.distance * zoom_factor).clamp(0.5, 10000.0);
        }

        // 3. Pan
        let rotation = Quat::from_euler(glam::EulerRot::YXZ, camera.yaw, -camera.pitch, 0.0);
        let right = rotation * Vec3::X;
        let up = rotation * Vec3::Y;

        if pan_delta_screen != Vec2::ZERO {
            // Pan speed scales with distance so it feels natural at any zoom level
            let pan_dist_factor = camera.distance;
            let pan_movement =
                (right * -pan_delta_screen.x + up * pan_delta_screen.y) * pan_dist_factor;
            camera.target += pan_movement;
        }

        // 4. Update Position
        let offset = rotation * Vec3::new(0.0, 0.0, camera.distance);
        camera.position = camera.target + offset;
    }
}
