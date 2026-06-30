use std::sync::Arc;
use ash::prelude::VkResult;
use ash::vk;
use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{AllocationCreateDesc, AllocationScheme};
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::{ElementState, MouseButton, MouseScrollDelta};
use winit::keyboard::KeyCode;

use crate::rendering::renderer::MAX_FRAME_LATENCY;
use crate::vulkan::core::VkCore;
use crate::vulkan::resources::{CommandPool, buffer::VkBuffer};

pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub fov: f32,
    pub pitch: f32,
    pub yaw: f32,
    pub distance: f32,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_buffers: Vec<VkBuffer<CameraUniform>>,
    aspect_ratio: f32,
}

impl Camera {
    pub fn new(
        vk_core: &Arc<VkCore>,
        world_size: &Vec3,
        window_size: &PhysicalSize<u32>,
        command_pool: &CommandPool
    ) -> VkResult<Self>
    {
        let target = Vec3::new(world_size.x / 2.0, world_size.y / 2.0, world_size.z / 2.0);

        // Isometric-ish setup
        let pitch = 60.0f32.to_radians();
        let yaw = 60.0f32.to_radians();

        // Distance calc
        let fov = 45.0f32.to_radians();
        let bounding_radius = (world_size.length() / 2.0) *0.25;
        let distance = bounding_radius / (fov / 2.0).tan();

        // Initial position
        let rotation = Quat::from_euler(glam::EulerRot::YXZ, yaw, -pitch, 0.0);
        let offset = rotation * Vec3::new(0.0, 0.0, distance);
        let position = target + offset;

        let aspect_ratio = window_size.width as f32 / window_size.height as f32;
        let camera_uniform = CameraUniform::new();
        let mut camera_buffers = Vec::with_capacity(MAX_FRAME_LATENCY);

        for _ in 0..MAX_FRAME_LATENCY {
            let camera_buffer = VkBuffer::new(
                &vk_core,
                &[camera_uniform],
                vk::BufferCreateInfo::default()
                    .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .size(std::mem::size_of::<CameraUniform>() as u64),
                AllocationCreateDesc{
                    name: "Camera uniform buffer",
                    requirements: vk::MemoryRequirements::default(),
                    location: MemoryLocation::CpuToGpu,
                    linear: false,
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged
                },
                command_pool.vk_cmd_pool(),
                *vk_core.graphics_queue()
            ).expect("Failed to create camera uniform buffer");

            camera_buffers.push(camera_buffer);
        }

        Ok(
            Self {
                camera_buffers,
                position,
                target,
                fov,
                distance,
                pitch,
                yaw,
                aspect_ratio,
                camera_controller: CameraController::new(1.0, 0.005, 0.002),
                camera_uniform,
            }
        )
    }

    pub fn update(&mut self, _dt: f32, _screen_size: &Vec2) {
        let (delta_yaw, delta_pitch, zoom_amount, pan_delta_screen) = self.camera_controller.process_frame_deltas();

        // 1. Orbit
        self.yaw += delta_yaw;
        self.pitch += delta_pitch;

        // Clamp pitch to avoid flipping
        const PITCH_LIMIT: f32 = 89.0f32 * (std::f32::consts::PI / 180.0);
        self.pitch = self.pitch.clamp(0.1, PITCH_LIMIT);

        // 2. Zoom (Exponential / Multiplicative)
        if zoom_amount != 0.0 {
            // 0.90 base means ~10% zoom per scroll "tick"
            // If zoom_amount is positive (scrolling up), we get closer (multiply by < 1.0)
            let zoom_factor = 0.95f32.powf(zoom_amount);
            self.distance = (self.distance * zoom_factor).clamp(0.5, 10000.0);
        }

        // 3. Pan
        let rotation = Quat::from_euler(glam::EulerRot::YXZ, self.yaw, -self.pitch, 0.0);
        let right = rotation * Vec3::X;
        let up = rotation * Vec3::Y;

        if pan_delta_screen != Vec2::ZERO {
            // Pan speed scales with distance so it feels natural at any zoom level
            let pan_dist_factor = self.distance;
            let pan_movement = (right * -pan_delta_screen.x + up * pan_delta_screen.y) * pan_dist_factor;
            self.target += pan_movement;
        }

        // 4. Update Position
        let offset = rotation * Vec3::new(0.0, 0.0, self.distance);
        self.position = self.target + offset;
    }

    // Standard Matrix builders
    pub fn build_view_projection_matrix(&mut self, screen_size: &Vec2) -> (Mat4, Mat4) {
        self.aspect_ratio = screen_size.x / screen_size.y;
        let view = Mat4::look_at_rh(self.position, self.target, Vec3::Y);
        let near = 0.5;
        let far = self.distance * 10.0;
        let projection = Mat4::perspective_rh(self.fov, self.aspect_ratio, near, far);
        let correction = Mat4::from_cols(
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, -1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.5, 0.0),
            Vec4::new(0.0, 0.0, 0.5, 1.0),
        );
        let final_proj = correction * projection;
        self.camera_uniform.update_view_projection(&view, &final_proj);
        (view, final_proj)
    }

    // Pass-throughs
    pub fn move_camera(&mut self, key: KeyCode, is_pressed: bool) {
        self.camera_controller.handle_keyboard(key, is_pressed);
    }
    pub fn zoom_camera(&mut self, mouse_scroll_delta: MouseScrollDelta) {
        self.camera_controller.handle_scroll(mouse_scroll_delta);
    }
    pub fn handle_mouse_input(&mut self, button: &MouseButton, state: &ElementState) {
        self.camera_controller.handle_mouse_click(button, state);
    }
    pub fn update_mouse_position(&mut self, pos: PhysicalPosition<f64>) {
        self.camera_controller.update_mouse_position(Vec2::new(pos.x as f32, pos.y as f32));
    }
    pub fn set_camera_zoom_position(&mut self, pos: Option<PhysicalPosition<f64>>) {
        if let Some(p) = pos { self.update_mouse_position(p); }
    }
    #[allow(unused)]
    pub fn screen_to_world(&self, screen_size: &Vec2, screen_pos: &Vec2) -> Option<Vec2> {
        let (view, proj) = self.get_matrices_for_calculation(screen_size);
        let view_proj_inv = (proj * view).inverse();
        let ndc_x = (screen_pos.x / screen_size.x) * 2.0 - 1.0;
        let ndc_y = (screen_pos.y / screen_size.y) * 2.0 - 1.0;
        let world_near = view_proj_inv * Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
        let world_far = view_proj_inv * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
        let near_p = world_near.truncate() / world_near.w;
        let far_p = world_far.truncate() / world_far.w;
        let dir = (far_p - near_p).normalize();
        let denom = Vec3::Z.dot(dir); // Assuming Z-up plane for world interaction
        if denom.abs() < 1e-6 { return None; }
        let t = -(Vec3::Z.dot(near_p)) / denom;
        if t < 0.0 { return None; }
        let hit = near_p + dir * t;
        Some(Vec2::new(hit.x, hit.y))
    }
    fn get_matrices_for_calculation(&self, screen_size: &Vec2) -> (Mat4, Mat4) {
        let view = Mat4::look_at_rh(self.position, self.target, Vec3::Y);
        let aspect_ratio = screen_size.x / screen_size.y;
        let projection = Mat4::perspective_rh(self.fov, aspect_ratio, 0.5, self.distance * 10.0);
        let correction = Mat4::from_cols(
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, -1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.5, 0.0),
            Vec4::new(0.0, 0.0, 0.5, 1.0),
        );
        (view, correction * projection)
    }
    pub fn get_uniform(&self) -> &CameraUniform { &self.camera_uniform }
    pub fn buffer(&self, frame_index: usize) -> &VkBuffer<CameraUniform> { &self.camera_buffers[frame_index] }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CameraUniform {
    view: [[f32; 4]; 4],
    projection: [[f32; 4]; 4]
}
impl CameraUniform {
    pub fn new() -> Self {
        Self { view: Mat4::IDENTITY.to_cols_array_2d(), projection: Mat4::IDENTITY.to_cols_array_2d() }
    }
    pub fn update_view_projection(&mut self, view: &Mat4, projection: &Mat4) {
        self.view = view.transpose().to_cols_array_2d();
        self.projection = projection.transpose().to_cols_array_2d();
    }
}

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
}
