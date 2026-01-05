use std::sync::Arc;
use ash::prelude::VkResult;
use ash::vk;
use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{AllocationCreateDesc, AllocationScheme};
use winit::dpi::{PhysicalPosition, PhysicalSize};



pub struct Camera {
    pub position: Vec3, // This is the "Target" on the ground
    pub zoom: f32,
    pub pitch: f32,     // Angle looking down (radians)
    pub yaw: f32,       // Rotation around Y axis (radians)
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_buffers: Vec<VkBuffer>
}

impl Camera {
    pub fn new(
        vk_core: &Arc<VkCore>,
        world_size: &Vec3,
        window_size: &PhysicalSize<u32>,
        command_pool: &CommandPool
    ) -> VkResult<Self>
    {
        // 1. Initialize center position
        let position = Vec3::new(
            world_size.x / 2.0,
            world_size.y / 2.0,
            world_size.z / 2.0
        );

        // 2. Setup standard Isometric angles
        // Pitch: 60 degrees down looks good for modern ARPGs
        // Yaw: 45 degrees gives the classic diamond tile look
        let pitch = 60.0f32.to_radians();
        let yaw = 45.0f32.to_radians();

        // 3. Calculate the required zoom
        // In Orthographic, zoom = pixels per world unit essentially
        let screen_width = window_size.width as f32;
        let screen_height = window_size.height as f32;

        let zoom_x = screen_width / world_size.x;
        let zoom_y = screen_height / world_size.y;
        let zoom = zoom_x.min(zoom_y) * 0.9;

        let camera_uniform = CameraUniform::new();
        let mut camera_buffers = Vec::with_capacity(MAX_FRAME_LATENCY);

        // 4. Create Buffers (Standard Vulkan boilerpate)
        for _ in 0..MAX_FRAME_LATENCY {
            let camera_buffer = VkBuffer::new(
                &vk_core,
                &[camera_uniform],
                vk::BufferCreateInfo::default()
                    .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .size(size_of::<CameraUniform>() as u64),
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
                zoom,
                pitch,
                yaw,
                camera_controller: CameraController::new(250.0, 0.1),
                camera_uniform,
            }
        )
    }

    pub fn build_view_projection_matrix(&mut self, screen_size: &Vec2) -> (Mat4, Mat4) {
        // 1. Calculate View Matrix (The "Look At")
        // We act as if 'self.position' is the target on the ground.
        // We move the camera backwards based on the rotation.
        let rotation = Quat::from_euler(glam::EulerRot::YXZ, self.yaw, -self.pitch, 0.0);

        // Offset camera by an arbitrary distance "back" from the target.
        // In Orthographic, this distance doesn't affect sprite size, only clipping.
        let camera_offset = rotation * Vec3::new(0.0, 0.0, 1000.0);
        let eye_position = self.position + camera_offset;

        let view = Mat4::look_at_rh(
            eye_position,
            self.position,
            Vec3::Y
        );

        // 2. Calculate Projection Matrix (Orthographic)
        let screen_width = screen_size.x;
        let screen_height = screen_size.y;

        // The zoom variable controls how much of the world we see.
        let half_width = screen_width / (2.0 * self.zoom);
        let half_height = screen_height / (2.0 * self.zoom);

        let projection = Mat4::orthographic_rh(
            -half_width,
            half_width,
            -half_height,
            half_height,
            0.1,    // Near plane (must be > 0 for some culling logic)
            5000.0, // Far plane (enough to see the ground)
        );

        // 3. Vulkan Correction Matrix
        // Flip Y (Vulkan Y is down) and map Z from [-1, 1] to [0, 1]
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

    pub fn update(&mut self, dt: f32, screen_size: &Vec2) {
        // 1. Handle Movement (WASD)
        // We want 'Up' to move visually 'Up' on screen, regardless of camera rotation.
        let forward = Vec3::new(-self.yaw.sin(), self.yaw.cos(), 0.0).normalize_or_zero();
        let right = Vec3::new(self.yaw.cos(), self.yaw.sin(), 0.0).normalize_or_zero();

        // Since this is isometric on Z-plane, we map Y inputs to the projected Forward vector on ground
        let move_speed = self.camera_controller.speed * dt / self.zoom;
        let mut delta = Vec3::ZERO;

        if self.camera_controller.is_up_pressed { delta += forward * move_speed; }
        if self.camera_controller.is_down_pressed { delta -= forward * move_speed; }
        if self.camera_controller.is_right_pressed { delta += right * move_speed; }
        if self.camera_controller.is_left_pressed { delta -= right * move_speed; }

        self.position += delta;

        // 2. Handle Zoom-to-Cursor
        if self.camera_controller.scroll_delta != 0.0 {
            // A. Get world pos under mouse BEFORE zoom
            let world_mouse_before = self.screen_to_world(screen_size, &self.camera_controller.mouse_position);

            // B. Apply Zoom
            let zoom_factor = 1.0 + (self.camera_controller.scroll_delta * self.camera_controller.zoom_sensitivity);
            self.zoom = (self.zoom * zoom_factor).clamp(0.5, 200.0);

            // C. Get world pos under mouse AFTER zoom
            let world_mouse_after = self.screen_to_world(screen_size, &self.camera_controller.mouse_position);

            // D. Shift camera target to compensate
            // This makes the world point stay under the mouse cursor
            if let (Some(before), Some(after)) = (world_mouse_before, world_mouse_after) {
                let diff = before - after;
                self.position += Vec3::new(diff.x, diff.y, 0.0);
            }

            self.camera_controller.scroll_delta = 0.0;
        }
    }

    /// Converts a screen position (pixels) to a World Position on the Z=0 plane.
    /// Returns None if the ray doesn't hit the ground plane.
    pub fn screen_to_world(&self, screen_size: &Vec2, screen_pos: &Vec2) -> Option<Vec2> {
        let (view, proj) = self.get_matrices_for_calculation(screen_size);
        let view_proj_inv = (proj * view).inverse();

        // 1. Convert to NDC (-1 to 1)
        let ndc_x = (screen_pos.x / screen_size.x) * 2.0 - 1.0;
        let ndc_y = (screen_pos.y / screen_size.y) * 2.0 - 1.0; // Winit Y is down, but we want mathematical Y

        // 2. Unproject two points: one at near plane (z=0) and one at far plane (z=1)
        // Note: In Vulkan, Near is 0, Far is 1.
        let ndc_near = Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
        let ndc_far = Vec4::new(ndc_x, ndc_y, 1.0, 1.0);

        let world_near = view_proj_inv * ndc_near;
        let world_far = view_proj_inv * ndc_far;

        // Perspective divide (normalize w)
        let near_point = world_near.truncate() / world_near.w;
        let far_point = world_far.truncate() / world_far.w;

        // 3. Ray-Plane Intersection (Plane Normal = Z-Up)
        // Ray Origin = near_point
        // Ray Dir = (far_point - near_point)
        let ray_origin = near_point;
        let ray_dir = (far_point - near_point).normalize();

        // Plane Z = 0. Normal is (0, 0, 1)
        let plane_normal = Vec3::Z;
        let plane_d = 0.0;

        let denom = plane_normal.dot(ray_dir);

        // Check if ray is parallel to plane
        if denom.abs() < 1e-6 {
            return None;
        }

        let t = -(plane_normal.dot(ray_origin) + plane_d) / denom;

        // If t < 0, intersection is behind the camera
        if t < 0.0 {
            return None;
        }

        let intersection = ray_origin + ray_dir * t;
        Some(Vec2::new(intersection.x, intersection.y))
    }

    // Helper to get matrices without updating the uniform buffer (for math calcs)
    fn get_matrices_for_calculation(&self, screen_size: &Vec2) -> (Mat4, Mat4) {
        let rotation = Quat::from_euler(glam::EulerRot::YXZ, self.yaw, -self.pitch, 0.0);
        let camera_offset = rotation * Vec3::new(0.0, 0.0, 1000.0);
        let eye_position = self.position + camera_offset;

        let view = Mat4::look_at_rh(eye_position, self.position, Vec3::Y);

        let screen_width = screen_size.x;
        let screen_height = screen_size.y;
        let half_width = screen_width / (2.0 * self.zoom);
        let half_height = screen_height / (2.0 * self.zoom);

        let projection = Mat4::orthographic_rh(
            -half_width, half_width, -half_height, half_height, 0.1, 5000.0,
        );

        let correction = Mat4::from_cols(
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, -1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.5, 0.0),
            Vec4::new(0.0, 0.0, 0.5, 1.0),
        );

        (view, correction * projection)
    }

    // Pass-through functions
    pub fn move_camera(&mut self, key: KeyCode, is_pressed: bool) {
        self.camera_controller.move_camera(key, is_pressed);
    }

    pub fn zoom_camera(&mut self, mouse_scroll_delta: MouseScrollDelta) {
        self.camera_controller.zoom_camera(mouse_scroll_delta);
    }

    pub fn set_camera_zoom_position(&mut self, pos: Option<PhysicalPosition<f64>>) {
        self.camera_controller.set_camera_zoom_position(pos);
    }

    pub fn get_uniform(&self) -> &CameraUniform { &self.camera_uniform }
    pub fn buffer(&self, frame_index: usize) -> &VkBuffer { &self.camera_buffers[frame_index] }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CameraUniform {
    view: [[f32; 4]; 4],
    projection: [[f32; 4]; 4]
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            // Initialize with the identity matrix.
            view: Mat4::IDENTITY.to_cols_array_2d(),
            projection: Mat4::IDENTITY.to_cols_array_2d(),
        }
    }

    pub fn update_view_projection(&mut self, view: &Mat4, projection: &Mat4) {
        self.view = view.transpose().to_cols_array_2d();
        self.projection = projection.transpose().to_cols_array_2d();
    }
}

use winit::event::{MouseScrollDelta};
use winit::keyboard::{KeyCode};
use crate::renderer::renderer::{MAX_FRAME_LATENCY};
use crate::vk_core::vk_core::VkCore;
use crate::vk_utils::{CommandPool};
use crate::vk_utils::vk_buffer::VkBuffer;

#[derive(Debug)]
pub struct CameraController {
    is_up_pressed: bool,
    is_down_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
    speed: f32,
    zoom_sensitivity: f32,
    scroll_delta: f32,
    mouse_position: glam::Vec2,
}

impl CameraController {
    fn new(speed: f32, zoom_sensitivity: f32) -> Self {
        Self {
            is_up_pressed: false,
            is_down_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
            speed,
            zoom_sensitivity,
            scroll_delta: 0.0,
            mouse_position: Vec2::ZERO,
        }
    }

    fn move_camera(&mut self, key: KeyCode, is_pressed: bool) -> bool {
        match key {
            KeyCode::KeyW => { self.is_up_pressed = is_pressed; true }
            KeyCode::KeyS => { self.is_down_pressed = is_pressed; true }
            KeyCode::KeyA => { self.is_left_pressed = is_pressed; true }
            KeyCode::KeyD => { self.is_right_pressed = is_pressed; true }
            _ => false,
        }
    }

    pub fn zoom_camera(&mut self, mouse_scroll_delta: MouseScrollDelta){
        self.scroll_delta += match mouse_scroll_delta {
            MouseScrollDelta::LineDelta(_, y) => y,
            MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
        };
    }

    pub fn set_camera_zoom_position(&mut self, pos: Option<PhysicalPosition<f64>>) {
        if let Some(position) = pos {
            self.mouse_position = Vec2::new(position.x as f32, position.y as f32);
        }
    }
}