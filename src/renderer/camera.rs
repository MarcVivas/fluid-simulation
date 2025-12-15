use std::sync::Arc;
use ash::prelude::VkResult;
use ash::vk;
use ash::vk::Extent2D;
use glam::{Mat4, Vec2, Vec3};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{AllocationCreateDesc, AllocationScheme};
use winit::dpi::{PhysicalPosition, PhysicalSize};

pub struct Camera {
    pub position: Vec3,
    pub zoom: f32,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_buffers: Vec<VkBuffer>
}

impl Camera {
    pub fn new(
        vk_core: &Arc<VkCore>,
        world_size: &Vec2,
        window_size: &PhysicalSize<u32>,
        command_pool: &CommandPool
    ) -> VkResult<Self>
    {
        let min_x = 0.0;
        let max_x = world_size.x;
        let min_y = 0.0;
        let max_y = world_size.y;

        let position = Vec3::new(
            (min_x + max_x) / 2.0,
            (min_y + max_y) / 2.0,
            0.0
        );

        // 3. Calculate the required zoom to fit the object on screen
        let world_width = world_size.x as u32;
        let world_height = world_size.y as u32;


        let screen_width = window_size.width;
        let screen_height = window_size.height;

        // Calculate zoom based on width and height, pick the smaller one to ensure it all fits
        let zoom_x = screen_width / world_width;
        let zoom_y = screen_height / world_height;
        let zoom = zoom_x.min(zoom_y) as f32 * 0.9; // Use 90% of the screen for some padding

        // Create the Camera controller and the initial uniform data
        let camera_uniform = CameraUniform::new();


        let mut camera_buffers = Vec::with_capacity(MAX_FRAME_LATENCY);

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
                command_pool.vk_cmd_pool()
            ).expect("Failed to create camera uniform buffer");

            camera_buffers.push(camera_buffer);
        }





        Ok(
            Self {
                camera_buffers,
                position,
                zoom,
                camera_controller: CameraController::new(250.0, 0.1),
                camera_uniform,
            }
        )
    }

    pub fn build_view_projection_matrix(&mut self, screen_size: &Vec2) -> (Mat4, Mat4) {
        // Create view matrix - invert camera position to move the world opposite to camera
        let view = Mat4::from_translation(-self.position);

        let screen_width = screen_size.x;
        let screen_height = screen_size.y;

        // Create symmetric orthographic projection centered around origin
        let half_width = screen_width / (2.0 * self.zoom);
        let half_height = screen_height / (2.0 * self.zoom);

        let projection = Mat4::orthographic_rh(
            -half_width,  // left
            half_width,   // right
            -half_height, // bottom
            half_height,  // top
            -1000.0, // near
            1000.0,  // far
        );

        let correction = Mat4::from_cols(
            glam::Vec4::new(1.0, 0.0, 0.0, 0.0),
            glam::Vec4::new(0.0, -1.0, 0.0, 0.0), // Flip Y
            glam::Vec4::new(0.0, 0.0, 0.5, 0.0),  // Scale Z (OpenGL -1..1 to Vulkan 0..1)
            glam::Vec4::new(0.0, 0.0, 0.5, 1.0),  // Shift Z
        );

        let final_proj = correction * projection;

        self.camera_uniform.update_view_projection(&view, &final_proj);
        (view, final_proj)
    }


    pub fn move_camera(&mut self, key: KeyCode, is_pressed: bool) {
        self.camera_controller.move_camera(key, is_pressed);
    }

    pub fn zoom_camera(&mut self, mouse_scroll_delta: MouseScrollDelta) {
        self.camera_controller.zoom_camera(mouse_scroll_delta);
    }

    pub fn set_camera_zoom_position(&mut self, pos: Option<PhysicalPosition<f64>>) {
        self.camera_controller.set_camera_zoom_position(pos);
    }

    pub fn update(&mut self, dt: f32, screen_size: &Vec2) {
        let move_speed = self.camera_controller.speed * dt / self.zoom;

        if self.camera_controller.is_up_pressed { self.position.y += move_speed; }
        if self.camera_controller.is_down_pressed { self.position.y -= move_speed; }
        if self.camera_controller.is_right_pressed { self.position.x += move_speed; }
        if self.camera_controller.is_left_pressed { self.position.x -= move_speed; }

        if self.camera_controller.scroll_delta != 0.0 {
            // 1. Get the world coordinates of the mouse before zooming
            let mouse_world_pos_before_zoom = self.screen_to_world(screen_size, &self.camera_controller.mouse_position);

            // 2. Calculate the new zoom level
            let zoom_factor = 1.0 + (self.camera_controller.scroll_delta * self.camera_controller.zoom_sensitivity);
            self.zoom *= zoom_factor;
            // Clamp the zoom to prevent it from becoming too small or large
            self.zoom = self.zoom.clamp(0.1, 100.0);

            // 3. Get the world coordinates of the mouse after zooming
            let mouse_world_pos_after_zoom = self.screen_to_world(screen_size, &self.camera_controller.mouse_position);

            // 4. Calculate the difference (how much the world shifted under the cursor)
            let world_delta = mouse_world_pos_before_zoom - mouse_world_pos_after_zoom;

            // 5. Adjust the camera position to counteract the shift
            self.position += Vec3::new(world_delta.x, world_delta.y, 0.0);

            // 6. Reset the scroll delta
            self.camera_controller.scroll_delta = 0.0;
        }
    }

    pub fn screen_to_world(&self, screen_size: &Vec2, screen_pos: &Vec2) -> Vec2 {
        // Convert screen coordinates to normalized device coordinates (-1 to 1)
        let ndc_x = (screen_pos.x / screen_size.x) * 2.0 - 1.0;
        let ndc_y = 1.0 - (screen_pos.y / screen_size.y) * 2.0; // Flip Y axis

        // Convert NDC to world coordinates
        let half_width = screen_size.x / (2.0 * self.zoom);
        let half_height = screen_size.y / (2.0 * self.zoom);

        let world_x = self.position.x + ndc_x * half_width;
        let world_y = self.position.y + ndc_y * half_height;

        Vec2::new(world_x, world_y)
    }

  
    pub fn get_uniform(&self) -> &CameraUniform {
        &self.camera_uniform
    }
    
    pub fn buffer(&self, frame_index: usize) -> &VkBuffer {
        &self.camera_buffers[frame_index]
    }
    
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
use crate::renderer::renderer::{Renderer, MAX_FRAME_LATENCY};
use crate::vk_core::vk_core::VkCore;
use crate::vk_utils::{CommandPool, DescriptorPool, DescriptorSet, PipelineLayout};
use crate::vk_utils::vk_buffer::VkBuffer;

#[derive(Debug)]
pub struct CameraController {
    is_up_pressed: bool,
    is_down_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
    speed: f32,
    zoom_sensitivity: f32,
    scroll_delta: f32, // New field to store scroll amount
    mouse_position: glam::Vec2, // Track mouse position for zoom-to-cursor
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
            mouse_position: glam::Vec2::ZERO,
        }
    }

    fn move_camera(&mut self, key: KeyCode, is_pressed: bool) -> bool {
        match key {
            KeyCode::KeyW => {
                self.is_up_pressed = is_pressed;
                true
            }
            KeyCode::KeyS => {
                self.is_down_pressed = is_pressed;
                true
            }
            KeyCode::KeyA => {
                self.is_left_pressed = is_pressed;
                true
            }
            KeyCode::KeyD => {
                self.is_right_pressed = is_pressed;
                true
            }
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
        let position = pos.unwrap();
        self.mouse_position = glam::Vec2::new(position.x as f32, position.y as f32);
    }


}