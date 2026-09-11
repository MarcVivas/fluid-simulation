use anyhow::{Context, Result};
use ash::vk;
use std::sync::Arc;
use winit::window::Window;

use crate::backends::vulkan::particles::ParticleRenderInput;
use crate::backends::vulkan::particles::ParticleRenderer;
use crate::backends::vulkan::rendering::main_pass::MainPass;
use crate::backends::vulkan::runtime::core::VulkanContext;
use crate::backends::vulkan::runtime::core::surface::Surface;
use crate::backends::vulkan::runtime::frame::frame_pacer::FramePacer;
use crate::backends::vulkan::runtime::graphics::frame_renderer::FrameRenderer;
use crate::backends::vulkan::runtime::swapchain::frame_status::FrameStatus;
use crate::viewer::camera::CameraUniform;

pub struct VulkanWorldRenderer {
    renderer: FrameRenderer,
    particle_renderer: ParticleRenderer,
    current_image_index: Option<u32>,
    main_pass: MainPass,
}

impl VulkanWorldRenderer {
    pub fn new(
        vk_core: Arc<VulkanContext>,
        window: &winit::window::Window,
        surface: Surface,
        frames_in_flight: usize,
    ) -> Result<Self> {
        let renderer = FrameRenderer::new(vk_core.clone(), window, surface, frames_in_flight)
            .context("Failed to create renderer")?;

        let particle_renderer = ParticleRenderer::new(vk_core.clone(), renderer.render_target())
            .context("Failed to create particle renderer")?;

        let main_pass = MainPass::new();

        Ok(Self {
            renderer,
            current_image_index: None,
            particle_renderer,
            main_pass,
        })
    }

    /// Begins the render frame. Returns true if valid, false if out-of-date/minimized.
    pub fn begin_frame(&mut self, window: &Window, frame_pacer: &FramePacer) -> Result<bool> {
        match self.renderer.begin_frame(window, frame_pacer)? {
            FrameStatus::Ready { image_index } => {
                self.current_image_index = Some(image_index);
                Ok(true)
            }
            FrameStatus::OutOfDate => {
                self.current_image_index = None;
                Ok(false)
            }
        }
    }

    pub fn update(
        &mut self,
        frame_pacer: &FramePacer,
        camera_uniform: &CameraUniform,
        render_data: &ParticleRenderInput,
    ) -> Result<()> {
        if let Some(image_index) = self.current_image_index {
            self.renderer
                .record_frame(frame_pacer, image_index, &self.main_pass, |cb| {
                    self.particle_renderer.draw(
                        cb,
                        render_data.total_particles,
                        render_data,
                        camera_uniform,
                    );
                })?;
        }

        Ok(())
    }

    pub fn resize_window(&mut self, window: &Window, frame_pacer: &FramePacer) -> Result<()> {
        self.renderer
            .resize_window(window, frame_pacer.frames_in_flight())
    }

    pub fn submit_and_present(
        &mut self,
        compute_semaphore: vk::SemaphoreSubmitInfo,
        frame_pacer: &FramePacer,
    ) -> Result<()> {
        if let Some(image_index) = self.current_image_index {
            self.renderer
                .submit_and_present(&[compute_semaphore], frame_pacer, image_index)?;
        }
        Ok(())
    }

    pub fn viewport_size(&self) -> glam::Vec2 {
        let extent = self.renderer.render_target().resolution();
        glam::Vec2::new(extent.width as f32, extent.height as f32)
    }

    pub fn previous_frame_wait(
        &self,
        frame_pacer: &FramePacer,
    ) -> Option<vk::SemaphoreSubmitInfo<'static>> {
        self.renderer.previous_frame_wait(frame_pacer)
    }
}
