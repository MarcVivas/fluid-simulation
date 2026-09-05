use std::sync::Arc;
use anyhow::{Context, Result};
use ash::vk;
use winit::window::Window;

use crate::{rendering::{camera::Camera, camera_controller::CameraController, main_pass::MainPass, particles::ParticleDrawer, }, vulkan::{core::{VulkanContext, surface::Surface}, frame::frame_pacer::FramePacer, graphics::renderer::Renderer, swapchain::frame_status::FrameStatus}, world::particles::ParticleRenderData};

pub struct SimulationVisualizer {
    renderer: Renderer,
    camera_controller: CameraController,
    current_render_data: Option<ParticleRenderData>,
    render_receiver: crossbeam_channel::Receiver<ParticleRenderData>,
    particle_drawer: ParticleDrawer,
    current_image_index: Option<u32>, 
    main_pass: MainPass
}

impl SimulationVisualizer{
    pub fn new(
        vk_core: Arc<VulkanContext>, 
        window: &winit::window::Window, 
        surface: Surface, 
        world_size: &glam::Vec3,
        render_receiver: crossbeam_channel::Receiver<ParticleRenderData>,
        frames_in_flight: usize,
    ) -> Result<Self> {
        let renderer = Renderer::new(vk_core.clone(), window, surface, frames_in_flight).context("Failed to create renderer")?;

        let camera = Camera::new(&vk_core, world_size, &window.inner_size(), renderer.command_pool(), frames_in_flight)
            .context("Failed to create camera")?;
        
        let camera_controller = CameraController::new(camera);
        let particle_drawer = ParticleDrawer::new(vk_core.clone(), renderer.render_target())
            .context("Failed to create particle drawer")?;

        let main_pass = MainPass::new();

        Ok(Self { renderer, current_render_data: None, camera_controller, render_receiver, current_image_index: None, particle_drawer, main_pass })
    }

    /// Begins the visualizer frame. Returns true if valid, false if out-of-date/minimized.
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

    pub fn update(&mut self, frame_pacer: &FramePacer) -> Result<()>{
        self.receive_latest_render_data();

        // Update camera
        self.update_camera(&frame_pacer)?;

        if let Some(image_index) = self.current_image_index {
            let camera_uniform = self.camera_controller.uniform_data();
            
            self.renderer.record_frame(frame_pacer, image_index, &self.main_pass, |cb| {
                    if let Some(render_data) = &self.current_render_data {
                        self.particle_drawer.draw(cb, render_data.total_particles, render_data, camera_uniform);
                    }
                }
            )?;
        }

        Ok(())
    }

    fn update_camera(&mut self, frame_pacer: &FramePacer) -> Result<()> {
        let extent = self.renderer.render_target().resolution();
        let screen_size = glam::Vec2::new(extent.width as f32, extent.height as f32);
        self.camera_controller.update(screen_size, frame_pacer)?;
        Ok(())
    }

    fn receive_latest_render_data(&mut self) {
        if let Ok(latest_render_data) = self.render_receiver.try_recv() {
            self.current_render_data = Some(latest_render_data);
        }
    }


    pub fn resize_window(&mut self, window: &Window, frame_pacer: &FramePacer) -> Result<()>{
        self.renderer.resize_window(window, frame_pacer.frames_in_flight())
    }


    pub fn submit_and_present(&mut self, compute_semaphore: vk::SemaphoreSubmitInfo, frame_pacer: &FramePacer) -> Result<()>{
          if let Some(image_index) = self.current_image_index {
              self.renderer.submit_and_present(
                  compute_semaphore,
                  frame_pacer,
                  image_index,
              )?;
          }
          Ok(())
    }

    pub fn camera_controller(&mut self) -> &mut CameraController {
        &mut self.camera_controller
    }

 

}
