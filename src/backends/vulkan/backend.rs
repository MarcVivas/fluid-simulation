use std::sync::Arc;

use anyhow::{Context, Result};
use ash::vk;
use winit::window::Window;

use crate::backends::gpu_backend::GpuBackend;
use crate::backends::vulkan::particles::physics::ParticlePhysics;
use crate::backends::vulkan::rendering::VulkanWorldRenderer;
use crate::backends::vulkan::runtime::compute::ComputeExecutor;
use crate::backends::vulkan::runtime::core::VulkanContext;
use crate::backends::vulkan::runtime::core::surface::Surface;
use crate::backends::vulkan::runtime::frame::frame_pacer::FramePacer;
use crate::backends::vulkan::runtime::profiler::GpuProfiler;
use crate::backends::vulkan::world_resources::VulkanWorldResources;
use crate::viewer::camera::CameraUniform;
use crate::world::World;

pub struct VulkanBackend {
    vk_context: Arc<VulkanContext>,
    world_renderer: VulkanWorldRenderer,
    particle_physics: ParticlePhysics,
    gpu_profiler: GpuProfiler,
    frame_pacer: FramePacer,
    gpu_world: VulkanWorldResources,
}

impl VulkanBackend {
    pub fn new(
        vk_context: Arc<VulkanContext>,
        window: &Window,
        surface: Surface,
        world: &World,
    ) -> Result<Self> {
        let frame_pacer = FramePacer::new(3);
        let frames_in_flight = frame_pacer.frames_in_flight();

        let compute_executor = ComputeExecutor::new(vk_context.clone(), frames_in_flight)
            .context("Failed to create compute executor")?;

        let gpu_world =
            VulkanWorldResources::new(&vk_context, world, compute_executor.command_pool())
                .context("Failed to create GPU world resources")?;

        let particle_physics = ParticlePhysics::new(
            vk_context.clone(),
            &gpu_world,
            world.search_radius(),
            compute_executor,
        )
        .context("Failed to create particle physics")?;

        let world_renderer =
            VulkanWorldRenderer::new(vk_context.clone(), window, surface, frames_in_flight)
                .context("Failed to create world renderer")?;

        let gpu_profiler = GpuProfiler::new(vk_context.clone(), 100, frames_in_flight)
            .context("Failed to create GPU profiler")?;

        Ok(Self {
            vk_context,
            world_renderer,
            particle_physics,
            gpu_profiler,
            frame_pacer,
            gpu_world,
        })
    }
}

impl GpuBackend for VulkanBackend {
    fn begin_frame(&mut self, window: &Window) -> Result<Option<glam::Vec2>> {
        if !self.world_renderer.begin_frame(window, &self.frame_pacer)? {
            return Ok(None);
        }

        Ok(Some(self.world_renderer.viewport_size()))
    }

    fn execute_frame(&mut self, camera: &CameraUniform, paused: bool) -> Result<()> {
        let device = self.vk_context.device();

        self.gpu_profiler.print_metrics(device, &self.frame_pacer);
        self.gpu_profiler
            .reset_on_host(device, self.frame_pacer.ring_index());

        let compute_finished = self
            .particle_physics
            .compute_finished_semaphore(&self.frame_pacer);

        self.particle_physics.update(
            &self.vk_context,
            &mut self.gpu_world,
            &self.frame_pacer,
            &self.gpu_profiler,
            paused,
        )?;

        let render_data = self.gpu_world.extract_render_data();
        self.world_renderer
            .update(&self.frame_pacer, camera, &render_data)?;

        self.particle_physics.submit_update(
            &self.frame_pacer,
            self.world_renderer
                .previous_frame_wait(&self.frame_pacer)
                .as_slice(),
        )?;
        self.world_renderer
            .submit_and_present(compute_finished, &self.frame_pacer)?;

        self.frame_pacer.advance();
        Ok(())
    }

    fn resize(&mut self, window: &Window) -> Result<()> {
        self.world_renderer.resize_window(window, &self.frame_pacer)
    }

    fn shutdown(&self) -> Result<()> {
        match unsafe { self.vk_context.device().device_wait_idle() } {
            Ok(()) | Err(vk::Result::ERROR_DEVICE_LOST) => Ok(()),
            Err(error) => Err(error).context("Failed to wait for Vulkan work during shutdown"),
        }
    }
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        if let Err(err) = self.shutdown() {
            eprintln!("Error during VulkanBackend shutdown: {:?}", err);
        }
    }
}
