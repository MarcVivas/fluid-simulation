use std::{sync::Arc};

use anyhow::{Context, Result};
use ash::vk;

use crate::{vulkan::{compute::ComputeEngine, core::VulkanContext, frame::frame_pacer::FramePacer, profiler::GpuProfiler}, world::{World, particles::ParticleRenderData}};

pub struct Simulation {
    world: World,
    compute_engine: ComputeEngine,
    paused: bool,

    render_sender: crossbeam_channel::Sender<ParticleRenderData>,
}

impl Simulation {
    pub fn new(vk_core: Arc<VulkanContext>, world_size: glam::Vec3, frames_in_flight: usize, render_sender: crossbeam_channel::Sender<ParticleRenderData>,) -> Result<Self> {

        let compute_engine = ComputeEngine::new(vk_core.clone(), frames_in_flight).context("Failed to create the compute engine")?;
        let world = World::new(&vk_core, &world_size, &compute_engine)
            .context("Failed to create world")?;

        Ok(Self { world, compute_engine, paused: true, render_sender})
    }

    
    pub fn update(&mut self, vk_core: &VulkanContext, frame_pacer: &FramePacer, gpu_profiler: &GpuProfiler) -> Result<()>{
        if self.paused {
            // Submit nothing
            self.compute_engine.record_commands(frame_pacer, |_cmd|{})?;
            return Ok(()); 
        }

        self.compute_engine.record_commands(frame_pacer, |cmd_buffer| { 
            self.world.update(vk_core, cmd_buffer, gpu_profiler, frame_pacer);
        })?;


        self.send_render_data()?;
        Ok(())
    }

    /// Submits the recorded compute work to the queue
    pub fn submit_update(&self, pacer: &FramePacer) -> Result<()> {
        // Both paused (empty command buffer) and active states submit and signal the timeline
        self.compute_engine.submit_to_queue(pacer, &[], true).context("Simulation subimt queue")?;
        Ok(())
    }

    pub fn compute_finished_semaphore(&self, frame_pacer: &FramePacer) -> vk::SemaphoreSubmitInfo<'static>{
        self.compute_engine.compute_finished_semaphore(frame_pacer, vk::PipelineStageFlags2::TASK_SHADER_EXT)
    }

    pub fn send_render_data(&self) -> Result<()>{
        self.render_sender.try_send(self.world.extract_render_data())?;
        Ok(())
    }

    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }
  
}
