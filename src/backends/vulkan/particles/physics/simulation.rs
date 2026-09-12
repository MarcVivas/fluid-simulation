use std::sync::Arc;

use anyhow::{Context, Result};
use ash::vk;

use crate::backends::vulkan::particles::physics::neighbors::NeighborList;
use crate::backends::vulkan::particles::physics::octree::Octree;
use crate::backends::vulkan::particles::physics::ParticleSolver;
use crate::backends::vulkan::runtime::compute::ComputeExecutor;
use crate::backends::vulkan::runtime::core::VulkanContext;
use crate::backends::vulkan::runtime::frame::frame_pacer::FramePacer;
use crate::backends::vulkan::runtime::profiler::GpuProfiler;
use crate::backends::vulkan::world_resources::VulkanWorldResources;

/// Coordinates Vulkan particle physics and its compute submissions; does not render.
pub struct ParticlePhysics {
    solver: ParticleSolver,
    octree: Octree,
    neighbor_list: NeighborList,
    compute_executor: ComputeExecutor,
}

impl ParticlePhysics {
    pub fn new(
        vk_context: Arc<VulkanContext>,
        gpu_world: &VulkanWorldResources,
        search_radius: f32,
        compute_executor: ComputeExecutor,
    ) -> Result<Self> {
        let command_pool = compute_executor.command_pool();
        let solver = ParticleSolver::new(
            &vk_context,
            command_pool,
            gpu_world.particles(),
            Octree::max_levels(),
            search_radius,
        )?;
        let octree = Octree::new(
            &vk_context,
            command_pool,
            gpu_world.particles().len() as u32,
        )?;
        let neighbor_list = NeighborList::new(
            &vk_context,
            command_pool,
            gpu_world.particles().len(),
            octree.max_expected_leaves(),
            octree.n_crit(),
            Octree::max_levels(),
        )?;

        Ok(Self {
            octree,
            neighbor_list,
            solver,
            compute_executor,
        })
    }

    fn record_update(
        &mut self,
        vk_context: &VulkanContext,
        gpu_world: &mut VulkanWorldResources,
        frame_pacer: &FramePacer,
        gpu_profiler: &GpuProfiler,
        paused: bool,
    ) -> Result<()> {
        if paused {
            self.compute_executor
                .record_commands(frame_pacer, |_cmd| {})?;
            return Ok(());
        }

        self.compute_executor
            .record_commands(frame_pacer, |cmd_buffer| {
                let bounds = gpu_world.bounds();

                self.solver.update(
                    vk_context,
                    cmd_buffer,
                    gpu_world.particles_mut(),
                    bounds.world_size,
                    bounds.world_min,
                    &mut self.octree,
                    &self.neighbor_list,
                    gpu_profiler,
                    frame_pacer,
                );
            })?;

        Ok(())
    }

    pub fn update(
        &mut self,
        vk_context: &VulkanContext,
        gpu_world: &mut VulkanWorldResources,
        frame_pacer: &FramePacer,
        gpu_profiler: &GpuProfiler,
        paused: bool,
    ) -> Result<()> {
        self.record_update(vk_context, gpu_world, frame_pacer, gpu_profiler, paused)?;
        Ok(())
    }

    pub fn submit_update(
        &self,
        frame_pacer: &FramePacer,
        waits: &[vk::SemaphoreSubmitInfo],
    ) -> Result<()> {
        self.compute_executor
            .submit_to_queue(frame_pacer, waits, true)
            .context("Simulation submit queue")?;
        Ok(())
    }

    pub fn compute_finished_semaphore(
        &self,
        frame_pacer: &FramePacer,
    ) -> vk::SemaphoreSubmitInfo<'static> {
        self.compute_executor
            .compute_finished_semaphore(frame_pacer, vk::PipelineStageFlags2::TASK_SHADER_EXT)
    }
}
