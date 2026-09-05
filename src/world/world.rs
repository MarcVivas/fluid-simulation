use std::sync::Arc;
use glam::{Vec3};
use crate::simulation::neighbor_list::{NeighborList};
use crate::simulation::octree::octree::Octree;
use crate::vulkan::compute::ComputeEngine;
use crate::vulkan::frame::frame_pacer::{FramePacer};
use crate::vulkan::profiler::GpuProfiler;
use crate::vulkan::commands::CommandBuffer;
use crate::world::particles::{Particles, ParticleRenderData};
use crate::simulation::physics_engine::PhysicsEngine;
use crate::vulkan::core::VulkanContext;
use crate::world::particles::ParticleInitPreset;

pub struct World{
    world_size: f32,
    world_min: glam::Vec4,
    physics_engine: PhysicsEngine,
    particle_system: Particles,
    octree: Octree,
    neighbor_list: NeighborList,
}

const NUM_PARTICLES: usize = 1000000; 

impl World{
    pub fn new(vk_core: &Arc<VulkanContext>, world_max: &Vec3, compute_engine: &ComputeEngine) -> anyhow::Result<Self>{

        let world_min = glam::Vec4::new(0., 0., 0.0, 0.);
        let world_size = world_max.max_element();
        let search_radius = 1.7f32;

        let particle_system = Particles::new(
            NUM_PARTICLES,
            &world_max,
            &vk_core,
            compute_engine.command_pool(),
            ParticleInitPreset::DoubleDamBreak,
            search_radius
        )?;


        let cmd_pool = compute_engine.command_pool();
        
        let octree = Octree::new(vk_core, cmd_pool, particle_system.len() as u32)?;
        let neighbor_list = NeighborList::new(vk_core, cmd_pool, NUM_PARTICLES, octree.max_expected_leaves(), vk_core.device_properties().subgroup_size(), Octree::max_levels())?;

        let physics_engine = PhysicsEngine::new(vk_core, cmd_pool, &particle_system, Octree::max_levels(), search_radius)?;


        Ok(Self {
            particle_system,
            physics_engine,
            world_size,
            world_min,
            octree,
            neighbor_list
        })
    }

    pub fn world_size(&self) -> f32 {
        self.world_size
    }

    pub fn world_min(&self) -> glam::Vec4 {
        self.world_min
    }

    /// Updates the world
    pub fn update(&mut self, vk_core: &VulkanContext, cmd_buffer: &CommandBuffer, gpu_profiler: &GpuProfiler, frame_pacer: &FramePacer){
        let world_min = self.world_min();
        let world_size = self.world_size();
        self.physics_engine.update(
             vk_core,
             cmd_buffer,
             &mut self.particle_system,
             world_size,
             world_min,
             &mut self.octree,
             &self.neighbor_list,
             gpu_profiler,
             frame_pacer
        );   
    }

    pub fn extract_render_data(&self) -> ParticleRenderData {
        self.particle_system.extract_render_data()
    }
}
