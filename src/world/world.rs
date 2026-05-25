use std::sync::Arc;
use ash::vk;
use glam::{Vec3};
use crate::physics_engine::neighbor_list::{NeighborList};
use crate::renderer::renderer::Renderer;
use crate::utils::data_structures::octree::octree::Octree;
use crate::utils::gpu_profiler::GpuProfiler;
use crate::world::world_objects::particles::{Particles, ParticleRenderData};
use crate::physics_engine::PhysicsEngine;
use crate::vulkan::vk_core::VkCore;
use crate::compute::ComputeEngine;

pub struct World{
    world_size: f32,
    world_min: glam::Vec4,
    physics_engine: PhysicsEngine,
    particle_system: Particles,
    octree: Octree,
    neighbor_list: NeighborList,
}

const NUM_PARTICLES: usize = 400000; //8193;

impl World{
    pub fn new(vk_core: &Arc<VkCore>, world_max: &Vec3, compute_engine: &ComputeEngine, renderer: &Renderer) -> Self{

        let world_min = glam::Vec4::new(0., 0., 0.0, 0.);
        let world_size = world_max.max_element();
        
        let particle_system = Particles::new(
            NUM_PARTICLES,
            &world_max,
            &vk_core,
            renderer.command_pool(),
        ).expect("Failed to create particle system");

        let search_radius = 1.8f32;

        let cmd_pool = compute_engine.command_pool();
        
        let octree = Octree::new(vk_core, cmd_pool, particle_system.len() as u32);
        
        let physics_engine = PhysicsEngine::new(vk_core, cmd_pool, &particle_system, Octree::max_levels(), search_radius).unwrap();

        let neighbor_list = NeighborList::new(vk_core, cmd_pool, NUM_PARTICLES as u32, vk_core.subgroup_size(), Octree::max_levels());

        Self {
            particle_system,
            physics_engine,
            world_size,
            world_min,
            octree,
            neighbor_list
        }
    }

    pub fn world_size(&self) -> f32 {
        self.world_size
    }

    pub fn world_min(&self) -> glam::Vec4 {
        self.world_min
    }

    /// Updates the world
    pub fn update(&mut self, vk_core: &Arc<VkCore>, compute_engine: &ComputeEngine, first_frame: bool, gpu_profiler: &GpuProfiler){
        let world_min = self.world_min();
        let world_size = self.world_size();

        compute_engine.record_commands(|cmd_buffer| {
            if !first_frame {
                let acquire_from_graphics = [vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::MESH_SHADER_EXT)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                    .dst_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ)
                    .src_queue_family_index(vk_core.graphics_queue_family_index())
                    .dst_queue_family_index(vk_core.compute_queue_family_index())
                    .buffer(self.particle_system.buffers().positions_buffer.current().vk_buffer())
                    .size(vk::WHOLE_SIZE)];
    
                cmd_buffer.pipeline_memory_barrier2(vk_core.device(), &acquire_from_graphics, &[]);
            } 
            
            self.physics_engine.update(
                 vk_core,
                 cmd_buffer,
                 &mut self.particle_system,
                 world_size,
                 world_min,
                 &mut self.octree,
                 &self.neighbor_list,
                 gpu_profiler
            );

            let release_to_graphics =[vk::BufferMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::NONE) // NONE for release operations
                .dst_access_mask(vk::AccessFlags2::NONE)       // NONE for release operations
                .src_queue_family_index(vk_core.compute_queue_family_index())
                .dst_queue_family_index(vk_core.graphics_queue_family_index())
                .buffer(self.particle_system.buffers().positions_buffer.current().vk_buffer())
                .size(vk::WHOLE_SIZE)];
                        
            cmd_buffer.pipeline_memory_barrier2(vk_core.device(), &release_to_graphics, &[]);
        });


        
      
    }

    pub fn extract_render_data(&self, compute_paused: bool) -> ParticleRenderData {
        let iterations = self.physics_engine.solver_iterations() as usize;
        self.particle_system.extract_render_data(iterations, compute_paused)
    }
}
