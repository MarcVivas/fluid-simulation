use ash::vk;

use crate::physics_engine::{PhysicsConfig};
use crate::physics_engine::neighbor_list::NeighborList;
use crate::utils::data_structures::octree::octree::{Octree};
use crate::utils::gpu_algorithms::hilbert_encoding::HilbertEncoder;
use crate::utils::gpu_profiler::GpuProfiler;
use crate::vulkan::vk_utils::CommandBuffer;
use crate::world::world_objects::{particles::Particles, particles::RearrangingSystem};
use crate::utils::gpu_algorithms::{sorting::kv_radix_sort::GpuKVRadixSort};
use crate::physics_engine::integration::{Integrator, UpdateVelocitiesSystem};
use crate::physics_engine::position_based_fluids::{DensityComputeSystem, VelocityRefiningSystem, VorticityForceComputeSystem};
use crate::physics_engine::position_based_dynamics::{ConstraintSolverSystem};
use crate::traits::{GpuTask};

use crate::vulkan::vk_core::VkCore;
use std::sync::Arc;

pub struct PhysicsEngine {
    physics_config: PhysicsConfig,
    sorting_system: GpuKVRadixSort<u32>,
    integration_system: Integrator,
    rearranging_system: RearrangingSystem,
    density_compute_system: DensityComputeSystem,
    constraint_solver_system: ConstraintSolverSystem,
    update_velocities_system: UpdateVelocitiesSystem,
    velocity_refining_system: VelocityRefiningSystem,
    vorticity_force_compute_system: VorticityForceComputeSystem,
    hilbert_encoder: HilbertEncoder
}

impl PhysicsEngine {
    pub fn new(
        vk_core: &Arc<VkCore>,
        cmd_pool: vk::CommandPool,
        particles: &Particles,
        max_levels: u32,
        search_radius: f32,
        super_cluster_size: u32
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let max_objects: u32 = particles.len() as u32;
        let integration_system = Integrator::new(vk_core)?;
        let sorting_system = GpuKVRadixSort::new(vk_core, cmd_pool, max_objects, None)?;
        let rearranging_system = RearrangingSystem::new(vk_core)?;
        let density_compute_system = DensityComputeSystem::new(vk_core, super_cluster_size)?;
        let constraint_solver_system = ConstraintSolverSystem::new(vk_core, super_cluster_size)?;
        let update_velocities_system = UpdateVelocitiesSystem::new(vk_core)?;
        let velocity_refining_system = VelocityRefiningSystem::new(vk_core)?;
        let vorticity_force_compute_system = VorticityForceComputeSystem::new(vk_core)?;
        let hilbert_encoder = HilbertEncoder::new(vk_core, max_levels)?;
        
        let physics_config = PhysicsConfig::new(search_radius);

        Ok(Self {
            physics_config,
            integration_system,
            sorting_system,
            rearranging_system,
            constraint_solver_system,
            update_velocities_system,
            density_compute_system,
            velocity_refining_system,
            vorticity_force_compute_system,
            hilbert_encoder
        })
    }

    pub fn update(
        &mut self,
        vk_core: &Arc<VkCore>,
        command_buffer: &CommandBuffer,
        particles: &mut Particles,
        world_size: f32,
        world_min: glam::Vec4,
        octree: &mut Octree,
        neighbor_list: &NeighborList,
        gpu_profiler: &GpuProfiler,
    ) { 
        
        let delta_time = self.physics_config.time_step;
        let world_max = world_min + world_size;
        let world_max = &glam::Vec3::new(world_max.x, world_max.y, world_max.z);


        let particle_data = particles.buffers();
        let device = vk_core.device();
        let vk_cmd_buffer = command_buffer.vk_cmd_buffer();
        
        gpu_profiler.profile_scope(device, vk_cmd_buffer, Integrator::profiling_label(), ||{
            self.integration_system.execute(
                vk_core,
                particles,
                delta_time,
                world_max,
                command_buffer,
            );
        });

       
       
        
        gpu_profiler.profile_scope(device, vk_cmd_buffer, HilbertEncoder::profiling_label(), ||{
            self.hilbert_encoder.dispatch(
                vk_core,
                particle_data.morton_codes_buffer.len() as u32,
                world_min,
                world_size,
                particle_data.positions_buffer.current(),
                &particle_data.morton_codes_buffer,
                &particle_data.object_indices_buffer,
                command_buffer,
            );
        });
        
        gpu_profiler.profile_scope(device, vk_cmd_buffer, GpuKVRadixSort::<u32>::profiling_label(), ||{
            self.sorting_system.sort(
                vk_core,
                &particle_data.morton_codes_buffer,
                &particle_data.object_indices_buffer,
                command_buffer,
                particle_data.morton_codes_buffer.len()
            );
        });
        
        gpu_profiler.profile_scope(device, vk_cmd_buffer, RearrangingSystem::profiling_label(), ||{
            self.rearranging_system
                .execute(vk_core, particle_data, command_buffer);
        });
        

        particles.buffers_mut().swap();

        gpu_profiler.profile_scope(device, vk_cmd_buffer, "Octree construction", ||{
            octree.build(vk_core, command_buffer, &particles.buffers().morton_codes_buffer, false);
        });

        let search_radius = self.physics_config.search_radius;
        
        gpu_profiler.profile_scope(device, vk_cmd_buffer, "Neighbor list construction", ||{
            neighbor_list.build(vk_core, command_buffer, octree, particles.buffers(), search_radius, world_min, world_size);
        });
       
        

        for _ in 0..self.physics_config.solver_iterations {

            
            gpu_profiler.profile_scope(device, vk_cmd_buffer, "Density compute", ||{
                self.density_compute_system.execute(
                    vk_core,
                    command_buffer,
                    octree,
                    neighbor_list,
                    particles,
                    &self.physics_config,
                );
            });
            
            
            gpu_profiler.profile_scope(device, vk_cmd_buffer, "Constraint solver", ||{
                self.constraint_solver_system.execute(
                    vk_core,
                    command_buffer,
                    octree,
                    neighbor_list,
                    particles,
                    &self.physics_config,
                );
            });

            particles.buffers_mut().positions_buffer.swap();
            
        }
        

        self.update_velocities_system.execute(
            vk_core,
            command_buffer,
            particles,
            delta_time,
            world_max,
        );

        /*
        self.velocity_refining_system.execute(
            vk_core,
            command_buffer,
            particles,
            spatial_grid,
            &self.physics_config,
        );
        
        particles.buffers_mut().velocities.swap();

        
        self.vorticity_force_compute_system.execute(
            vk_core,
            command_buffer,
            particles,
            spatial_grid,
            &self.physics_config,
        );
        */
            
         
        
            
        
    }
    
    /// Returns the number of solver iterations
    pub fn solver_iterations(&self) -> usize {
        self.physics_config.solver_iterations
    }
}
