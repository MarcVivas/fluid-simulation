use ash::vk;
use crate::algorithms::hilbert_encoding::HilbertEncoder;
use crate::algorithms::sorting::kv_radix_sort::GpuKVRadixSort;
use crate::simulation::constraints::ConstraintSolver;
use crate::simulation::fluids::density_compute::DensityCompute;
use crate::simulation::fluids::velocity_refiner::VelocityRefiner;
use crate::simulation::fluids::vorticity_force_compute::VorticityForceCompute;
use crate::simulation::integration::{Integrator, VelocityUpdater};
use crate::simulation::neighbor_list::NeighborList;
use crate::simulation::octree::octree::Octree;
use crate::simulation::physics_config::PhysicsConfig;
use crate::vulkan::frame::frame_pacer::FramePacer;
use crate::vulkan::profiler::GpuProfiler;
use crate::vulkan::commands::CommandBuffer;
use crate::world::{particles::Particles, particles::ParticleReorderer};

use crate::vulkan::shaders::traits::{GpuTask};

use crate::vulkan::core::VulkanContext;
use std::sync::Arc;

pub struct PhysicsEngine {
    physics_config: PhysicsConfig,
    sorter: GpuKVRadixSort<u32>,
    integrator: Integrator,
    particle_reorderer: ParticleReorderer,
    density_compute: DensityCompute,
    constraint_solver: ConstraintSolver,
    velocity_updater: VelocityUpdater,
    velocity_refiner: VelocityRefiner,
    vorticity_force_compute: VorticityForceCompute,
    hilbert_encoder: HilbertEncoder
}

impl PhysicsEngine {
    pub fn new(
        vk_core: &Arc<VulkanContext>,
        cmd_pool: vk::CommandPool,
        particles: &Particles,
        max_levels: u32,
        search_radius: f32,
    ) -> anyhow::Result<Self> {

        let max_objects: u32 = particles.len() as u32;
        let integrator = Integrator::new(vk_core)?;
        let sorter = GpuKVRadixSort::new(vk_core, cmd_pool, max_objects, None)?;
        let particle_reorderer = ParticleReorderer::new(vk_core)?;
        let density_compute = DensityCompute::new(vk_core)?;
        let constraint_solver = ConstraintSolver::new(vk_core)?;
        let velocity_updater = VelocityUpdater::new(vk_core)?;
        let velocity_refiner = VelocityRefiner::new(vk_core)?;
        let vorticity_force_compute = VorticityForceCompute::new(vk_core)?;
        let hilbert_encoder = HilbertEncoder::new(vk_core, max_levels)?;
        
        let physics_config = PhysicsConfig::new(search_radius);

        Ok(Self {
            physics_config,
            integrator,
            sorter,
            particle_reorderer,
            constraint_solver,
            velocity_updater,
            density_compute,
            velocity_refiner,
            vorticity_force_compute,
            hilbert_encoder
        })
    }

    pub fn update(
        &mut self,
        vk_core: &VulkanContext,
        command_buffer: &CommandBuffer,
        particles: &mut Particles,
        world_size: f32,
        world_min: glam::Vec4,
        octree: &mut Octree,
        neighbor_list: &NeighborList,
        gpu_profiler: &GpuProfiler,
        frame_pacer: &FramePacer
    ) { 
        
        let delta_time = self.physics_config.time_step;
        let world_max = world_min + world_size;
        let world_max = &glam::Vec3::new(world_max.x, world_max.y, world_max.z);


        let particle_data = particles.buffers();
        let device = vk_core.device();
        let vk_cmd_buffer = command_buffer.vk_cmd_buffer();
        let ring_idx = frame_pacer.ring_index();
        
        gpu_profiler.profile_scope(device, vk_cmd_buffer, Integrator::profiling_label(), ring_idx, ||{
            self.integrator.execute(
                vk_core,
                particles,
                delta_time,
                world_max,
                command_buffer,
            );
        });

       
       
        
        gpu_profiler.profile_scope(device, vk_cmd_buffer, HilbertEncoder::profiling_label(), ring_idx, ||{
            self.hilbert_encoder.dispatch(
                vk_core,
                particle_data.hilbert_keys.len() as u32,
                world_min,
                world_size,
                particle_data.positions_buffer.current(),
                &particle_data.hilbert_keys,
                &particle_data.particle_indexes,
                command_buffer,
            );
        });
        
        gpu_profiler.profile_scope(device, vk_cmd_buffer, GpuKVRadixSort::<u32>::profiling_label(), ring_idx, ||{
            self.sorter.sort(
                vk_core,
                &particle_data.hilbert_keys,
                &particle_data.particle_indexes,
                command_buffer,
                particle_data.hilbert_keys.len()
            );
        });
        
        gpu_profiler.profile_scope(device, vk_cmd_buffer, ParticleReorderer::profiling_label(), ring_idx, ||{
            self.particle_reorderer
                .execute(vk_core, particle_data, command_buffer);
        });
        

        particles.buffers_mut().swap();

        gpu_profiler.profile_scope(device, vk_cmd_buffer, "Octree construction", ring_idx, ||{
            octree.build(vk_core, command_buffer, &particles.buffers().hilbert_keys, false, world_min, world_size);
        });

        let search_radius = self.physics_config.search_radius;
        
        gpu_profiler.profile_scope(device, vk_cmd_buffer, "Neighbor list construction", ring_idx, ||{
            neighbor_list.build(vk_core, command_buffer, octree, particles.buffers(), search_radius, world_min, world_size);
        });
       
        

        for i in 0..self.physics_config.solver_iterations {

            let density_label = format!("Density compute {}", i);
            gpu_profiler.profile_scope(device, vk_cmd_buffer, &density_label, ring_idx, ||{
                self.density_compute.execute(
                    vk_core,
                    command_buffer,
                    neighbor_list,
                    particles,
                    &self.physics_config,
                );
            });
            
            let constraint_label = format!("Constraint solver {}", i);
            
            gpu_profiler.profile_scope(device, vk_cmd_buffer, &constraint_label, ring_idx, ||{
                self.constraint_solver.execute(
                    vk_core,
                    command_buffer,
                    neighbor_list,
                    particles,
                    &self.physics_config,
                );
            });

            particles.buffers_mut().positions_buffer.swap();
            
        }
        

        self.velocity_updater.execute(
            vk_core,
            command_buffer,
            particles,
            delta_time,
            world_max,
        );

       
    
        self.velocity_refiner.execute(
            vk_core,
            command_buffer,
            particles,
            neighbor_list,
            &self.physics_config,
        );
        
        particles.buffers_mut().velocities.swap();
  
       
        
        self.vorticity_force_compute.execute(
            vk_core,
            command_buffer,
            particles,
            neighbor_list,
            &self.physics_config,
        ); 
        
        
    }
    
    /// Returns the number of solver iterations
    pub fn solver_iterations(&self) -> usize {
        self.physics_config.solver_iterations
    }
}
