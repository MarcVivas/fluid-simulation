use std::sync::Arc;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use crate::vulkan::compute::{ComputePass, ComputeSystemBuilder};
use crate::simulation::physics_config::PhysicsConfig;
use crate::simulation::neighbor_list::NeighborList;
use crate::simulation::octree::octree::Octree;
use crate::vulkan::shaders::{ShaderCompileTimeConstants, ShaderModule};
use crate::world::{particles::Particles};
use crate::vulkan::core::VkCore;
use crate::vulkan::resources::{CommandBuffer, compute_buffer_barrier};

pub struct ConstraintSolver {
    constraint_solver_pass: ComputePass,
    #[allow(unused)]
    constraint_solver_shader: ShaderModule,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, Default)]
struct ConstraintSolverPushConstants {
    super_clusters: u64,
    super_cluster_neighbors: u64,
    leaf_particles: u64,
    unsorted_leaf_particles: u64,
    leaf_count: u64,
    particle_to_neighborhood: u64, 
    num_workgroups: u32,
    num_elements: u32,
    kernel_radius: f32,
    rest_density: f32,
    reversed_rest_density: f32,
    poly6_constant: f32,
    kernel_radius_2: f32,
    spiky_constant: f32,
    k: f32,
    delta_q_squared: f32,
    n: u32,
    _padding: u32
}

impl ConstraintSolver {
    pub fn new(vk_core: &Arc<VkCore>, super_cluster_size: u32) -> Result<Self, Box<dyn std::error::Error>> {
        let (constraint_solver_pass, constraint_solver_shader) = ComputeSystemBuilder::new(vk_core.clone(), "constraint_solver")
            .entry_points(&["main"])
            .compile_time_constants(
                ShaderCompileTimeConstants::new()
                    .add("THREAD_GROUP_SIZE", super_cluster_size)
            )
            .push_constants::<ConstraintSolverPushConstants>()
            // Read positions 
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Write positions
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Densities
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Fluid lambdas
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            .build_with_single_pass()?;

        Ok(Self { constraint_solver_pass, constraint_solver_shader })
    }

    pub fn execute(&self, vk_core: &Arc<VkCore>, command_buffer: &CommandBuffer, octree: &Octree, neighbor_list: &NeighborList, particles: &Particles, physics_config: &PhysicsConfig) {
        let device = vk_core.device();
        let num_elements = particles.len() as u32;


        let particle_buffers = particles.buffers();
        let (read_positions, write_positions) = particle_buffers.positions_buffer.read_write();
        let (read_positions, write_positions) = (read_positions.vk_buffer(), write_positions.vk_buffer());
        let densities = particle_buffers.densities.vk_buffer();
        let lambdas = particle_buffers.lambdas.vk_buffer();


        let thread_group_size = neighbor_list.super_cluster_size();
        let num_workgroups = (num_elements + thread_group_size - 1) / thread_group_size;
        
        let push_constants = ConstraintSolverPushConstants {
            super_clusters: neighbor_list.super_clusters().address(),
            super_cluster_neighbors: neighbor_list.super_cluster_neighbors().address(),
            particle_to_neighborhood: neighbor_list.particle_to_neighborhood().address(),
            num_elements,
            kernel_radius: physics_config.kernel_radius,
            rest_density: physics_config.rest_density,
            reversed_rest_density: physics_config.reversed_rest_density,
            poly6_constant: physics_config.kernel_poly6,
            kernel_radius_2: physics_config.kernel_radius_2,
            spiky_constant: physics_config.kernel_spiky_grad,
            k: physics_config.k,
            delta_q_squared: physics_config.delta_q_squared,
            n: physics_config.n,
            unsorted_leaf_particles: octree.data().unsorted_leaf_particles().address(),
            leaf_count: octree.data().leaf_count().address(),
            num_workgroups,
            leaf_particles: octree.data().leaf_particles().address(),
            ..Default::default()
        };
        
        let buffers = [read_positions, write_positions, densities, lambdas];
        let images = [];


        self.constraint_solver_pass.dispatch_compute(
            vk_core,
            command_buffer,
            [num_workgroups, 1, 1],
            &buffers,
            &images,
            bytemuck::bytes_of(&push_constants)
        );
        
        
        let buffer_barriers = [
            compute_buffer_barrier(
                read_positions,
                vk::AccessFlags2::SHADER_STORAGE_READ,
                vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::SHADER_STORAGE_READ,
            ),
            compute_buffer_barrier(
                write_positions,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            )
        ];

       
        command_buffer.pipeline_memory_barrier(device, &buffer_barriers, &[]);
    }
    
}
