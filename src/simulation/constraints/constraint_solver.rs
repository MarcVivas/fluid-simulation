use std::sync::Arc;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use crate::vulkan::compute::{ComputePass, ComputeSystemBuilder};
use crate::simulation::physics_config::PhysicsConfig;
use crate::simulation::neighbor_list::NeighborList;
use crate::vulkan::shaders::{ShaderCompileTimeConstants, ShaderModule};
use crate::world::{particles::Particles};
use crate::vulkan::core::VulkanContext;
use crate::vulkan::commands::{CommandBuffer, compute_buffer_barrier};

pub struct ConstraintSolver {
    constraint_solver_pass: ComputePass,
    #[allow(unused)]
    constraint_solver_shader: ShaderModule,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, Default)]
struct ConstraintSolverPushConstants {
    src_positions: vk::DeviceAddress,
    dst_positions: vk::DeviceAddress,
    densities: vk::DeviceAddress,
    lambdas: vk::DeviceAddress,
    particle_to_neighborhood: vk::DeviceAddress, 
    neighbor_particle_indices: vk::DeviceAddress,
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

const THREAD_GROUP_SIZE: u32 = 64;

impl ConstraintSolver {
    pub fn new(vk_core: &Arc<VulkanContext>) -> anyhow::Result<Self> {
        let (constraint_solver_pass, constraint_solver_shader) = ComputeSystemBuilder::new(vk_core.clone(), "constraint_solver")
            .entry_points(&["main"])
            .compile_time_constants(
                ShaderCompileTimeConstants::new()
                    .add("THREAD_GROUP_SIZE", THREAD_GROUP_SIZE)
            )
            .push_constants::<ConstraintSolverPushConstants>()
            .build_with_single_pass()?;

        Ok(Self { constraint_solver_pass, constraint_solver_shader })
    }

    pub fn execute(&self, vk_core: &VulkanContext, command_buffer: &CommandBuffer, neighbor_list: &NeighborList, particles: &Particles, physics_config: &PhysicsConfig) {
        let device = vk_core.device();
        let num_elements = particles.len() as u32;


        let particle_buffers = particles.buffers();
        let (read_positions, write_positions) = particle_buffers.positions_buffer.read_write();
        let densities = &particle_buffers.densities;
        let lambdas = &particle_buffers.lambdas;


        let num_workgroups = (num_elements + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;
        
        let push_constants = ConstraintSolverPushConstants {
            dst_positions: write_positions.address(),
            src_positions: read_positions.address(),
            densities: densities.address(),
            lambdas: lambdas.address(),
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
            num_workgroups,
            neighbor_particle_indices: neighbor_list.neighbor_particle_indices().address(),
            ..Default::default()
        };

        self.constraint_solver_pass.dispatch_compute(
            vk_core,
            command_buffer,
            [num_workgroups, 1, 1],
            &[],
            &[],
            bytemuck::bytes_of(&push_constants)
        );
        
        
        let buffer_barriers = [
            compute_buffer_barrier(
                read_positions.vk_buffer(),
                vk::AccessFlags2::SHADER_STORAGE_READ,
                vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::SHADER_STORAGE_READ,
            ),
            compute_buffer_barrier(
                write_positions.vk_buffer(),
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            )
        ];

       
        command_buffer.pipeline_memory_barrier(device, &buffer_barriers, &[]);
    }
    
}
