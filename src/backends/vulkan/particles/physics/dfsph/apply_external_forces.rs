const SHADER: crate::backends::vulkan::runtime::shaders::ShaderCode =
    crate::shader!("particles/physics/dfsph/apply_external_forces");

use crate::backends::vulkan::particles::ParticleStorage;
use crate::backends::vulkan::runtime::commands::CommandBuffer;
use crate::backends::vulkan::runtime::commands::compute_buffer_barrier;
use crate::backends::vulkan::runtime::compute::ComputePass;
use crate::backends::vulkan::runtime::compute::ComputeSystemBuilder;
use crate::backends::vulkan::runtime::core::VulkanContext;
use crate::backends::vulkan::runtime::shaders::ShaderModule;
use crate::backends::vulkan::runtime::shaders::SpecializationConstants;
use crate::world::WorldBounds;
use crate::world::particles::physics_config::PhysicsConfig;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::Vec4Swizzles;
use std::sync::Arc;

pub struct ApplyExternalForces {
    apply_external_forces_compute_pass: ComputePass,
    #[allow(unused)]
    apply_external_forces_compute_shader: ShaderModule,
}

const THREAD_GROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Debug, Copy, Clone, Zeroable, Pod, Default)]
struct ApplyExternalPushConstants {
    positions: vk::DeviceAddress,
    velocities: vk::DeviceAddress,

    world_min: glam::Vec3,
    delta_time: f32,
    world_max: glam::Vec3,
    num_elements: u32,

    wall_repulsion_inverse_distance: f32,
    wall_repulsion_acceleration: f32,
    velocity_decay: f32,
    _padding: u32,
}

impl ApplyExternalForces {
    pub fn new(vk_context: &Arc<VulkanContext>) -> anyhow::Result<Self> {
        let (apply_external_forces_compute_pass, apply_external_forces_compute_shader) =
            ComputeSystemBuilder::new(vk_context.clone(), SHADER)
                .entry_points(&["main"])
                .specialization(SpecializationConstants::default().u32(THREAD_GROUP_SIZE))
                .push_constants::<ApplyExternalPushConstants>()
                .build_with_single_pass()?;
        Ok(Self {
            apply_external_forces_compute_pass,
            apply_external_forces_compute_shader,
        })
    }

    pub fn execute(
        &mut self,
        vk_context: &VulkanContext,
        command_buffer: &CommandBuffer,
        particles: &ParticleStorage,
        physics_config: &PhysicsConfig,
        world_bounds: &WorldBounds
    ) {
        let device = vk_context.device();
        let particle_data = particles.buffers();
        let num_elements = particle_data.hilbert_keys.len() as u32;

        // Dispatch exactly based on particle count
        let num_workgroups = (num_elements + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;
        
        let push_constants = ApplyExternalPushConstants {
            positions: particle_data.positions_buffer.current().address(),
            velocities: particle_data.velocities.current().address(),
            world_min: world_bounds.world_min.xyz(),
            delta_time: physics_config.time_step,
            world_max: world_bounds.world_min.xyz() + world_bounds.world_size,
            num_elements,
            wall_repulsion_inverse_distance: 1.0 / physics_config.wall_repulsion_distance.max(1e-6),
            wall_repulsion_acceleration: physics_config.wall_repulsion_acceleration,
            velocity_decay: (-physics_config.velocity_damping_rate.max(0.0) * physics_config.time_step).exp(),
            ..Default::default()
        };

        self.apply_external_forces_compute_pass.dispatch_compute(
            vk_context,
            command_buffer,
            [num_workgroups, 1, 1],
            &[],
            &[],
            bytemuck::bytes_of(&push_constants),
        );

        let buffer_barriers = [
            compute_buffer_barrier(
                particle_data.velocities.current().vk_buffer(),
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            ),
        ];
        command_buffer.pipeline_memory_barrier(device, &buffer_barriers, &[]);
    }
}

#[test]
fn shader_interface() {
    crate::backends::vulkan::runtime::shaders::shader_code::assert_interface(
        SHADER,
        5,
        &["main"],
        1,
    );
}
