use ash::vk;

/// Vulkan render input produced by the Vulkan simulation backend.
pub struct ParticleRenderInput {
    pub positions_buffer: vk::Buffer,
    pub velocities_buffer: vk::Buffer,
    pub positions_address: vk::DeviceAddress,
    pub velocities_address: vk::DeviceAddress,
    pub total_particles: usize,
}
