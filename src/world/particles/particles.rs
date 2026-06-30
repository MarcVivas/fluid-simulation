use std::error::Error;
use std::sync::Arc;
use ash::vk;
use glam::{Vec3, Vec4};
use rand::Rng;
use crate::vulkan::core::VkCore;
use crate::vulkan::resources::buffer::{VkBuffer, PingPong};

pub struct Particles {
    total_particles: usize,
    #[allow(unused)]
    max_radius: f32,
    buffers: ParticleData,

}

pub struct ParticleData {
    pub positions_buffer: PingPong<VkBuffer<Vec4>>,
    pub previous_positions_buffer: PingPong<VkBuffer<Vec4>>,
    pub velocities: PingPong<VkBuffer<Vec4>>,
    pub lambdas: VkBuffer<f32>,
    pub densities: VkBuffer<f32>,
    pub vorticity: VkBuffer<glam::Vec4>,
    pub hilbert_keys: VkBuffer<u32>,
    pub particle_indexes: VkBuffer<u32>,
}

pub struct ParticleRenderData {
    pub positions_buffer: vk::Buffer,
    pub velocities: vk::Buffer,
    pub total_particles: usize,
}

impl ParticleData {
    pub fn swap(&mut self) {
        self.positions_buffer.swap();
        self.previous_positions_buffer.swap();
        self.velocities.swap();
    }
}

impl Particles {
    pub fn new(
        num_particles: usize,
        world_dim: &Vec3,
        vk_core: &Arc<VkCore>,
        cmd_pool: vk::CommandPool,
    ) -> Result<Self, Box<dyn Error>> {
        let mut random_number_generator = rand::rng();

        let mut max_radius :f32  = 0.0;
        
        let mut positions: Vec<Vec4> = Vec::with_capacity(num_particles);
        let mut previous_positions: Vec<Vec4> = Vec::with_capacity(num_particles);
        let mut velocities: Vec<Vec4> = Vec::with_capacity(num_particles);

        (0..num_particles/2).for_each(|_| {

            // Generate a random position
            let x_pos = random_number_generator.random_range(0.0..world_dim.x);
            let y_pos = random_number_generator.random_range(0.0..world_dim.y);
            let z_pos = random_number_generator.random_range(0.0..=world_dim.z);
            let radius = random_number_generator.random_range(0.35..=0.35) as f32;
            max_radius = max_radius.max(radius);
            let position = Vec4::new(x_pos, y_pos, z_pos, radius);
            positions.push(position);
            previous_positions.push(position);

            // Generate a random velocity
            static MAX_VELOCITY: f32 = 5.0;
            let x = random_number_generator.random_range(0.0..MAX_VELOCITY);
            let y = random_number_generator.random_range(0.0..MAX_VELOCITY);
            let z = random_number_generator.random_range(0.0..MAX_VELOCITY);
            let velocity = Vec4::new(x, y, z, 0.0);
            velocities.push(velocity);
        });

        (num_particles/2..num_particles).for_each(|_| {

            // Generate a random position
            let x_pos = random_number_generator.random_range(0.0..world_dim.x);
            let y_pos = random_number_generator.random_range(0.0..world_dim.y/2.0);
            let z_pos = random_number_generator.random_range(world_dim.z/2.0..=world_dim.z);
            let radius = random_number_generator.random_range(0.35..=0.35) as f32;
            max_radius = max_radius.max(radius);
            let position = Vec4::new(x_pos, y_pos, z_pos, radius);
            positions.push(position);
            previous_positions.push(position);

            // Generate a random velocity
            static MAX_VELOCITY: f32 = 5.0;
            let x = random_number_generator.random_range(0.0..MAX_VELOCITY);
            let y = random_number_generator.random_range(0.0..MAX_VELOCITY);
            let z = random_number_generator.random_range(0.0..MAX_VELOCITY);
            let velocity = Vec4::new(x, y, z, 0.0);
            velocities.push(velocity);
        });


        let particle_system_buffers = create_particle_data(
            vk_core,
            cmd_pool,
            *vk_core.compute_queue(),
            &positions,
            &previous_positions,
            &velocities,
        )?;

        Ok(
            Self {
                total_particles: num_particles,
                buffers: particle_system_buffers,
                max_radius
            }
        )
    }

    pub fn positions_buffer(&self) -> &VkBuffer<Vec4> {
        &self.buffers.positions_buffer.current()
    }

    pub fn len(&self) -> usize {
        self.total_particles
    }

    pub fn buffers(&self) -> &ParticleData {
        &self.buffers
    }

    pub fn buffers_mut(&mut self) -> &mut ParticleData {
        &mut self.buffers
    }

    pub fn max_radius(&self) -> f32 {
        self.max_radius
    }

    pub fn extract_render_data(&self) -> ParticleRenderData{
        ParticleRenderData {
            positions_buffer: self.buffers().positions_buffer.current().vk_buffer(),
            velocities: self.buffers().velocities.next().vk_buffer(),
            total_particles: self.total_particles,
        }
    }
}

fn create_particle_data(
    vk_core: &Arc<VkCore>,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    positions: &[Vec4],
    previous_positions: &[Vec4],
    velocities: &[Vec4],
) -> Result<ParticleData, Box<dyn Error>> {
    let len = positions.len();

    let positions_buffer = PingPong::new_vk_buffer(
        vk_core,
        positions,
        "Particle positions buffer",
        command_pool,
        queue
    )?;

    let previous_positions_buffer =  PingPong::new_vk_buffer(
        vk_core,
        previous_positions,
        "Particle previous positions buffer",
        command_pool,
        queue
    )?;

    let velocities = PingPong::new_vk_buffer(
        vk_core,
        velocities,
        "Particle velocities buffer",
        command_pool,
        queue
    )?;

    let density_constraints = VkBuffer::new_gpu_only_uninitialized(
        vk_core,
        len,
        "Particle density constraints buffer",
    )?;

    let densities = VkBuffer::new_gpu_only_uninitialized(
        vk_core,
        len,
        "Particle densities",
    )?;

    let vorticity = VkBuffer::new_gpu_only_uninitialized(
        vk_core,
        len,
        "Particles vorticity",
    )?;

    let hilbert_keys = VkBuffer::new_gpu_only_uninitialized(
        vk_core,
        len,
        "Particle hilbert keys",
    )?;

    let particle_indexes = VkBuffer::new_gpu_only_uninitialized(
        vk_core,
        len,
        "Particle indexes buffer",
    )?;

    let particle_system_buffers = ParticleData {
        positions_buffer,
        previous_positions_buffer,
        velocities,
        lambdas: density_constraints,
        densities,
        vorticity,
        hilbert_keys,
        particle_indexes,
    };

    Ok(particle_system_buffers)
}
