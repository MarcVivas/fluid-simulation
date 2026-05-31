use std::error::Error;
use std::sync::Arc;
use ash::vk;
use glam::{Vec3};
use rand::Rng;
use crate::components::{Density, FluidLambda, MortonCode, Position, Velocity, Vorticity};
use crate::vulkan::vk_core::VkCore;
use crate::utils::data_structures::{ping_pong::PingPong};
use crate::vulkan::vk_utils::{create_ping_pong_buffer, VkBuffer};

pub struct Particles {
    total_particles: usize,
    #[allow(unused)]
    max_radius: f32,
    buffers: ParticleData,

}

pub struct ParticleData {
    pub positions_buffer: PingPong<VkBuffer<Position>>,
    pub previous_positions_buffer: PingPong<VkBuffer<Position>>,
    pub velocities: PingPong<VkBuffer<Velocity>>,
    pub lambdas: VkBuffer<FluidLambda>,
    pub densities: VkBuffer<Density>,
    pub vorticity: VkBuffer<Vorticity>,
    pub morton_codes_buffer: VkBuffer<MortonCode>,
    pub object_indices_buffer: VkBuffer<u32>,
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

    /// Predicts which buffer will be the output after 'n' swaps
    pub fn predict_final_indices(&self, solver_iterations: usize) -> (usize, usize) {
        let mut pos_idx = self.positions_buffer.current_index();
        let mut vel_idx = self.velocities.current_index();

        // Simulate global swap
        pos_idx = (pos_idx + 1) % 2;
        vel_idx = (vel_idx + 1) % 2;

        // Simulate solver
        for _ in 0..solver_iterations {
            pos_idx = (pos_idx + 1) % 2;
        }

        // Simulate final velocity swap
        vel_idx = (vel_idx + 1) % 2;

        (pos_idx, vel_idx)

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

        let mut positions: Vec<Position> = Vec::with_capacity(num_particles);
        let mut previous_positions: Vec<Position> = Vec::with_capacity(num_particles);
        let mut velocities: Vec<Velocity> = Vec::with_capacity(num_particles);

        (0..num_particles/2).for_each(|_| {

            // Generate a random position
            let x_pos = random_number_generator.random_range(0.0..world_dim.x);
            let y_pos = random_number_generator.random_range(0.0..world_dim.y);
            let z_pos = random_number_generator.random_range(0.0..=world_dim.z);
            let radius = random_number_generator.random_range(0.35..=0.35) as f32;
            max_radius = max_radius.max(radius);
            let position = Position::new(x_pos, y_pos, z_pos, radius);
            positions.push(position);
            previous_positions.push(position);

            // Generate a random velocity
            static MAX_VELOCITY: f32 = 5.0;
            let x = random_number_generator.random_range(0.0..MAX_VELOCITY);
            let y = random_number_generator.random_range(0.0..MAX_VELOCITY);
            let z = random_number_generator.random_range(0.0..MAX_VELOCITY);
            let velocity = Velocity::new(x, y, z, 0.0);
            velocities.push(velocity);
        });

        (num_particles/2..num_particles).for_each(|_| {

            // Generate a random position
            let x_pos = random_number_generator.random_range(0.0..world_dim.x);
            let y_pos = random_number_generator.random_range(0.0..world_dim.y/2.0);
            let z_pos = random_number_generator.random_range(world_dim.z/2.0..=world_dim.z);
            let radius = random_number_generator.random_range(0.35..=0.35) as f32;
            max_radius = max_radius.max(radius);
            let position = Position::new(x_pos, y_pos, z_pos, radius);
            positions.push(position);
            previous_positions.push(position);

            // Generate a random velocity
            static MAX_VELOCITY: f32 = 5.0;
            let x = random_number_generator.random_range(0.0..MAX_VELOCITY);
            let y = random_number_generator.random_range(0.0..MAX_VELOCITY);
            let z = random_number_generator.random_range(0.0..MAX_VELOCITY);
            let velocity = Velocity::new(x, y, z, 0.0);
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

    pub fn positions_buffer(&self) -> &VkBuffer<Position> {
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

    pub fn extract_render_data(&self, solver_iterations: usize, compute_paused: bool) -> ParticleRenderData{
        if compute_paused{
            // Return current
            return ParticleRenderData {
                positions_buffer: self.buffers().positions_buffer.current().vk_buffer(),
                velocities: self.buffers.velocities.current().vk_buffer(),
                total_particles: self.total_particles
            };
        }
        let (pos_idx, vel_idx) = self.buffers.predict_final_indices(solver_iterations);
        let final_positions_buffer = self.buffers.positions_buffer.from_index(pos_idx);

        let final_velocities_buffer = self.buffers.velocities.from_index(vel_idx);
        ParticleRenderData {
            positions_buffer: final_positions_buffer.vk_buffer(),
            velocities: final_velocities_buffer.vk_buffer(),
            total_particles: self.total_particles,
        }
    }
}

fn create_particle_data(
    vk_core: &Arc<VkCore>,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    positions: &[Position],
    previous_positions: &[Position],
    velocities: &[Velocity],
) -> Result<ParticleData, Box<dyn Error>> {
    let len = positions.len();

    let positions_buffer = create_ping_pong_buffer(
        vk_core,
        positions,
        "Particle positions buffer",
        command_pool,
        queue
    )?;

    let previous_positions_buffer =  create_ping_pong_buffer(
        vk_core,
        previous_positions,
        "Particle previous positions buffer",
        command_pool,
        queue
    )?;

    let velocities = create_ping_pong_buffer(
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

    let morton_codes_buffer = VkBuffer::new_gpu_only_uninitialized(
        vk_core,
        len,
        "Particle morton codes buffer",
    )?;

    let object_indices_buffer = VkBuffer::new_gpu_only_uninitialized(
        vk_core,
        len,
        "Particle object indices buffer",
    )?;

    let particle_system_buffers = ParticleData {
        positions_buffer,
        previous_positions_buffer,
        velocities,
        lambdas: density_constraints,
        densities,
        vorticity,
        morton_codes_buffer,
        object_indices_buffer,
    };

    Ok(particle_system_buffers)
}
