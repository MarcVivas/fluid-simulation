use std::error::Error;
use std::sync::Arc;
use ash::vk;
use glam::{uvec3, Vec3};
use crate::vk_core::VkCore;
use crate::vk_utils::{VkBuffer, create_gpu_only_buffer};

pub struct SpatialGrid {
    grid_buffers: SpatialGridBuffers,
    cell_size: f32,
}

pub struct SpatialGridBuffers {
    pub cell_starts: VkBuffer,
    pub cell_ends: VkBuffer,
}

impl SpatialGridBuffers{
    pub fn map_capacity(&self) -> usize{
        self.cell_starts.len()
    }
}

impl SpatialGridBuffers {
    pub fn new(vk_core: &Arc<VkCore>, command_pool: vk::CommandPool, grid_size: &glam::UVec3) -> Result<Self, Box<dyn Error>>{
        
        
        // The capacity is the total cells in the grid
        let capacity = (grid_size.x * grid_size.y * grid_size.z);
        
        
        let cell_starts_vec = vec![0; capacity as usize];
        let cell_ends_vec = cell_starts_vec.clone();
        
        let cell_starts = create_gpu_only_buffer(
            vk_core,
            &cell_starts_vec,
            "Cell starts buffer",
            command_pool, 
            *vk_core.compute_queue()
        )?;
        
        let cell_ends = create_gpu_only_buffer(
            vk_core,
            &cell_ends_vec,
            "Cell ends buffer",
            command_pool, 
            *vk_core.compute_queue()
        )?;
        
        Ok(Self {cell_starts,cell_ends})
        
    }
}

impl SpatialGrid {
    pub fn new(vk_core: &Arc<VkCore>, command_pool: vk::CommandPool, max_radius: f32, world_size: &Vec3) -> Self {
        let cell_size = Self::compute_cell_size(max_radius);
        
        assert!(world_size.x == world_size.y && world_size.y == world_size.z);
        let grid_size = uvec3(
            ((world_size.x / cell_size).ceil() as u32).next_power_of_two(),
             ((world_size.y / cell_size).ceil() as u32).next_power_of_two(),
              ((world_size.z / cell_size).ceil() as u32).next_power_of_two()
        );
        let grid_buffers = SpatialGridBuffers::new(vk_core, command_pool, &grid_size).unwrap();
        Self{grid_buffers, cell_size}
    }
    
    fn compute_cell_size(max_radius: f32) -> f32{
        max_radius * 2.2
    }
    
    pub fn cell_size(&self) -> f32{
        self.cell_size
    }
    
    pub fn buffers(&self) -> &SpatialGridBuffers{
        &self.grid_buffers
    }
    
}
