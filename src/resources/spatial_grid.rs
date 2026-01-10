use std::error::Error;
use std::sync::Arc;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use crate::vk_core::VkCore;
use crate::vk_utils::{VkBuffer, create_gpu_only_buffer};

pub struct SpatialGrid {
    grid_buffers: SpatialGridBuffers,
    max_occupied_cells: u32,
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

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct KeyValue{
    key: u32,
    value: u32,
}

impl SpatialGridBuffers {
    pub fn new(vk_core: &Arc<VkCore>, max_occupied_cells: u32, command_pool: vk::CommandPool) -> Result<Self, Box<dyn Error>>{
        let buffer_length = max_occupied_cells * 2;
        
        let cell_starts_vec = vec![KeyValue{key: 0, value: 0}; buffer_length as usize];
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
    pub fn new(vk_core: &Arc<VkCore>, command_pool: vk::CommandPool, max_occupied_cells: u32, max_radius: f32) -> Self {
        let grid_buffers = SpatialGridBuffers::new(vk_core, max_occupied_cells, command_pool).unwrap();
        let cell_size = Self::compute_cell_size(max_radius);
        Self{grid_buffers, max_occupied_cells, cell_size}
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
