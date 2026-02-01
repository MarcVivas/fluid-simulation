use std::error::Error;
use std::sync::Arc;
use ash::vk;
use ash::vk::ImageViewCreateInfo;
use glam::{uvec3, UVec3, Vec3};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{AllocationCreateDesc, AllocationScheme};
use crate::vk_core::VkCore;
use crate::vk_utils::{VkImage, ImageView};

pub struct SpatialGridBuffers {
    // The grid texture will store the cell indexes (cell_starts, cell_ends)
    pub grid_texture: VkImage,
    pub grid_texture_view: ImageView,
}

pub const EMPTY_CELL: u32 = 0xFFFFFFFF;

// If you change this, you have to change it as well in the SpatialGrid.slang shader
const NUM_VARIABLES_PER_CELL: u32 = 3;
impl SpatialGridBuffers {
    pub fn new(vk_core: &Arc<VkCore>, grid_size: &UVec3) -> Result<Self, Box<dyn Error>>{
        
        let extent = vk::Extent3D {
            // The width is multiplied by 3 because we need to store cell starts, ends and neighbors bits in the same buffer
            width: grid_size.x * NUM_VARIABLES_PER_CELL,
            height: grid_size.y,
            depth: grid_size.z,
        };

        let grid_texture = VkImage::new(
            vk_core,
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_3D)
                .extent(extent)
                .format(vk::Format::R32_UINT)
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            &AllocationCreateDesc{
                name: "Cell indexes",
                requirements: vk::MemoryRequirements::default(),
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged
            }
        )?;

        let grid_texture_view = ImageView::new(
            vk_core.clone(),
            &ImageViewCreateInfo::default()
                .image(grid_texture.vk_image())
                .view_type(vk::ImageViewType::TYPE_3D)
                .format(vk::Format::R32_UINT)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1)
                )
        )?;
        
        
        Ok(
            Self
            {
                grid_texture,
                grid_texture_view,
            }
        )
        
    }
    
    
}

pub struct SpatialGrid {
    grid_buffers: SpatialGridBuffers,
    cell_size: f32,
    grid_size: UVec3,
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
        let grid_buffers = SpatialGridBuffers::new(vk_core, &grid_size).unwrap();
        Self{grid_buffers, cell_size, grid_size}
    }
    
    fn compute_cell_size(max_radius: f32) -> f32{
        max_radius * 2.0
    }
    
    pub fn cell_size(&self) -> f32{
        self.cell_size
    }
    
    /// Returns the number of bits used to encode the morton code
    /// e.g. if grid size is 1024 
    /// 1024 = 2^10
    /// 1024.trailing_zeros() = 10
    /// 10 * number of dimensions = 30 bits
    pub fn num_bits_needed_for_morton_codes(&self) -> u32{
        self.grid_size.x.trailing_zeros() * 3
    }
    
    pub fn grid_size(&self) -> &UVec3{
        &self.grid_size
    }
    
    pub fn buffers(&self) -> &SpatialGridBuffers{
        &self.grid_buffers
    }
    
}
