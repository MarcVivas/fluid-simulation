use std::sync::Arc;
use ash::prelude::VkResult;
use ash::vk;
use ash::vk::{DescriptorSetLayoutBinding, PushConstantRange};
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{AllocationCreateDesc, AllocationScheme};
use crate::compute::{ComputeCommandPool, ComputePass};
use crate::systems::IntegrationPushConstants;
use crate::vk_core::VkCore;
use crate::vk_utils::{CommandBuffer, DescriptorSetLayoutConfig, PipelineLayout, ShaderModule};
use crate::vk_utils::vk_buffer::VkBuffer;

const BITS_PER_PASS: u32 = 4;
const BIN_COUNT: u32 = 1 << BITS_PER_PASS;
const ELEMENTS_PER_THREAD: u32 = 4;
const THREADS_PER_GROUP: u32 = 128;
const NUM_PASSES: u32 = 32 / BITS_PER_PASS; // 8 passes

const BLOCK_SIZE: u32 = THREADS_PER_GROUP * ELEMENTS_PER_THREAD;

const MAX_THREAD_GROUPS: u32 = 800;

pub struct SortingSystem {
    counting_pass: ComputePass,
    scan_pass: ComputePass,
    scatter_pass: ComputePass,
    reduce_pass: ComputePass,
    scan_add_pass: ComputePass,
    sorting_shader: ShaderModule,
    sorting_data: SortingData
}

struct SortingData {
    num_keys: u32,
    histogram_buffer: VkBuffer,
    keys_b: VkBuffer,
    payload_b: VkBuffer,
    reduce_table: VkBuffer,
    scan_scratch: VkBuffer,
}

impl SortingData {
    pub fn new(vk_core: &Arc<VkCore>, command_pool: vk::CommandPool, max_keys: u32) -> Result<Self, Box<dyn std::error::Error>> {

        let histogram = vec![0u32; Self::calculate_histogram_len(max_keys) as usize];
        let keys_b = vec![0u32; max_keys as usize];
        let payload_b = vec![0u32; max_keys as usize];
        let reduce_table = vec![0u32; Self::calculate_reduce_table_len(max_keys) as usize];
        let scan_scratch = vec![0u32; Self::calculate_scan_scratch_len(max_keys) as usize];
        
        let histogram_buffer = VkBuffer::new(
            vk_core,
            &histogram,
            vk::BufferCreateInfo::default()
                .size((size_of::<u32>() * histogram.len()) as vk::DeviceSize)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            AllocationCreateDesc{
                name: "Histogram buffer",
                requirements: vk::MemoryRequirements::default(),
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged
            },
            command_pool,
            *vk_core.compute_queue()
        )?;
        
        let keys_b_buffer = VkBuffer::new(
            vk_core,
            &keys_b,
            vk::BufferCreateInfo::default()
                .size((size_of::<u32>() * keys_b.len()) as vk::DeviceSize)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            AllocationCreateDesc{
                name: "Keys b buffer",
                requirements: vk::MemoryRequirements::default(),
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged
            },
            command_pool,
            *vk_core.compute_queue()
        )?;
        let payload_b_buffer = VkBuffer::new(
            vk_core,
            &payload_b,
            vk::BufferCreateInfo::default()
                .size((size_of::<u32>() * payload_b.len()) as vk::DeviceSize)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            AllocationCreateDesc{
                name: "Payload b buffer",
                requirements: vk::MemoryRequirements::default(),
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged
            },
            command_pool,
            *vk_core.compute_queue()
        )?;
        
        let reduce_table_buffer = VkBuffer::new(
            vk_core,
            &reduce_table,
            vk::BufferCreateInfo::default()
                .size((size_of::<u32>() * reduce_table.len()) as vk::DeviceSize)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            AllocationCreateDesc{
                name: "Reduce table buffer",
                requirements: vk::MemoryRequirements::default(),
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged
            },
            command_pool,
            *vk_core.compute_queue()
        )?;

        let scan_scratch_buffer = VkBuffer::new(
            vk_core,
            &scan_scratch,
            vk::BufferCreateInfo::default()
                .size((size_of::<u32>() * scan_scratch.len()) as vk::DeviceSize)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            AllocationCreateDesc{
                name: "Reduce table buffer",
                requirements: vk::MemoryRequirements::default(),
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged
            },
            command_pool,
            *vk_core.compute_queue()
        )?;
        
        Ok(
            Self{
                num_keys: max_keys,
                histogram_buffer,
                keys_b: keys_b_buffer,
                payload_b: payload_b_buffer,
                reduce_table: reduce_table_buffer,
                scan_scratch: scan_scratch_buffer
            }  
        )
        
    }

    pub fn calculate_histogram_len(num_keys: u32) -> u64 {
        let block_size = ELEMENTS_PER_THREAD * THREADS_PER_GROUP;

        // Calculate the number of blocks (ceiling division)
        let num_blocks = (num_keys + block_size - 1) / block_size;

        // Size in bytes: 16 bins * NumBlocks * 4 bytes (u32)
        (BIN_COUNT * num_blocks) as u64
    }
    
    pub fn calculate_reduce_table_len(num_keys: u32) -> u64 {
        let block_size = ELEMENTS_PER_THREAD * THREADS_PER_GROUP;
        let num_blocks = (num_keys + block_size - 1) / block_size;
        let num_reduced_blocks = (num_blocks + block_size - 1) / block_size;

        let reduce_size = BIN_COUNT as u64 * num_reduced_blocks as u64;
        reduce_size
    }
    
    pub fn calculate_scan_scratch_len(num_keys: u32) -> u64 {
        Self::calculate_reduce_table_len(num_keys)
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Zeroable, Pod)]
struct SortingPushConstants {
    pub num_keys: u32,
    pub num_blocks_per_thread_group: i32,
    pub num_thread_groups: u32,
    pub num_thread_groups_with_additional_blocks: u32,
    pub num_reduce_thread_group_per_bin: u32,
    pub num_scan_values: u32,
    pub shift: u32,
    pub padding: u32, // Keep alignment happy
}

impl SortingSystem {
    pub fn new(vk_core: &Arc<VkCore>, compute_command_pool: &ComputeCommandPool, max_keys: u32) -> Result<Self, Box<dyn std::error::Error>> {
        let sorting_data = SortingData::new(vk_core, compute_command_pool.vk_cmd_pool(), max_keys)?;

        let parallel_sort_shader_module = ShaderModule::new(vk_core.clone(), "parallel_sort");

        let count_shader_stage_create_infos = vk::PipelineShaderStageCreateInfo::default()
            .module(parallel_sort_shader_module.vk_shader_module())
            .name(c"count")
            .stage(vk::ShaderStageFlags::COMPUTE);
        
        let scan_shader_stage_create_infos = vk::PipelineShaderStageCreateInfo::default()
            .module(parallel_sort_shader_module.vk_shader_module())
            .name(c"scan_reduce_table")
            .stage(vk::ShaderStageFlags::COMPUTE);
        
        let scatter_shader_stage_create_infos = vk::PipelineShaderStageCreateInfo::default()
            .module(parallel_sort_shader_module.vk_shader_module())
            .name(c"scatter")
            .stage(vk::ShaderStageFlags::COMPUTE);
        
        let reduce_shader_stage_create_infos = vk::PipelineShaderStageCreateInfo::default()
            .module(parallel_sort_shader_module.vk_shader_module())
            .name(c"reduce")
            .stage(vk::ShaderStageFlags::COMPUTE);
        
        let scan_add_shader_stage_create_infos = vk::PipelineShaderStageCreateInfo::default()
            .module(parallel_sort_shader_module.vk_shader_module())
            .name(c"scan_add")
            .stage(vk::ShaderStageFlags::COMPUTE);
        
        let bindings = [
            // Source key
            DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Destination key
            DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Source payload
            DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Destination payload
            DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Scratch buffer for histogram
            DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Scratch buffer reduce table
            DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Scan scratch buffer
            DescriptorSetLayoutBinding::default()
                .binding(6)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let descriptor_set_layout_config = [DescriptorSetLayoutConfig{
            bindings: &bindings,
            flags: Some(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
        }];

        let push_constant_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(size_of::<SortingPushConstants>() as u32)];

       
        
        let counting_pass = Self::create_pass(vk_core, compute_command_pool, count_shader_stage_create_infos, &descriptor_set_layout_config, &push_constant_ranges)?;
        let scan_pass = Self::create_pass(vk_core, compute_command_pool, scan_shader_stage_create_infos, &descriptor_set_layout_config, &push_constant_ranges)?;
        let scatter_pass = Self::create_pass(vk_core, compute_command_pool, scatter_shader_stage_create_infos, &descriptor_set_layout_config, &push_constant_ranges)?;
        let reduce_pass = Self::create_pass(vk_core, compute_command_pool, reduce_shader_stage_create_infos, &descriptor_set_layout_config, &push_constant_ranges)?;
        let scan_add_pass = Self::create_pass(vk_core, compute_command_pool, scan_add_shader_stage_create_infos, &descriptor_set_layout_config, &push_constant_ranges)?;
        

        Ok(
            Self {
                sorting_shader: parallel_sort_shader_module,
                counting_pass,
                scan_pass,
                scatter_pass,
                reduce_pass,
                scan_add_pass,
                sorting_data
            }
        )
    }
    
    fn create_pass(vk_core: &Arc<VkCore>, compute_command_pool: &ComputeCommandPool, pipeline_info: vk::PipelineShaderStageCreateInfo, descriptor_set_layout_config: &[DescriptorSetLayoutConfig], push_constant_ranges: &[PushConstantRange]) -> VkResult<ComputePass> {
        let pipeline_layout = PipelineLayout::new(
            vk_core.clone(),
            descriptor_set_layout_config,
            push_constant_ranges
        )?;
        
        let compute_pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(pipeline_info)
            .layout(pipeline_layout.vk_pipeline_layout());
        
        let count_pipeline = unsafe {
            vk_core.device().create_compute_pipelines(
                vk::PipelineCache::null(),
                &[compute_pipeline_info],
                None
            ).expect("Failed to create compute pipeline")[0]
        };

        let compute_pass = ComputePass::new(
            vk_core.clone(),
            compute_command_pool,
            count_pipeline,
            pipeline_layout,
        );
        
        compute_pass
        
    }
    
    pub fn sort(&self, vk_core: &Arc<VkCore>, keys: &VkBuffer, payload: &VkBuffer, compute_command_pool: &ComputeCommandPool){
        for i in 0..NUM_PASSES {
            let shift = i * BITS_PER_PASS;

            let (src_keys, dst_keys, src_payload, dst_payload) = 
                if i % 2 == 0 {
                    (keys, &self.sorting_data.keys_b, payload, &self.sorting_data.payload_b)
                }
                else {
                    (&self.sorting_data.keys_b, keys, &self.sorting_data.payload_b, payload)
                };
            
            // Calculate dispatch info
            let push_constants = get_dispatch_data(self.sorting_data.num_keys, shift);
            let push_constants_bytes = bytemuck::bytes_of(&push_constants);
            
            let descriptor_buffer_infos = [
                vk::DescriptorBufferInfo::default().buffer(src_keys.vk_buffer()).range(vk::WHOLE_SIZE),
                vk::DescriptorBufferInfo::default().buffer(dst_keys.vk_buffer()).range(vk::WHOLE_SIZE),
                vk::DescriptorBufferInfo::default().buffer(src_payload.vk_buffer()).range(vk::WHOLE_SIZE),
                vk::DescriptorBufferInfo::default().buffer(dst_payload.vk_buffer()).range(vk::WHOLE_SIZE),
                vk::DescriptorBufferInfo::default().buffer(self.sorting_data.histogram_buffer.vk_buffer()).range(vk::WHOLE_SIZE),
                vk::DescriptorBufferInfo::default().buffer(self.sorting_data.reduce_table.vk_buffer()).range(vk::WHOLE_SIZE),
                vk::DescriptorBufferInfo::default().buffer(self.sorting_data.scan_scratch.vk_buffer()).range(vk::WHOLE_SIZE),
            ];

            let descriptor_writes = [
                // Assuming bindings 0-6 are contiguous in the set layout
                vk::WriteDescriptorSet::default()
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&descriptor_buffer_infos[0..7]),
            ];
            
            
            
            // Pass 1: Count bits. Generates the histogram of every block
            // 4 bits per pass -> 2^4=16 possible bins, binary numbers
            self.counting_pass.execute(
                [push_constants.num_thread_groups, 1, 1],
                &[],
                |device, cmd_buffer, pipeline_layout |{
                    cmd_buffer.push_constants(
                        device,
                        pipeline_layout.vk_pipeline_layout(),
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        push_constants_bytes
                    );

                    // PUSH the descriptor directly
                    unsafe {
                        vk_core.push_descriptor().cmd_push_descriptor_set(
                            cmd_buffer.vk_cmd_buffer(),
                            vk::PipelineBindPoint::COMPUTE,
                            pipeline_layout.vk_pipeline_layout(),
                            0,
                            &descriptor_writes,
                        );
                    }
                    barrier(vk_core, cmd_buffer);
                }
            );
            
            
            // Pass 2: Reduce (For large arrays). Group blocks together (e.g 8 blocks = 1 big block) and sum their histograms
            // This reads the large SumTable and creates a smaller "ReduceTable"
            self.reduce_pass.execute(
                [push_constants.num_scan_values, 1, 1],
                &[],
                |device, cmd_buffer, pipeline_layout| {
                    // 1. Push Constants 
                    // (Must update because we are in a new pipeline bind, though data is same)
                    cmd_buffer.push_constants(
                        device,
                        pipeline_layout.vk_pipeline_layout(),
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        push_constants_bytes
                    );

                    // 2. Push Descriptors
                    unsafe {
                        vk_core.push_descriptor().cmd_push_descriptor_set(
                            cmd_buffer.vk_cmd_buffer(),
                            vk::PipelineBindPoint::COMPUTE,
                            pipeline_layout.vk_pipeline_layout(),
                            0,
                            &descriptor_writes,
                        );
                    }

                    // 3. Barrier
                    // Critical: The next pass (Scan) needs to read the ReduceTable we just wrote.
                    barrier(vk_core, cmd_buffer);
                }
            );
            
            
            // Pass 3: Scan to know where each bin starts globally
            // Scans the ReduceTable.
            self.scan_pass.execute(
                [1, 1, 1],
                &[],
                |device, cmd, layout| {
                    cmd.push_constants(device, layout.vk_pipeline_layout(), vk::ShaderStageFlags::COMPUTE, 0, push_constants_bytes);
                    unsafe {
                        vk_core.push_descriptor().cmd_push_descriptor_set(
                            cmd.vk_cmd_buffer(), vk::PipelineBindPoint::COMPUTE,
                            layout.vk_pipeline_layout(), 0, &descriptor_writes
                        );
                    }
                    barrier(vk_core, cmd);
                }
            );
            
            
            // Pass 4: Scan and add. Convert the big block offsets back to the smaller blocks
            self.scan_add_pass.execute(
                [push_constants.num_scan_values, 1, 1],
                &[],
                |device, cmd, layout| {
                    cmd.push_constants(device, layout.vk_pipeline_layout(), vk::ShaderStageFlags::COMPUTE, 0, push_constants_bytes);
                    unsafe {
                        vk_core.push_descriptor().cmd_push_descriptor_set(
                            cmd.vk_cmd_buffer(), vk::PipelineBindPoint::COMPUTE,
                            layout.vk_pipeline_layout(), 0, &descriptor_writes
                        );
                    }
                    barrier(vk_core, cmd);
                }
            );
            
            
            // Pass 5: Scatter. Read the offsets from the scan and move the keys and payloads to the correct position.
            self.scatter_pass.execute(
                [push_constants.num_thread_groups, 1, 1],
                &[],
                |device, cmd, layout| {
                    cmd.push_constants(device, layout.vk_pipeline_layout(), vk::ShaderStageFlags::COMPUTE, 0, push_constants_bytes);
                    unsafe {
                        vk_core.push_descriptor().cmd_push_descriptor_set(
                            cmd.vk_cmd_buffer(), vk::PipelineBindPoint::COMPUTE,
                            layout.vk_pipeline_layout(), 0, &descriptor_writes
                        );
                    }
                    barrier(vk_core, cmd);
                }
            );
        }

        let keys = keys.read_back::<u32>(vk_core, compute_command_pool.vk_cmd_pool()).unwrap();
        let is_sorted = keys.is_sorted();
        assert!(is_sorted);
    }
}

fn get_dispatch_data(num_keys: u32, shift: u32) -> SortingPushConstants {

    let total_blocks = calculate_num_blocks(num_keys, BLOCK_SIZE);

    // 1. Cap the thread groups
    let mut num_thread_groups = total_blocks;
    if num_thread_groups > MAX_THREAD_GROUPS {
        num_thread_groups = MAX_THREAD_GROUPS;
    }

    let blocks_per_thread_group = total_blocks / num_thread_groups;
    let mut num_thread_groups_with_additional_blocks = total_blocks % num_thread_groups;

    if total_blocks < MAX_THREAD_GROUPS {
        num_thread_groups_with_additional_blocks = 0;
    }

    // 2. Calculate Reduce / Scan parameters
    // We need one reduced value for every BIN.
    // Even if we only have 1 thread group, we have 16 Bins, so we need 16 reduced groups.
    let num_reduce_thread_groups = BIN_COUNT * ((num_thread_groups + BLOCK_SIZE - 1) / BLOCK_SIZE);
    let num_reduce_thread_group_per_bin = num_reduce_thread_groups / BIN_COUNT;

    SortingPushConstants {
        num_keys,
        num_blocks_per_thread_group: blocks_per_thread_group as i32,
        num_thread_groups,
        num_thread_groups_with_additional_blocks,
        num_reduce_thread_group_per_bin,
        num_scan_values: num_reduce_thread_groups, // Was num_thread_groups
        shift,
        padding: 0
    }
}

fn calculate_num_blocks(num_keys: u32, block_size: u32) -> u32 {
    (num_keys + block_size - 1) / block_size
}

fn barrier(vk_core: &Arc<VkCore>, cmd_buffer: &CommandBuffer){

    
    let memory_barrier = vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE);    
    let dependency_info = vk::DependencyInfo::default()
        .memory_barriers(std::slice::from_ref(&memory_barrier));
    
    cmd_buffer.pipeline_barrier2(vk_core.device(), &dependency_info);
    
}