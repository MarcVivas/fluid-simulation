use crate::vulkan::core::vk_core::VulkanContext;
use crate::vulkan::buffers::AllocatedBuffer;
use ash::vk;
use anyhow::{anyhow, Result};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{AllocationCreateDesc, AllocationScheme};
use std::marker::PhantomData;
use std::ptr;
use std::sync::Arc;

pub struct VkBuffer<T: Copy> {
    buffer: AllocatedBuffer,
    len: usize,
    _phantom_data: PhantomData<T>,
    address: u64, // Gpu address
}

impl<T: Copy> VkBuffer<T> {
    pub fn new(
        vk_core: &Arc<VulkanContext>,
        data: &[T],
        buffer_create_info: vk::BufferCreateInfo,
        allocation_create_desc: AllocationCreateDesc,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<Self> {
        let len = data.len();
        let buffer_size = (len * size_of::<T>()) as vk::DeviceSize;
        let device = vk_core.device();

        let staging_buffer = Self::create_staging_buffer(vk_core, buffer_size)?;

        // Copy data to the staging buffer
        let staging_data_ptr = staging_buffer
            .allocation()?
            .mapped_ptr()
            .ok_or_else(|| anyhow!("failed to get mapped staging-buffer pointer"))?
            .as_ptr() as *mut u8;

        unsafe {
            ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                staging_data_ptr,
                buffer_size as usize,
            );
        }

        // Create the gpu only buffer
        let buffer_create_info = vk::BufferCreateInfo {
            usage: buffer_create_info.usage
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            ..buffer_create_info
        };

        let allocated_buffer = AllocatedBuffer::new(
            vk_core.clone(),
            &buffer_create_info,
            &allocation_create_desc,
        )?;

        // Copy data from the staging buffer to gpu only buffer
        unsafe {
            let fence_info = vk::FenceCreateInfo::default();
            let fence = device.create_fence(&fence_info, None)?;

            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_pool(command_pool)
                .command_buffer_count(1);

            let command_buffer = device.allocate_command_buffers(&alloc_info)?[0];

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            device.begin_command_buffer(command_buffer, &begin_info)?;

            let copy_region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: buffer_size,
            };

            device.cmd_copy_buffer(
                command_buffer,
                staging_buffer.vk_buffer(),
                allocated_buffer.vk_buffer(),
                &[copy_region],
            );

            device.end_command_buffer(command_buffer)?;

            let command_buffer_submit_infos =
                [vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer)];

            let submit_info =
                vk::SubmitInfo2::default().command_buffer_infos(&command_buffer_submit_infos);

            device.queue_submit2(queue, &[submit_info], fence)?;

            // Wait for just this operation, not the whole queue
            device.wait_for_fences(&[fence], true, u64::MAX)?;

            // Clean up temp resources
            device.destroy_fence(fence, None);
            device.free_command_buffers(command_pool, &[command_buffer]);
        }

        let address: u64 = Self::get_address(vk_core, allocated_buffer.vk_buffer());

        Ok(Self {
            buffer: allocated_buffer,
            len,
            _phantom_data: PhantomData,
            address: address,
        })
    }

    /// To create buffers with uninitialized data.
    pub fn new_uninitialized(
        vk_core: &Arc<VulkanContext>,
        len: usize,
        buffer_create_info: vk::BufferCreateInfo,
        allocation_create_desc: AllocationCreateDesc,
    ) -> Result<Self> {
        // Create the gpu buffer
        let buffer_create_info = vk::BufferCreateInfo {
            usage: buffer_create_info.usage
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            ..buffer_create_info
        };

        let allocated_buffer = AllocatedBuffer::new(
            vk_core.clone(),
            &buffer_create_info,
            &allocation_create_desc,
        )?;

        let address: u64 = Self::get_address(vk_core, allocated_buffer.vk_buffer());

        Ok(Self {
            buffer: allocated_buffer,
            len,
            _phantom_data: PhantomData,
            address,
        })
    }

    /// Creates a buffer with uninitialized data that can be used for GPU only operations.
    pub fn new_gpu_only_uninitialized(
        vk_core: &Arc<VulkanContext>,
        len: usize,
        name: &str,
    ) -> Result<Self> {
        let graphics_family = vk_core.graphics_queue_family_index();
        let compute_family = vk_core.compute_queue_family_index();

        let (sharing_mode, queue_family_indices) = if graphics_family == compute_family {
            (vk::SharingMode::EXCLUSIVE, vec![])
        } else {
            (
                vk::SharingMode::CONCURRENT,
                vec![graphics_family, compute_family],
            )
        };

        let buffer_create_info = vk::BufferCreateInfo::default()
            .size((len * size_of::<T>()) as vk::DeviceSize)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            )
            .sharing_mode(sharing_mode)
            .queue_family_indices(&queue_family_indices);

        let allocation_create_desc = AllocationCreateDesc {
            name,
            requirements: vk::MemoryRequirements::default(),
            location: MemoryLocation::GpuOnly,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };

        Self::new_uninitialized(vk_core, len, buffer_create_info, allocation_create_desc)
    }

    /// Creates a buffer with the given data that can be used for GPU only operations.
    pub fn new_gpu_only(
        vk_core: &Arc<VulkanContext>,
        data: &[T],
        name: &str,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<Self> {
        let graphics_family = vk_core.graphics_queue_family_index();
        let compute_family = vk_core.compute_queue_family_index();

        let (sharing_mode, queue_family_indices) = if graphics_family == compute_family {
            (vk::SharingMode::EXCLUSIVE, vec![])
        } else {
            (
                vk::SharingMode::CONCURRENT,
                vec![graphics_family, compute_family],
            )
        };

        Self::new(
            vk_core,
            data,
            vk::BufferCreateInfo::default()
                .size((data.len() * size_of::<T>()) as vk::DeviceSize)
                .usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_SRC
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                )
                .sharing_mode(sharing_mode)
                .queue_family_indices(&queue_family_indices),
            AllocationCreateDesc {
                name,
                requirements: vk::MemoryRequirements::default(),
                location: MemoryLocation::GpuOnly,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            },
            command_pool,
            queue,
        )
    }

    fn create_staging_buffer(
        vk_core: &Arc<VulkanContext>,
        buffer_size: vk::DeviceSize,
    ) -> Result<AllocatedBuffer> {
        let staging_buffer_create_info = vk::BufferCreateInfo {
            size: buffer_size,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..vk::BufferCreateInfo::default()
        };

        let staging_alloc_create_desc = AllocationCreateDesc {
            name: "Staging buffer memory allocation",
            requirements: vk::MemoryRequirements::default(),
            location: MemoryLocation::CpuToGpu, // Host visible memory
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };

        let staging_buffer = AllocatedBuffer::new(
            vk_core.clone(),
            &staging_buffer_create_info,
            &staging_alloc_create_desc,
        );

        staging_buffer
    }

    pub fn vk_buffer(&self) -> vk::Buffer {
        self.buffer.vk_buffer()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// Efficiently writes data to the mapped memory.
    /// The buffer must have been created with MemoryLocation::CpuToGpu
    pub fn update(&self, data: &T) -> Result<()> {
        let allocation = self.buffer.allocation()?;

        // CpuToGpu is usually persistently mapped.
        // If unmapped, you might need allocation.map() here.
        if let Some(ptr) = allocation.mapped_ptr() {
            unsafe {
                let data_ptr = ptr.as_ptr() as *mut T;
                ptr::copy_nonoverlapping(data, data_ptr, 1);
            }
        } else {
            return Err(anyhow!("cannot update buffer: memory is not mapped"));
        }
        Ok(())
    }

    #[allow(unused)]
    pub fn read_back(
        &self,
        vk_core: &Arc<VulkanContext>,
        command_pool: vk::CommandPool,
    ) -> Result<Vec<T>> {
        let device = vk_core.device();
        let buffer_size = (self.len * size_of::<T>()) as vk::DeviceSize;

        // 1. Create a "Read" Staging Buffer (GpuToCpu)
        let staging_buffer_create_info = vk::BufferCreateInfo::default()
            .size(buffer_size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let staging_alloc_desc = AllocationCreateDesc {
            name: "Readback staging buffer",
            requirements: vk::MemoryRequirements::default(),
            location: MemoryLocation::GpuToCpu, // Optimized for reading back to CPU
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };

        let staging_buffer = AllocatedBuffer::new(
            vk_core.clone(),
            &staging_buffer_create_info,
            &staging_alloc_desc,
        )?;

        // 2. Record Copy Command
        unsafe {
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_pool(command_pool)
                .command_buffer_count(1);

            let cmd = device.allocate_command_buffers(&alloc_info)?[0];

            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            // --- IMPORTANT: Sync2 Barrier ---
            // Ensure the compute shader is actually DONE writing before we start the copy
            let barrier = vk::BufferMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .buffer(self.buffer.vk_buffer())
                .size(vk::WHOLE_SIZE);

            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().buffer_memory_barriers(&[barrier]),
            );

            let copy_region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: buffer_size,
            };

            device.cmd_copy_buffer(
                cmd,
                self.buffer.vk_buffer(),
                staging_buffer.vk_buffer(),
                &[copy_region],
            );

            device.end_command_buffer(cmd)?;

            // 3. Submit and Wait
            let command_buffer_submit_infos =
                [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];

            let submit_info =
                vk::SubmitInfo2::default().command_buffer_infos(&command_buffer_submit_infos);

            device.queue_submit2(*vk_core.compute_queue(), &[submit_info], fence)?;
            device.wait_for_fences(&[fence], true, u64::MAX)?;

            // 4. Map and Copy to Vec
            let mut result = Vec::with_capacity(self.len);
            let ptr = staging_buffer
                .allocation()?
                .mapped_ptr()
                .ok_or_else(|| anyhow!("Failed to map readback memory"))?
                .as_ptr();

            ptr::copy_nonoverlapping(ptr as *const T, result.as_mut_ptr(), self.len);
            result.set_len(self.len);

            // Cleanup
            device.destroy_fence(fence, None);
            device.free_command_buffers(command_pool, &[cmd]);

            Ok(result)
        }
    }

    pub fn address(&self) -> u64 {
        self.address
    }

    fn get_address(vk_core: &Arc<VulkanContext>, buffer: vk::Buffer) -> u64 {
        unsafe {
            vk_core
                .buffer_device_address_loader()
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        }
    }
}
