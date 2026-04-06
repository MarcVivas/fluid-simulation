use std::sync::Arc;
use crate::vulkan::{vk_core::VkCore, vk_utils::QueryPool};
use ash::vk;

pub struct GpuProfiler {
    query_pools: Vec<QueryPool>,
    timestamp_period: f32,
    query_count: u32,
    frame_index: usize,
}

impl GpuProfiler {
    pub fn new(vk_core: Arc<VkCore>, max_zones: u32, frames_in_flight: usize) -> Self {
        let properties = unsafe {
            vk_core
                .instance()
                .get_physical_device_properties(*vk_core.physical_device())
        };

        let timestamp_period = properties.limits.timestamp_period;

        let query_count = max_zones * 2; // 2 timestamps per zone
        let create_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(query_count);

        let mut query_pools = Vec::with_capacity(frames_in_flight);

        for _ in 0..frames_in_flight {
            let query_pool = QueryPool::new(vk_core.clone(), create_info).unwrap();
            query_pools.push(query_pool);
        }


        Self {
            query_pools,
            timestamp_period,
            query_count,
            frame_index: 0
        }
    }

    pub fn timestamp(&self, device: &ash::Device, cb: vk::CommandBuffer, index: u32) {
        assert!(
            index < self.query_count,
            "timestamp index {index} out of range"
        );

        unsafe {
            device.cmd_write_timestamp(
                cb,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE, // Ensure work is done
                self.query_pools[0].vk_query_pool(),
                index,
            );
        }
    }

    pub fn set_frame_index(&mut self, frame_index: usize){
        self.frame_index = frame_index;
    }

    pub fn reset(&self, device: &ash::Device, command_buffer: vk::CommandBuffer) {
        unsafe {
            device.cmd_reset_query_pool(command_buffer, self.query_pools[0].vk_query_pool(), 0, self.query_count);
        }
    }

    pub fn get_results(
        &self,
        device: &ash::Device,
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let mut results = vec![0u64; self.query_count as usize];
        unsafe {
            let _ = device.get_query_pool_results(
                self.query_pools[0].vk_query_pool(),
                0,
                &mut results,
                vk::QueryResultFlags::TYPE_64,
            )?;
        }

        // Convert pairs of timestamps to milliseconds
        Ok(results
            .chunks_exact(2)
            .map(|chunk| {
                let diff = chunk[1].saturating_sub(chunk[0]);
                (diff as f64 * self.timestamp_period as f64) / 1_000_000.0
            })
            .collect())
    }
}
