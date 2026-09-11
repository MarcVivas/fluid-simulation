use crate::backends::vulkan::runtime::core::VulkanContext;
use crate::backends::vulkan::runtime::frame::frame_pacer::FramePacer;
use crate::backends::vulkan::runtime::profiler::profiling_zones::GpuProfilingZones;
use crate::backends::vulkan::runtime::queries::QueryPool;
use anyhow::Result;
use ash::vk;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub struct GpuProfiler {
    query_pools: Vec<QueryPool>,
    timestamp_period: f32,
    query_count: u32,
    zones: Mutex<GpuProfilingZones>,
}

impl GpuProfiler {
    pub fn new(
        vk_core: Arc<VulkanContext>,
        max_zones: u32,
        frames_in_flight: usize,
    ) -> Result<Self> {
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
            let query_pool = QueryPool::new(vk_core.clone(), create_info)?;
            query_pools.push(query_pool);
        }

        Ok(Self {
            query_pools,
            timestamp_period,
            query_count,
            zones: Mutex::new(GpuProfilingZones::new(query_count)),
        })
    }

    // Call this every time at the start of the frame
    pub fn reset(&self, device: &ash::Device, command_buffer: vk::CommandBuffer, ring_idx: usize) {
        let pool = &self.query_pools[ring_idx];
        unsafe {
            device.cmd_reset_query_pool(command_buffer, pool.vk_query_pool(), 0, self.query_count);
        }
    }

    pub fn reset_on_host(&self, device: &ash::Device, ring_idx: usize) {
        let pool = &self.query_pools[ring_idx];

        unsafe {
            (device.fp_v1_2().reset_query_pool)(
                device.handle(),
                pool.vk_query_pool(),
                0,
                self.query_count,
            );
        }
    }

    pub fn begin_timestamp_zone(
        &self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        label: &str,
        ring_idx: usize,
    ) -> u32 {
        let start_index = self
            .zones
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get_or_create_zone_index(label);
        unsafe {
            device.cmd_write_timestamp(
                cb,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.query_pools[ring_idx].vk_query_pool(),
                start_index,
            );
        }
        start_index
    }

    pub fn end_timestamp_zone(
        &self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        idx: u32,
        ring_idx: usize,
    ) {
        let zone_start_index = idx;
        let zone_end_index = zone_start_index + 1;

        let pool = &self.query_pools[ring_idx];

        unsafe {
            device.cmd_write_timestamp(
                cb,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                pool.vk_query_pool(),
                zone_end_index,
            );
        }
    }

    pub fn profile_scope<F, R>(
        &self,
        device: &ash::Device,
        cmd_buffer: vk::CommandBuffer,
        label: &str,
        ring_idx: usize,
        scope: F,
    ) -> R
    where
        F: FnOnce() -> R,
    {
        let idx = self.begin_timestamp_zone(device, cmd_buffer, label, ring_idx);
        let res = scope();
        self.end_timestamp_zone(device, cmd_buffer, idx, ring_idx);
        res
    }

    pub fn get_results(
        &self,
        device: &ash::Device,
        frame_pacer: &FramePacer,
    ) -> Result<HashMap<String, f64>> {
        let frames_in_flight = self.query_pools.len();
        let total_frames_processed = frame_pacer.total_frames_processed();
        if total_frames_processed < frames_in_flight as u64 {
            return Ok(HashMap::new());
        }

        let zones_lock = self
            .zones
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let active_queries = zones_lock.get_active_query_count();

        if active_queries == 0 {
            return Ok(HashMap::new());
        }

        let ring_idx = frame_pacer.ring_index();
        let pool = &self.query_pools[ring_idx];
        let mut results = vec![0u64; active_queries as usize];

        unsafe {
            let _ = device.get_query_pool_results(
                pool.vk_query_pool(),
                0,
                &mut results,
                vk::QueryResultFlags::TYPE_64,
            )?;
        }

        // Convert pairs of timestamps to milliseconds
        let mut processed_results: HashMap<String, f64> = HashMap::new();
        for (label, &start_idx) in zones_lock.all_zones() {
            let start = results[start_idx as usize];
            let end = results[start_idx as usize + 1];
            let ms =
                (end.saturating_sub(start) as f64 * self.timestamp_period as f64) / 1_000_000.0;
            processed_results.insert(label.clone(), ms);
        }

        Ok(processed_results)
    }

    /// Print profiler metrics
    pub fn print_metrics(&self, device: &ash::Device, frame_pacer: &FramePacer) {
        let timings = self.get_results(device, frame_pacer).unwrap_or_default();
        for (label, time) in timings {
            if time != 0.0 {
                println!("Pass {}: {:.4} ms", label, time);
            }
        }
    }
}
