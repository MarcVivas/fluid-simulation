use std::sync::{Arc, Mutex, atomic::{self, AtomicUsize}};
use crate::{utils::gpu_profiler::gpu_profiling_zones::GpuProfilingZones, vulkan::{vk_core::VkCore, vk_utils::QueryPool}};
use ash::vk;
use std::collections::HashMap;


pub struct GpuProfiler {
    query_pools: Vec<QueryPool>,
    timestamp_period: f32,
    query_count: u32,
    frame_index: AtomicUsize,
    zones: Mutex<GpuProfilingZones>,
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
            frame_index: AtomicUsize::new(0),
            zones: Mutex::new(GpuProfilingZones::new(query_count)),
        }
    }
    
    // Call this every time at the start of the frame
    pub fn reset(&self, device: &ash::Device, command_buffer: vk::CommandBuffer) {
        let pool = &self.query_pools[self.frame_index.load(atomic::Ordering::Relaxed)];
        unsafe {
            device.cmd_reset_query_pool(command_buffer, pool.vk_query_pool(), 0, self.query_count);
        }
    }
    
    pub fn reset_on_host(&self, device: &ash::Device) {
        let frame_idx = self.frame_index.load(atomic::Ordering::Relaxed);
        let pool = &self.query_pools[frame_idx];
        
        unsafe {
            (device.fp_v1_2().reset_query_pool)(
                device.handle(),
                pool.vk_query_pool(), 
                0, 
                self.query_count
            );
        }
    }
    
    pub fn begin_timestamp_zone(&self, device: &ash::Device, cb: vk::CommandBuffer, label: &str){
        let start_index = self.zones.lock().unwrap().get_or_create_zone_index(label);
        unsafe {
            device.cmd_write_timestamp(
                cb,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.query_pools[self.frame_index.load(atomic::Ordering::Relaxed)].vk_query_pool(),
                start_index,
            );
        }
    }
    
    pub fn end_timestamp_zone(&self, device: &ash::Device, cb: vk::CommandBuffer, label: &str) {
        
        let zone_start_index = *self.zones.lock().unwrap().get_zone_index(label).expect(format!("The zone with the given label {} didn't exist.", label).as_str());
        let zone_end_index = zone_start_index + 1; 
        
        let pool = &self.query_pools[self.frame_index.load(atomic::Ordering::Relaxed)];

        unsafe {
            device.cmd_write_timestamp(
                cb,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE, 
                pool.vk_query_pool(),
                zone_end_index,
            );
        }
    }

    pub fn set_frame_index(&self, frame_index: usize, total_frames_processed: u64){
        if total_frames_processed < self.query_pools.len() as u64 {
            return;
        }
        self.frame_index.store(frame_index, atomic::Ordering::Relaxed);
    }



    pub fn get_results(
        &self,
        device: &ash::Device,
        total_frames_processed: u64,
    ) -> Result<HashMap<String, f64>, Box<dyn std::error::Error>> {
        let frames_in_flight = self.query_pools.len();
        if total_frames_processed < frames_in_flight as u64 {
            return Ok(HashMap::new())
        }
        
        let zones_lock = self.zones.lock().unwrap();
        let active_queries = zones_lock.get_active_query_count();

        if active_queries == 0 {
            return Ok(HashMap::new());
        }

        let frame_idx = self.frame_index.load(atomic::Ordering::Relaxed);
        let pool = &self.query_pools[frame_idx];
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
            let ms = (end.saturating_sub(start) as f64 * self.timestamp_period as f64) / 1_000_000.0;
            processed_results.insert(label.clone(), ms);
        }
        
        Ok(processed_results)
    }
}
