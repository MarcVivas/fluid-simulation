use std::sync::Arc;

use engine::compute::ComputeEngine;
use engine::utils::gpu_algorithms::sorting::kv_radix_sort::{GpuKVRadixSort, RadixSortPayload};
use engine::vulkan::headless::VkHeadless;
use engine::vulkan::vk_core::VkCore;
use engine::vulkan::vk_utils::VkBuffer;
use rand::Rng;
use rand::rngs::ThreadRng;


struct KvRadixSortTest<T: Copy + std::fmt::Debug + std::cmp::PartialEq + RadixSortPayload> {
    sorter: GpuKVRadixSort<T>,
    input_keys: Vec<u32>,
    input_payload: Vec<T>,
    keys_buffer: VkBuffer<u32>,
    payload_buffer: VkBuffer<T>,
    count_buffer: VkBuffer<u32>
}

impl <T: Copy + std::fmt::Debug + std::cmp::PartialEq + RadixSortPayload> KvRadixSortTest<T>{
    pub fn new(vk_core: &Arc<VkCore>, engine: &ComputeEngine, count: u32, mut rng: ThreadRng, payload_generator: impl Fn(u32) -> T) -> Self
    {
        let input_keys: Vec<u32> = (0..count)
            .map(|_|{
                return rng.random::<u32>();
            }).collect();
        
        let input_payload: Vec<T> = (0..count).map(&payload_generator).collect();

        let queue = *vk_core.compute_queue();
        let cmd_pool = engine.command_pool();
        
        let keys_buffer: VkBuffer<u32> = VkBuffer::new_gpu_only(vk_core, &input_keys, "Keys", cmd_pool, queue).unwrap();
        let payload_buffer: VkBuffer<T> = VkBuffer::new_gpu_only(vk_core, &input_payload, "Payload", cmd_pool, queue).unwrap();
        let count_buffer : VkBuffer<u32> = VkBuffer::new_gpu_only(vk_core, &vec![input_keys.len() as u32], "Count buffer", cmd_pool, queue).unwrap();

        let sorter = GpuKVRadixSort::new(vk_core, cmd_pool, count, Some(32)).unwrap();
        
        Self {
           sorter,
           input_keys,
           input_payload,
           keys_buffer,
           payload_buffer,
           count_buffer
        }
    }

    pub fn run_test(&self, vk_core: &Arc<VkCore>, engine: &ComputeEngine, indirect_dispatch: bool){

        engine.record_commands(|cmd| {
            if indirect_dispatch{
                self.sorter.sort_indirect(vk_core, self.count_buffer.address(), &self.keys_buffer, &self.payload_buffer, cmd);
            }
            else {
                self.sorter.sort(vk_core, &self.keys_buffer, &self.payload_buffer, cmd, self.input_keys.len());
            }
        });

        engine.submit_without_signaling();

        self.validate(vk_core, engine);
    }

    fn validate(&self, vk_core: &Arc<VkCore>, engine: &ComputeEngine,){
        let mut expected: Vec<(u32, T)> = self.input_keys.clone().into_iter()
            .zip(self.input_payload.clone().into_iter())
            .collect();
        
        expected.sort_by_key(|(key, _)| *key);

        
        let result_keys: Vec<u32> = self.keys_buffer.read_back(vk_core, engine.command_pool()).unwrap();
        let result_payload: Vec<T> = self.payload_buffer.read_back(vk_core, engine.command_pool()).unwrap();

        for i in 0..self.input_keys.len() as usize {
              assert_eq!(
                  result_keys[i], 
                  expected[i].0, 
                  "Key mismatch at index {}", i
              );
              assert_eq!(
                  result_payload[i], 
                  expected[i].1, 
                  "Payload mismatch at index {}", i
              );
        }
    }
}



#[test]
pub fn kv_radix_sort_test(){
    VkHeadless::run(|engine, vk_core, rng|{
        let count = 100024;
        let kv_radix_sort_test = KvRadixSortTest::<u32>::new(vk_core, engine, count, rng, |i| i);
        kv_radix_sort_test.run_test(vk_core, engine, false);       
    });
}

#[test]
pub fn kv_radix_sort_indirect_test(){
    VkHeadless::run(|engine, vk_core, rng|{
        let count = 200024;
        let kv_radix_sort_test = KvRadixSortTest::<u32>::new(vk_core, engine, count, rng, |i| i);
        kv_radix_sort_test.run_test(vk_core, engine, true);   
    });
}


#[test]
pub fn kv_radix_sort_indirect_uvec2_payload_test(){
    VkHeadless::run(|engine, vk_core, rng|{
        let count = 100023;
        let kv_radix_sort_test = KvRadixSortTest::<glam::UVec2>::new(vk_core, engine, count, rng, |i| glam::UVec2::new(i, i));
        kv_radix_sort_test.run_test(vk_core, engine, true);    
    });
}