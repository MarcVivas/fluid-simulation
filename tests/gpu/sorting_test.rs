
use engine::components::MortonCode;
use engine::utils::gpu_algorithms::sorting::kv_radix_sort::GpuKVRadixSort;
use engine::vulkan::vk_utils::VkBuffer;
use rand::Rng;

use crate::gpu::TestContext;

#[test]
pub fn test_gpu_kv_radix_sort(){
    TestContext::run_gpu_test(|engine, vk_core|{
        let count = 100024;
        
        let mut rng = rand::rng();
        
        let input_keys: Vec<MortonCode> = (0..count)
            .map(|_|{
                return rng.random::<MortonCode>();
            }).collect();
        
        let input_payload: Vec<MortonCode> = (0..count).map(|i| i as MortonCode).collect();
        
        let keys_buffer: VkBuffer<MortonCode> = VkBuffer::new_gpu_only(vk_core, &input_keys, "Keys", engine.command_pool(), *vk_core.compute_queue()).unwrap();
        let payload_buffer: VkBuffer<MortonCode> = VkBuffer::new_gpu_only(vk_core, &input_payload, "Payload", engine.command_pool(), *vk_core.compute_queue()).unwrap();
        
        let sorting_system = GpuKVRadixSort::new(vk_core, count, Some(32)).unwrap();
    
        engine.record_commands(|cmd| {
            sorting_system.sort(vk_core, &keys_buffer, &payload_buffer, cmd);
        });
        
        engine.submit_to_queue(&[]);
        
        let mut expected: Vec<(MortonCode, u32)> = input_keys.into_iter()
            .zip(input_payload.into_iter())
            .collect();
        
        expected.sort_by_key(|(morton_code, _)| *morton_code);
        
        let result_keys: Vec<MortonCode> = keys_buffer.read_back(vk_core, engine.command_pool()).unwrap();
        let result_payload: Vec<u32> = payload_buffer.read_back(vk_core, engine.command_pool()).unwrap();
        
        for i in 0..count as usize {
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
    })
}