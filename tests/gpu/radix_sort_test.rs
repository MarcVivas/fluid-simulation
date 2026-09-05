use engine::{algorithms::sorting::radix_sort::RadixSort, vulkan::{headless::VkHeadless, buffers::VkBuffer}};
use rand::Rng;

#[test]
pub fn test_radix_sort(){
    VkHeadless::run(|engine, vk_core, mut rng|{
        let frame_pacer = engine::vulkan::frame::frame_pacer::FramePacer::new(1);
        let count = 1400024;
                
        let mut input_array: Vec<u32> = (0..count)
            .map(|_|{
                return rng.random::<u32>();
            }).collect();
        
        
        let input_buffer: VkBuffer<u32> = VkBuffer::new_gpu_only(vk_core, &input_array, "Input", engine.command_pool(), *vk_core.compute_queue()).unwrap();
        
        let radix_sort = RadixSort::new(vk_core, count, Some(32)).unwrap();
    
        engine.record_commands(&frame_pacer, |cmd| {
            radix_sort.sort(vk_core, &input_buffer, cmd);
        }).expect("failed to record radix-sort commands");
        
        engine.submit_without_signaling(&frame_pacer)
            .expect("failed to submit radix-sort commands");
        
        
        input_array.sort();
        
        let actual_result: Vec<u32> = input_buffer.read_back(vk_core, engine.command_pool()).unwrap();

        assert_eq!(input_array, actual_result);
        
        
    })
}
