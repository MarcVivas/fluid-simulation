use std::sync::Arc;

use engine::{vulkan::compute::ComputeEngine, algorithms::exclusive_prefix_sum::ExclusivePrefixSum, vulkan::{headless::VkHeadless, core::VkCore, resources::buffer::VkBuffer}};
use rand::{Rng, rngs::ThreadRng};



#[test]
pub fn small_exclusive_prefix_sum(){
    VkHeadless::run(|engine, vk_core, rng|{
        run_prefix_sum_test(vk_core, engine, rng, 26);
    });
}

#[test]
pub fn medium_size_exclusive_prefix_sum(){
    VkHeadless::run(|engine, vk_core, rng|{
        run_prefix_sum_test(vk_core, engine, rng, 26000);
    });
}

#[test]
pub fn large_exclusive_prefix_sum(){
    VkHeadless::run(|engine, vk_core, rng|{
        run_prefix_sum_test(vk_core, engine, rng, 21_600_000);
    });
}

fn generate_random_test_case(vk_core: &Arc<VkCore>, engine: &ComputeEngine, num_elements: u32, rng: &mut ThreadRng) -> (Vec<u32>, VkBuffer<u32>) {
    let data: Vec<u32> = (0..num_elements).map(
        |_|{
            rng.random_range(0..=1)
        }
    ).collect();
    let numbers_buffer: VkBuffer<u32> = VkBuffer::new_gpu_only(
        vk_core,
        &data,
        "Numbers buffer",
        engine.command_pool(),
        *vk_core.compute_queue()
    ).unwrap(); 
    
    (data, numbers_buffer)
}


fn cpu_exclusive_prefix_sum(data: &mut Vec<u32>){
    // 0, 1, 2 -> 0, 0, 1
    let mut preceeding_sum: u32 = 0;
    for x in data {
        let temp_val = *x;
        *x = preceeding_sum;
        preceeding_sum += temp_val;
    }
}

fn run_prefix_sum_test(vk_core: &Arc<VkCore>, engine: &ComputeEngine, mut rng: ThreadRng, num_elements: u32){
    let exclusive_prefix_sum = ExclusivePrefixSum::new(vk_core, num_elements);
    
    let (mut data, data_buffer): (Vec<u32>, VkBuffer<u32>) = generate_random_test_case(vk_core, engine, num_elements, &mut rng);
    
    engine.record_commands(|cmd_buffer|{
        exclusive_prefix_sum.dispatch(vk_core, cmd_buffer, &data_buffer, &data_buffer);
    });
    
    engine.submit_without_signaling();
    
    cpu_exclusive_prefix_sum(&mut data);
    let actual = data_buffer.read_back(vk_core, engine.command_pool()).unwrap();
    
    assert_eq!(data, actual);
}