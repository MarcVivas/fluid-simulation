use std::sync::Arc;

use gpu_fluid_simulation::backends::vulkan::algorithms::sorting::kv_radix_sort::GpuKVRadixSort;
use gpu_fluid_simulation::backends::vulkan::algorithms::sorting::kv_radix_sort::RadixSortPayload;
use gpu_fluid_simulation::backends::vulkan::runtime::buffers::VkBuffer;
use gpu_fluid_simulation::backends::vulkan::runtime::compute::ComputeExecutor;
use gpu_fluid_simulation::backends::vulkan::runtime::core::VulkanContext;
use gpu_fluid_simulation::backends::vulkan::runtime::frame::frame_pacer::FramePacer;
use gpu_fluid_simulation::backends::vulkan::runtime::headless::VkHeadless;
use rand::Rng;
use rand::rngs::ThreadRng;

struct KvRadixSortTest<T: Copy + std::fmt::Debug + std::cmp::PartialEq + RadixSortPayload> {
    sorter: GpuKVRadixSort<T>,
    input_keys: Vec<u32>,
    input_payload: Vec<T>,
    keys_buffer: VkBuffer<u32>,
    payload_buffer: VkBuffer<T>,
    count_buffer: VkBuffer<u32>,
}

impl<T: Copy + std::fmt::Debug + std::cmp::PartialEq + RadixSortPayload> KvRadixSortTest<T> {
    pub fn new(
        vk_context: &Arc<VulkanContext>,
        engine: &ComputeExecutor,
        count: u32,
        mut rng: ThreadRng,
        payload_generator: impl Fn(u32) -> T,
    ) -> Self {
        let input_keys: Vec<u32> = (0..count)
            .map(|_| {
                return rng.random::<u32>();
            })
            .collect();

        let input_payload: Vec<T> = (0..count).map(&payload_generator).collect();

        let queue = *vk_context.compute_queue();
        let cmd_pool = engine.command_pool();

        let keys_buffer: VkBuffer<u32> =
            VkBuffer::new_gpu_only(vk_context, &input_keys, "Keys", cmd_pool, queue).unwrap();
        let payload_buffer: VkBuffer<T> =
            VkBuffer::new_gpu_only(vk_context, &input_payload, "Payload", cmd_pool, queue).unwrap();
        let count_buffer: VkBuffer<u32> = VkBuffer::new_gpu_only(
            vk_context,
            &vec![input_keys.len() as u32],
            "Count buffer",
            cmd_pool,
            queue,
        )
        .unwrap();

        let sorter = GpuKVRadixSort::new(vk_context, cmd_pool, count, Some(32)).unwrap();

        Self {
            sorter,
            input_keys,
            input_payload,
            keys_buffer,
            payload_buffer,
            count_buffer,
        }
    }

    pub fn run_test(
        &self,
        vk_context: &Arc<VulkanContext>,
        engine: &ComputeExecutor,
        frame_pacer: &FramePacer,
        indirect_dispatch: bool,
    ) {
        engine
            .record_commands(&frame_pacer, |cmd| {
                if indirect_dispatch {
                    self.sorter.sort_indirect(
                        vk_context,
                        self.count_buffer.address(),
                        &self.keys_buffer,
                        &self.payload_buffer,
                        cmd,
                    );
                } else {
                    self.sorter.sort(
                        vk_context,
                        &self.keys_buffer,
                        &self.payload_buffer,
                        cmd,
                        self.input_keys.len(),
                    );
                }
            })
            .expect("failed to record radix-sort commands");

        engine
            .submit_without_signaling(&frame_pacer)
            .expect("failed to submit radix-sort commands");

        self.validate(vk_context, engine);
    }

    fn validate(&self, vk_context: &Arc<VulkanContext>, engine: &ComputeExecutor) {
        let mut expected: Vec<(u32, T)> = self
            .input_keys
            .clone()
            .into_iter()
            .zip(self.input_payload.clone().into_iter())
            .collect();

        expected.sort_by_key(|(key, _)| *key);

        let result_keys: Vec<u32> = self
            .keys_buffer
            .read_back(vk_context, engine.command_pool())
            .unwrap();
        let result_payload: Vec<T> = self
            .payload_buffer
            .read_back(vk_context, engine.command_pool())
            .unwrap();

        for i in 0..self.input_keys.len() as usize {
            assert_eq!(result_keys[i], expected[i].0, "Key mismatch at index {}", i);
            assert_eq!(
                result_payload[i], expected[i].1,
                "Payload mismatch at index {}",
                i
            );
        }
    }
}

#[test]
pub fn kv_radix_sort_test() {
    VkHeadless::run(|engine, vk_context, rng| {
        let frame_pacer = gpu_fluid_simulation::backends::vulkan::runtime::frame::frame_pacer::FramePacer::new(1);
        let count = 1400024;
        let kv_radix_sort_test = KvRadixSortTest::<u32>::new(vk_context, engine, count, rng, |i| i);
        kv_radix_sort_test.run_test(vk_context, engine, &frame_pacer, false);
    });
}

#[test]
pub fn kv_radix_sort_indirect_test() {
    VkHeadless::run(|engine, vk_context, rng| {
        let frame_pacer = gpu_fluid_simulation::backends::vulkan::runtime::frame::frame_pacer::FramePacer::new(1);
        let count = 2100024;
        let kv_radix_sort_test = KvRadixSortTest::<u32>::new(vk_context, engine, count, rng, |i| i);
        kv_radix_sort_test.run_test(vk_context, engine, &frame_pacer, true);
    });
}

#[test]
pub fn kv_radix_sort_indirect_uvec2_payload_test() {
    VkHeadless::run(|engine, vk_context, rng| {
        let frame_pacer = gpu_fluid_simulation::backends::vulkan::runtime::frame::frame_pacer::FramePacer::new(1);
        let count = 1000023;
        let kv_radix_sort_test =
            KvRadixSortTest::<glam::UVec2>::new(vk_context, engine, count, rng, |i| {
                glam::UVec2::new(i, i)
            });
        kv_radix_sort_test.run_test(vk_context, engine, &frame_pacer, true);
    });
}
