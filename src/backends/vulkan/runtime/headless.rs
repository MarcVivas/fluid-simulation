use crate::backends::vulkan::runtime::compute::ComputeExecutor;
use crate::backends::vulkan::runtime::core::VulkanContext;
use crate::backends::vulkan::runtime::core::init_headless;
use rand::rngs::ThreadRng;
use std::sync::{Arc, Mutex, OnceLock};

pub struct VkHeadless {
    pub vk_core: Arc<VulkanContext>,
    pub engine: ComputeExecutor,
    pub mutex: Mutex<()>, // Mutex to prevent multiple tests use the same queue
}

impl VkHeadless {
    fn new() -> &'static VkHeadless {
        static CONTEXT: OnceLock<VkHeadless> = OnceLock::new();
        CONTEXT.get_or_init(|| {
            // Initialize headless vulkan
            let vk_core =
                Arc::new(init_headless().expect("failed to initialize headless Vulkan context"));

            // Initialize the compute engine
            let engine = ComputeExecutor::new(vk_core.clone(), Self::frames_in_flight())
                .expect("Failed to create Compute engine");

            // Return the TestContext
            VkHeadless {
                vk_core: vk_core,
                mutex: Mutex::new(()),
                engine,
            }
        })
    }

    pub fn run<F>(code: F)
    where
        F: FnOnce(&ComputeExecutor, &Arc<VulkanContext>, ThreadRng),
    {
        let ctx = VkHeadless::new();
        let _lock = ctx
            .mutex
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let vk_core = &ctx.vk_core;
        let engine = &ctx.engine;
        let rng = rand::rng();
        code(engine, vk_core, rng);
    }

    pub fn frames_in_flight() -> usize {
        1
    }
}
