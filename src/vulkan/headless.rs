use std::sync::{Arc, Mutex, OnceLock};
use crate::{compute::ComputeEngine, vulkan::vk_core::{VkCore, init_headless}};
use rand::rngs::ThreadRng;

pub struct VkHeadless {
    pub vk_core: Arc<VkCore>,
    pub engine: ComputeEngine,
    pub mutex: Mutex<()>    // Mutex to prevent multiple tests use the same queue
}

impl VkHeadless {
    fn new() -> &'static VkHeadless {
        static CONTEXT: OnceLock<VkHeadless> = OnceLock::new();
        CONTEXT.get_or_init(||{
            // Initialize headless vulkan
            let vk_core = Arc::new(init_headless());
            
            // Initialize the compute engine
            let engine = ComputeEngine::new(vk_core.clone(), 1)
                .expect("Failed to create Compute engine");   
            
            // Return the TestContext
            VkHeadless {
                vk_core: vk_core,
                mutex: Mutex::new(()),
                engine
            }
        })
    }
    
    pub fn run<F>(code: F)
        where
            F: FnOnce(&ComputeEngine, &Arc<VkCore>, ThreadRng)
    {
        let ctx = VkHeadless::new();
        let _lock = ctx.mutex.lock().unwrap_or_else(|poisoned| {
            poisoned.into_inner()
        });
        let vk_core = &ctx.vk_core;
        let engine = &ctx.engine;
        let rng = rand::rng();
        code(engine, vk_core, rng);
    }
}







