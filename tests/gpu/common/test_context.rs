use std::sync::{Arc, Mutex, OnceLock};
use engine::{compute::ComputeEngine, vulkan::vk_core::{VkCore, init_headless}};
use rand::rngs::ThreadRng;

pub struct TestContext {
    pub vk_core: Arc<VkCore>,
    pub engine: ComputeEngine,
    pub mutex: Mutex<()>    // Mutex to prevent multiple tests use the same queue
}

impl TestContext {
    fn new() -> &'static TestContext {
        static CONTEXT: OnceLock<TestContext> = OnceLock::new();
        CONTEXT.get_or_init(||{
            // Initialize headless vulkan
            let vk_core = Arc::new(init_headless());
            
            // Initialize the compute engine
            let engine = ComputeEngine::new(vk_core.clone(), 1)
                .expect("Failed to create Compute engine");   
            
            // Return the TestContext
            TestContext {
                vk_core: vk_core,
                mutex: Mutex::new(()),
                engine
            }
        })
    }
    
pub fn run_gpu_test<F>(test_logic: F)
        where
            F: FnOnce(&ComputeEngine, &Arc<VkCore>, ThreadRng)
    {
        let ctx = TestContext::new();
        let _lock = ctx.mutex.lock().unwrap_or_else(|poisoned| {
            poisoned.into_inner()
        });
        let vk_core = &ctx.vk_core;
        let engine = &ctx.engine;
        let rng = rand::rng();
        test_logic(engine, vk_core, rng);
    }
}





