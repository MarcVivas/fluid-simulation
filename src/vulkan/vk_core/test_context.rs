use std::sync::{Arc, Mutex, OnceLock};
use crate::vulkan::vk_core::{VkCore, init_headless};

pub struct TestContext {
    pub vk_core: Arc<VkCore>,
    pub mutex: Mutex<()>    // Mutex to prevent multiple tests use the same queue
}

pub fn get_test_context() -> &'static TestContext {
    static CONTEXT: OnceLock<TestContext> = OnceLock::new();
    CONTEXT.get_or_init(||{
        // Initialize headless vulkan
        let vk_core = init_headless();
        
        // Return the TestContext
        TestContext { 
            vk_core: Arc::new(vk_core),
            mutex: Mutex::new(())
        }
    })
}