use ash::vk;

pub struct RenderConfig {
    clear_value: vk::ClearValue,
    depth_clear: vk::ClearValue,
}

impl RenderConfig {
    pub fn new(clear_value: vk::ClearValue, depth_clear: vk::ClearValue) -> Self {
        Self {
            clear_value,
            depth_clear
        }
    }

    pub fn clear_value(&self) -> &vk::ClearValue {
        &self.clear_value
    }

    pub fn depth_clear(&self) -> &vk::ClearValue {
        &self.depth_clear
    }
}
