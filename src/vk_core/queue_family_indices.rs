pub struct QueueFamilyIndices {
    pub graphics_family: u32, 
    pub compute_family: u32,
}

impl QueueFamilyIndices {
    pub fn new(graphics_family: u32, compute_family: u32) -> Self {
        Self { graphics_family, compute_family }
    }
}