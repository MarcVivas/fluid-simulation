use bytemuck::{Pod, Zeroable};
use glam::Vec4;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct PositionComponent {
    pub value: Vec4
}

impl PositionComponent {
    pub fn new(x: f32, y: f32, z: f32, w: Option<f32>) -> Self {
        Self { value: Vec4::new(x, y, z, w.unwrap_or(1.0)) }
    }
}