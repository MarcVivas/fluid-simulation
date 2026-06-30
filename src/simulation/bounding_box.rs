use bytemuck::{Pod, Zeroable};
use glam::Vec4Swizzles;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BoundingBox{
    pub min: glam::Vec4,
    pub max: glam::Vec4,
}

impl BoundingBox {
    pub fn new(min: &glam::Vec4, max: &glam::Vec4) -> Self{
        Self { min: *min, max: *max }
    }

    pub fn intersects(&self, other_bounding_box: &BoundingBox) -> bool {
        self.min.xyz().cmple(other_bounding_box.max.xyz()).all() && self.max.xyz().cmpge(other_bounding_box.min.xyz()).all()
    }
}


