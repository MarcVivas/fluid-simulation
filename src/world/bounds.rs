use bytemuck::{Pod, Zeroable};
use glam::Vec4Swizzles;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, Default)]
pub struct BoundingBox {
    pub min: glam::Vec4,
    pub max: glam::Vec4,
}

impl BoundingBox {
    pub fn new(min: &glam::Vec4, max: &glam::Vec4) -> Self {
        Self {
            min: *min,
            max: *max,
        }
    }

    pub fn intersects(&self, other_bounding_box: &BoundingBox) -> bool {
        self.min.xyz().cmple(other_bounding_box.max.xyz()).all()
            && self.max.xyz().cmpge(other_bounding_box.min.xyz()).all()
    }
}


#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, Default)]
pub struct NodeBoundingBox {
    /// xyz = minimum corner, w = cube side length
    pub min_and_side_length: glam::Vec4,
}

impl NodeBoundingBox {
    pub fn new(
        min_corner: glam::Vec3,
        side_length: f32,
    ) -> Self {
        Self {
            min_and_side_length: min_corner.extend(side_length),
        }
    }
}

/// Backend-neutral world configuration and spatial bounds.
#[derive(Clone, Copy, Debug)]
pub struct WorldBounds {
    pub world_size: f32,
    pub world_min: glam::Vec4,
}

impl WorldBounds {
    pub fn new(world_max: &glam::Vec3) -> Self {
        Self {
            world_size: world_max.max_element(),
            world_min: glam::Vec4::ZERO,
        }
    }
}
