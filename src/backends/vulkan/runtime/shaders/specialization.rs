use ash::vk;

/// Owns the storage borrowed by Vulkan during pipeline creation.
#[derive(Default)]
pub struct SpecializationConstants {
    entries: Vec<vk::SpecializationMapEntry>,
    data: Vec<u8>,
}

impl SpecializationConstants {
    /// Values must be appended in Slang constant_id order, starting at zero.
    pub fn u32(mut self, value: u32) -> Self {
        let id = u32::try_from(self.entries.len()).expect("too many specialization constants");
        self.entries.push(vk::SpecializationMapEntry {
            constant_id: id,
            offset: u32::try_from(self.data.len()).expect("specialization data too large"),
            size: size_of::<u32>(),
        });
        self.data.extend_from_slice(&value.to_ne_bytes());
        self
    }

    pub fn info(&self) -> vk::SpecializationInfo<'_> {
        vk::SpecializationInfo::default()
            .map_entries(&self.entries)
            .data(&self.data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packs_u32_values_and_offsets() {
        let values = SpecializationConstants::default().u32(64).u32(128);
        assert_eq!(values.entries[0].constant_id, 0);
        assert_eq!(values.entries[1].constant_id, 1);
        assert_eq!(values.entries[1].offset, 4);
        assert_eq!(values.entries[1].size, 4);
        assert_eq!(
            values.data,
            [64u32.to_ne_bytes(), 128u32.to_ne_bytes()].concat()
        );
    }
}
