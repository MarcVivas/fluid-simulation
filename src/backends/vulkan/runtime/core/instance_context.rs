use ash::{Entry, Instance};

use crate::backends::vulkan::runtime::core::debug_messenger::DebugMessenger;

/// Owns the Vulkan loader, instance, and instance-level debug state.
pub struct VulkanInstance {
    _debug_messenger: Option<DebugMessenger>,
    instance: Instance,
    entry: Entry,
}

impl VulkanInstance {
    pub(crate) fn new(entry: Entry, instance: Instance) -> anyhow::Result<Self> {
        let debug_messenger = DebugMessenger::new(&entry, &instance)?;
        Ok(Self {
            _debug_messenger: debug_messenger,
            instance,
            entry,
        })
    }

    pub fn entry(&self) -> &Entry {
        &self.entry
    }
    pub fn raw(&self) -> &Instance {
        &self.instance
    }
}
