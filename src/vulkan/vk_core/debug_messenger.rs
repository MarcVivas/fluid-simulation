use std::ffi;
use ash::{vk, Entry};
use ash::ext::debug_utils;
use ash::vk::DebugUtilsMessengerEXT;


pub struct DebugMessenger{
    debug_messenger: DebugUtilsMessengerEXT,
    debug_utils_loader: debug_utils::Instance,
}

impl DebugMessenger {
    pub fn new(entry: &Entry, instance: &ash::Instance) -> Option<Self> {
        #[cfg(not(debug_assertions))]
        { return None }

        let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default().message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
        ).message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        ).pfn_user_callback(Some(Self::vulkan_debug_callback));

        let debug_utils_loader = debug_utils::Instance::new(entry, instance);
        let debug_callback = unsafe {
            debug_utils_loader.create_debug_utils_messenger(&debug_info, None)
        }.expect("failed to set up debug messenger");

        Some(Self {
            debug_utils_loader,
            debug_messenger: debug_callback,
        })
    }
    unsafe extern "system" fn vulkan_debug_callback(
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT,
        p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
        _user_data: *mut std::os::raw::c_void,
    ) -> vk::Bool32 {
        unsafe {
            let callback_data = *p_callback_data;
            let message_id_number = callback_data.message_id_number;

            let message_id_name = if callback_data.p_message_id_name.is_null() {
                std::borrow::Cow::from("")
            } else {
                ffi::CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
            };

            let message = if callback_data.p_message.is_null() {
                std::borrow::Cow::from("")
            } else {
                ffi::CStr::from_ptr(callback_data.p_message).to_string_lossy()
            };

            println!(
                "{message_severity:?}:\n\
                {message_type:?} [{message_id_name} ({message_id_number})] : {message}\n",
            );

            vk::FALSE
        }
    }
}

impl Drop for DebugMessenger {
    fn drop(&mut self) {
        unsafe {
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_messenger, None);
        }
    }
}