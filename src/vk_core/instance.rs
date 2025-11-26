use std::ffi::CStr;
use std::os::raw::c_char;
use ash::{vk, Entry, Instance};
use ash::ext::debug_utils;

const VALIDATION_LAYERS: [&CStr; 1] = [c"VK_LAYER_KHRONOS_validation"];
#[cfg(debug_assertions)]
const ENABLE_VALIDATION_LAYERS: bool = true;
#[cfg(not(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = false;

/// The instance is the connection between your application and the Vulkan library
pub fn create_instance(entry: &Entry, required_extensions: &[*const c_char]) -> Instance {

    let app_name = c"Vulkan";
    let app_info = vk::ApplicationInfo::default()
        .application_name(app_name)
        .application_version(0)
        .engine_name(app_name)
        .engine_version(0)
        .api_version(vk::API_VERSION_1_3);

    let create_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
        vk::InstanceCreateFlags::empty()
    }
    else {
        vk::InstanceCreateFlags::default()
    };

    let layers_names_raw: Vec<*const c_char> = if check_validation_support(entry)
        && ENABLE_VALIDATION_LAYERS
    {
        VALIDATION_LAYERS.iter()
            .map(|raw_name| raw_name.as_ptr())
            .collect()
    }
    else {
        vec![]
    };


    let mut extension_names = required_extensions.to_vec();


    #[cfg(debug_assertions)]
    {
        extension_names.push(debug_utils::NAME.as_ptr());
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        extension_names.push(ash::khr::portability_enumeration::NAME.as_ptr());
        extension_names.push(ash::khr::get_physical_device_properties2::NAME.as_ptr());
    }


    let instance_create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_layer_names(&layers_names_raw)
        .enabled_extension_names(&extension_names)
        .flags(create_flags);

    let instance: Instance = unsafe {
        entry.create_instance(&instance_create_info, None)
    }.expect("failed to create instance");

    instance


}

fn check_validation_support(entry: &Entry) -> bool {
    let available_layers = match unsafe { entry.enumerate_instance_layer_properties() } {
        Ok(layers) => layers,
        Err(_) => return false,
    };

    for required_layer_name in VALIDATION_LAYERS {

        let is_layer_found = available_layers
            .iter()
            .any(
                |layer_properties| {
                    let layer_name = unsafe { CStr::from_ptr(layer_properties.layer_name.as_ptr()) };
                    let layer_name_str = layer_name.to_str().unwrap();
                    required_layer_name.to_str().unwrap() == layer_name_str
                }
        );

        if !is_layer_found {
            return false;
        }
    }
    true

}
