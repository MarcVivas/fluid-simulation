use ash::vk;
use std::ffi::CStr;

macro_rules! define_vulkan_requirements {
    (
        standalone_extensions: [ $( $standalone_ext:expr ),* $(,)? ],

        features: {
            $(
                $field:ident : $type:ty => {
                    extension: $ext_opt:expr,
                    enable: $enable_expr:expr,
                    check: [ $( $check_field:ident == $check_val:expr ),* $(,)? ] $(,)?
                }
            ),* $(,)?
        }
    ) => {
        pub struct DeviceFeatures {
            $( pub $field: $type, )*
            pub base: vk::PhysicalDeviceFeatures,
        }

        impl Default for DeviceFeatures {
            fn default() -> Self {
                Self {
                    $( $field: $enable_expr, )*
                    base: vk::PhysicalDeviceFeatures {
                        shader_clip_distance: vk::TRUE,
                        shader_int64: vk::TRUE,
                        ..Default::default()
                    },
                }
            }
        }

        impl DeviceFeatures {
            pub fn required_extension_pointers(enable_swapchain: bool) -> Vec<*const std::ffi::c_char> {
                let mut list: Vec<&'static CStr> = Vec::new();

                $(
                    if let Some(ext) = $ext_opt {
                        list.push(ext);
                    }
                )*
                $(
                    list.push($standalone_ext);
                )*

                if enable_swapchain {
                    list.push(ash::khr::swapchain::NAME);
                }

                list.into_iter().map(|name| name.as_ptr()).collect()
            }

            pub fn query(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> Self {
                let mut features = Self::default();
                let mut features2 = vk::PhysicalDeviceFeatures2::default();

                $( features2 = features2.push_next(&mut features.$field); )*

                unsafe {
                    instance.get_physical_device_features2(physical_device, &mut features2);
                }
                features.base = features2.features;
                features
            }

            pub fn is_supported(&self) -> bool {
                true
                $(
                    && $( self.$field.$check_field == $check_val )&&*
                )*
                && self.base.shader_int64 == vk::TRUE
            }

            pub fn apply_to_device<'a>(&'a mut self, mut create_info: vk::DeviceCreateInfo<'a>) -> vk::DeviceCreateInfo<'a> {
                $( create_info = create_info.push_next(&mut self.$field); )*
                create_info
            }
        }
    };
}

define_vulkan_requirements! {
    standalone_extensions: [
        ash::khr::push_descriptor::NAME,
    ],

    features: {
        // Explicit <'static> lifetime specifiers added for Ash feature structs
        mesh_shader: vk::PhysicalDeviceMeshShaderFeaturesEXT<'static> => {
            extension: Some(ash::ext::mesh_shader::NAME),
            enable: vk::PhysicalDeviceMeshShaderFeaturesEXT::default().mesh_shader(true).task_shader(true),
            check: [mesh_shader == vk::TRUE, task_shader == vk::TRUE],
        },

        sync2: vk::PhysicalDeviceSynchronization2Features<'static> => {
            extension: None,
            enable: vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true),
            check: [synchronization2 == vk::TRUE],
        },
        // Build-time Slang shaders use LocalSizeId for specialized workgroup sizes.
        maintenance4: vk::PhysicalDeviceMaintenance4Features<'static> => {
            extension: None,
            enable: vk::PhysicalDeviceMaintenance4Features::default().maintenance4(true),
            check: [maintenance4 == vk::TRUE],
        },
        dynamic_rendering: vk::PhysicalDeviceDynamicRenderingFeatures<'static> => {
            extension: None,
            enable: vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true),
            check: [dynamic_rendering == vk::TRUE],
        },

        bda: vk::PhysicalDeviceBufferDeviceAddressFeatures<'static> => {
            extension: Some(ash::khr::buffer_device_address::NAME),
            enable: vk::PhysicalDeviceBufferDeviceAddressFeatures::default().buffer_device_address(true),
            check: [buffer_device_address == vk::TRUE],
        },
        timeline_semaphore: vk::PhysicalDeviceTimelineSemaphoreFeatures<'static> => {
            extension: None,
            enable: vk::PhysicalDeviceTimelineSemaphoreFeatures::default().timeline_semaphore(true),
            check: [timeline_semaphore == vk::TRUE],
        },
        scalar_alignment: vk::PhysicalDeviceScalarBlockLayoutFeatures<'static> => {
            extension: None,
            enable: vk::PhysicalDeviceScalarBlockLayoutFeatures::default().scalar_block_layout(true),
            check: [scalar_block_layout == vk::TRUE],
        },
        host_query_reset: vk::PhysicalDeviceHostQueryResetFeatures<'static> => {
            extension: None,
            enable: vk::PhysicalDeviceHostQueryResetFeatures::default().host_query_reset(true),
            check: [host_query_reset == vk::TRUE],
        },
    }
}
