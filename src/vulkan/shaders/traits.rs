pub trait ShaderName {
    fn shader_name() -> &'static str;
}

pub trait GpuTask {
    fn profiling_label() -> &'static str;
}