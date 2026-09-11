#[macro_export]
macro_rules! gpu_profile {
    ($device: expr, $profiler: expr, $cmd_buffer: expr, $profile_label: expr, $code: block ) => {
        $profiler.begin_timestamp_zone($device, $cmd_buffer, $profile_label);
        $code;
        $profiler.end_timestamp_zone($device, $cmd_buffer, $profile_label);
    };
}
