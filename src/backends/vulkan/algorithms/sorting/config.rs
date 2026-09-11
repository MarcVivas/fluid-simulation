// CPU allocation and dispatch settings. Keep these synchronized with the fixed
// constants at the top of radix_sort.shader.slang, kv_radix_sort_32.shader.slang,
// and kv_radix_sort_64.shader.slang. THREADS_PER_GROUP is THREAD_GROUP_SIZE in Slang.
// Four elements per thread is structural: shader loads explicitly access [0..3].
pub const BITS_PER_PASS: u32 = 4;
pub const BIN_COUNT: u32 = 1 << BITS_PER_PASS;
pub const ELEMENTS_PER_THREAD: u32 = 4;
pub const THREADS_PER_GROUP: u32 = 64;
pub const BLOCK_SIZE: u32 = THREADS_PER_GROUP * ELEMENTS_PER_THREAD;
pub const MAX_THREAD_GROUPS: u32 = 800;
