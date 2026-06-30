// Single pass prefix sum with decoupled look-back
// Algorithm based on https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back


struct PrefixSumPushConstants{
    uint32_t num_elements;

    Ptr<uint32_t, Access.Read> num_elements_ptr;
    
    // The actual array of numbers to perform the prefix sum
    // Size: num_elements
    Ptr<uint32_t, Access.Read> nums;

    // The result will be written here
    Ptr<uint32_t, Access.ReadWrite> out_nums;
    
    // The Atomic Counter (to assign logical block IDs at runtime)
    // Size: 1 uint
    uint32_t* sync_counter;
    
}

[[vk::push_constant]] PrefixSumPushConstants pc; 

// globallycoherent ensures look-back works without Interlocked inside the loop
// The Status Array (The "Flags" and "Values" from the paper)
// Size: num_thread_groups
// Lower 2 bits (Flags), Upper 30 bits (Values)
[[vk::binding(0, 0)]] globallycoherent RWStructuredBuffer<uint32_t> status_array;

static const uint32_t FLAG_NOT_READY = 0; // State X (No data has been processed for this tile yet.)
static const uint32_t FLAG_AGGREGATE = 1; // State A (The block has finished its local sum, but doesn't know the values before it.)
static const uint32_t FLAG_INCLUSIVE = 2; // State P (This block knows the total sum of everything from index 0 to its own end.)

#define NUM_WAVES_PER_GROUP THREAD_GROUP_SIZE/WAVE_SIZE
groupshared uint shared_sums[NUM_WAVES_PER_GROUP];

groupshared uint broadcast;

[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
func main(
    const uint32_t local_id : SV_GroupThreadID 
){

    const uint32_t num_elements = pc.num_elements_ptr != 0 ? pc.num_elements_ptr[0] : pc.num_elements;
    
    // Get IDs...
    const uint32_t group_id = get_thread_group_id(local_id); 
    const uint32_t global_id = group_id * THREAD_GROUP_SIZE + local_id;
    const uint32_t wave_id = local_id / WAVE_SIZE;
    
    // Get the value from global memory
    const uint32_t value = global_id < num_elements ? pc.nums[global_id] : 0;
    
    // Perform a warp level exclusive prefix sum
    const uint32_t wave_accumulated = WavePrefixSum(value);
    
    // The last thread has the total sum of the wave
    // Store it in shared memory 
    if(WaveGetLaneIndex() == WAVE_SIZE - 1){
        const uint32_t total_sum = wave_accumulated + value;
        shared_sums[wave_id] = total_sum;
    }
    
    GroupMemoryBarrierWithGroupSync();
    
    // The first wave computes the prefix sum of all the waves. 
    if(wave_id == 0){
        const uint32_t lane_id = WaveGetLaneIndex();
        // Guard: lane_id might be 32, but we might only have 8 waves
        const uint32_t shared_val = lane_id < NUM_WAVES_PER_GROUP ? shared_sums[lane_id] : 0;
        const uint32_t accumulated_val = WavePrefixSum(shared_val);
        
        if(lane_id < NUM_WAVES_PER_GROUP){
            shared_sums[lane_id] = accumulated_val;            
        }
    }
    
    GroupMemoryBarrierWithGroupSync();
    
    // Calculate the final value with the wave local value and the shared value
    const uint32_t final_local_value = shared_sums[wave_id] + wave_accumulated;
    const uint32_t total_local_sum = final_local_value + value;
    
    // Only the last thread of the thread group publishes the status
    if(local_id == THREAD_GROUP_SIZE - 1){
        
        // The first group status is always the final sum
        // The rest of the groups have only their partial thread group sums
        const uint32_t flag = group_id == 0 ? FLAG_INCLUSIVE : FLAG_AGGREGATE;

        // Pack value and flag and publish it in the status array
        InterlockedExchange(status_array[group_id], pack_value_and_flag(total_local_sum, flag));
    }
    
    
    // Look back loop
    // The first thread group does not need to look back
    // Decoupled look-back
    // Only one thread is enough to look-back
    if(local_id == THREAD_GROUP_SIZE - 1 && group_id > 0){
                
        uint32_t look_back_value = 0;
        int32_t i = int32_t(group_id) - 1;         
        
        while(i >= 0){

            uint32_t status_packed = status_array[i];
            
            // Extract lower 2 bits
            const uint32_t flag = status_packed & 3;
            // Extract the upper 30 bits
            const uint32_t value = status_packed >> 2;
            
            if(flag == FLAG_AGGREGATE){
                look_back_value += value;
                i--;
            }
            else if(flag == FLAG_INCLUSIVE){
                look_back_value += value;
                break;
            }
        }
        
        // Store the look back value and make it public to other threads in the thread group
        broadcast = look_back_value;
        
        // Update the thread group status
        InterlockedExchange(status_array[group_id], pack_value_and_flag(look_back_value + total_local_sum, FLAG_INCLUSIVE));
    }
    
    GroupMemoryBarrierWithGroupSync();
    
    uint32_t prev_prefix_sum = group_id == 0 ? 0 : broadcast;
    
    if(global_id >= num_elements) return;
    pc.out_nums[global_id] = final_local_value + prev_prefix_sum; 
    
}



// We manually assign thread groups ids since the GPU might not execute them in order 
[ForceInline]
func get_thread_group_id(const uint32_t local_id) -> uint32_t {
    // Only one thread of the thread group
    if(local_id == 0){
        uint32_t assigned_thread_group_id;
        InterlockedAdd(pc.sync_counter[0], 1, assigned_thread_group_id);
        broadcast = assigned_thread_group_id;
    }
    GroupMemoryBarrierWithGroupSync();
    return broadcast;
}

// Pack value and flag: (sum << 2) | flag
// Lower 2 bits (Flags), Upper 30 bits (Values)
[ForceInline]
func pack_value_and_flag(const uint32_t value, const uint32_t flag) -> uint32_t {
    return (value << 2) + flag;
}