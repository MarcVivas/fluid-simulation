
// -----------------------------------------------------------------------------
// Resources
// -----------------------------------------------------------------------------

struct SortingPushConstants {
    Ptr<uint32_t, Access.Read> src_keys;
    uint* dst_keys;
    uint* histogram;
    uint* reduce_table;
    uint* scan_scratch;

    uint num_keys;
    int  num_blocks_per_thread_group;
    uint num_thread_groups;
    uint num_thread_groups_with_additional_blocks;
    uint num_reduce_thread_group_per_bin;
    uint num_scan_values;
    uint shift_bit;
    uint padding;
};

[[vk::push_constant]] SortingPushConstants push_constants;


// -----------------------------------------------------------------------------
// Shared Memory (LDS)
// -----------------------------------------------------------------------------
groupshared uint group_shared_histogram[THREAD_GROUP_SIZE * BIN_COUNT];
groupshared uint group_shared_sums[THREAD_GROUP_SIZE];
// Offset cache to avoid loading the offsets all the time
groupshared uint shared_bin_offsets[THREAD_GROUP_SIZE];
// Local histogram for offset calculations
groupshared uint shared_local_histogram[BIN_COUNT];
// Scratch area for the scatter shader
groupshared uint shared_scratch[THREAD_GROUP_SIZE];

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

uint thread_group_reduce(uint local_sum, uint thread_local_id){
    // Wave local reduce
    uint wave_reduced = WaveActiveSum(local_sum);

    const uint wave_id = thread_local_id / WaveGetLaneCount();

    // The leader of the wave stores the reduction to shared memory
    if (WaveIsFirstLane()) {
        group_shared_sums[wave_id] = wave_reduced;
    }

    GroupMemoryBarrierWithGroupSync();

    // The first wave of threads sum up the reductions of the rest of the waves
    if(wave_id == 0){
        const uint value = (thread_local_id < THREAD_GROUP_SIZE / WaveGetLaneCount()) ? group_shared_sums[thread_local_id] : 0;
        wave_reduced = WaveActiveSum(value);
    }

    return wave_reduced;
}

uint thread_group_scan(const uint local_sum, const uint thread_local_id){
    // First, do the wave scan
    uint wave_prefix_sum = WavePrefixSum(local_sum);

    const uint wave_id = thread_local_id / WaveGetLaneCount();
    const uint lane_id = WaveGetLaneIndex();

    // The last element of the wave writes the wave prefix sum to shared memory
    if(lane_id == WaveGetLaneCount() - 1) {
        group_shared_sums[wave_id] = wave_prefix_sum + local_sum;
    }

    GroupMemoryBarrierWithGroupSync();

    // The first wave does the prefix sum of the partial wave sums
    if(wave_id == 0){
        const uint partial_wave_sum = group_shared_sums[thread_local_id];
        group_shared_sums[thread_local_id] = WavePrefixSum(partial_wave_sum);
    }

    GroupMemoryBarrierWithGroupSync();

    wave_prefix_sum += group_shared_sums[wave_id];

    return wave_prefix_sum;
}

void scan_helper(
    const uint thread_local_id,
    const uint thread_group_id,
    const uint base_index,
    const uint bin_offset,
    const bool add_partial_sums,
    const uint num_values_to_scan,
    uint32_t* scan_src_buffer,
    uint32_t* scan_dst_buffer,
    uint32_t* scan_scratch_buffer
){

    // Load shared memory
    for(uint i = 0; i < ELEMENTS_PER_THREAD; i++){
        const uint data_index = base_index + (i*THREAD_GROUP_SIZE) + thread_local_id;

        const uint col = ((i * THREAD_GROUP_SIZE) + thread_local_id) / ELEMENTS_PER_THREAD;
        const uint row = ((i * THREAD_GROUP_SIZE) + thread_local_id) % ELEMENTS_PER_THREAD;
        shared_scan_memory[row][col] = (data_index < num_values_to_scan) ? scan_src_buffer[bin_offset + data_index] : 0;
    }

	GroupMemoryBarrierWithGroupSync();

    uint local_sum = 0;
    // Calculate the local scan for the current thread
    for(uint i = 0; i < ELEMENTS_PER_THREAD; i++){
        const uint tmp = shared_scan_memory[i][thread_local_id];
        shared_scan_memory[i][thread_local_id] = local_sum;
        local_sum += tmp;
    }

    // Scan partial sums
    uint thread_group_sum = thread_group_scan(local_sum, thread_local_id);

    // Add reduced partial sums if requested
    uint partial_sum = 0;
    if (add_partial_sums){
        partial_sum = scan_scratch_buffer[thread_group_id];
    }

    // Add the thread group scanned value back in shared memory
    for(uint i = 0; i < ELEMENTS_PER_THREAD; i++){
        shared_scan_memory[i][thread_local_id] += thread_group_sum;
    }

    GroupMemoryBarrierWithGroupSync();

    // Write to scan dst
    for(uint i = 0; i < ELEMENTS_PER_THREAD; i++){
        const uint data_index = base_index + (i * THREAD_GROUP_SIZE) + thread_local_id;
        const uint col = ((i * THREAD_GROUP_SIZE) + thread_local_id) / ELEMENTS_PER_THREAD;
        const uint row = ((i * THREAD_GROUP_SIZE) + thread_local_id) % ELEMENTS_PER_THREAD;
        if(data_index < num_values_to_scan){
            scan_dst_buffer[bin_offset + data_index] = shared_scan_memory[row][col] + partial_sum;
        }
    }
}

// -----------------------------------------------------------------------------
// Entry Points
// -----------------------------------------------------------------------------

[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void count(const uint3 local_id : SV_GroupThreadID, const uint3 group_id : SV_GroupID)
{
    const uint thread_group_id = group_id.x;
    const uint thread_local_id = local_id.x;

    // Start by clearing the local counts in shared memory
    for (int i = 0; i < BIN_COUNT; i++) {
        group_shared_histogram[(i * THREAD_GROUP_SIZE) + thread_local_id] = 0;
    }

    // Wait for everyone to catch up
    GroupMemoryBarrierWithGroupSync();

    uint thread_group_block_start = BLOCK_SIZE * push_constants.num_blocks_per_thread_group * thread_group_id;
    uint num_blocks_to_process = push_constants.num_blocks_per_thread_group;

    if(thread_group_id >= push_constants.num_thread_groups - push_constants.num_thread_groups_with_additional_blocks){
        thread_group_block_start += (thread_group_id - (push_constants.num_thread_groups - push_constants.num_thread_groups_with_additional_blocks)) * BLOCK_SIZE;
        num_blocks_to_process++;
    }

    // Get the block starting index for this thread
    uint block_index = thread_group_block_start + thread_local_id;

    // For each block to process
    for(uint block_count = 0; block_count < num_blocks_to_process; block_count++, block_index += BLOCK_SIZE){
        uint data_index = block_index;

        // Pre-load the key values in order to hide some of the read latency
        uint local_src_keys[ELEMENTS_PER_THREAD];
        local_src_keys[0] = push_constants.src_keys[data_index];
        local_src_keys[1] = push_constants.src_keys[data_index + THREAD_GROUP_SIZE];
        local_src_keys[2] = push_constants.src_keys[data_index + THREAD_GROUP_SIZE * 2];
        local_src_keys[3] = push_constants.src_keys[data_index + THREAD_GROUP_SIZE * 3];

        // For each local value
        for(uint i = 0; i < ELEMENTS_PER_THREAD; i++){
            if(data_index < push_constants.num_keys){
                // data_index is not out of bounds

                // Shift + mask to remove the other values
                /*
                Example:
                Shifted: 0000 0000 0000 0000 0000 0000 0100 1101
                Mask:    0000 0000 0000 0000 0000 0000 0000 1111
                         ---------------------------------------
                Result:  0000 0000 0000 0000 0000 0000 0000 1101
                */
                const uint local_key = (local_src_keys[i] >> push_constants.shift_bit) & 0xf;
                InterlockedAdd(group_shared_histogram[local_key * THREAD_GROUP_SIZE + thread_local_id], 1);
                data_index += THREAD_GROUP_SIZE;
            }
        }
    }

	// Wait for all the subgroups to finish
	GroupMemoryBarrierWithGroupSync();

	if(thread_local_id < BIN_COUNT){
	    uint sum = 0;
	    for(uint i = 0; i < THREAD_GROUP_SIZE; i++){
	        sum += group_shared_histogram[thread_local_id * THREAD_GROUP_SIZE + i];
	    }
	    push_constants.histogram[thread_local_id * push_constants.num_thread_groups + thread_group_id] = sum;
	}

}

[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void reduce(const uint3 local_id : SV_GroupThreadID, const uint3 group_id : SV_GroupID)
{
    const uint thread_group_id = group_id.x;
    const uint thread_local_id = local_id.x;

    // Figure out what bin data we are reducing
    const uint bin_id = thread_group_id / push_constants.num_reduce_thread_group_per_bin;
    const uint bin_offset = bin_id * push_constants.num_thread_groups;

    // Get the base index for this thread group
    const uint base_index = (thread_group_id % push_constants.num_reduce_thread_group_per_bin) * ELEMENTS_PER_THREAD * THREAD_GROUP_SIZE;

    uint thread_group_sum = 0;
    for(uint i = 0; i < ELEMENTS_PER_THREAD; ++i){
        uint data_index = base_index + (i * THREAD_GROUP_SIZE) + thread_local_id;
        thread_group_sum += (data_index < push_constants.num_thread_groups) ? push_constants.histogram[bin_offset + data_index] : 0;
    }

    thread_group_sum = thread_group_reduce(thread_group_sum, thread_local_id);

    // The leader of the thread group writes the reduced sum for the bin
    if(thread_local_id == 0){
        push_constants.reduce_table[thread_group_id] = thread_group_sum;
    }

}

// Just a standard prefix scan for the histogram
groupshared int shared_scan_memory[ELEMENTS_PER_THREAD][THREAD_GROUP_SIZE];
[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void scan_reduce_table(const uint3 local_id : SV_GroupThreadID, const uint3 group_id : SV_GroupID)
{
    const uint base_index = ELEMENTS_PER_THREAD * THREAD_GROUP_SIZE * group_id.x;
    const uint bin_offset = 0;
    const bool add_partial_sums = false;
    scan_helper(
        local_id.x,
        group_id.x,
        base_index,
        bin_offset,
        add_partial_sums,
        push_constants.num_scan_values,
        push_constants.reduce_table,
        push_constants.reduce_table,
        push_constants.scan_scratch
    );

}

[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void scan_add(const uint3 local_id : SV_GroupThreadID, const uint3 group_id : SV_GroupID)
{
    // Adds the reduced scan results back to the main table
    const uint base_index = (group_id.x % push_constants.num_reduce_thread_group_per_bin) * ELEMENTS_PER_THREAD * THREAD_GROUP_SIZE;
    const uint bin_id = group_id.x / push_constants.num_reduce_thread_group_per_bin;
    const uint bin_offset = bin_id * push_constants.num_thread_groups;
    const bool add_partial_sums = true;
        scan_helper(
            local_id.x,
            group_id.x,
            base_index,
            bin_offset,
            add_partial_sums,
            push_constants.num_thread_groups,
            push_constants.histogram,
            push_constants.histogram,
            push_constants.reduce_table
        );
}

[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void scatter(const uint local_id : SV_GroupThreadID, const uint group_id : SV_GroupID)
{
    // Explanation:
    /*
     To make the writes fast, the thread groups sort the keys locally in Shared Memory before writing them to Global Memory.
     This is done to prevent scattered writes to global memory.
     E.g. Scattered writes
     Thread 0 has a key that belongs in Bin 15 (Address 9000).
     Thread 1 has a key that belongs in Bin 0 (Address 100).
     Thread 2 has a key that belongs in Bin 5 (Address 500).
     E.g Coalesced writes
     Threads 0-16 will all hold Bin 0 keys.
     Threads 17-25 will all hold Bin 1 keys.
    */

    // Load the bin offsets from histogram to shared memory for faster referencing
    if (local_id < BIN_COUNT){
        shared_bin_offsets[local_id] = push_constants.histogram[local_id * push_constants.num_thread_groups + group_id];
    }

    // Wait for all the threads of the group to be done
    GroupMemoryBarrierWithGroupSync();

    // Figure out this thread group's index into the block data (taking into account thread groups that need to do extra reads)
    uint thread_group_block_start = BLOCK_SIZE * push_constants.num_blocks_per_thread_group * group_id;
    uint num_blocks_to_process = push_constants.num_blocks_per_thread_group;

    if(group_id >= push_constants.num_thread_groups - push_constants.num_thread_groups_with_additional_blocks){
        thread_group_block_start += (group_id - (push_constants.num_thread_groups - push_constants.num_thread_groups_with_additional_blocks)) * BLOCK_SIZE;
        num_blocks_to_process++;
    }

    // Get the block starting index for this thread
    uint block_index = thread_group_block_start + local_id;

    // Count value occurrences
    uint new_count;
    for(uint block_count = 0; block_count < num_blocks_to_process; block_count++, block_index+=BLOCK_SIZE){
        uint data_index = block_index;

        // Pre-load values in local memory
        uint local_src_keys[ELEMENTS_PER_THREAD];
        local_src_keys[0] = push_constants.src_keys[data_index];
        local_src_keys[1] = push_constants.src_keys[data_index + THREAD_GROUP_SIZE];
        local_src_keys[2] = push_constants.src_keys[data_index + THREAD_GROUP_SIZE * 2];
        local_src_keys[3] = push_constants.src_keys[data_index + THREAD_GROUP_SIZE * 3];

        // For each element
        for (uint i = 0; i < ELEMENTS_PER_THREAD; i++){
            // Clear the local histogram
            if(local_id < BIN_COUNT){
                shared_local_histogram[local_id] = 0;
            }

            // Load the local key and value
            uint local_key = (data_index < push_constants.num_keys) ? local_src_keys[i] : 0xffffffff;

            // Sort the thread group keys locally in shared memory
            // Radix sort with 2 bits per pass and therefore 4 total bins (00, 01, 10, 11)
            // The numbers/keys to be sorted have just 4 bits, the shifted and masked values
            for(uint bit_shift = 0; bit_shift < BITS_PER_PASS; bit_shift += 2){
                // Get the key index (4 bits)
                const uint key_index = (local_key >> push_constants.shift_bit) & 0xf;
                // Get the 2 bit key from the key_index
                const uint bit_key = (key_index >> bit_shift) & 0x3;

                // Create a packed histogram
                // Instead of using 4 uint, the histogram bins are packed in a single uint
                // 8 bits per bin
                uint packed_histogram = 1U << (bit_key * 8);

                // Sum up all the packed keys (generates counted offsets up to current thread group)
                uint local_sum = thread_group_scan(packed_histogram, local_id);

                // The last thread stores the updated histogram counts for the thread group
                // Scratch = 0xsum3|sum2|sum1|sum0 for thread group
                if(local_id == (THREAD_GROUP_SIZE-1)){
                    const uint total_histogram_sum = local_sum + packed_histogram;
                    shared_scratch[0] = total_histogram_sum;
                }

                GroupMemoryBarrierWithGroupSync();

                // Load the sums value for the thread group
                packed_histogram = shared_scratch[0];

                // This line does a prefix sum of all the 4 bins inside the uint (packedHistogram = 0x sum2_1_0|sum1_0|sum0|0)
                packed_histogram = (packed_histogram << 8) + (packed_histogram << 16) + (packed_histogram << 24);

                // Calculate the proper offset for this thread's value
                local_sum += packed_histogram;

                // Calculate target offset
                const uint key_offset = (local_sum >> (bit_key * 8)) & 0xff;

                // Rearrange the keys (store, sync, load)
                // Store the key
                group_shared_sums[key_offset] = local_key;
                // Wait all threads of the group
                GroupMemoryBarrierWithGroupSync();
                // Pick up the actual key assigned to this thread
                local_key = group_shared_sums[local_id];
                GroupMemoryBarrierWithGroupSync();
            }

            // Need to recalculate the key_index on this thread now that values have been copied around the thread group
            const uint key_index = (local_key >> push_constants.shift_bit) & 0xf;

            // Reconstruct histogram
            InterlockedAdd(shared_local_histogram[key_index], 1);

            GroupMemoryBarrierWithGroupSync();

            // Prefix histogram
            const uint value = local_id < BIN_COUNT ? shared_local_histogram[local_id] : 0;
            const uint histogram_prefix_sum = WavePrefixSum(value);

            // Broadcast prefix sum via shared memory
            if(local_id < BIN_COUNT){
                shared_scratch[local_id] = histogram_prefix_sum;
            }

            const uint global_offset = shared_bin_offsets[key_index];

            GroupMemoryBarrierWithGroupSync();

            // Get the local offset (at this point the keys are all in increasing order from 0 -> num bins in localID 0 -> thread group size)
            const uint local_offset = local_id - shared_scratch[key_index];

            // Write to destination
            const uint total_offset = global_offset + local_offset;

            if(total_offset < push_constants.num_keys){
                push_constants.dst_keys[total_offset] = local_key;
            }

            GroupMemoryBarrierWithGroupSync();

            // Update the cached histogram for the next set of entries
            if(local_id < BIN_COUNT){
                shared_bin_offsets[local_id] += shared_local_histogram[local_id];
            }

            data_index += THREAD_GROUP_SIZE;
        }

    }
}
