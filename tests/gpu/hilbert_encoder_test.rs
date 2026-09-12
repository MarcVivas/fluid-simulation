use std::sync::Arc;

use gpu_fluid_simulation::backends::vulkan::algorithms::hilbert_encoding::HilbertEncoder;
use gpu_fluid_simulation::backends::vulkan::runtime::buffers::VkBuffer;
use gpu_fluid_simulation::backends::vulkan::runtime::compute::ComputeExecutor;
use gpu_fluid_simulation::backends::vulkan::runtime::core::VulkanContext;
use gpu_fluid_simulation::backends::vulkan::runtime::headless::VkHeadless;
use glam::Vec4;
use rand::Rng;
use rand::rngs::ThreadRng;

#[test]
pub fn test_hilbert_encoder() {
    VkHeadless::run(|engine, vk_context, mut rng| {
        let frame_pacer = gpu_fluid_simulation::backends::vulkan::runtime::frame::frame_pacer::FramePacer::new(1);
        let count = 1_000_000;
        let cmd_pool = engine.command_pool();

        // Define a perfect cubic world for the Hilbert grid
        let world_min = Vec4::new(-100.0, -100.0, -100.0, 0.0);
        let world_size = 500.0f32;
        let max_levels = 10; // For a 30-bit key

        let (hilbert_buffer, positions_buffer, points_indexes, original_points) =
            generate_test_data(vk_context, engine, &mut rng, count, world_min, world_size);

        let hilbert_encoder = HilbertEncoder::new(vk_context, max_levels).unwrap();

        engine
            .record_commands(&frame_pacer, |cmd| {
                hilbert_encoder.dispatch(
                    vk_context,
                    count as u32,
                    world_min,
                    world_size,
                    &positions_buffer,
                    &hilbert_buffer,
                    &points_indexes,
                    cmd,
                );
            })
            .expect("failed to record Hilbert-encoder commands");

        engine
            .submit_without_signaling(&frame_pacer)
            .expect("failed to submit Hilbert-encoder commands");
        unsafe {
            vk_context.device().device_wait_idle().unwrap();
        }

        let result_keys: Vec<u32> = hilbert_buffer.read_back(vk_context, cmd_pool).unwrap();
        let result_point_ids: Vec<u32> = points_indexes.read_back(vk_context, cmd_pool).unwrap();

        validate(&result_keys, &result_point_ids, &original_points, count);
    });
}

fn validate_length(result_keys: &Vec<u32>, result_point_ids: &Vec<u32>, count: usize) {
    assert_eq!(result_keys.len(), count, "Output keys length mismatch");
    assert_eq!(result_point_ids.len(), count, "Output IDs length mismatch");
}

fn validate_ids(result_point_ids: &Vec<u32>, count: usize) {
    for i in 0..count {
        assert_eq!(
            result_point_ids[i], i as u32,
            "Object ID mapping was corrupted at index {}",
            i
        );
    }
}

fn validate_determinism(result_keys: &Vec<u32>) {
    // We injected identical points at indices 0 and 1. They must produce identical keys.
    assert_eq!(
        result_keys[0], result_keys[1],
        "Determinism failed! Identical points produced different keys: {} vs {}",
        result_keys[0], result_keys[1]
    );
}

fn validate_bounds_and_clamping(result_keys: &Vec<u32>) {
    // They should clamp safely to valid Hilbert keys without overflowing.
    for &key in result_keys {
        assert!(
            key < (1u32 << 30u32),
            "Out-of-bounds (min) particle generated an invalid overflow key!"
        );
    }
}

fn validate_spatial_compactness(
    result_keys: &Vec<u32>,
    result_point_ids: &Vec<u32>,
    original_points: &Vec<Vec4>,
    count: usize,
) {
    // We zip the keys and IDs together and sort them by the Hilbert Key.
    let mut sorted_data: Vec<(&u32, &u32)> = result_keys
        .into_iter()
        .zip(result_point_ids.into_iter())
        .collect();
    sorted_data.sort_by_key(|&(key, _)| key);

    let mut hilbert_total_distance = 0.0;
    let mut random_total_distance = 0.0;

    for i in 0..(count - 1) {
        // Distance between consecutive particles in the HILBERT-SORTED array
        let p1_hilbert = original_points[*sorted_data[i].1 as usize].truncate();
        let p2_hilbert = original_points[*sorted_data[i + 1].1 as usize].truncate();
        hilbert_total_distance += p1_hilbert.distance(p2_hilbert);

        // Distance between consecutive particles in the RANDOM (unsorted) array
        let p1_random = original_points[i].truncate();
        let p2_random = original_points[i + 1].truncate();
        random_total_distance += p1_random.distance(p2_random);
    }

    let hilbert_avg_dist = hilbert_total_distance / count as f32;
    let random_avg_dist = random_total_distance / count as f32;

    // A functioning Hilbert curve should drastically reduce the physical distance
    // between neighboring elements in memory. Usually by a factor of 10x to 50x!
    assert!(
        hilbert_avg_dist < (random_avg_dist / 15.0),
        "Hilbert sorting did not significantly improve spatial locality!"
    );
}

fn validate(
    result_keys: &Vec<u32>,
    result_point_ids: &Vec<u32>,
    original_points: &Vec<Vec4>,
    count: usize,
) {
    validate_length(result_keys, result_point_ids, count);
    validate_ids(result_point_ids, count);
    validate_determinism(result_keys);
    validate_bounds_and_clamping(result_keys);
    validate_spatial_compactness(result_keys, result_point_ids, original_points, count);
}

fn generate_test_data(
    vk_context: &Arc<VulkanContext>,
    engine: &ComputeExecutor,
    rng: &mut ThreadRng,
    count: usize,
    world_min: Vec4,
    world_size: f32,
) -> (VkBuffer<u32>, VkBuffer<Vec4>, VkBuffer<u32>, Vec<Vec4>) {
    let mut input_points: Vec<Vec4> = Vec::with_capacity(count);

    // 1. Inject Determinism Test Points (Indices 0 and 1)
    let duplicate_point = Vec4::new(
        world_min.x + 10.0,
        world_min.y + 15.0,
        world_min.z + 20.0,
        1.0,
    );
    input_points.push(duplicate_point);
    input_points.push(duplicate_point);

    // 2. Inject Out-of-Bounds Test Points (Indices 2 and 3)
    input_points.push(Vec4::new(
        world_min.x - 5000.0,
        world_min.y - 5000.0,
        world_min.z - 5000.0,
        1.0,
    )); // Way below min
    input_points.push(Vec4::new(
        world_min.x + 5000.0,
        world_min.y + 5000.0,
        world_min.z + 5000.0,
        1.0,
    )); // Way above max

    // 3. Fill the rest with random data inside the world
    for _ in 4..count {
        input_points.push(Vec4::new(
            rng.random_range(world_min.x..(world_min.x + world_size)),
            rng.random_range(world_min.y..(world_min.y + world_size)),
            rng.random_range(world_min.z..(world_min.z + world_size)),
            1.0,
        ));
    }

    let points_buffer = VkBuffer::new_gpu_only(
        vk_context,
        &input_points,
        "Input Points",
        engine.command_pool(),
        *vk_context.compute_queue(),
    )
    .unwrap();

    let hilbert_buffer = VkBuffer::new_gpu_only(
        vk_context,
        &vec![0u32; count],
        "Hilbert keys Output",
        engine.command_pool(),
        *vk_context.compute_queue(),
    )
    .unwrap();

    let points_indexes = VkBuffer::new_gpu_only(
        vk_context,
        &vec![0u32; count],
        "Points indexes",
        engine.command_pool(),
        *vk_context.compute_queue(),
    )
    .unwrap();

    (hilbert_buffer, points_buffer, points_indexes, input_points)
}
