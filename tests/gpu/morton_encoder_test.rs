use engine::vulkan::vk_utils::VkBuffer;
use glam::Vec4;
use rand::Rng;
use engine::components::MortonCode;
use engine::utils::gpu_algorithms::morton_encoding::MortonEncoder;

use crate::gpu::TestContext;

const CELL_SIZE: f32 = 2.0;

#[test]
pub fn test_morton_encoder(){
    
    TestContext::run_gpu_test(|engine, vk_core|{
        let count = 100024;
        
        let mut rng = rand::rng();
        
        let input_points: Vec<Vec4> = (0..count)
                .map(|_| {
                    Vec4::new(
                        rng.random_range(0.0..1.0),
                        rng.random_range(0.0..1.0),
                        rng.random_range(0.0..1.0),
                        1.0,
                    )
                })
                .collect();
        
        // 2. Prepare GPU Buffers
        let points_buffer: VkBuffer<Vec4> = VkBuffer::new_gpu_only(
            vk_core, 
            &input_points, 
            "Input Points", 
            engine.command_pool(), 
            *vk_core.compute_queue()
        ).unwrap();
        
        let codes_buffer: VkBuffer<MortonCode> = VkBuffer::new_gpu_only(
            vk_core, 
            &vec![0u32; count], // Initialize with zeros
            "Morton Codes Output", 
            engine.command_pool(), 
            *vk_core.compute_queue()
        ).unwrap();
        
        let points_indexes: VkBuffer<u32> = VkBuffer::new_gpu_only(vk_core, &vec![0u32; count],
            "Points indexes",
            engine.command_pool(),
            *vk_core.compute_queue()
        ).unwrap();
        
        let morton_encoder = MortonEncoder::new(vk_core).unwrap();
        
        engine.record_commands(|cmd|{
            morton_encoder.execute(
                vk_core,
                count as u32,
                CELL_SIZE,
                &points_buffer,
                &codes_buffer,
                &points_indexes,
                cmd
            );
        });
        
        engine.submit_to_queue(&[]);
        
        let result_codes: Vec<MortonCode> = codes_buffer.read_back(vk_core, engine.command_pool()).unwrap();
        let result_point_ids: Vec<u32> = points_indexes.read_back(vk_core, engine.command_pool()).unwrap();
        
        for i in 0..count {
            let p = input_points[i];
            let expected = cpu_morton_encode(p.x, p.y, p.z, CELL_SIZE);
            
            assert_eq!(
                result_codes[i], 
                expected, 
                "Morton mismatch at index {}: Point({:?})", i, p
            );
            
            assert_eq!(
                result_point_ids[i],
                i as u32,
                "Point id mismatch at index {}: Point Id({})", i, i
            )
        }
    });

    
}

/// Verification logic using cell_size discretization
fn cpu_morton_encode(x: f32, y: f32, z: f32, cell_size: f32) -> u32 {
    // 1. Discretize based on cell size (matching your shader logic)
    // We use & 1023 to ensure we stay within 10 bits per axis (30 bits total)
    let ux = (x / cell_size).floor() as u32 & 1023;
    let uy = (y / cell_size).floor() as u32 & 1023;
    let uz = (z / cell_size).floor() as u32 & 1023;

    fn expand_bits(mut v: u32) -> u32 {
        v = (v | (v << 16)) & 0x030000FF;
        v = (v | (v << 8)) & 0x0300F00F;
        v = (v | (v << 4)) & 0x030C30C3;
        v = (v | (v << 2)) & 0x09249249;
        v
    }

    // Interleave bits: X is bit 2, Y is bit 1, Z is bit 0
    (expand_bits(ux) << 2) | (expand_bits(uy) << 1) | expand_bits(uz)
}