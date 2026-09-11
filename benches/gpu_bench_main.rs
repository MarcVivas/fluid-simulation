use criterion::{Criterion, criterion_group, criterion_main};

mod gpu_benches;

criterion_group!(
    name =  benches;
    config = Criterion::default();
    targets = gpu_benches::exclusive_prefix_sum_bench::bench_exclusive_prefix_sum, gpu_benches::density_bench::bench_density_compute, gpu_benches::physics_bench::bench_physics_engine
);
criterion_main!(benches);
