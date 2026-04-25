use criterion::{Criterion, criterion_group, criterion_main};

mod gpu_benches;

criterion_group!(
    name =  benches;
    config = Criterion::default();
    targets = gpu_benches::exclusive_prefix_sum_bench::bench_exclusive_prefix_sum
);
criterion_main!(benches);