# Fluid simulation
A highly optimized GPU fluid simulation. Particle based, uses an octree for neighbor search. 

## Code organization

See [the architecture guide](docs/architecture.md) for module responsibilities,
ownership, and the naming changes.

## Prerequisites
This is what I had to install to run the project. 
- Install [LLVM](https://github.com/llvm/llvm-project/releases)
- Install [Lunar Vulkan SDK](https://vulkan.lunarg.com/) 
- Your GPU must be compatible with mesh shaders, wave/subgroup/warp instructions among other things... 

Something else? I do not know. 

## Run

Slang shaders are compiled during the Cargo build and embedded in the executable.
The build downloads Slang if needed.

```bash
cargo run --release
``` 

## Test
```bash
cargo test
```

## Benchmark
```bash
cargo bench 
```

## Controls
| Input | Action |
| :---- | :----- |
| Mouse wheel | Zoom in/out |
| Left click and drag | Rotate camera |
| Right click and drag | Move camera |

## Performance
System:
- GPU: AMD 6800XT
- CPU: AMD Ryzen 7600
- OS: Linux


## References
- [Divergence-free SPH](https://dl.acm.org/doi/epdf/10.1145/2786784.2786796)
- [Cornerstone: Octree Construction Algorithms for Scalable Particle Simulations](https://arxiv.org/abs/2307.06345)
- [Single-pass Parallel Prefix Scan with Decoupled Look-back](https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf)
