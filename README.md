# High-Performance SGEMM Engine in CUDA C++

A progressive, hardware-aware implementation of Single-Precision General Matrix Multiplication (SGEMM) in CUDA C++, benchmarking iterative microarchitectural optimizations against NVIDIA's vendor library (`cuBLAS`).

[![Live Demo](https://img.shields.io/badge/Live_Demo-Interactive_Drag_Race-10b981?style=for-the-badge&logo=nvidia)](https://anushkaxx07.github.io/cuda-sgemm-optimization/)

> 🏎️ **Live Interactive Visualizer:** Experience the [CUDA SGEMM Grand Prix](https://anushkaxx07.github.io/cuda-sgemm-optimization/) in your browser! Watch all 5 GPU kernels race across a drag strip with velocities directly proportional to their real-world GFLOPS throughput—from the uncoalesced DRAM crawl (`01_naive`) to register-tiled hyperdrive (`04_reg_tile`) and NVIDIA `cuBLAS`.

![Benchmark Results](benchmark_plot.png)

## Performance Roadmap (Tesla T4 GPU)

| Kernel | Optimization Technique | GFLOPS | % of cuBLAS |
| :--- | :--- | :--- | :--- |
| **01_naive** | 1 thread per element, uncoalesced global memory reads | ~300 | ~4% |
| **02_coalesced** | Warp-aligned global memory coalescing | ~850 | ~12% |
| **03_shared_mem** | 2D on-chip shared memory tiling ($32 \times 32$) to reduce DRAM roundtrips | ~2,800 | ~40% |
| **04_reg_tile** | Register-level tiling (TM=8) to increase thread work and hide latency | ~5,200 | ~75% |
| **cuBLAS** | NVIDIA Vendor Library (Reference) | ~7,000 | 100% |

## Key GPU Concepts Demonstrated
- **Memory Coalescing:** Aligning contiguous 32-byte cache lines for warp threads.
- **Shared Memory Tiling:** Utilizing high-speed on-chip SRAM to reuse matrix sub-blocks.
- **Register Pressure & Latency Hiding:** Storing accumulators in hardware registers to maximize instruction-level parallelism (ILP).

## How to Build & Run
```bash
nvcc -O3 -arch=sm_75 benchmark.cu -lcublas -o benchmark
./benchmark
python3 plot_results.py
