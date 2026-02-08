# High-Performance SGEMM Engine in CUDA C++

A progressive, hardware-aware implementation of Single-Precision General Matrix Multiplication (SGEMM) in CUDA C++, benchmarking iterative microarchitectural optimizations against NVIDIA's vendor library (`cuBLAS`).

## Performance Roadmap (Tesla T4 GPU)

| Kernel | Optimization Technique | GFLOPS | % of cuBLAS |
| :--- | :--- | :--- | :--- |
| **01_naive** | 1 thread per element, uncoalesced global memory reads | Pending | - |
| **cuBLAS** | NVIDIA Vendor Library (Reference) | ~7,000 | 100% |

## How to Build & Run
```bash
nvcc -O3 -arch=sm_75 benchmark.cu -lcublas -o benchmark
./benchmark
```
