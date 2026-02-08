#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "src/01_naive.cuh"

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

void randomize_matrix(float* mat, int size) {
    for (int i = 0; i < size; ++i) mat[i] = static_cast<float>(rand()) / RAND_MAX;
}

// Clean template runner function
template <typename F>
float run_and_time(const char* name, F kernel_launch, cudaEvent_t start, cudaEvent_t stop, int M, int N, int K, int num_iters) {
    // Warmup
    kernel_launch();
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < num_iters; ++i) {
        kernel_launch();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / num_iters;
    double gflops = (2.0 * M * N * K * 1e-9) / (avg_ms * 1e-3);
    std::cout << name << " | Time: " << avg_ms << " ms | Performance: " << gflops << " GFLOPS\n";
    return static_cast<float>(gflops);
}

int main() {
    const int M = 2048, N = 2048, K = 2048;
    const size_t bytes_A = M * K * sizeof(float);
    const size_t bytes_B = K * N * sizeof(float);
    const size_t bytes_C = M * N * sizeof(float);

    std::cout << "Benchmarking Matrix Multiplication (M=N=K=" << M << ") on GPU...\n";

    // Allocate host memory
    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N);
    randomize_matrix(h_A.data(), M * K);
    randomize_matrix(h_B.data(), K * N);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes_A));
    CHECK_CUDA(cudaMalloc(&d_B, bytes_B));
    CHECK_CUDA(cudaMalloc(&d_C, bytes_C));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));

    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::ofstream out("results.csv");
    out << "Kernel,GFLOPS\n";

    // 1. Naive
    dim3 block1(32, 32);
    dim3 grid1((N + 31) / 32, (M + 31) / 32);
    float g1 = run_and_time("01_naive       ", [&]() { sgemm_naive<<<grid1, block1>>>(M, N, K, d_A, d_B, d_C); }, start, stop, M, N, K, 5);
    out << "01_naive," << g1 << "\n";

    // cuBLAS (Golden Reference)
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;
    float g_cublas = run_and_time("cuBLAS (NVIDIA)", [&]() {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    }, start, stop, M, N, K, 10);
    out << "cuBLAS," << g_cublas << "\n";
    cublasDestroy(handle);

    out.close();
    std::cout << "\nResults saved to results.csv!\n";

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
