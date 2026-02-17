#pragma once
#include <cuda_runtime.h>

// Global memory coalescing: threads in a warp access consecutive memory addresses
__global__ void sgemm_coalesced(int M, int N, int K, const float *A, const float *B, float *C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
