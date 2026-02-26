#pragma once
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

// Shared Memory Tiling: cache tiles of A & B on fast on-chip SRAM to reduce DRAM traffic
__global__ void sgemm_shared_mem(int M, int N, int K, const float *A, const float *B, float *C) {
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float sum = 0.0f;

    int num_tiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        // Cooperatively load 32x32 tile into shared memory
        if (row < M && (t * BLOCK_SIZE + threadIdx.x) < K)
            s_A[threadIdx.y][threadIdx.x] = A[row * K + t * BLOCK_SIZE + threadIdx.x];
        else
            s_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (t * BLOCK_SIZE + threadIdx.y) < K)
            s_B[threadIdx.y][threadIdx.x] = B[(t * BLOCK_SIZE + threadIdx.y) * N + col];
        else
            s_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads(); // Wait for all threads to finish loading the tile

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        __syncthreads(); // Wait before loading the next tile
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
