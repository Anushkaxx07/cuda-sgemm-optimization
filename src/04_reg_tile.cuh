#pragma once
#include <cuda_runtime.h>

#define BM 64
#define BN 64
#define BK 8
#define TM 8

// 1D/2D Register Tiling: Each thread computes multiple results (TM) in hardware registers
__global__ void sgemm_reg_tile(int M, int N, int K, const float *A, const float *B, float *C) {
    __shared__ float s_A[BM][BK];
    __shared__ float s_B[BK][BN];

    int cRow = blockIdx.y * BM + (threadIdx.x / (BN / 1)) * TM;
    int cCol = blockIdx.x * BN + (threadIdx.x % BN);

    float threadResults[TM] = {0.0f};

    int num_tiles = (K + BK - 1) / BK;

    for (int t = 0; t < num_tiles; ++t) {
        // Load into shared memory
        int tid = threadIdx.x;
        if ((blockIdx.y * BM + tid / BK) < M && (t * BK + tid % BK) < K)
            s_A[tid / BK][tid % BK] = A[(blockIdx.y * BM + tid / BK) * K + t * BK + tid % BK];
        else
            s_A[tid / BK][tid % BK] = 0.0f;

        if ((t * BK + tid / BN) < K && (blockIdx.x * BN + tid % BN) < N)
            s_B[tid / BN][tid % BN] = B[(t * BK + tid / BN) * N + blockIdx.x * BN + tid % BN];
        else
            s_B[tid / BN][tid % BN] = 0.0f;

        __syncthreads();

        // Compute using registers
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float b_val = s_B[k][threadIdx.x % BN];
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
                threadResults[tm] += s_A[(threadIdx.x / BN) * TM + tm][k] * b_val;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
        if ((cRow + tm) < M && cCol < N) {
            C[(cRow + tm) * N + cCol] = threadResults[tm];
        }
    }
}
