#include "gemm/sgemm_gpu_coalescing.cuh"


template<int BLOCK_SIZE>
__global__ void sgemm_gpu_coalescing(float* a, float* b, float* c, int M, int K, int N) {
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.x / BLOCK_SIZE; // warp内相同
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x % BLOCK_SIZE; // warp内递增

    if (row < M && col < N) {
        float tmp = 0.0f;
        for (int k = 0; k < K; ++k) {
            tmp += a[OFFSET(row, k, K)] * b[OFFSET(k, col, N)];
        }

        c[OFFSET(row, col, N)] = tmp;
    }
}

// 1024 per block


void lunch_sgemm_gpu_coalescing(float* a, float* b, float* c, int M, int K, int N, cudaStream_t stream) {
    static constexpr int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE * BLOCK_SIZE);
    dim3 grid(CEIL_DIV(N, BLOCK_SIZE), CEIL_DIV(M, BLOCK_SIZE));

    sgemm_gpu_coalescing<BLOCK_SIZE><<<grid, block, 0, stream>>>(a, b, c, M, K, N);
}