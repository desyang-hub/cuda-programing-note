#include "gemm/gemm.cuh"

__global__ void sgemm_gpu(float* __restrict__ a, float* __restrict__ b, float* __restrict__ c, const int M, const int K, const int N) {
    // 每个线程处理c中一个位置的计算

    int col = threadIdx.x + blockIdx.x * blockDim.x; // warp内每次增加1
    int row = threadIdx.y + blockIdx.y * blockDim.y; // warp内相同

    // __shared__ float sum[1] = 0.0f;
    
    if (col < N && row < M) {
        float sum = 0.0f;
        
        #pragma unroll
        for (int i = 0; i < K; ++i) {
            // row * K + i;  i * K + col
            sum += a[OFFSET(row, i, K)] * b[OFFSET(i, col, N)];
        }

        // 填写结果
        c[OFFSET(row, col, N)] = sum;
    }
    
}

void lunch_sgemm_gpu(float* a, float* b, float* c, int M, int K, int N, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    sgemm_gpu<<<grid, block, 0, stream>>>(a, b, c, M, K, N);
}