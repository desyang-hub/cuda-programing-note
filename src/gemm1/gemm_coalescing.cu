#include "gemm1/gemm_coalescing.cuh"

template <const uint BLOCKSIZE>
__global__ void sgemm_coalescing(int M, int N, int K, float alpha, 
    const float* __restrict__ A,
    const float* __restrict__ B,
    float beta, 
    float* __restrict__ C) 
{

    const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
    
    if (x < M && y < N) {
      float tmp = 0.0;
      for (int i = 0; i < K; ++i) {
        tmp += A[x * K + i] * B[i * N + y];
      }
      C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}



void lunch_sgemm_coalescing(int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C, cudaStream_t stream) 
{
    // gridDim stays the same
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    // make blockDim 1-dimensional, but don't change number of threads
    dim3 blockDim(32 * 32);
    sgemm_coalescing<32><<<gridDim, blockDim, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
}