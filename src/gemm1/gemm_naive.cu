#include "gemm1/gemm_naive.cuh"

__global__ void sgemm_naive(int M, int N, int K, float alpha, 
    const float* __restrict__ A,
    const float* __restrict__ B, 
    float beta, 
    float* __restrict__ C) 
{
    // compute position in C that this thread is responsible for
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    // `if` condition is necessary for when M or N aren't multiples of 32.
    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
        tmp += A[x * K + i] * B[i * N + y];
        }
        // C = α*(A@B)+β*C
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

void lunch_sgemm_naive(int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C, cudaStream_t stream) 
{
    dim3 block(32, 32);
    dim3 grid(CEIL_DIV(M, 32), CEIL_DIV(N, 32));

    sgemm_naive<<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
}