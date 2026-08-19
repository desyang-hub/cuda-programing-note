#include "sgemm/sgemm_coalescing.cuh"

// 这里对于B的访问是可以合并的，在物理上是连续的，在一行上访问B上的列
// template<int BLOCKSIZE>
// __global__ void sgemm_coalescing(int M, int N, int K, float alpha, const float *A,
//     const float *B, float beta, float *C) 
// {

//     const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
//     const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

//     if (x < M && y < N) {
//         float tmp = 0.0;
//         for (int i = 0; i < K; ++i) {
//             tmp += A[x * K + i] * B[i * N + y];
//         }

//         C[x * N + y] = alpha * tmp + beta * C[x * N + y];
//     }
// }

// 这里的合并访存使用了将数据划分为
template<size_t BLOCKSIZE>
__global__ void sgemm_coalescing(int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C) {

    const int col = blockIdx.x * BLOCKSIZE + threadIdx.x % BLOCKSIZE;
    const int row = blockIdx.y * BLOCKSIZE + threadIdx.x / BLOCKSIZE; // 32 warp 下，col是连续的

    if (row < M && col < N) {
        float r_tmp = 0.0;
        for (int k = 0; k < K; ++k) {
            r_tmp += A[OFFSET(row, k, K)] * B[OFFSET(k, col, N)]; // col连续，那么这里的访存就是连续的，可以进行指令合并成一条，从而降低访存开销
        }

        int c_id = OFFSET(row, col, N);
        C[c_id] = alpha * r_tmp + beta * C[c_id];
    }
}

void lunch_gemm_coalescing(int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C, cudaStream_t stream) 
{
    // gridDim stays the same
    dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
    // make blockDim 1-dimensional, but don't change number of threads
    dim3 blockDim(32 * 32);
    sgemm_coalescing<32><<<gridDim, blockDim, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
}