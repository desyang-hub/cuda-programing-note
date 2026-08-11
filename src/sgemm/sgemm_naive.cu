#include "sgemm/sgemm_naive.cuh"

// __global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
//     const float *B, float beta, float *C) {
//     const int row = blockIdx.x * blockDim.x + threadIdx.x;
//     const int col = blockIdx.y * blockDim.y + threadIdx.y;

//     // `if` condition is necessary for when M or N aren't multiples of 32.
//     if (col < N && row < M) {
//         float tmp = 0.0;
//         for (int i = 0; i < K; ++i) {
//             tmp += A[OFFSET(row, i, K)] * B[OFFSET(i, col, N)];
//         }
//         // C = α*(A@B)+β*C
//         int c_id = OFFSET(row, col, N);
//         C[c_id] = alpha * tmp + beta * C[c_id];
//     }
// }



// void lunch_gemm_naive(int M, int N, int K, float alpha, const float *A,
//     const float *B, float beta, float *C, cudaStream_t stream) 
// {
//     dim3 block(32, 32);
//     dim3 grid(CEIL_DIV(M, 32), CEIL_DIV(N, 32));


//     sgemm_naive<<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
// }



__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C) {
    
    const int row = threadIdx.y + blockIdx.y * blockDim.y;
    const int col = threadIdx.x + blockIdx.x * blockDim.x;

    // `if` condition is necessary for when M or N aren't multiples of 32.
    if (col < N && row < M) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[OFFSET(row, i, K)] * B[OFFSET(i, col, N)];
        }
        // C = α*(A@B)+β*C
        int c_id = OFFSET(row, col, N);
        C[c_id] = alpha * tmp + beta * C[c_id];
    }
}



void lunch_gemm_naive(int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C, cudaStream_t stream) 
{
    dim3 block(32, 32);
    dim3 grid(CEIL_DIV(N, 32), CEIL_DIV(M, 32));


    sgemm_naive<<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
}