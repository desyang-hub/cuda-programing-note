#include "gemm1/gemm_cpu.cuh"

void gemm_cpu(int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C) 
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int id = OFFSET(i, j, N);
            for (int k = 0; k < K; ++k) {
                int a_id = OFFSET(i, k, K);
                int b_id = OFFSET(k, j, N);
                C[id] += A[a_id] * B[b_id];
            }
            C[id] = alpha * C[id] + beta;
        }
    }
}

void lunch_sgemm_cpu(int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C, cudaStream_t) 
{
    gemm_cpu(M, N, K, alpha, A, B, beta, C);
}