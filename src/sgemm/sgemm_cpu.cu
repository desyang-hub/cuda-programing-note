#include "sgemm/sgemm_cpu.cuh"

// 矩阵乘法cpu实现
void gemm_cpu(int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C) {

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float tmp = 0.0f;
            for (int k = 0; k < K; ++k) {
                tmp += A[OFFSET(i, k, K)] * B[OFFSET(k, j, N)];
            }

            int c_id = OFFSET(i, j, N);
            C[c_id] = alpha * tmp + beta * C[c_id];
        }
    }
}

void lunch_gemm_cpu(int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C, cudaStream_t) {

    }
