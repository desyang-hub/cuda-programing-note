#include "gemm/gemm.cuh"

#include <iostream>

void sgemm_cpu(float* a, float* b, float* c, int M, int K, int N, cudaStream_t) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += a[OFFSET(i, k, K)] * b[OFFSET(k, j, N)];
            }
            c[OFFSET(i, j, N)] = sum;
        }
    }
}



void lunch_sgemm_cpu(float* a, float* b, float* c, int M, int K, int N, cudaStream_t stream) {
    sgemm_cpu(a, b, c, M, K, N, stream);
}