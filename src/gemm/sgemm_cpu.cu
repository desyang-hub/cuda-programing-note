#include "gemm/gemm.cuh"

#include <iostream>

void sgemm_cpu(float* a, float* b, float* c, int M, int K, int N, cudaStream_t) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += a[OFFSET(i, k, K)] * b[OFFSET(k, j, N)];
                // std::cout << i << "," << k << " " << k << "," << j << std::endl;
                // std::cout << "addr: " << OFFSET(i, k, K) << " " << b[OFFSET(k, j, N)] << std::endl;
                // std::cout << a[OFFSET(i, k, K)] << " " << b[OFFSET(k, j, N)] << std::endl;
            }
            c[OFFSET(i, j, N)] = sum;
            // std::cout << "sum" << sum << std::endl;
        }
    }
}



void lunch_sgemm_cpu(float* a, float* b, float* c, int M, int K, int N, cudaStream_t stream) {
    sgemm_cpu(a, b, c, M, K, N, stream);
}