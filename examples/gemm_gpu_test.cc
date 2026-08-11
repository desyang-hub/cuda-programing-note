#include "gemm/gemm.cuh"
#include "gemm/sgemm_gpu_v1.cuh"
#include "gemm/sgemm_gpu_v2.cuh"
#include "gemm/sgemm_gpu_v3.cuh"
#include "gemm/sgemm_gpu_ai.cuh"
#include "gemm1/gemm_naive.cuh"
#include "cuda_utils.h"
#include <iostream>

int main(int argc, char const *argv[])
{
    
    int M = 4096;
    int K = 4096;
    int N = 4096;

    CudaMallocHostGuard<float> h_a(M * K);
    CudaMallocHostGuard<float> h_b(K * N);
    CudaMallocHostGuard<float> h_c(M * N);

    for (int i = 0; i < M * K; ++i) h_a[i] = 1;
    for (int i = 0; i < K * N; ++i) h_b[i] = 2;
    for (int i = 0; i < M * N; ++i) h_c[i] = 2;

    // for (int i = 0; i < M; ++i) {
    //     for (int j = 0; j < N; ++j) {
    //         if (h_c[OFFSET(i, j, N)] != 2.0f * K) {
    //             std::cout << h_c[OFFSET(i, j, N)] << std::endl;
    //         }
    //     }
    // }
    // std::cout << "split line ====" << std::endl;

    CudaMallocGuard<float> d_a(M * K);
    CudaMallocGuard<float> d_b(K * N);
    CudaMallocGuard<float> d_c(M * N);

    // move
    h_a.copy_to_device(d_a.get());
    h_b.copy_to_device(d_b.get());
    

    CudaEvent start, end;
    cudaEventRecord(start.get());
    lunch_sgemm_naive(M, N, K, 1, d_a.get(), d_b.get(), 0, d_c.get());
    // lunch_sgemm_gpu(d_a.get(), d_b.get(), d_c.get(), M, K, N);
    // lunch_sgemm_gpu_v1(d_a.get(), d_b.get(), d_c.get(), M, K, N);
    // lunch_sgemm_gpu_v2(d_a.get(), d_b.get(), d_c.get(), M, K, N);
    // lunch_sgemm_gpu_v3(d_a.get(), d_b.get(), d_c.get(), M, K, N);
    // launch_sgemm_optimized(d_a.get(), d_b.get(), d_c.get(), M, K, N);
    cudaEventRecord(end.get());

    cudaEventSynchronize(end.get());
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start.get(), end.get()));
    std::cout << "spends times: " << ms << " ms" << std::endl;

    float total_flops = static_cast<float>(2) * M * K * N;
    std::cout << total_flops << " Flops" << std::endl;
    std::cout << "score : " << total_flops / (1e9 * ms) << " TFlops/s" << std::endl;
    

    h_c.copy_from_device(d_c.get());

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (h_c[OFFSET(i, j, N)] != 2 * K) {
                std::cout << OFFSET(i, j, N) << std::endl;
                std::cout << h_c[OFFSET(i, j, N)] << std::endl;
                std::cout << "verify failed !" << std::endl;
                return -1;
            }
        }
    }
    std::cout << "verify success." << std::endl;

    return 0;
}
