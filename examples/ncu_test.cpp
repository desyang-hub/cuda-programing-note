#include "sgemm/sgemm.cuh"

#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <functional>

const int M = 4096;
const int K = 4096;
const int N = 4096;

static constexpr float ALPHA        = 1.0f;
static constexpr float BETA         = 0.0f;

// ============================================================
//  工具
// ============================================================
static void fill_random(float *p, size_t n) {
    for (size_t i = 0; i < n; ++i)
        p[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
}


static inline double gemm_flops(int M, int N, int K) {
    return 2.0 * M * N * K;
}

// ============================================================
//  宏
// ============================================================
#define CUBLAS_CHECK(call)                                                   \
    do {                                                                     \
        cublasStatus_t s = (call);                                           \
        if (s != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuBLAS Error %s:%d %d\n",                      \
                    __FILE__, __LINE__, (int)s);                             \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// ============================================================
//  cuBLAS Wrapper（适配 Row-Major SGEMM 接口）
// ============================================================
static cublasHandle_t g_cublas_handle = nullptr;

void cublas_sgemm_rowmajor(int M, int N, int K, float alpha,
    const float *A, const float *B,
    float beta, float *C, cudaStream_t stream)
{
// Row-major C = αAB + βC  ⟺  Col-major Cᵀ = αBᵀAᵀ + βCᵀ
// cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
// 这里把 row-major 的 (M,N,K) 映射为 col-major 的 (N,M,K)
CUBLAS_CHECK(cublasSetStream(g_cublas_handle, stream));
CUBLAS_CHECK(cublasSgemm(g_cublas_handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      N, M, K,
      &alpha, B, N, A, K,
      &beta, C, N));
}

template<class Func>
void test(Func func, const char* name) {
    const size_t sA = (size_t)M * K * sizeof(float);
    const size_t sB = (size_t)K * N * sizeof(float);
    const size_t sC = (size_t)M * N * sizeof(float);

    srand(42);
    std::vector<float> hA(M*K), hB(K*N), hC(M*N);
    fill_random(hA.data(), hA.size());
    fill_random(hB.data(), hB.size());
    fill_random(hC.data(), hC.size());

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, sA));
    CUDA_CHECK(cudaMalloc(&dB, sB));
    CUDA_CHECK(cudaMalloc(&dC, sC));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), sA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), sB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, hC.data(), sC, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ★ Warmup: 至少 5 次 ★
    for (int i = 0; i < 5; ++i)
        func(M, N, K, ALPHA, dA, dB, BETA, dC, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ★ 多次计时取最小值 ★
    constexpr int TIMED_ITERS = 20;
    cudaEvent_t evS, evE;
    CUDA_CHECK(cudaEventCreate(&evS));
    CUDA_CHECK(cudaEventCreate(&evE));

    float min_ms = 1e9f;
    for (int i = 0; i < TIMED_ITERS; ++i) {
        CUDA_CHECK(cudaEventRecord(evS, stream));
        func(M, N, K, ALPHA, dA, dB, BETA, dC, stream);
        CUDA_CHECK(cudaEventRecord(evE, stream));
        CUDA_CHECK(cudaEventSynchronize(evE));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, evS, evE));
        min_ms = std::min(min_ms, ms);
    }

    double tflops = gemm_flops(M, N, K) / (min_ms * 1e-3) / 1e12;

    printf("[%s] %.3f ms | %.2f TFLOPS\n", name, min_ms, tflops);

    CUDA_CHECK(cudaEventDestroy(evS));
    CUDA_CHECK(cudaEventDestroy(evE));
    CUDA_CHECK(cudaStreamDestroy(stream));
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

int main(int argc, char const *argv[])
{

    // ★ 创建全局 cuBLAS handle ★
    CUBLAS_CHECK(cublasCreate(&g_cublas_handle));

    // test(lunch_sgemm_sharedMemTMTN, "lunch_sgemm_sharedMemTMTN");

    // test(lunch_sgemm_1DBlocktiling, "sgemm_1DBlocktiling");

    // test(lunch_sgemm_2DBlocktiling, "sgemm_2DBlocktiling");
    
    test(lunch_gemm_naive, "gemm_naive");
    test(lunch_gemm_coalescing, "gemm_coalescing");
    test(lunch_sgemm_sharedMemTiling, "sgemm_sharedMemTiling");

    // test(lunch_gemm_05, "lunch_gemm_05");

    test(cublas_sgemm_rowmajor, "cublas_sgemm_rowmajor");

    CUBLAS_CHECK(cublasDestroy(g_cublas_handle));

    return 0;
}
