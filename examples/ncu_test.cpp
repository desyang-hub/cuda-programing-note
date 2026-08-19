#include "sgemm/sgemm.cuh"

#include <vector>
#include <iostream>

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

template<class Func>
void test(Func func) {
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

    // ── 校验 ──
    // func(M, N, K, ALPHA, dA, dB, BETA, dC, 0);
    // CUDA_CHECK(cudaDeviceSynchronize());
    // res.passed = verify_with_cublas(handle, M, N, K,
    //                                 ALPHA, BETA, dA, dB, dC, dC);
    // if (!res.passed) {
    //     cudaFree(dA); cudaFree(dB); cudaFree(dC);
    //     return res;
    // }

    // // 恢复原始 C
    // CUDA_CHECK(cudaMemcpy(dC, hC.data(), sC, cudaMemcpyHostToDevice));

    // // ── Warmup ──
    cudaStream_t stream = 0;
    // for (int i = 0; i < WARMUP_ITERS; ++i)
    //     func(M, N, K, ALPHA, dA, dB, BETA, dC, stream);
    // CUDA_CHECK(cudaStreamSynchronize(stream));

    // ── Timed Runs ──
    cudaEvent_t evS, evE;
    CUDA_CHECK(cudaEventCreate(&evS));
    CUDA_CHECK(cudaEventCreate(&evE));

    float ms;
    CUDA_CHECK(cudaEventRecord(evS, stream));
    func(M, N, K, ALPHA, dA, dB, BETA, dC, stream);
    CUDA_CHECK(cudaEventRecord(evE, stream));
    CUDA_CHECK(cudaEventSynchronize(evE));
    CUDA_CHECK(cudaEventElapsedTime(&ms, evS, evE));

    std::cout << "kernel compute times: " << ms << " ms" << std::endl;
    std::cout << "M: " << M << " K: " << K << " N: " << N << std::endl;

    std::cout << (gemm_flops(M, N, K) / (ms * 1e-3) / 1e12) << " TFlops" << std::endl;
}

int main(int argc, char const *argv[])
{

    test(lunch_sgemm_sharedMemTMTN);

    return 0;
}
