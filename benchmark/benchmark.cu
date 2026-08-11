#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <functional>

#include "cuda_utils.h"
#include "sgemm/sgemm.cuh"

// ============================================================
//  通用 SGEMM Kernel 接口签名
//  所有被测 kernel 必须遵循此签名
// ============================================================
using SgemmKernelFn = std::function<void(
    int M, int N, int K,
    float alpha, const float *A,
    const float *B, float beta, float *C,
    cudaStream_t stream)>;

struct KernelEntry {
    std::string  name;       // 显示名称
    SgemmKernelFn fn;        // kernel 启动函数
};

// ============================================================
//  配置
// ============================================================
static constexpr int   WARMUP_ITERS = 10;
static constexpr int   BENCH_ITERS  = 100;
static constexpr float ALPHA        = 1.0f;
static constexpr float BETA         = 0.0f;
static constexpr float REL_TOL      = 5e-4f;
static constexpr float ABS_TOL      = 1e-4f;

struct MatSize { int M, N, K; };

// ── 在此添加待测矩阵尺寸 ──
static const std::vector<MatSize> BENCH_SIZES = {
    {  256,   256,   256},
    {  512,   512,   512},
    { 1024,  1024,  1024},
    { 2048,  2048,  2048},
    // { 4096,  4096,  4096},
    // { 8192,  8192,  8192},
};

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
//  cuBLAS Golden Reference (Row-Major 适配)
// ============================================================
static bool verify_with_cublas(cublasHandle_t handle,
                                int M, int N, int K,
                                float alpha, float beta,
                                const float *dA, const float *dB,
                                const float *dC_gpu,
                                const float *dC_orig)
{
    size_t sizeC = (size_t)M * N * sizeof(float);
    float *dC_ref;
    CUDA_CHECK(cudaMalloc(&dC_ref, sizeC));
    CUDA_CHECK(cudaMemcpy(dC_ref, dC_orig, sizeC, cudaMemcpyDeviceToDevice));

    // row-major C = αAB+βC  ⟺  col-major Cᵀ = αBᵀAᵀ+βCᵀ
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha, dB, N, dA, K,
                             &beta, dC_ref, N));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> hRef(M * N), hGpu(M * N);
    CUDA_CHECK(cudaMemcpy(hRef.data(), dC_ref, sizeC, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hGpu.data(), dC_gpu, sizeC, cudaMemcpyDeviceToHost));

    size_t fails = 0;
    float maxRel = 0, maxAbs = 0;
    size_t worstIdx = 0;
    for (size_t i = 0; i < (size_t)M * N; ++i) {
        float a = fabsf(hRef[i] - hGpu[i]);
        float d = fmaxf(fabsf(hRef[i]), 1.0f);
        float r = a / d;
        if (a > ABS_TOL && r > REL_TOL) ++fails;
        if (r > maxRel) { maxRel = r; worstIdx = i; }
        maxAbs = fmaxf(maxAbs, a);
    }

    bool pass = (fails == 0);
    printf("    [Verify] %s  MaxRel=%.2e  MaxAbs=%.2e  Fails=%zu/%zu",
           pass ? "✅" : "❌", maxRel, maxAbs, fails, (size_t)M * N);
    if (!pass)
        printf("  Worst@(%zu,%zu) ref=%.4e gpu=%.4e",
               worstIdx / N, worstIdx % N, hRef[worstIdx], hGpu[worstIdx]);
    printf("\n");

    CUDA_CHECK(cudaFree(dC_ref));
    return pass;
}

// ============================================================
//  单个 Kernel × 单个 Size 的 Benchmark
// ============================================================
struct BenchResult {
    float mean_ms, min_ms, max_ms, std_ms;
    double tflops;
    bool   passed;
};

static BenchResult bench_one(cublasHandle_t handle,
                              const KernelEntry &ke,
                              const MatSize &sz)
{
    const int M = sz.M, N = sz.N, K = sz.K;
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

    BenchResult res{};

    // ── 校验 ──
    ke.fn(M, N, K, ALPHA, dA, dB, BETA, dC, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    res.passed = verify_with_cublas(handle, M, N, K,
                                    ALPHA, BETA, dA, dB, dC, dC);
    if (!res.passed) {
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        return res;
    }

    // 恢复原始 C
    CUDA_CHECK(cudaMemcpy(dC, hC.data(), sC, cudaMemcpyHostToDevice));

    // ── Warmup ──
    cudaStream_t stream = 0;
    for (int i = 0; i < WARMUP_ITERS; ++i)
        ke.fn(M, N, K, ALPHA, dA, dB, BETA, dC, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ── Timed Runs ──
    cudaEvent_t evS, evE;
    CUDA_CHECK(cudaEventCreate(&evS));
    CUDA_CHECK(cudaEventCreate(&evE));

    std::vector<float> times(BENCH_ITERS);
    for (int i = 0; i < BENCH_ITERS; ++i) {
        CUDA_CHECK(cudaEventRecord(evS, stream));
        ke.fn(M, N, K, ALPHA, dA, dB, BETA, dC, stream);
        CUDA_CHECK(cudaEventRecord(evE, stream));
        CUDA_CHECK(cudaEventSynchronize(evE));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, evS, evE));
        times[i] = ms;
    }

    double sum = 0, sq = 0;
    float mn = times[0], mx = times[0];
    for (auto t : times) {
        sum += t; sq += (double)t * t;
        mn = std::min(mn, t); mx = std::max(mx, t);
    }
    double mean = sum / BENCH_ITERS;
    double var  = sq / BENCH_ITERS - mean * mean;

    res.mean_ms = (float)mean;
    res.min_ms  = mn;
    res.max_ms  = mx;
    res.std_ms  = (float)std::sqrt(std::max(var, 0.0));
    res.tflops  = gemm_flops(M, N, K) / (mean * 1e-3) / 1e12;

    CUDA_CHECK(cudaEventDestroy(evS));
    CUDA_CHECK(cudaEventDestroy(evE));
    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
    return res;
}

// ============================================================
//  批量运行入口
// ============================================================
static void run_all_benchmarks(const std::vector<KernelEntry> &kernels,
                                const std::vector<MatSize> &sizes)
{
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    for (const auto &ke : kernels) {
        printf("\n╔══════════════════════════════════════════════════════╗\n");
        printf("║  Kernel: %-44s ║\n", ke.name.c_str());
        printf("╚══════════════════════════════════════════════════════╝\n");
        printf("%-5s %-5s %-5s | %8s %8s %8s %8s | %9s\n",
               "M","N","K","Mean","Min","Max","Std","TFLOPS");
        printf("──────────────────────────────────────────────────────────────\n");

        for (const auto &sz : sizes) {
            BenchResult r = bench_one(handle, ke, sz);
            if (r.passed) {
                printf("%-5d %-5d %-5d | %7.3fms %7.3fms %7.3fms %7.3fms | %8.3f\n",
                       sz.M, sz.N, sz.K,
                       r.mean_ms, r.min_ms, r.max_ms, r.std_ms, r.tflops);
            } else {
                printf("%-5d %-5d %-5d | ❌ VERIFICATION FAILED — skipped\n",
                       sz.M, sz.N, sz.K);
            }
            fflush(stdout);
        }
    }

    CUBLAS_CHECK(cublasDestroy(handle));
}

// ============================================================
//  ★ 在此声明你的所有 kernel ★
// ============================================================
void lunch_gemm_naive(int M, int N, int K, float alpha, const float *A,
                       const float *B, float beta, float *C, cudaStream_t s);
void lunch_sgemm_tiled(int M, int N, int K, float alpha, const float *A,
                       const float *B, float beta, float *C, cudaStream_t s);
void lunch_sgemm_warp(int M, int N, int K, float alpha, const float *A,
                      const float *B, float beta, float *C, cudaStream_t s);
// ... 继续添加

// ============================================================
//  Main
// ============================================================
int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | SMs: %d | Clock: %.2f GHz\n",
           prop.name, prop.multiProcessorCount, prop.clockRate / 1e6);
    printf("Warmup: %d | Iters: %d | Tol: REL=%.0e ABS=%.0e\n",
           WARMUP_ITERS, BENCH_ITERS, REL_TOL, ABS_TOL);

    // ★ 注册所有待测 kernel —— 增删只改这里 ★
    std::vector<KernelEntry> kernels = {
        {"sgemm_naive",  lunch_gemm_naive},
        {"sgemm_coalescing", lunch_gemm_coalescing},
        {"sgemm_03", lunch_gemm_03},
        // {"sgemm_tiled",  lunch_sgemm_tiled},
        // {"sgemm_warp",   lunch_sgemm_warp},
        // {"sgemm_ldg",    lunch_sgemm_ldg},
        // {"sgemm_async",  lunch_sgemm_async},
    };

    run_all_benchmarks(kernels, BENCH_SIZES);
    return 0;
}