#include "gemm1/gemm_cpu.cuh"
#include "gemm1/gemm_naive.cuh"
#include "gemm1/gemm_coalescing.cuh"
#include "cuda_utils.h"

#include <iostream>
#include <random>
#include <cmath>
#include <functional>
#include <string>
#include <vector>
#include <limits>

using CallFunc = std::function<void(int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C, cudaStream_t stream)>;

// ==================== CPU 参考实现 ====================
static void sgemm_cpu(const float* A, const float* B, float* C,
                      int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ==================== 浮点比较工具 ====================
static bool verify_result(const float* gpu_c, const float* cpu_c,
                          int M, int N, float tol = 1e-2f) {
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    int err_count = 0;

    for (int i = 0; i < M * N; ++i) {
        float abs_err = std::fabs(gpu_c[i] - cpu_c[i]);
        float denom = std::max(std::fabs(cpu_c[i]), 1.0f);
        float rel_err = abs_err / denom;

        max_abs_err = std::max(max_abs_err, abs_err);
        max_rel_err = std::max(max_rel_err, rel_err);

        if (abs_err > tol * denom) {
            if (err_count < 5) {
                int row = i / N, col = i % N;
                std::cerr << "  MISMATCH at (" << row << "," << col << "): "
                          << "gpu=" << gpu_c[i] << " cpu=" << cpu_c[i]
                          << " abs_err=" << abs_err << " rel_err=" << rel_err
                          << std::endl;
            }
            ++err_count;
        }
    }

    std::cout << "  max_abs_err=" << max_abs_err
              << "  max_rel_err=" << max_rel_err
              << "  mismatches=" << err_count << "/" << (M * N)
              << std::endl;

    return err_count == 0;
}

// ==================== 随机初始化 ====================
static void init_random(float* data, size_t n, std::mt19937& rng) {
    // 使用 [-0.5, 0.5] 范围，避免累加溢出且接近真实负载
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (size_t i = 0; i < n; ++i) {
        data[i] = dist(rng);
    }
}

// ==================== 性能基准测试 ====================
struct BenchResult {
    std::string name;
    float ms;
    float tflops;
    bool passed;
};

static BenchResult benchmark_kernel(
    const std::string& name,
    CallFunc launch_fn,
    float* d_a, float* d_b, float* d_c,
    int M, int K, int N,
    int warmup = 3, int repeat = 10)
{
    BenchResult result{name, 0, 0, true};
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        launch_fn(M, N, K, 1, d_a, d_b, 0, d_c, stream);
    }
    cudaStreamSynchronize(stream);

    // Timed runs
    CudaEvent start, end;
    cudaEventRecord(start.get(), stream);
    for (int i = 0; i < repeat; ++i) {
        launch_fn(M, N, K, 1, d_a, d_b, 0, d_c, stream);
    }
    cudaEventRecord(end.get(), stream);
    cudaEventSynchronize(end.get());

    float total_ms;
    cudaEventElapsedTime(&total_ms, start.get(), end.get());
    result.ms = total_ms / repeat;

    double flops = 2.0 * M * K * N;
    result.tflops = static_cast<float>(flops / (result.ms * 1e9));

    cudaStreamDestroy(stream);
    return result;
}

// ==================== Main ====================
int main(int argc, char const* argv[]) {
    std::mt19937 rng(42);

    // ============================================================
    // Phase 1: 正确性验证（小矩阵 + 随机数据 + CPU 对照）
    // ============================================================
    const int TEST_M = 256, TEST_K = 256, TEST_N = 256;
    std::cout << "========== Phase 1: Correctness Check (" 
              << TEST_M << "x" << TEST_K << "x" << TEST_N << ") ==========" << std::endl;

    CudaMallocHostGuard<float> h_a(TEST_M * TEST_K);
    CudaMallocHostGuard<float> h_b(TEST_K * TEST_N);
    CudaMallocHostGuard<float> h_c_gpu(TEST_M * TEST_N);
    CudaMallocHostGuard<float> h_c_cpu(TEST_M * TEST_N);

    init_random(h_a.get(), TEST_M * TEST_K, rng);
    init_random(h_b.get(), TEST_K * TEST_N, rng);

    // CPU reference
    sgemm_cpu(h_a.get(), h_b.get(), h_c_cpu.get(), TEST_M, TEST_K, TEST_N);

    CudaMallocGuard<float> d_a(TEST_M * TEST_K);
    CudaMallocGuard<float> d_b(TEST_K * TEST_N);
    CudaMallocGuard<float> d_c(TEST_M * TEST_N);

    h_a.copy_to_device(d_a.get());
    h_b.copy_to_device(d_b.get());

    // 定义所有待测 kernel
    struct KernelEntry {
        std::string name;
        std::function<void(int M, int N, int K, float alpha, const float *A,
            const float *B, float beta, float *C, cudaStream_t stream)> fn;
    };

    std::vector<KernelEntry> kernels = {
        // {"sgemm_cpu", lunch_sgemm_cpu},
        {"sgemm_naive", lunch_sgemm_naive},
        {"sgemm_coalescing", lunch_sgemm_coalescing},
        // {"sgemm_gpu_v3", lunch_sgemm_gpu_v3},
        // {"sgemm_gpu_ai",  launch_sgemm_optimized},
    };

    bool all_passed = true;
    for (auto& ke : kernels) {
        std::cout << "[Verify] " << ke.name << " ... ";
        cudaMemset(d_c.get(), 0, TEST_M * TEST_N * sizeof(float));
        ke.fn(TEST_M, TEST_N, TEST_K, 1, d_a.get(), d_b.get(), 0, d_c.get(), 0);
        cudaDeviceSynchronize();
        h_c_gpu.copy_from_device(d_c.get());

        bool ok = verify_result(h_c_gpu.get(), h_c_cpu.get(), TEST_M, TEST_N);
        std::cout << (ok ? "PASS ✓" : "FAIL ✗") << std::endl;
        if (!ok) all_passed = false;
    }

    if (!all_passed) {
        std::cerr << "\n❌ Some kernels failed verification. Aborting benchmark." << std::endl;
        return -1;
    }
    std::cout << "\n✅ All kernels passed verification.\n" << std::endl;

    // ============================================================
    // Phase 2: 性能基准测试（大矩阵 + 随机数据）
    // ============================================================
    const int BENCH_M = 4096, BENCH_K = 4096, BENCH_N = 4096;
    std::cout << "========== Phase 2: Benchmark (" 
              << BENCH_M << "x" << BENCH_K << "x" << BENCH_N 
              << ", random data) ==========" << std::endl;

    CudaMallocHostGuard<float> bh_a(BENCH_M * BENCH_K);
    CudaMallocHostGuard<float> bh_b(BENCH_K * BENCH_N);
    init_random(bh_a.get(), BENCH_M * BENCH_K, rng);
    init_random(bh_b.get(), BENCH_K * BENCH_N, rng);

    CudaMallocGuard<float> bd_a(BENCH_M * BENCH_K);
    CudaMallocGuard<float> bd_b(BENCH_K * BENCH_N);
    CudaMallocGuard<float> bd_c(BENCH_M * BENCH_N);

    bh_a.copy_to_device(bd_a.get());
    bh_b.copy_to_device(bd_b.get());

    std::vector<BenchResult> results;
    for (auto& ke : kernels) {
        std::cout << "[Bench] " << ke.name << " ... " << std::flush;
        auto r = benchmark_kernel(ke.name, ke.fn,
                                  bd_a.get(), bd_b.get(), bd_c.get(),
                                  BENCH_M, BENCH_K, BENCH_N);
        std::cout << r.ms << " ms | " << r.tflops << " TFLOPS" << std::endl;
        results.push_back(r);
    }

    // Summary table
    std::cout << "\n========== Summary ==========" << std::endl;
    std::cout << "Kernel          | Time(ms) | TFLOPS   " << std::endl;
    std::cout << "----------------|----------|----------" << std::endl;
    for (auto& r : results) {
        printf("%-16s| %8.3f | %8.3f\n", r.name.c_str(), r.ms, r.tflops);
    }

    return 0;
}