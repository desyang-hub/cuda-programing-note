#include "kernels/vector_add.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

int main() {
    constexpr int N = 1 << 20; // 约 100 万元素
    constexpr size_t BYTES = N * sizeof(float);

    // ---- 主机端分配与初始化 ----
    float* h_a = new float[N];
    float* h_b = new float[N];
    float* h_c = new float[N];

    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    // ---- 设备端内存管理 ----
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, BYTES);
    cudaMalloc(&d_b, BYTES);
    cudaMalloc(&d_c, BYTES);

    cudaMemcpy(d_a, h_a, BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, BYTES, cudaMemcpyHostToDevice);

    // ---- 启动 Kernel ----
    launchVectorAdd(d_a, d_b, d_c, N);

    // ---- 结果拷回并验证 ----
    cudaMemcpy(h_c, d_c, BYTES, cudaMemcpyDeviceToHost);

    bool passed = true;
    for (int i = 0; i < N; ++i) {
        float expected = static_cast<float>(i) + static_cast<float>(i * 2);
        if (fabsf(h_c[i] - expected) > 1e-5f) {
            fprintf(stderr, "Verification FAILED at index %d: got %f, expected %f\n",
                    i, h_c[i], expected);
            passed = false;
            break;
        }
    }
    printf("%s\n", passed ? "✅ Verification PASSED" : "❌ Verification FAILED");

    // ---- 清理 ----
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return passed ? 0 : 1;
}