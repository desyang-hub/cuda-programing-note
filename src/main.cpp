#include "kernels/vector_add.cuh"
#include "kernels/vecAdd.cuh"
#include "kernels/err_kernel.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <iostream>

#include "cuda_utils.h"

void test_vec_add() {
    // 初始化主机内存
    int N = 1 << 20; // 元素个数
    int TOTAL_BYTES = N * sizeof(float);
    float* A = new float[N];
    std::unique_ptr<float[]> pA(A);
    float* B = new float[N];
    std::unique_ptr<float[]> pB(B);
    float* C = new float[N];
    std::unique_ptr<float[]> pC(C);

    for (int i = 0; i < N; ++i) {
        A[i] = i;
        B[i] = 2 * i;
    }

    // 初始化CUDA空间
    CudaMallocGuard gA(TOTAL_BYTES);
    CudaMallocGuard gB(TOTAL_BYTES);
    CudaMallocGuard gC(TOTAL_BYTES);

    // H2D
    CUDA_CHECK(cudaMemcpy(gA.get(), A, TOTAL_BYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gB.get(), B, TOTAL_BYTES, cudaMemcpyHostToDevice));

    // kernel
    lunchVecAdd(gA.get(), gB.get(), gC.get(), N);
    // 查看内核抛出异常
    CUDA_CHECK(cudaGetLastError());

    // D2H
    CUDA_CHECK(cudaMemcpy(C, gC.get(), TOTAL_BYTES, cudaMemcpyDeviceToHost));

    // compare
    bool pass = true;
    for (int i = 0; i < N; ++i) {
        if (fabsf(C[i] - (static_cast<float>(i) + static_cast<float>(2 * i))) > 1e-5f) {
            pass = false;
        }
    }

    if (pass) {
        printf("verify success.\n");
    }
    else {
        printf("verify failed!\n");
    }
}


int main() {
    test_vec_add();
    lunch_err_kernel();

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