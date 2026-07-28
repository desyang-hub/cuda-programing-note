#include "kernels/vector_add.cuh"
#include <cstdio>

// Kernel 定义：仅在本翻译单元内可见
__global__ void vectorAddKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 主机端包装函数：封装 kernel 启动配置
void launchVectorAdd(const float* d_a, const float* d_b, float* d_c, int n) {
    constexpr int BLOCK_SIZE = 256;
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    vectorAddKernel<<<grid_size, BLOCK_SIZE>>>(d_a, d_b, d_c, n);

    // 检查 kernel 启动是否出错
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
    }
}