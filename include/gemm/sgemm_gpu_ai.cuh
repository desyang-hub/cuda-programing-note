#pragma once

#include <cuda_runtime.h>

void launch_sgemm_optimized(float* A, float* B, float* C,
    int M, int K, int N, cudaStream_t stream=0);