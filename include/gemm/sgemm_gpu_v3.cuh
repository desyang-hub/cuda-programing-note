#pragma once

#include <cuda_runtime.h>

void lunch_sgemm_gpu_v3(float* a, float* b, float* c, int M, int K, int N, cudaStream_t stream=0);