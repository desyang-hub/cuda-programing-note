#pragma once

#include <cuda_runtime.h>

void lunchVecAdd(const float* A, const float* B, float* C, int n, cudaStream_t stream = nullptr);