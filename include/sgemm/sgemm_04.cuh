#pragma once

#include "cuda_utils.h"

void lunch_gemm_04(int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C, cudaStream_t stream=0);