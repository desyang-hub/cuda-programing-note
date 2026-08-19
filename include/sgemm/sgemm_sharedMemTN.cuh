#pragma once

#include "cuda_utils.h"

// 这个kernel要做的是每个线程计算TN个元素的结果
void lunch_sgemm_sharedMemTN(int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C, cudaStream_t stream=0);