#pragma once

#include "cuda_utils.h"

// 这个kernel要做的是每个线程加载多个元素，计算TMxTN个结果
void lunch_sgemm_sharedMemTMTN(int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C, cudaStream_t stream=0);