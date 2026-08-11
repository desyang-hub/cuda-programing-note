#pragma once
#include "cuda_utils.h"

#define OFFSET(row, col, cols) ((cols) * (row) + col)
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#define LOAD_FLOAT4(dst, src)       \
    do {                            \
        (dst)[0] = (src)[0];        \
        (dst)[1] = (src)[1];        \
        (dst)[2] = (src)[2];        \
        (dst)[3] = (src)[3];        \
    } while(0)

/// @brief cpu矩阵乘法
/// @param a 
/// @param b 
/// @param c 
/// @param M 
/// @param K 
/// @param N 
void lunch_sgemm_cpu(float* a, float* b, float* c, int M, int K, int N, cudaStream_t=0); // M * K and K * N



void lunch_sgemm_gpu(float* a, float* b, float* c, int M, int K, int N, cudaStream_t=0);