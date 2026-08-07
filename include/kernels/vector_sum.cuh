#pragma once

#include <cuda_runtime.h>

/// @brief 使用GPU来进行reduction运算
/// @param arry 数组
/// @param n 数组长度
/// @return 
double lunch_vector_sum(double* arry, int n, cudaStream_t stream=0);