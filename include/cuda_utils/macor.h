#pragma once

#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(expr_to_check) do {            \
    cudaError_t result  = expr_to_check;          \
    if(result != cudaSuccess)                     \
    {                                             \
        fprintf(stderr,                           \
                "CUDA Runtime Error: %s:%i:%d = %s\n", \
                __FILE__,                         \
                __LINE__,                         \
                result,\
                cudaGetErrorString(result));      \
    }                                             \
} while(0)


#define OFFSET(row, col, cols) ((cols) * (row) + col)
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)