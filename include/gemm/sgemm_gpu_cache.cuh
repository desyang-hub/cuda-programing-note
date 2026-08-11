#pragma once

#include "cuda_utils.h"

void lunch_sgemm_gpu_cache(float* a, float* b, float* c, int M, int K, int N, cudaStream_t=0);