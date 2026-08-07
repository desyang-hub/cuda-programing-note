#pragma once

void lunch_sgemm_gpu_v1(float* a, float* b, float* c, int M, int K, int N, cudaStream_t=0);