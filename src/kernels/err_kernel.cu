#include "kernels/err_kernel.cuh"
#include "cuda_utils.h"

__global__ void k()
{}

void lunch_err_kernel() {
    k<<<8192, 4096>>>(); // Invalid block size
    CUDA_CHECK(cudaGetLastError());
}

// env CUDA_LOG_FILE=cudaLog.txt ./bin