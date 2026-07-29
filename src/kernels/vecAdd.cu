#include "kernels/vecAdd.cuh"

__global__ void vec_add(const float* A, const float* B, float* C, int n) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n) {
        C[id] = A[id] + B[id];
    }
}

void lunchVecAdd(const float* A, const float* B, float* C, int n, cudaStream_t stream) {
    const int BLOCK_SIZE = 256;
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (stream) {
        vec_add<<<grid_size, BLOCK_SIZE, 0, stream>>>(A, B, C, n);
    }
    else {
        vec_add<<<grid_size, BLOCK_SIZE>>>(A, B, C, n);
    }
    
}