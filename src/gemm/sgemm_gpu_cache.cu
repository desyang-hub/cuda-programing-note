#include "gemm/sgemm_gpu_cache.cuh"


template<int BLOCKSIZE>
__global__ void sgemm_gpu_cache(float* A, float* B, float* C, int M, int K, int N) {

    //step one: the output block that we want to compute in this threadblock
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    // step 2: allocate buffer for current block in fast shared mem
    // shared mem is shared between all threads in a block
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    //step 3: the inner row & col that we're accessing in this thread
    const uint threadCol = threadIdx.x % BLOCKSIZE;
    const uint threadRow = threadIdx.x / BLOCKSIZE;

    // advance pointers to the starting positions
    A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
    B += cCol * BLOCKSIZE;                        // row=0, col=cCol
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

    float tmp = 0.0;
    // the outer loop advances A along the columns and B along
    // the rows until we have fully calculated the result in C.
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        // Have each thread load one of the elements in A & B from
        // global memory into shared memory.
        // Make the threadCol (=threadIdx.x) the consecutive index
        // to allow global memory access coalescing
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

        // block threads in this block until cache is fully populated
        __syncthreads();

        // advance pointers onto next chunk
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        // execute the dotproduct on the currently cached block
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
            tmp += As[threadRow * BLOCKSIZE + dotIdx] *
                    Bs[dotIdx * BLOCKSIZE + threadCol];
        }
        // need to sync again at the end, to avoid faster threads
        // fetching the next block into the cache before slower threads are done
        __syncthreads();
    }

    C[threadRow * N + threadCol] = tmp;
}

void lunch_sgemm_gpu_cache(float* A, float* B, float* C, int M, int K, int N, cudaStream_t stream) {
    static constexpr int BLOCKSIZE = 32;
    dim3 block(BLOCKSIZE * BLOCKSIZE);
    dim3 grid(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE));

    sgemm_gpu_cache<BLOCKSIZE><<<grid, block, 0, stream>>>(A, B, C, M, K, N);
}