#include "sgemm/sgemm_03.cuh"


// template<int BLOCKSIZE>
// __global__ void gemm_03(int M, int N, int K, float alpha, const float *A,
//     const float *B, float beta, float *C)
// {
//     //step one: the output block that we want to compute in this threadblock
//     const uint b_row = blockIdx.x;
//     const uint b_col = blockIdx.y;

//     // step 2: allocate buffer for current block in fast shared mem
//     // shared mem is shared between all threads in a block
//     __shared__ float As[BLOCKSIZE * BLOCKSIZE];
//     __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

//     //step 3: the inner row & col that we're accessing in this thread
//     const uint t_col = threadIdx.x % BLOCKSIZE;
//     const uint t_row = threadIdx.x / BLOCKSIZE;

//     // advance pointers to the starting positions
//     A += b_row * BLOCKSIZE * K;                    // row=cRow, col=0
//     B += b_col * BLOCKSIZE;                        // row=0, col=cCol
//     C += b_row * BLOCKSIZE * N + b_col * BLOCKSIZE; // row=cRow, col=cCol
//     //step 4:
//     float tmp = 0.0;
//     for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
//         // in step 5
//         // Have each thread load one of the elements in A & B
//         // Make the threadCol (=threadIdx.x) the consecutive index
//         // to allow global memory access coalescing
//         As[t_row * BLOCKSIZE + t_col] = A[t_row * K + t_col];
//         Bs[t_row * BLOCKSIZE + t_col] = B[t_row * N + t_col];

//         // block threads in this block until cache is fully populated
//         __syncthreads();

//         A += BLOCKSIZE;
//         B += BLOCKSIZE * N;
//         // step 6:
//         // execute the dotproduct on the currently cached block
//         for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
//             tmp += As[t_row * BLOCKSIZE + dotIdx] *
//                     Bs[dotIdx * BLOCKSIZE + t_col];
//         }
//         // step 7: 
//         // need to sync again at the end, to avoid faster threads
//         // fetching the next block into the cache before slower threads are done
//         __syncthreads();
//     }
//     C[t_row * N + t_col] =
//         alpha * tmp + beta * C[t_row * N + t_col];

// }


// void lunch_gemm_03(int M, int N, int K, float alpha, const float *A,
//     const float *B, float beta, float *C, cudaStream_t stream) 
// {
//     dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
//     dim3 blockDim(32 * 32);
//     // L1 cache becomes useless, since we access GMEM only via SMEM, so we carve
//     // out all of L1 to SMEM. This doesn't currently make a difference, since
//     // occupancy is limited by reg and thread count, but it's good to do anyway.
//     // cudaFuncSetAttribute(gemm_03<32>,
//     //                      cudaFuncAttributePreferredSharedMemoryCarveout,
//     //                      cudaSharedmemCarveoutMaxShared);
//     gemm_03<32><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  
// }




// template<int BLOCKSIZE>
// __global__ void gemm_03(int M, int N, int K, float alpha, const float *A,
//     const float *B, float beta, float *C)
// {
//     //step one: the output block that we want to compute in this threadblock
//     const int tx = threadIdx.x;
//     const int ty = threadIdx.y;
//     const int bx = blockIdx.x;
//     const int by = blockIdx.y;

//     // step 2: allocate buffer for current block in fast shared mem
//     // shared mem is shared between all threads in a block
//     __shared__ float s_a[BLOCKSIZE][BLOCKSIZE];
//     __shared__ float s_b[BLOCKSIZE][BLOCKSIZE];

//     // 每个线程加载一个数据即可
//     int g_row = by * blockDim.y + ty;
//     int g_col = bx * blockDim.x + tx;

//     //step 4:
//     float tmp = 0.0;

//     #pragma unroll
//     for (int bk = 0; bk < CEIL_DIV(K, BLOCKSIZE); ++bk) {
        
//         int a_k = bk * BLOCKSIZE + tx;
//         s_a[ty][tx] = (g_row < M && a_k < K) ? A[OFFSET(g_row, a_k, K)] : 0.0f;

//         int b_k = bk * BLOCKSIZE + ty;
//         s_b[ty][tx] = (b_k < K && g_col < N) ? B[OFFSET(b_k, g_col, N)] : 0.0f;

//         // 等待数据加载完成
//         __syncthreads();

//         // 每个线程只计算一个结果即可
//         #pragma unroll
//         for (int k = 0; k < BLOCKSIZE; ++k) {
//             tmp += s_a[ty][k] * s_b[k][tx];
//         }

//         __syncthreads();
//     }

//     if (g_col < N && g_row < M)
//         C[OFFSET(g_row, g_col, N)] = alpha * tmp + beta * C[OFFSET(g_row, g_col, N)];

// }


template<int BLOCKSIZE>
__global__ void gemm_03(int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C) 
{
    // 每个线程加载一个shared 元素
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    // shared Mem
    __shared__ float s_a[BLOCKSIZE][BLOCKSIZE];
    __shared__ float s_b[BLOCKSIZE][BLOCKSIZE];

    // global m, n
    const int g_row = by * BLOCKSIZE + ty;
    const int g_col = bx * BLOCKSIZE + tx;

    float tmp = 0.0;

    const int mk = M * K;
    const int kn = K * N;

    for (int bk = 0; bk < (K + BLOCKSIZE - 1) / BLOCKSIZE; ++bk) {
        int g_a_col = bk * BLOCKSIZE + tx;
        int a_id = OFFSET(g_row, g_a_col, K);
        s_a[ty][tx] = a_id < mk ? A[a_id] : 0;

        int g_b_row = bk * BLOCKSIZE + ty;
        int b_id = OFFSET(g_b_row, g_col, N);
        s_b[ty][tx] = b_id < kn ? B[b_id] : 0;

        // 等待所有数据加载完成
        __syncthreads();

        // 每个线程计算一个结果，每次计算BLOCKSIZE 个 k
        for (int k = 0; k < BLOCKSIZE; ++k) {
            tmp += s_a[ty][k] * s_b[k][tx];
        }

        __syncthreads();

    }

    if (g_row < M && g_col < N) {
        const int c_id = OFFSET(g_row, g_col, N);
        C[c_id] = alpha * tmp + beta * C[c_id];
    }
}


void lunch_gemm_03(int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C, cudaStream_t stream) 
{
    dim3 blockDim(32, 32);
    dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
    
    gemm_03<32><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}