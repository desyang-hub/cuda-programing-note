#include "sgemm/sgemm_sharedMemTiling.cuh"

// 这个程序主要是通过shared memory tiling 来进行kernel加速

// 将数据拆分，每个block加载一部分数据，每个block只计算C的一部分

// template<size_t BLOCKSIZE>
// __global__ void sgemm_sharedMemTiling(int M, int N, int K, float alpha, const float *A,
//     const float *B, float beta, float *C) {

//         const int tx = threadIdx.x; // 连续
//         const int ty = threadIdx.y;

//         // 每个线程加载一个s_a, s_b, 计算一个r_c
//         __shared__ float s_a[BLOCKSIZE][BLOCKSIZE];
//         __shared__ float s_b[BLOCKSIZE][BLOCKSIZE];

//         float r_c = 0.0f;

//         // 计算block在全局的起始点坐标，全局是M,N
//         int row = blockIdx.y * BLOCKSIZE + ty;
//         int col = blockIdx.x * BLOCKSIZE + tx;

//         size_t mk = M * K;
//         size_t kn = K * N;

//         // 每个循环处理A的BK个列(B的BK个行)那么最少需要 (K + BK - 1) / BK 次循环
//         for (int bk = 0; bk < CEIL_DIV(K, BLOCKSIZE); ++bk) {
//             // block内的每个线程加载一个数据，确保shared mem的数据全部加载完成

//             // 计算全局a的坐标
//             int a_col = bk * BLOCKSIZE + tx;
//             int a_id = OFFSET(row, a_col, K);
//             s_a[ty][tx] = a_id < mk ? A[a_id] : 0;

//             // 计算全局b的坐标
//             int b_row = bk * BLOCKSIZE + ty;
//             int b_id = OFFSET(b_row, col, N);
//             s_b[ty][tx] = b_id < kn ? B[b_id] : 0;

//             // 等待一个block中的所有线程将s_a, s_b的数据填充完毕
//             __syncthreads();

//             // 开始计算
//             for (int k = 0; k < BLOCKSIZE; ++k) {
//                 r_c += s_a[ty][k] * s_b[k][tx];
//             }

//             // 为了避免没有计算完就开始新一轮数据加载这里需要等待所有数据计算完成
//             __syncthreads();
//         }

//         // 每个线程计算一个结果
//         if (row < M && col < N) {
//             int c_id = OFFSET(row, col, N);
//             C[c_id] = alpha * r_c + beta * C[c_id];
//         }        
//     }


    // template <const int BLOCKSIZE>
    // __global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
    //                                        const float *A, const float *B,
    //                                        float beta, float *C) {
    //   // the output block that we want to compute in this threadblock
    //   const uint cRow = blockIdx.x;
    //   const uint cCol = blockIdx.y;
    
    //   // allocate buffer for current block in fast shared mem
    //   // shared mem is shared between all threads in a block
    //   __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    //   __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];
    
    //   // the inner row & col that we're accessing in this thread
    //   const uint threadCol = threadIdx.x % BLOCKSIZE;
    //   const uint threadRow = threadIdx.x / BLOCKSIZE;
    
    //   // advance pointers to the starting positions
    //   A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
    //   B += cCol * BLOCKSIZE;                        // row=0, col=cCol
    //   C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol
    
    //   float tmp = 0.0;
    //   for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    //     // Have each thread load one of the elements in A & B
    //     // Make the threadCol (=threadIdx.x) the consecutive index
    //     // to allow global memory access coalescing
    //     As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    //     Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];
    
    //     // block threads in this block until cache is fully populated
    //     __syncthreads();
    //     A += BLOCKSIZE;
    //     B += BLOCKSIZE * N;
    
    //     // execute the dotproduct on the currently cached block
    //     for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
    //       tmp += As[threadRow * BLOCKSIZE + dotIdx] *
    //              Bs[dotIdx * BLOCKSIZE + threadCol];
    //     }
    //     // need to sync again at the end, to avoid faster threads
    //     // fetching the next block into the cache before slower threads are done
    //     __syncthreads();
    //   }
    //   C[threadRow * N + threadCol] =
    //       alpha * tmp + beta * C[threadRow * N + threadCol];
    // }

// void lunch_sgemm_sharedMemTiling(int M, int N, int K, float alpha, const float *A,
//     const float *B, float beta, float *C, cudaStream_t stream) {
//         dim3 block(32, 32);
//         dim3 grid(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
//         sgemm_sharedMemTiling<32><<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
//     }



// 换成一维数组，速度有明显提升

template<size_t BLOCKSIZE>
__global__ void sgemm_sharedMemTiling(int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C) {

        const int tx = threadIdx.x; // 连续

        // 每个线程加载一个s_a, s_b, 计算一个r_c
        __shared__ float s_a[BLOCKSIZE * BLOCKSIZE];
        __shared__ float s_b[BLOCKSIZE * BLOCKSIZE];

        float r_c = 0.0f;

        const int s_row = tx / BLOCKSIZE;
        const int s_col = tx % BLOCKSIZE;

        // 计算block在全局的起始点坐标，全局是M,N
        int row = blockIdx.y * BLOCKSIZE + s_row;
        int col = blockIdx.x * BLOCKSIZE + s_col;

        // 每个循环处理A的BK个列(B的BK个行)那么最少需要 (K + BK - 1) / BK 次循环
        for (int bk = 0; bk < CEIL_DIV(K, BLOCKSIZE); ++bk) {
            // block内的每个线程加载一个数据，确保shared mem的数据全部加载完成
            // 计算全局a的坐标
            int a_col = bk * BLOCKSIZE + s_col;
            int a_id = OFFSET(row, a_col, K);
            s_a[tx] = (row < M && a_col < K) ? A[a_id] : 0;

            // 计算全局b的坐标
            int b_row = bk * BLOCKSIZE + s_row;
            int b_id = OFFSET(b_row, col, N);
            s_b[tx] = (b_row < K && col < N) ? B[b_id] : 0;

            // 等待一个block中的所有线程将s_a, s_b的数据填充完毕
            __syncthreads();

            // 开始计算
            for (int k = 0; k < BLOCKSIZE; ++k) {
                // r_c += s_a[ty][k] * s_b[k][tx];
                r_c += s_a[OFFSET(s_row, k, BLOCKSIZE)] * s_b[OFFSET(k, s_col, BLOCKSIZE)];
            }

            // 为了避免没有计算完就开始新一轮数据加载这里需要等待所有数据计算完成
            __syncthreads();
        }

        // 每个线程计算一个结果
        if (row < M && col < N) {
            int c_id = OFFSET(row, col, N);
            if (row < M && col < N) {
                C[c_id] = alpha * r_c + beta * C[c_id];
            }
        }        
    }


void lunch_sgemm_sharedMemTiling(int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C, cudaStream_t stream) {
        const int BLOCKSIZE = 32;
        dim3 block(BLOCKSIZE, BLOCKSIZE);
        dim3 grid(CEIL_DIV(N, BLOCKSIZE), CEIL_DIV(M, BLOCKSIZE));
        sgemm_sharedMemTiling<BLOCKSIZE><<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
    }