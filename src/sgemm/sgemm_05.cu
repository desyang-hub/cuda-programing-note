#include "sgemm/sgemm_05.cuh"

#include <cassert>

// template <const int BM, const int BN, const int BK, const int TM, const int TN>
// __global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
// sgemm2DBlocktiling(int M, int N, int K, float alpha, 
//     const float *A, const float *B, float beta, float *C) {
//     const uint cRow = blockIdx.y;
//     const uint cCol = blockIdx.x;

//     const uint totalResultsBlocktile = BM * BN;
//     // A thread is responsible for calculating TM*TN elements in the blocktile
//     const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

//     // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
//     assert(numThreadsBlocktile == blockDim.x);

//     // BN/TN are the number of threads to span a column
//     const int threadCol = threadIdx.x % (BN / TN);
//     const int threadRow = threadIdx.x / (BN / TN);

//     // allocate space for the current blocktile in smem
//     __shared__ float As[BM * BK];
//     __shared__ float Bs[BK * BN];

//     // Move blocktile to beginning of A's row and B's column
//     A += cRow * BM * K;
//     B += cCol * BN;
//     C += cRow * BM * N + cCol * BN;

//     // calculating the indices that this thread will load into SMEM
//     const uint innerRowA = threadIdx.x / BK;
//     const uint innerColA = threadIdx.x % BK;
//     // calculates the number of rows of As that are being loaded in a single step
//     // by a single block
//     const uint strideA = numThreadsBlocktile / BK;
//     const uint innerRowB = threadIdx.x / BN;
//     const uint innerColB = threadIdx.x % BN;
//     // for both As and Bs we want each load to span the full column-width, for
//     // better GMEM coalescing (as opposed to spanning full row-width and iterating
//     // across columns)
//     const uint strideB = numThreadsBlocktile / BN;

//     // allocate thread-local cache for results in registerfile
//     float threadResults[TM * TN] = {0.0};
//     // register caches for As and Bs
//     float regM[TM] = {0.0};
//     float regN[TN] = {0.0};

//     // outer-most loop over block tiles
//     for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
//         // populate the SMEM caches
//         for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
//         As[(innerRowA + loadOffset) * BK + innerColA] =
//             A[(innerRowA + loadOffset) * K + innerColA];
//         }
//         for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
//         Bs[(innerRowB + loadOffset) * BN + innerColB] =
//             B[(innerRowB + loadOffset) * N + innerColB];
//         }
//         __syncthreads();

//         // advance blocktile
//         A += BK;     // move BK columns to right
//         B += BK * N; // move BK rows down

//         // calculate per-thread results
//         for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
//         // block into registers
//         for (uint i = 0; i < TM; ++i) {
//             regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
//         }
//         for (uint i = 0; i < TN; ++i) {
//             regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
//         }
//         for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
//             for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
//             threadResults[resIdxM * TN + resIdxN] +=
//                 regM[resIdxM] * regN[resIdxN];
//             }
//         }
//         }
//         __syncthreads();
//     }

//     // write out the results
//     for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
//         for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
//         C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
//             alpha * threadResults[resIdxM * TN + resIdxN] +
//             beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
//         }
//     }
// }



// void lunch_gemm_05(int M, int N, int K, float alpha, const float *A,
//     const float *B, float beta, float *C, cudaStream_t stream) 
// {
//     const uint BK = 8;
//     const uint TM = 8;
//     const uint TN = 8;
//     if (M >= 128 and N >= 128) {
//         const uint BM = 128;
//         const uint BN = 128;
//         dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
//         dim3 blockDim((BM * BN) / (TM * TN));
//         sgemm2DBlocktiling<BM, BN, BK, TM, TN>
//         <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
//     } else {
//         // this is a hacky solution to the underlying problem
//         // of not having proper bounds checking in the kernel
//         const uint BM = 64;
//         const uint BN = 64;
//         dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
//         dim3 blockDim((BM * BN) / (TM * TN));
//         sgemm2DBlocktiling<BM, BN, BK, TM, TN>
//         <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
//     }

// }


// 这个实验是通过增加每个线程加载元素的个数和计算量来提升性能

// template <const int BM, const int BN, const int BK, const int TM, const int TN>
// __global__ void sgemm2DBlocktiling(int M, int N, int K, float alpha, 
//     const float *A, const float *B, float beta, float *C) {

//     const int tx = threadIdx.x;
//     const int ty = threadIdx.y;
//     const int bx = blockIdx.x;
//     const int by = blockIdx.y;

//     const int tid = tx + ty * blockDim.x;

//     // shared mem
//     __shared__ float s_a[BM][BK];
//     __shared__ float s_b[BK][BN];

//     float r_c[TM][TN] = {0.0};
//     float r_a[TM];
//     float r_b[TN];

//     const int threadTotal = blockDim.x * blockDim.y;
//     const int thread_sa_load_num = BM * BK / threadTotal;
//     const int thread_sb_load_num = BK * BN / threadTotal;

//     for (int bk = 0; bk < CEIL_DIV(K, BK); ++bk) {
//         // 先将需要加载的数据全局加载完成 一共有 BM * BK 个数据，每个线程加载多少呢？
        
//         // load s_a
//         for (int i = 0; i < thread_sa_load_num; ++i) {
//             int s_a_id = tid * thread_sa_load_num + i;
//             int s_a_row = s_a_id / BK;
//             int s_a_col = s_a_id % BK;

//             int g_a_row = by * BM + s_a_row;
//             int g_a_col = bk * BK + s_a_col;
//             s_a[s_a_row][s_a_col] = A[OFFSET(g_a_row, g_a_col, K)];
//         }

//         // load s_b
//         for (int i = 0; i < thread_sb_load_num; ++i) {
//             int s_b_id = tid * thread_sb_load_num + i;
//             int s_b_row = s_b_id / BN;
//             int s_b_col = s_b_id % BN;

//             int g_b_row = bk * BK + s_b_row;
//             int g_b_col = bx * BN + s_b_col;
//             s_b[s_b_row][s_b_col] = B[OFFSET(g_b_row, g_b_col, N)];
//         }

//         __syncthreads();

        
//         // 开始计算 每个线程需要计算 TM * TN 个元素

//         for (int k = 0; k < BK; ++k) {
//             // 加载所需元素
//             for (int i = 0; i < TM; ++i) {
//                 r_a[i] = s_a[ty * TM + i][k];
//             }
//             for (int j = 0; j < TN; ++j) {
//                 r_b[j] = s_b[k][tx * TN + j];
//             }

//             for (int i = 0; i < TM; ++i) {
//                 for (int j = 0; j < TN; ++j) {
//                     r_c[i][j] += r_a[i] * r_b[j];
//                 }
//             }
//         }

//         __syncthreads();
//     }


//     for (int i = 0; i < TM; ++i) {
//         const int g_c_row = by * BM + ty * TM + i;
//         for (int j = 0; j < TN; ++j) {
//             const int g_c_col = bx * BN + tx * TN + j;
//             int c_id = OFFSET(g_c_row, g_c_col, N);

//             // 计算结果并写回
//             C[c_id] = alpha * r_c[i][j] + beta * C[c_id];
//         }
//     }

// }


template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm2DBlocktiling(int M, int N, int K, float alpha, 
    const float *A, const float *B, float beta, float *C) {

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int tid = tx + ty * blockDim.x;

    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    float r_c[TM][TN] = {0.0};
    float r_a[TM];
    float r_b[TN];

    const int threadTotal = blockDim.x * blockDim.y;
    const int thread_sa_load_num = BM * BK / threadTotal;
    const int thread_sb_load_num = BK * BN / threadTotal;

    for (int bk = 0; bk < CEIL_DIV(K, BK); ++bk) {
        // 加载数据到共享内存（保持不变）
        for (int i = 0; i < thread_sa_load_num; ++i) {
            int s_a_id = tid * thread_sa_load_num + i;
            int s_a_row = s_a_id / BK;
            int s_a_col = s_a_id % BK;
            int g_a_row = by * BM + s_a_row;
            int g_a_col = bk * BK + s_a_col;
            s_a[s_a_row][s_a_col] = (g_a_row < M && g_a_col < K) ? 
                A[OFFSET(g_a_row, g_a_col, K)] : 0.0f;
        }

        for (int i = 0; i < thread_sb_load_num; ++i) {
            int s_b_id = tid * thread_sb_load_num + i;
            int s_b_row = s_b_id / BN;
            int s_b_col = s_b_id % BN;
            int g_b_row = bk * BK + s_b_row;
            int g_b_col = bx * BN + s_b_col;
            s_b[s_b_row][s_b_col] = (g_b_row < K && g_b_col < N) ? 
                B[OFFSET(g_b_row, g_b_col, N)] : 0.0f;
        }

        __syncthreads();

        // 计算：每次加载一个 k 到寄存器并立即计算
        for (int k = 0; k < BK; ++k) {
            // 加载 A 的一列到寄存器
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                r_a[i] = s_a[ty * TM + i][k];
            }
            
            // 加载 B 的一行到寄存器
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                r_b[j] = s_b[k][tx * TN + j];
            }
            
            // 计算外积
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    r_c[i][j] += r_a[i] * r_b[j];
                }
            }
        }

        __syncthreads();
    }

    // 写回结果（保持不变）
    for (int i = 0; i < TM; ++i) {
        const int g_c_row = by * BM + ty * TM + i;
        if (g_c_row >= M) continue;
        for (int j = 0; j < TN; ++j) {
            const int g_c_col = bx * BN + tx * TN + j;
            if (g_c_col >= N) continue;
            int c_id = OFFSET(g_c_row, g_c_col, N);
            C[c_id] = alpha * r_c[i][j] + beta * C[c_id];
        }
    }
}


void lunch_gemm_05(int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C, cudaStream_t stream) 
{
    const uint BK = 8;
    const uint TM = 8;
    const uint TN = 8;
    if (M >= 128 and N >= 128) {
        const uint BM = 128;
        const uint BN = 128;
        dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
        dim3 blockDim(BN / TN, BM / TM);
        sgemm2DBlocktiling<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    } else {
        // this is a hacky solution to the underlying problem
        // of not having proper bounds checking in the kernel
        const uint BM = 64;
        const uint BN = 64;
        dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
        dim3 blockDim(BN / TN, BM / TM);
        sgemm2DBlocktiling<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    }

}



// template <const int BM, const int BN, const int BK, const int TM, const int TN>
// __global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
// sgemm2DBlocktiling_optimized(int M, int N, int K, float alpha, 
//     const float *A, const float *B, float beta, float *C) {
    
//     const uint cRow = blockIdx.y;
//     const uint cCol = blockIdx.x;
//     const uint tid = threadIdx.x;
    
//     const uint totalResultsBlocktile = BM * BN;
//     const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);
    
//     const int threadCol = tid % (BN / TN);
//     const int threadRow = tid / (BN / TN);
    
//     // 一维共享内存
//     __shared__ float As[BM * BK];
//     __shared__ float Bs[BK * BN];
    
//     // 指针偏移
//     const float* A_ptr = A + cRow * BM * K;
//     const float* B_ptr = B + cCol * BN;
//     float* C_ptr = C + cRow * BM * N + cCol * BN;
    
//     // 加载索引计算（注释版本的优化方法）
//     const uint innerRowA = tid / BK;
//     const uint innerColA = tid % BK;
//     const uint strideA = numThreadsBlocktile / BK;
    
//     const uint innerRowB = tid / BN;
//     const uint innerColB = tid % BN;
//     const uint strideB = numThreadsBlocktile / BN;
    
//     // 寄存器缓存
//     float threadResults[TM * TN] = {0.0};
//     float regM[TM];
//     float regN[TN];
    
//     // 主循环
//     for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
//         // 加载 A - 使用指针偏移
//         #pragma unroll
//         for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
//             uint idx = (innerRowA + loadOffset) * BK + innerColA;
//             uint g_idx = (innerRowA + loadOffset) * K + innerColA;
//             As[idx] = (innerRowA + loadOffset < BM && innerColA < BK) ? 
//                 A_ptr[g_idx] : 0.0f;
//         }
        
//         // 加载 B - 使用指针偏移
//         #pragma unroll
//         for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
//             uint idx = (innerRowB + loadOffset) * BN + innerColB;
//             uint g_idx = (innerRowB + loadOffset) * N + innerColB;
//             Bs[idx] = (innerRowB + loadOffset < BK && innerColB < BN) ? 
//                 B_ptr[g_idx] : 0.0f;
//         }
        
//         __syncthreads();
        
//         // 计算 - 使用一维索引访问共享内存
//         #pragma unroll
//         for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
//             // 加载到寄存器
//             #pragma unroll
//             for (uint i = 0; i < TM; ++i) {
//                 regM[i] = As[((threadRow * TM + i) * BK) + dotIdx];
//             }
//             #pragma unroll
//             for (uint i = 0; i < TN; ++i) {
//                 regN[i] = Bs[(dotIdx * BN) + (threadCol * TN + i)];
//             }
            
//             // 计算外积
//             #pragma unroll
//             for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
//                 #pragma unroll
//                 for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
//                     threadResults[resIdxM * TN + resIdxN] += 
//                         regM[resIdxM] * regN[resIdxN];
//                 }
//             }
//         }
        
//         // 移动指针
//         A_ptr += BK;
//         B_ptr += BK * N;
        
//         __syncthreads();
//     }
    
//     // 写回结果
//     #pragma unroll
//     for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
//         #pragma unroll
//         for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
//             uint row = threadRow * TM + resIdxM;
//             uint col = threadCol * TN + resIdxN;
//             if (row < BM && col < BN) {
//                 C_ptr[row * N + col] = 
//                     alpha * threadResults[resIdxM * TN + resIdxN] +
//                     beta * C_ptr[row * N + col];
//             }
//         }
//     }
// }


// void lunch_gemm_05(int M, int N, int K, float alpha, const float *A,
//     const float *B, float beta, float *C, cudaStream_t stream) 
// {
//     const uint BK = 8;
//     const uint TM = 8;
//     const uint TN = 8;
//     if (M >= 128 && N >= 128) {
//         const uint BM = 128;
//         const uint BN = 128;
//         dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
//         dim3 blockDim((BM * BN) / (TM * TN));  // 一维线程块
//         sgemm2DBlocktiling_optimized<BM, BN, BK, TM, TN>
//         <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
//     } else {
//         const uint BM = 64;
//         const uint BN = 64;
//         dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
//         dim3 blockDim((BM * BN) / (TM * TN));  // 一维线程块
//         sgemm2DBlocktiling_optimized<BM, BN, BK, TM, TN>
//         <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
//     }
// }