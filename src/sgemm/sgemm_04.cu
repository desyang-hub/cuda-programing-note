#include "sgemm/sgemm_04.cuh"

// template <const int BM, const int BN, const int BK, const int TM>
// __global__ void gemm_04(int M, int N, int K, float alpha,
//                                    const float *A, const float *B, float beta,
//                                    float *C) {
//   // If we flip x and y here we get ~30% less performance for large matrices.
//   // The current, 30% faster configuration ensures that blocks with sequential
//   // blockIDs access columns of B sequentially, while sharing the same row of A.
//   // The slower configuration would share columns of A, but access into B would
//   // be non-sequential. So the faster configuration has better spatial locality
//   // and hence a greater L2 hit rate.
//   const uint cRow = blockIdx.y;
//   const uint cCol = blockIdx.x;

//   // each warp will calculate 32*TM elements, with 32 being the columnar dim.
//   const int threadCol = threadIdx.x % BN;
//   const int threadRow = threadIdx.x / BN;

//   // allocate space for the current blocktile in SMEM
//   __shared__ float As[BM * BK];
//   __shared__ float Bs[BK * BN];

//   // Move blocktile to beginning of A's row and B's column
//   A += cRow * BM * K;
//   B += cCol * BN;
//   C += cRow * BM * N + cCol * BN;

//   // todo: adjust this to each thread to load multiple entries and
//   // better exploit the cache sizes
//   assert(BM * BK == blockDim.x);
//   assert(BN * BK == blockDim.x);
//   const uint innerColA = threadIdx.x % BK; // warp-level GMEM coalescing
//   const uint innerRowA = threadIdx.x / BK;
//   const uint innerColB = threadIdx.x % BN; // warp-level GMEM coalescing
//   const uint innerRowB = threadIdx.x / BN;

//   // allocate thread-local cache for results in registerfile
//   float threadResults[TM] = {0.0};

//   // outer loop over block tiles
//   for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
//     // populate the SMEM caches
//     As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
//     Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
//     __syncthreads();

//     // advance blocktile
//     A += BK;
//     B += BK * N;

//     // calculate per-thread results
//     for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
//       // we make the dotproduct loop the outside loop, which facilitates
//       // reuse of the Bs entry, which we can cache in a tmp var.
//       float tmpB = Bs[dotIdx * BN + threadCol];
//       for (uint resIdx = 0; resIdx < TM; ++resIdx) {
//         threadResults[resIdx] +=
//             As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
//       }
//     }
//     __syncthreads();
//   }

//   // write out the results
//   for (uint resIdx = 0; resIdx < TM; ++resIdx) {
//     C[(threadRow * TM + resIdx) * N + threadCol] =
//         alpha * threadResults[resIdx] +
//         beta * C[(threadRow * TM + resIdx) * N + threadCol];
//   }
// }



// void lunch_gemm_04(int M, int N, int K, float alpha, const float *A,
//     const float *B, float beta, float *C, cudaStream_t stream) 
// {
//     const uint BM = 64;
//     const uint BN = 64;
//     const uint BK = 8;
//     const uint TM = 8;
//     dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
//     dim3 blockDim((BM * BN) / TM);
//     gemm_04<BM, BN, BK, TM><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
// }


template <const int BM, const int BN, const int BK, const int TM>
__global__ void gemm_04(int M, int N, int K, float alpha,
    const float *A, const float *B, float beta, float *C) 
{
    const int by = blockIdx.y;
    const int bx = blockIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int tid = tx + ty * blockDim.x;

    // 64 行 8 列，对应 ty, tx
    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    // 计算 当前线程处理的 shared Mem 的坐标
    const int a_patch_row = tid / BK;
    const int a_patch_col = tid % BK;

    const int b_patch_row = tid / BN;
    const int b_patch_col = tid % BN;

    // 计算当前线程处理的全局 row, col
    const int g_row = BM * by + a_patch_row;
    const int g_col = BN * bx + b_patch_col;

    // allocate thread-local cache for results in registerfile
    float threadResults[TM] = {0.0};

    const int mk = M * K;
    const int kn = K * N;

    for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
        // 先加载数据，每个线程加载一个
        int a_g_col = bk * BK + a_patch_col;
        int a_id = OFFSET(g_row, a_g_col, K);
        s_a[a_patch_row][a_patch_col] = a_id < mk ? A[a_id] : 0;

        int b_g_row = bk * BK + b_patch_row;
        int b_id = OFFSET(b_g_row, g_col, N);
        s_b[b_patch_row][b_patch_col] = b_id < kn ? B[b_id] : 0;

        // 同步，确保所有数据加载完成
        __syncthreads();

        // 每个线程处理TM个元素，block 一共计算 BM * BN 个元素，每个线程处理 TM个元素
        for (int  i = 0; i < TM; ++i) {
            // block内 每个线程处理8行，上方有 ty * TM 个已经被计算
            int s_a_row = ty * TM + i;
            for (int k = 0; k < BK; ++k) {
                threadResults[i] += s_a[s_a_row][k] * s_b[k][b_patch_col];
            }
        }

        __syncthreads(); // 计算完成同步，避免还没有计算完成又开始加载数据导致覆盖
    }

    // const int mn = M * N;

    // 将线程计算的结果写回到C中
    for (int i = 0; i < TM; ++i) {
        int c_id = OFFSET(by * BM + ty * TM + i, g_col, N);
        C[c_id] = alpha * threadResults[i] + beta * C[c_id];
    }
}


// 上一个版本，每个线程计算一个结果，这里我们要做每个线程做多个处理
void lunch_gemm_04(int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C, cudaStream_t stream) 
{
    const uint BM = 64;
    const uint BN = 64;
    const uint BK = 8;
    const uint TM = 8;

    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim(BN, BM / TM); // BM * BN 是块的规模， / TM 意思是每个线程处理TM个元素，需要这么多线程
    gemm_04<BM, BN, BK, TM><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}