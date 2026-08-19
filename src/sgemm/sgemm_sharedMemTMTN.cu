#include "sgemm/sgemm_sharedMemTMTN.cuh"

// template<size_t BM, size_t BN, size_t BK, size_t TM, size_t TN>
// __global__ void sgemm_sharedMemTMTN(int M, int N, int K, float alpha,
//     const float *A, const float *B, float beta, float *C)
// {
//     constexpr int THREADS_X = BN / TN;
//     constexpr int THREADS_Y = BM / TM;
//     constexpr int TOTAL_THREADS = THREADS_X * THREADS_Y;

//     static_assert(BM % TM == 0 && BN % TN == 0,
//                   "BM must be divisible by TM, BN by TN");
//     static_assert((BM * BK) % TOTAL_THREADS == 0,
//                   "BM*BK must be evenly divisible by thread count");
//     static_assert((BK * BN) % TOTAL_THREADS == 0,
//                   "BK*BN must be evenly divisible by thread count");

//     const int tx  = threadIdx.x;
//     const int ty  = threadIdx.y;
//     const int tid = tx + ty * blockDim.x;

//     __shared__ float s_a[BM][BK + 1];  // ✅ padding
//     __shared__ float s_b[BK][BN];  // ✅ padding

//     float r_c[TM][TN] = {0.0f};

//     constexpr int SA_PER_THREAD = (BM * BK) / TOTAL_THREADS;
//     constexpr int SB_PER_THREAD = (BK * BN) / TOTAL_THREADS;

//     for (int bk = 0; bk < CEIL_DIV(K, BK); ++bk) {
//         // ── Load A tile ──
//         #pragma unroll
//         for (int i = 0; i < SA_PER_THREAD; ++i) {
//             int sa_id     = tid * SA_PER_THREAD + i;
//             int s_a_row   = sa_id / BK;
//             int s_a_col   = sa_id % BK;
//             int g_a_row   = blockIdx.y * BM + s_a_row;
//             int g_a_col   = bk * BK + s_a_col;
//             s_a[s_a_row][s_a_col] = (g_a_row < M && g_a_col < K)
//                 ? A[OFFSET(g_a_row, g_a_col, K)] : 0.f;
//         }

//         // ── Load B tile ──
//         #pragma unroll
//         for (int i = 0; i < SB_PER_THREAD; ++i) {
//             int sb_id     = tid * SB_PER_THREAD + i;
//             int s_b_row   = sb_id / BN;
//             int s_b_col   = sb_id % BN;
//             int g_b_row   = bk * BK + s_b_row;
//             int g_b_col   = blockIdx.x * BN + s_b_col;
//             s_b[s_b_row][s_b_col] = (g_b_row < K && g_b_col < N)
//                 ? B[OFFSET(g_b_row, g_b_col, N)] : 0.f;
//         }

//         __syncthreads();

//         // ── Compute TM×TN per thread ──
//         #pragma unroll
//         for (int i = 0; i < TM; ++i) {
//             #pragma unroll
//             for (int j = 0; j < TN; ++j) {
//                 #pragma unroll
//                 for (int k = 0; k < BK; ++k) {
//                     r_c[i][j] += s_a[ty * TM + i][k] * s_b[k][tx * TN + j];
//                 }
//             }
//         }

//         __syncthreads();
//     }

//     // ── Write back ──
//     #pragma unroll
//     for (int i = 0; i < TM; ++i) {
//         int row = blockIdx.y * BM + ty * TM + i;
//         #pragma unroll
//         for (int j = 0; j < TN; ++j) {
//             int col = blockIdx.x * BN + tx * TN + j;
//             if (row < M && col < N) {
//                 int c_id = OFFSET(row, col, N);
//                 C[c_id] = alpha * r_c[i][j] + beta * C[c_id];
//             }
//         }
//     }
// }

template<size_t BM, size_t BN, size_t BK, size_t TM, size_t TN>
__global__ void sgemm_sharedMemTMTN(int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C) {

        // 使用编译器常量，能够减少运行耗时，提升性能
        constexpr int TOTAL_THREADS = (BN / TN) * (BM / TM);
        constexpr int SA_PER_THREAD = (BM * BK) / TOTAL_THREADS;
        constexpr int SB_PER_THREAD = (BK * BN) / TOTAL_THREADS;

        static_assert((BM * BK) % TOTAL_THREADS == 0);
        static_assert((BK * BN) % TOTAL_THREADS == 0);

        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int tid = tx + ty * blockDim.x;

        __shared__ float s_a[BM][BK + 1];
        __shared__ float s_b[BK][BN];

        float r_c[TM][TN] = {0.0f};

        // 计算每个线程需要加载多少个s_a
        int sa_id_start = SA_PER_THREAD * tid;
        int sb_id_start = SB_PER_THREAD * tid;

        #pragma unroll
        for (int bk = 0; bk < CEIL_DIV(K, BK); ++bk) {
            // 每个线程加载sa_perthreds个元素
            #pragma unroll
            for (int i = 0; i < SA_PER_THREAD; ++i) {
                int sa_id = sa_id_start + i;
                int s_a_row = sa_id / BK;
                int s_a_col = sa_id % BK;

                int a_row = blockIdx.y * BM + s_a_row;
                int a_col = bk * BK + s_a_col;
                s_a[s_a_row][s_a_col] = (a_row < M && a_col < K) ? A[OFFSET(a_row, a_col, K)] : 0;
            }

            #pragma unroll
            for (int i = 0; i < SB_PER_THREAD; ++i) {
                int sb_id = sb_id_start + i;
                int s_b_row = sb_id / BN;
                int s_b_col = sb_id % BN;

                int b_row = bk * BK + s_b_row;
                int b_col = blockIdx.x * BN + s_b_col;
                s_b[s_b_row][s_b_col] = (b_row < K && b_col < N) ? B[OFFSET(b_row, b_col, N)] : 0;
            }

            // 等待数据全部加载完成
            __syncthreads();

            // 开始TM * TN 个元素的计算，朴素版
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                const int s_a_row = ty * TM + i;

                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    const int s_b_col = tx * TN + j;

                    #pragma unroll
                    for (int k = 0; k < BK; ++k) {
                        r_c[i][j] += s_a[s_a_row][k] * s_b[k][s_b_col];
                    }
                }
            }

            __syncthreads();
        }

        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            const int row = blockIdx.y * BM + ty * TM + i;
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int col = blockIdx.x * BN + tx * TN + j;
                if (row < M && col < N) {
                    const int c_id = OFFSET(row, col, N);
                    C[c_id] = alpha * r_c[i][j] + beta * C[c_id];
                }
            }
        }
    }

// v1 版本
template<size_t BM, size_t BN, size_t BK, size_t TM, size_t TN>
__global__ void sgemm_sharedMemTMTN_v1(int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C) {

        // 使用编译器常量，能够减少运行耗时，提升性能
        constexpr int TOTAL_THREADS = (BN / TN) * (BM / TM);
        constexpr int SA_PER_THREAD = (BM * BK) / TOTAL_THREADS;
        constexpr int SB_PER_THREAD = (BK * BN) / TOTAL_THREADS;

        static_assert((BM * BK) % TOTAL_THREADS == 0);
        static_assert((BK * BN) % TOTAL_THREADS == 0);

        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int tid = tx + ty * blockDim.x;

        __shared__ float s_a[BM][BK + 1];
        __shared__ float s_b[BK][BN];

        float r_c[TM][TN] = {0.0f};
        float r_a[TM];
        float r_b[TN];

        // 计算每个线程需要加载多少个s_a
        int sa_id_start = SA_PER_THREAD * tid;
        int sb_id_start = SB_PER_THREAD * tid;

        #pragma unroll
        for (int bk = 0; bk < CEIL_DIV(K, BK); ++bk) {
            // 每个线程加载sa_perthreds个元素
            #pragma unroll
            for (int i = 0; i < SA_PER_THREAD; ++i) {
                int sa_id = sa_id_start + i;
                int s_a_row = sa_id / BK;
                int s_a_col = sa_id % BK;

                int a_row = blockIdx.y * BM + s_a_row;
                int a_col = bk * BK + s_a_col;
                s_a[s_a_row][s_a_col] = (a_row < M && a_col < K) ? A[OFFSET(a_row, a_col, K)] : 0;
            }

            #pragma unroll
            for (int i = 0; i < SB_PER_THREAD; ++i) {
                int sb_id = sb_id_start + i;
                int s_b_row = sb_id / BN;
                int s_b_col = sb_id % BN;

                int b_row = bk * BK + s_b_row;
                int b_col = blockIdx.x * BN + s_b_col;
                s_b[s_b_row][s_b_col] = (b_row < K && b_col < N) ? B[OFFSET(b_row, b_col, N)] : 0;
            }

            // 等待数据全部加载完成
            __syncthreads();

            // 开始TM * TN 个元素的计算，朴素版
            #pragma unroll
            for (int k = 0; k < BK; ++k) {

                #pragma unroll
                for (int i = 0; i < TM; ++i) {
                    r_a[i] = s_a[ty * TM + i][k];
                }

                #pragma unroll
                for (int i = 0; i < TN; ++i) {
                    r_b[i] = s_b[k][tx * TN + i];
                }

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

        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            const int row = blockIdx.y * BM + ty * TM + i;
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int col = blockIdx.x * BN + tx * TN + j;
                if (row < M && col < N) {
                    const int c_id = OFFSET(row, col, N);
                    C[c_id] = alpha * r_c[i][j] + beta * C[c_id];
                }
            }
        }
    }

void lunch_sgemm_sharedMemTMTN(int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C, cudaStream_t stream) {
        const int BK = 8;
        const int TM = 8;
        const int TN = 8;

        if (M >= 128 && N >= 128) {
            const int BM = 128;
            const int BN = 128;
            dim3 block(CEIL_DIV(BN, TN), CEIL_DIV(BM, TM));
            dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
            sgemm_sharedMemTMTN_v1<BM, BN, BK, TM, TN>
            <<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
        }
        else {
            const int BM = 64;
            const int BN = 64;
            dim3 block(CEIL_DIV(BN, TN), CEIL_DIV(BM, TM));
            dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
            sgemm_sharedMemTMTN<BM, BN, BK, TM, TN>
            <<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
        }
    }

