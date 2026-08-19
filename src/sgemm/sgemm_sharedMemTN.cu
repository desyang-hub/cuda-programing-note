#include "sgemm/sgemm_sharedMemTN.cuh"

// 此处还是每个线程加载一个shared_mem 元素，但是每个线程需要计算TN个r_c结果

template<size_t BM, size_t BN, size_t BK, size_t TN>
__global__ void sgemm_sharedMemTN(int M, int N, int K, float alpha,
    const float *A, const float *B, float beta, float *C)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = tx + ty * blockDim.x;

    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    float r_c[TN] = {0.0};

    const int s_a_row = tid / BK;
    const int s_a_col = tid % BK;

    const int s_b_row = tid / BN;
    const int s_b_col = tid % BN;

    int a_row = blockIdx.y * BM + s_a_row;
    int b_col = blockIdx.x * BN + s_b_col;

    for (int bk = 0; bk < CEIL_DIV(K, BK); ++bk) {
        // 加载数据
        int a_col = bk * BK + s_a_col;
        int a_id = OFFSET(a_row, a_col, K);
        s_a[s_a_row][s_a_col] = (a_row < M && a_col < K) ? A[a_id] : 0;

        int b_row = bk * BK + s_b_row;
        int b_id = OFFSET(b_row, b_col, N);
        s_b[s_b_row][s_b_col] = (b_row < K && b_col < N) ? B[b_id] : 0;

        __syncthreads();

        for (int j = 0; j < TN; ++j) {
            const int sb_col = tx * TN + j;
            for (int k = 0; k < BK; ++k) {
                r_c[j] += s_a[ty][k] * s_b[k][sb_col];
            }
        }

        __syncthreads();
    }

    const int row = blockIdx.y * BM + ty;
    for (int j = 0; j < TN; ++j) {
        const int g_col = blockIdx.x * BN + tx * TN + j;

        if (row < M && g_col < N) {
            const int c_id = OFFSET(row, g_col, N);
            C[c_id] = alpha * r_c[j] + beta * C[c_id];
        }
    }
}

// template<size_t BM, size_t BN, size_t BK, size_t TN>
// __global__ void sgemm_sharedMemTN(int M, int N, int K, float alpha,
//     const float *A, const float *B, float beta, float *C)
// {
//     const int tx  = threadIdx.x;
//     const int ty  = threadIdx.y;
//     const int tid = tx + ty * blockDim.x;

//     static_assert(BM == BN, "BM must equal BN");

//     __shared__ float s_a[BM][BK];
//     __shared__ float s_b[BK][BN];

//     float r_c[TN] = {0.0f};

//     // 加载阶段的分工索引
//     const int s_a_row = tid / BK;
//     const int s_a_col = tid % BK;
//     const int s_b_row = tid / BN;
//     const int s_b_col = tid % BN;

//     for (int bk = 0; bk < CEIL_DIV(K, BK); ++bk) {
//         // ✅ Fix 1: 用 s_a_row 计算全局行
//         int g_a_row = blockIdx.y * BM + s_a_row;
//         int a_col   = bk * BK + s_a_col;
//         s_a[s_a_row][s_a_col] = (g_a_row < M && a_col < K)
//             ? A[OFFSET(g_a_row, a_col, K)] : 0.f;

//         // ✅ Fix 2: 用 s_b_col 计算全局列
//         int b_row   = bk * BK + s_b_row;
//         int g_b_col = blockIdx.x * BN + s_b_col;
//         s_b[s_b_row][s_b_col] = (b_row < K && g_b_col < N)
//             ? B[OFFSET(b_row, g_b_col, N)] : 0.f;

//         __syncthreads();

//         // ✅ Fix 3: 用 ty 作为计算行
//         #pragma unroll
//         for (int j = 0; j < TN; ++j) {
//             const int sb_col = tx * TN + j;
//             #pragma unroll
//             for (int k = 0; k < BK; ++k) {
//                 r_c[j] += s_a[ty][k] * s_b[k][sb_col];
//             }
//         }

//         __syncthreads();
//     }

//     // 写回（这部分原本就是正确的）
//     int row = blockIdx.y * BM + ty;
//     for (int i = 0; i < TN; ++i) {
//         int b_col = blockIdx.x * BN + tx * TN + i;
//         if (row < M && b_col < N) {
//             int c_id = OFFSET(row, b_col, N);
//             C[c_id] = alpha * r_c[i] + beta * C[c_id];
//         }
//     }
// }

void lunch_sgemm_sharedMemTN(int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C, cudaStream_t stream) {
        const int BM = 64;
        const int BN = 64;
        const int BK = 8;
        const int TN = 8;
        dim3 block(CEIL_DIV(BN, TN), BM);
        dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

        sgemm_sharedMemTN<BM, BN, BK, TN>
        <<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
    }