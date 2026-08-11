#include "gemm/sgemm_gpu_v1.cuh"
#include "gemm/gemm.cuh"


// __global__ void sgemm_gpu_v1(float* __restrict__ a, float* __restrict__ b, float* __restrict__ c, int M, int K, int N) {

//     const int BM = 128;
//     const int BN = 128;
//     const int BK = 8;
//     const int TM = 8;
//     const int TN = 8;

//     const int bx = blockIdx.x;
//     const int by = blockIdx.y;
//     const int tx = threadIdx.x;
//     const int ty = threadIdx.y;
//     const int tid = ty * blockDim.x + tx;

//     __shared__ float s_a[BM][BK];
//     __shared__ float s_b[BK][BN];

//     float r_c[TM][TN] = {0.0};

//     int load_a_smem_m = tid >> 1;  // tid/2, row of s_a
//     int load_a_smem_k = (tid & 1) << 2;  // (tid % 2 == 0) ? 0 : 4, col of s_a
//     int load_b_smem_k = tid >> 5;   // tid/32, row of s_b
//     int load_b_smem_n = (tid & 31) << 2;  // (tid % 32) * 4, col of s_b

//     int load_a_gmem_m = by * BM + load_a_smem_m;  // global row of a
//     int load_b_gmem_n = bx * BN + load_b_smem_n;  // global col of b

//     for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
//         int load_a_gmem_k = bk * BK + load_a_smem_k;   // global col of a
//         int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
//         FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr]);
//         int load_b_gmem_k = bk * BK + load_b_smem_k;   // global row of b
//         int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
//         FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr]);

//         __syncthreads();

//         #pragma unroll
//         for (int k = 0; k < BK; k++) {
//             #pragma unroll
//             for (int m = 0; m < TM; m++) {
//                 #pragma unroll
//                 for (int n = 0; n < TN; n++) {
//                     int comp_a_smem_m = ty * TM + m;
//                     int comp_b_smem_n = tx * TN + n;
//                     r_c[m][n] += s_a[comp_a_smem_m][k] * s_b[k][comp_b_smem_n];
//                 }
//             }
//         }

//         __syncthreads();
//     }

//     #pragma unroll
//     for (int i = 0; i < TM; i++) {
//         int store_c_gmem_m = by * BM + ty * TM + i;
//         #pragma unroll
//         for (int j = 0; j < TN; j += 4) {
//             int store_c_gmem_n = bx * BN + tx * TN + j;
//             int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
//             FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][j]);
//         }
//     }
// }


// __global__ void sgemm_gpu_v1(float* __restrict__ a, float* __restrict__ b, float* __restrict__ c, int M, int K, int N) {

//     const int BM = 128;
//     const int BN = 128;
//     const int BK = 8;
//     const int TM = 8;
//     const int TN = 8;

//     int tx = threadIdx.x;
//     int ty = threadIdx.y;
//     int bx = blockIdx.x;
//     int by = blockIdx.y;

//     int tid = tx + bx * blockDim.x;

//     __shared__ float s_a[BM][BK];
//     __shared__ float s_b[BK][BN];

//     float r_c[TM][TN] = {0.0f};

//     // 这里算的只是块内的坐标
//     // 以下用于将tid映射到float4的加载索引
//     int a_patch_row = tid / 2;
//     int a_patch_col = (tid % 2) * 4;

//     // 以下用于将tid映射到float4的加载索引
//     int b_patch_row = tid / 32;
//     int b_patch_col = (tid % 32) * 4;

//     // 计算实际的a,b上的索引
//     int m = by * BM + a_patch_row;
//     int n = bx * BN + b_patch_col;

//     for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
//         // 计算实际的索引
//         int k = bk * BK + a_patch_col;
//         int a_id = OFFSET(m, k, K);
//         FLOAT4(s_a[b_patch_row][b_patch_col]) = FLOAT4(a[a_id]);

//         k = bk * BK + b_patch_col;
//         int b_id = OFFSET(k, n, N);
//         FLOAT4(s_b[b_patch_row][b_patch_col]) = FLOAT4(b[b_id]);

//         // 等待数据加载完成
//         __syncthreads();

//         for (int i = 0; i < TM; ++i) {
//             for (int j = 0; j < TN; ++j) {
//                 for (int tk = 0; tk < BK; ++tk) {
//                     int tm = i + TM * ty;
//                     int tn = j + TN * bk;
//                     r_c[i][j] += s_a[tm][tk] * s_b[tk][tn];
//                 }
//             }
//         }

//     }

//     __syncthreads();
//     // 将计算的结果填充到输出中

//     for (int i = 0; i < TM; ++i) {
//         // by * BM 计算的是块偏移 ty * TM 计算的是线程偏移， i是线程内偏移，由于每个线程内需要处理的是TM * TN 个元素的计算和写入，
//         // 这里计算的是c的绝对行
//         int store_c_m = by * BM + ty * TM + i;

//         for (int j = 0; j < TN; j += 4) { // 每次写入4个
//             // 这里计算的是c的绝对列数
//             int store_c_n = bx * BN + tx * TN + j;
//             int store_c_id = OFFSET(store_c_m, store_c_n, N);

//             // 计算真实的坐标
//             FLOAT4(c[store_c_id]) = FLOAT4(r_c[i][j]);
//         }
//     }
// }


// void lunch_sgemm_gpu_v1(float* a, float* b, float* c, int M, int K, int N, cudaStream_t stream) {
//     // const int outer_repeat = 10, inner_repeat = 1;
//     const int BM = 128, BN = 128, TM = 8, TN = 8;

//     // const int M = 512, N = 512, K = 512;
//     dim3 blockDim(BN / TN, BM / TM); // 每个线程处理TN * TM 个元素， 那么只需要 BN / TN, BM / TM个线程即可
//     dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

//     sgemm_gpu_v1<<<gridDim, blockDim, 0, stream>>>(a, b, c, M, K, N);
// }

// __global__ void sgemm_gpu_v1(float* __restrict__ a, float* __restrict__ b, float* __restrict__ c, const int M, int K, int N) { 
//     // 每个thread要处理TM * TN 个元素
//     const int BM = 128;
//     const int BN = 128;
//     const int BK = 8;
//     const int TM = 8;
//     const int TN = 8;

//     // 申请足够的空间, 存储一个block计算所需要的所有空间，用于存储a, b
//     __shared__ float s_a[BM][BK];
//     __shared__ float s_b[BK][BN];

//     // 单个线程内的计算结果
//     float r_c[TM][TN] = {0.0f};

//     const int tx = threadIdx.x;
//     const int ty = threadIdx.y;
//     const int bx = blockIdx.x;
//     const int by = blockIdx.y;
//     const int tid = tx + ty * blockDim.x; // col + row * cols;

//     // 计算tid对应的a,b矩阵的下标  计算相对于 s_a的坐标
//     int s_a_row = tid / 2;
//     int s_a_col = (tid % 2) * 4;

//     int s_b_row = tid / 32;
//     int s_b_col = (tid % 32) * 4;

//     // 计算tid对应的绝对的 m, n 的位置
//     int a_m = by * BM + s_a_row;
//     int b_n = bx * BN + s_b_col;

//     // 需要遍历所有的k，
//     for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
//         // 加载数据
//         int a_k = bk * BK + s_a_col;
//         int a_id = OFFSET(a_m, a_k, K);
//         FLOAT4(s_a[s_a_row][s_a_col]) = FLOAT4(a[a_id]);

//         int b_k = bk * BK + s_b_row;
//         int b_id = OFFSET(b_k, b_n, N);
//         FLOAT4(s_b[s_b_row][s_b_col]) = FLOAT4(b[b_id]);

//         // 必须要完成所需要计算数据的加载
//         __syncthreads();

//         // 这里做的是单个线程要做的事情(对8*8个元素计算获得结果)
//         // 这里对同一个元素复用了8次，刚好也是带来~8倍的性能提升关键
//         for (int i = 0; i < TM; ++i) {
//             for (int j = 0; j < TN; ++j) {
//                 // 计算实际的坐标
//                 for (int k = 0; k < BK; ++k) {

//                     int actual_m = ty * TM + i;
//                     int actual_n = tx * TN + j;

//                     r_c[i][j] += s_a[actual_m][k] * s_b[k][actual_n];
//                 }
//             }
//         }
//     }

//     __syncthreads();

//     // 将当前线程计算的结果存入到c中，由于每个线程的处理位置不同，所以没有竞争关系
//     for (int i = 0; i < TM; ++i) {
//         int m = by * BM + ty * TM + i;
//         for (int j = 0; j < TN; j += 4) {
//             int n = bx * BN + tx * TN + j;
//             int c_id = OFFSET(m, n, N);
//             FLOAT4(c[c_id]) = FLOAT4(r_c[i][j]);
//         }
//     }
// }

// void lunch_sgemm_gpu_v1(float* a, float* b, float* c, int M, int K, int N, cudaStream_t stream) {
//     const int BM = 128;
//     const int BN = 128;

//     const int TM = 8;
//     const int TN = 8;

//     dim3 block(BN / TN, BM / TM); // 实际 16 * 16 每个thread 处理 TM*TN个元素的计算和写回
//     dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);


//     sgemm_gpu_v1<<<grid, block, 0, stream>>>(a, b, c, M, K, N);
// }


// 重新写一遍

__global__ void sgemm_gpu_v1(
    float* __restrict__ a, float* __restrict__ b, float* __restrict__ c, int M, int K, int N) {
    const int BM = 128;
    const int BN = 128;
    const int TM = 8;
    const int TN = 8;
    const int BK = 8;

    __shared__ float s_a[BM][BK]; // 每个block要处理的a数据将会存储到这里
    __shared__ float s_b[BK][BN]; // 每个block要处理的b数据将会存储到这里

    // 用于存储每个线程累加的结果
    float r_c[TM][TN] = {0.0};

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = tx + ty * blockDim.x;

    // 用于计算当前线程所需要处理的数据的偏移量，这个偏移量是相对于block而言的
    const int patch_a_row = tid / 2;
    const int patch_a_col = (tid % 2) * 4; // 这个计算的是列偏移

    const int patch_b_row = tid / 32;
    const int patch_b_col = (tid % 32) * 4;

    // 计算当前线程正在处理的真实的 行和列
    const int row = by * BM + patch_a_row;
    const int col = bx * BN + patch_b_col;

    // TK太小了，无法一次将所有的计算完成，我们分多次进行
    #pragma unroll
    for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) { // 向上取整，确保所有的k被计算并累加
        // 先将数据加载到s_a, s_b, 需要先计算真实的位置
        const int a_col = bk * BK + patch_a_col;
        const int a_id = OFFSET(row, a_col, K);
        FLOAT4(s_a[patch_a_row][patch_a_col]) = FLOAT4(a[a_id]);

        const int b_row = bk * BK + patch_b_row;
        const int b_id = OFFSET(b_row, col, N);
        FLOAT4(s_b[patch_b_row][patch_b_col]) = FLOAT4(b[b_id]);

        // 等待数据加载完成
        __syncthreads();

        // 每个thread完成计算，和朴素版的计算方法一致
        for (int i = 0; i < TM; ++i) {
            for (int j = 0; j < TN; ++j) {
                for (int k = 0; k < BK; ++k) {
                    // 计算相对s_a,s_b的坐标

                    const int t_row = ty * TM + i;
                    const int t_col = tx * TN + j;

                    r_c[i][j] += s_a[t_row][k] * s_b[k][t_col];
                }
            }
        }

        __syncthreads();
    }

    // __syncthreads();

    // 将结果写入c
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        // 计算全局所在行
        const int t_row = by * BM + ty * TM + i;

        for (int j = 0; j < TN; j += 4) { // 每次移动四个
            const int t_col = bx * BN + tx * TN + j;
            const int c_id = OFFSET(t_row, t_col, N);

            FLOAT4(c[c_id]) = FLOAT4(r_c[i][j]);
        }
    }
}


__global__ void sgemm_gpu_v1_fixed(
    float* __restrict__ a, float* __restrict__ b, float* __restrict__ c, int M, int K, int N) {
    const int BM = 128;
    const int BN = 128;
    const int TM = 8;
    const int TN = 8;
    const int BK = 16;

    __shared__ float s_a[BM][BK]; // 每个block要处理的a数据将会存储到这里
    __shared__ float s_b[BK][BN]; // 每个block要处理的b数据将会存储到这里

    // 用于存储每个线程累加的结果
    float r_c[TM][TN] = {0.0};

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = tx + ty * blockDim.x;

    // 用于计算当前线程所需要处理的数据的偏移量，这个偏移量是相对于block而言的
    const int patch_a_row = tid / 2;
    const int patch_a_col = (tid % 2) * 8; // 这个计算的是列偏移

    const int patch_b_row = tid / 16;
    const int patch_b_col = (tid % 16) * 8;

    // 计算当前线程正在处理的真实的 行和列
    const int row = by * BM + patch_a_row;
    const int col = bx * BN + patch_b_col;

    // TK太小了，无法一次将所有的计算完成，我们分多次进行
    #pragma unroll
    for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) { // 向上取整，确保所有的k被计算并累加
        // 先将数据加载到s_a, s_b, 需要先计算真实的位置
        const int a_col = bk * BK + patch_a_col;
        int a_id = OFFSET(row, a_col, K);
        // FLOAT4(s_a[patch_a_row][patch_a_col]) = FLOAT4(a[a_id]);

        const int b_row = bk * BK + patch_b_row;
        int b_id = OFFSET(b_row, col, N);
        // FLOAT4(s_b[patch_b_row][patch_b_col]) = FLOAT4(b[b_id]);

        for (int i = 0; i < 2; ++i) {
            FLOAT4(s_a[patch_a_row][patch_a_col + i*4]) = FLOAT4(a[a_id]);
            FLOAT4(s_b[patch_b_row][patch_b_col + i*4]) = FLOAT4(b[b_id]);
            a_id += 4;
            b_id += 4;
        }

        // 等待数据加载完成
        __syncthreads();

        // 每个thread完成计算，和朴素版的计算方法一致
        for (int i = 0; i < TM; ++i) {
            for (int j = 0; j < TN; ++j) {
                for (int k = 0; k < BK; ++k) {
                    // 计算相对s_a,s_b的坐标

                    const int t_row = ty * TM + i;
                    const int t_col = tx * TN + j;

                    r_c[i][j] += s_a[t_row][k] * s_b[k][t_col];
                }
            }
        }

        __syncthreads();
    }

    // __syncthreads();

    // 将结果写入c
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        // 计算全局所在行
        const int t_row = by * BM + ty * TM + i;

        for (int j = 0; j < TN; j += 4) { // 每次移动四个
            const int t_col = bx * BN + tx * TN + j;
            const int c_id = OFFSET(t_row, t_col, N);

            FLOAT4(c[c_id]) = FLOAT4(r_c[i][j]);
        }
    }
}

__global__ void sgemm_gpu_v1_repeat(
    float* __restrict__ a, 
    float* __restrict__ b, 
    float* __restrict__ c, 
    int M, int K, int N) {

    // block tile size
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    
    // register tile size => 每个线程要计算8x8个 C中的结果
    const int TM = 8; // tile size 8x8
    const int TN = 8;

    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    float r_c[TM][TN] = {0.0};

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tid = tx + ty * blockDim.x;


    // 计算当前线程需要处理的元素位置 m, n
    int a_batch_row = tid / 2;
    int a_batch_col = (tid % 2) * 4;

    int b_batch_row = tid / 32;
    int b_batch_col = (tid % 32) * 4;

    // 计算实际的位置
    int row = by * BM + a_batch_row;
    int col = bx * BN + b_batch_col;

    // 计算8x8个元素
    for (int bk = 0; bk < CEIL_DIV(K, BK); ++bk) {

        // 循环内每次计算的是 a的一列和 b的一行的乘积， 结果是8x8的矩阵，每轮只计算单个元素的乘积，并累加到r_c

        // 计算真实要加载的 a 的坐标
        int a_k = bk * BK + a_batch_col;
        int a_id = OFFSET(row, a_k, K);
        FLOAT4(s_a[a_batch_row][a_batch_col]) = FLOAT4(a[a_id]);

        int b_k = bk * BK + b_batch_row;
        int b_id = OFFSET(b_k, col, N);
        FLOAT4(s_b[b_batch_row][b_batch_col]) = FLOAT4(b[b_id]);

        // 这一轮加载了所需元素, 以上步骤，每个线程都会加载一部分数据，
        __syncthreads();
        // 同步，此时，s_a和s_b都加载满了数据

        // 每个线程对自己的8x8个元素结果计算，注意这个只是一个k的结果，外层继续遍历，直到所有结果都计算完成
        // 对 8x8 个元素进行计算
        for (int i = 0; i < TM; ++i) {
            for (int j = 0; j < TN; ++j) {
                int t_row = ty * TM + i;
                int t_col = tx * TN + j;
                for (int k = 0; k < BK; ++k) {
                    
                    r_c[i][j] += s_a[t_row][k] * s_b[k][t_col];
                }
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        // 计算实际的行
        int row = by * BM + ty * TM + i;
        for (int j = 0; j < TN; j += 4) {
            int col = bx * BN + tx * TN + j;
            int c_id = OFFSET(row, col, N);
            FLOAT4(c[c_id]) = FLOAT4(r_c[i][j]);
        }
    }
}


__global__ void sgemm_gpu_v1_repeat1(
    float* __restrict__ a, 
    float* __restrict__ b, 
    float* __restrict__ c, 
    int M, int K, int N) {

    const int BM = 128;
    const int BN = 128;
    const int BK = 16;

    // tile block
    const int TM = 8;
    const int TN = 8;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    // block内 线程id
    const int tid = tx + ty * blockDim.x;

    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    // 寄存器，累加保存
    float r_c[TM][TN] = {0.0};

    // 128 x 16 一共 16x16 个 block 内 每个线程需要加载 8 个float
    int a_patch_row = tid / 2; // 每行2个线程
    int a_patch_col = (tid % 2) * 8;

    // 128 / 8 = 16 block 内的 row, col 偏移
    int b_patch_row = tid / 16;
    int b_patch_col = (tid % 16) * 8;

    // 全局偏移 + 块内偏移
    int g_row = by * BM + a_patch_row;
    int g_col = bx * BN + b_patch_col;
    
    // 计算每个 8x8 结果，需要遍历这个数值在 k 上的所有列

    // 单个循环计算 k=8, 所以最多只需要 (k + BK - 1) / BK 次即可全部计算完成
    for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
        
        int g_a_col = bk * BK + a_patch_col;
        int a_id = OFFSET(g_row, g_a_col, K);
        // FLOAT4(s_a[a_patch_row][a_patch_col]) = FLOAT4(a[a_id]);

        int g_b_row = bk * BK + b_patch_row;
        int b_id = OFFSET(g_b_row, g_col, N);
        // FLOAT4(s_b[b_patch_row][b_patch_col]) = FLOAT4(b[b_id]);

        // 每个线程加载8个float
        for (int i = 0; i < 2; ++i) {
            FLOAT4(s_a[a_patch_row][a_patch_col + i * 4]) = FLOAT4(a[a_id + i * 4]);
            FLOAT4(s_b[b_patch_row][b_patch_col + i * 4]) = FLOAT4(b[b_id + i * 4]);
        }

        __syncthreads();

        for (int i = 0; i < TM; ++i) {
            int t_row = ty * TM + i;
            for (int j = 0; j < TN; ++j) {
                int t_col = tx * TN + j;
                for (int k = 0; k < BK; ++k) {
                    r_c[i][j] += s_a[t_row][k] * s_b[k][t_col];
                }
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        int row = by * BM + ty * TM + i;
        for (int j = 0; j < TN; j+=4) {
            int col = bx * BN + tx * TN + j;
            int c_id = OFFSET(row, col, N);
            FLOAT4(c[c_id]) = FLOAT4(r_c[i][j]);
        }
    }
}


void lunch_sgemm_gpu_v1(float* a, float* b, float* c, int M, int K, int N, cudaStream_t stream) {
    const int BM = 128;
    const int BN = 128;
    const int TM = 8; // tile size 8x8
    const int TN = 8;

    dim3 block(BN / TN, BM / TM); // 16 * 16 => 256 threads/block per
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm_gpu_v1<<<grid, block, 0, stream>>>(a, b, c, M, K, N);
    // sgemm_gpu_v1_repeat1<<<grid, block, 0, stream>>>(a, b, c, M, K, N);
}





// __global__ void sgemm_gpu_v1(
//     float* __restrict__ a, float* __restrict__ b, float* __restrict__ c,
//     int M, int K, int N)
// {
//     const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;

//     __shared__ float s_a[BM][BK];
//     __shared__ float s_b[BK][BN];

//     float r_c[TM][TN] = {0.0f};

//     const int tx = threadIdx.x;
//     const int ty = threadIdx.y;
//     const int bx = blockIdx.x;
//     const int by = blockIdx.y;
//     const int tid = ty * blockDim.x + tx;

//     // ---- 加载分工（保持 BK=8 原版，已验证正确）----
//     const int pa_row = tid >> 1;
//     const int pa_col = (tid & 1) << 2;
//     const int pb_row = tid >> 5;
//     const int pb_col = (tid & 31) << 2;

//     const int a_base_m = by * BM + pa_row;
//     const int b_base_n = bx * BN + pb_col;

//     for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
//         // ---- Load（不变）----
//         int a_k = bk * BK + pa_col;
//         FLOAT4(s_a[pa_row][pa_col]) = FLOAT4(a[a_base_m * K + a_k]);

//         int b_k = bk * BK + pb_row;
//         FLOAT4(s_b[pb_row][pb_col]) = FLOAT4(b[b_k * N + b_base_n]);

//         __syncthreads();

//         // ★ 关键改动：k-outer + 寄存器缓存 + 全 unroll ★
//         #pragma unroll
//         for (int k = 0; k < BK; ++k) {
//             // A 的一列片段 → 寄存器
//             float ra[TM];
//             #pragma unroll
//             for (int m = 0; m < TM; ++m)
//                 ra[m] = s_a[ty * TM + m][k];

//             // B 的一行片段 → 寄存器
//             float rb[TN];
//             #pragma unroll
//             for (int n = 0; n < TN; ++n)
//                 rb[n] = s_b[k][tx * TN + n];

//             // 外积累加
//             #pragma unroll
//             for (int m = 0; m < TM; ++m)
//                 #pragma unroll
//                 for (int n = 0; n < TN; ++n)
//                     r_c[m][n] += ra[m] * rb[n];
//         }

//         __syncthreads();
//     }

//     // ---- Store（不变）----
//     #pragma unroll
//     for (int i = 0; i < TM; ++i) {
//         int grow = by * BM + ty * TM + i;
//         #pragma unroll
//         for (int j = 0; j < TN; j += 4) {
//             int gcol = bx * BN + tx * TN + j;
//             FLOAT4(c[grow * N + gcol]) = FLOAT4(r_c[i][j]);
//         }
//     }
// }


// void lunch_sgemm_gpu_v1(float* a, float* b, float* c, int M, int K, int N, cudaStream_t stream) {
//     const int BM = 128;
//     const int BN = 128;
//     const int TM = 8;
//     const int TN = 8;

//     dim3 block(BN / TN, BM / TM); // 16 * 16 => 256 threads/block per
//     dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

//     sgemm_gpu_v1<<<grid, block, 0, stream>>>(a, b, c, M, K, N);
// }



// ==================== cp.async 工具函数 ====================
// __device__ __forceinline__ void cp_async_16B(void* smem_ptr, const void* gmem_ptr) {
//     uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
//     asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" 
//                  :: "r"(smem_addr), "l"(gmem_ptr));
// }

// __device__ __forceinline__ void cp_async_commit() {
//     asm volatile("cp.async.commit_group;\n");
// }

// template<int N>
// __device__ __forceinline__ void cp_async_wait() {
//     asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
// }

// // ==================== Kernel ====================
// __global__ void sgemm_gpu_v1(
//     float* __restrict__ a, float* __restrict__ b, float* __restrict__ c,
//     int M, int K, int N)
// {
//     const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;

//     // Double buffer: 2 × (s_a + s_b)
//     __shared__ float s_a[2][BM][BK];  // 2 × 128×8 = 2×4KB = 8KB
//     __shared__ float s_b[2][BK][BN];  // 2 × 8×128 = 2×4KB = 8KB
//     // Total: 16KB shared memory

//     float r_c[TM][TN] = {0.0f};

//     const int tx = threadIdx.x;   // 0..15
//     const int ty = threadIdx.y;   // 0..15
//     const int bx = blockIdx.x;
//     const int by = blockIdx.y;
//     const int tid = ty * blockDim.x + tx;  // 0..255

//     // ---- Load A mapping (same as before) ----
//     const int pa_row = tid >> 1;         // 0..127
//     const int pa_col = (tid & 1) << 2;   // 0 or 4

//     // ---- Load B mapping (same as before) ----
//     const int pb_row = tid >> 5;         // 0..7
//     const int pb_col = (tid & 31) << 2;  // 0,4,8,...,124

//     const int a_base_m = by * BM + pa_row;
//     const int b_base_n = bx * BN + pb_col;
//     const int num_k_tiles = (K + BK - 1) / BK;

//     // ---- Prologue: async load tile 0 into buffer 0 ----
//     {
//         int a_k = pa_col;
//         int a_addr = a_base_m * K + a_k;
//         cp_async_16B(&s_a[0][pa_row][pa_col], &a[a_addr]);

//         int b_k = pb_row;
//         int b_addr = b_k * N + b_base_n;
//         cp_async_16B(&s_b[0][pb_row][pb_col], &b[b_addr]);
//     }
//     cp_async_commit();

//     // ---- Main loop ----
//     for (int kt = 0; kt < num_k_tiles; ++kt) {
//         int cur_buf = kt & 1;         // current compute buffer
//         int nxt_buf = (kt + 1) & 1;   // next load buffer

//         // Wait for current tile to arrive
//         cp_async_wait<0>();
//         __syncthreads();

//         // Prefetch next tile into nxt_buf (overlaps with compute below)
//         if (kt + 1 < num_k_tiles) {
//             int next_k = (kt + 1) * BK;

//             int a_k = next_k + pa_col;
//             int a_addr = a_base_m * K + a_k;
//             cp_async_16B(&s_a[nxt_buf][pa_row][pa_col], &a[a_addr]);

//             int b_k = next_k + pb_row;
//             int b_addr = b_k * N + b_base_n;
//             cp_async_16B(&s_b[nxt_buf][pb_row][pb_col], &b[b_addr]);
//         }
//         cp_async_commit();

//         // ---- Compute on cur_buf (same i-j-k as your original) ----
//         #pragma unroll
//         for (int i = 0; i < TM; ++i) {
//             #pragma unroll
//             for (int j = 0; j < TN; ++j) {
//                 #pragma unroll
//                 for (int k = 0; k < BK; ++k) {
//                     int comp_a_row = ty * TM + i;
//                     int comp_b_col = tx * TN + j;
//                     r_c[i][j] += s_a[cur_buf][comp_a_row][k] 
//                                * s_b[cur_buf][k][comp_b_col];
//                 }
//             }
//         }

//         __syncthreads();
//     }

//     // ---- Store C (unchanged) ----
//     #pragma unroll
//     for (int i = 0; i < TM; ++i) {
//         int grow = by * BM + ty * TM + i;
//         #pragma unroll
//         for (int j = 0; j < TN; j += 4) {
//             int gcol = bx * BN + tx * TN + j;
//             FLOAT4(c[grow * N + gcol]) = FLOAT4(r_c[i][j]);
//         }
//     }
// }

// void lunch_sgemm_gpu_v1(float* a, float* b, float* c, int M, int K, int N, cudaStream_t stream) {
//     const int BM = 128, BN = 128, TM = 8, TN = 8;
//     dim3 block(BN / TN, BM / TM);  // 16×16 = 256 threads
//     dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
//     sgemm_gpu_v1<<<grid, block, 0, stream>>>(a, b, c, M, K, N);
// }