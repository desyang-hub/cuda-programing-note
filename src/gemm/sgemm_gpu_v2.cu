#include "gemm/gemm.cuh"

#include "gemm/sgemm_gpu_v2.cuh"

__global__ void sgemm_gpu_v2(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[BK][BM];
    __shared__ float s_b[BK][BN];

    float r_load_a[4];
    float r_load_b[4];
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0};

    int load_a_smem_m = tid >> 1;
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_k = tid >> 5;
    int load_b_smem_n = (tid & 31) << 2;

    // 当前正在处理的行
    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {

        // 加载线程计算需要用到的数据
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

        // 列行存储
        s_a[load_a_smem_k    ][load_a_smem_m] = r_load_a[0];
        s_a[load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);

        __syncthreads();

        
        #pragma unroll
        for (int tk = 0; tk < BK; tk++) {
            // 加载计算所需要的数据

            // 加载到register上进行运算，减少shared mem的访存次数
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[tk][ty * TM / 2         ]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[tk][ty * TM / 2 + BM / 2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[tk][tx * TN / 2         ]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[tk][tx * TN / 2 + BN / 2]);

            // tm, tn 是相对坐标，指的是每个线程需要计算的8*8个元素的坐标

            // 这个循环计算的是单个位置乘法并与写入位置做加法的过程，随着tk遍历，所有元素都做好乘法并加到指定位置
            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                #pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
                }
            }
        }

        __syncthreads();
    }

    // for (int i = 0; i < TM; ++i) {
    //     const int t_row = by * BM + ty * (TM / 2) + (i >> 2) * (BM / 2) + (i & 3);
    //     for (int half = 0; half < 2; ++half) {
    //         const int t_col = bx * BN + tx * (TN / 2) + half * (BN / 2);
    //         FLOAT4(c[OFFSET(t_row, t_col, N)]) = FLOAT4(r_c[i][half * 4]);
    //     }
    // }

    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
    }
    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
    }

    // #pragma unroll
    // for (int i = 0; i < TM; i++) {
    //     int m = by * BM + ty * TM / 2 + (i >= TM/2 ? BM/2 : 0) + (i % (TM/2));
    //     int n_base = bx * BN + tx * TN / 2;
    //     int addr = OFFSET(m, n_base, N);
    //     FLOAT4(c[addr])        = FLOAT4(r_c[i][0]);
    //     FLOAT4(c[addr + BN/2]) = FLOAT4(r_c[i][4]);
    // }
}

void lunch_sgemm_gpu_v2(float* a, float* b, float* c, int M, int K, int N, cudaStream_t stream) {

    const int BM = 128;
    const int BN = 128;
    const int TM = 8;
    const int TN = 8;

    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm_gpu_v2<<<grid, block, 0, stream>>>(a, b, c, M, K, N);
}