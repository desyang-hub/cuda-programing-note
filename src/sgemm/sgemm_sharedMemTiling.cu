#include "sgemm/sgemm_sharedMemTiling.cuh"

// 这个程序主要是通过shared memory tiling 来进行kernel加速

// 将数据拆分，每个block加载一部分数据，每个block只计算C的一部分

template<size_t BLOCKSIZE>
__global__ void sgemm_sharedMemTiling(int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C) {

        const int tx = threadIdx.x; // 连续
        const int ty = threadIdx.y;

        // 每个线程加载一个s_a, s_b, 计算一个r_c
        __shared__ float s_a[BLOCKSIZE][BLOCKSIZE];
        __shared__ float s_b[BLOCKSIZE][BLOCKSIZE];

        float r_c = 0.0f;

        // 计算block在全局的起始点坐标，全局是M,N
        int row = blockIdx.y * BLOCKSIZE + ty;
        int col = blockIdx.x * BLOCKSIZE + tx;

        size_t mk = M * K;
        size_t kn = K * N;

        // 每个循环处理A的BK个列(B的BK个行)那么最少需要 (K + BK - 1) / BK 次循环
        for (int bk = 0; bk < CEIL_DIV(K, BLOCKSIZE); ++bk) {
            // block内的每个线程加载一个数据，确保shared mem的数据全部加载完成

            // 计算全局a的坐标
            int a_col = bk * BLOCKSIZE + tx;
            int a_id = OFFSET(row, a_col, K);
            s_a[ty][tx] = a_id < mk ? A[a_id] : 0;

            // 计算全局b的坐标
            int b_row = bk * BLOCKSIZE + ty;
            int b_id = OFFSET(b_row, col, N);
            s_b[ty][tx] = b_id < kn ? B[b_id] : 0;

            // 等待一个block中的所有线程将s_a, s_b的数据填充完毕
            __syncthreads();

            // 开始计算
            for (int k = 0; k < BLOCKSIZE; ++k) {
                r_c += s_a[ty][k] * s_b[k][tx];
            }

            // 为了避免没有计算完就开始新一轮数据加载这里需要等待所有数据计算完成
            __syncthreads();
        }

        // 每个线程计算一个结果
        if (row < M && col < N) {
            int c_id = OFFSET(row, col, N);
            C[c_id] = alpha * r_c + beta * C[c_id];
        }        
    }

void lunch_sgemm_sharedMemTiling(int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C, cudaStream_t stream) {
        dim3 block(32, 32);
        dim3 grid(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
        sgemm_sharedMemTiling<32><<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
    }