// ============================================================
// 配置参数
// ============================================================
#define BM 128          // Block tile M
#define BN 128          // Block tile N
#define BK 8            // Block tile K (每次从global加载的K维度宽度)
#define TM 8            // 每线程计算的M方向元素数
#define TN 8            // 每线程计算的N方向元素数
// 线程块: (BM/TM) * (BN/TN) = 16 * 16 = 256 threads

// ============================================================
// 工具宏
// ============================================================
#define OFFSET(r, c, ld) ((r) * (ld) + (c))

// ============================================================
// Kernel
// ============================================================
__global__ void sgemm_optimized(
    const float* __restrict__ A,   // M x K, row-major
    const float* __restrict__ B,   // K x N, row-major
    float*       __restrict__ C,   // M x N, row-major
    const int M, const int K, const int N)
{
    // ---- 线程在block内的逻辑坐标 ----
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int thread_row = tid / (BN / TN);   // 0..15
    const int thread_col = tid % (BN / TN);   // 0..15

    // ---- Block在矩阵中的起始位置 ----
    const int block_row_start = blockIdx.y * BM;
    const int block_col_start = blockIdx.x * BN;

    // ---- Shared memory: 双缓冲 ----
    __shared__ float As[2][BM * BK];  // A tile: BM x BK
    __shared__ float Bs[2][BK * BN];  // B tile: BK x BN

    // ---- 寄存器: 累加器 TM x TN ----
    float reg_C[TM][TN] = {};

    // ---- 寄存器: 当前K-slice的A列和B行 ----
    float reg_A[TM];
    float reg_B[TN];

    // ---- 全局内存加载参数 ----
    // A tile: BM x BK = 128 x 8 = 1024 floats, 256 threads → 每线程4个float
    // B tile: BK x BN = 8 x 128 = 1024 floats, 256 threads → 每线程4个float
    const int A_TILE_ROW_STRIDE = BM * BK / (BM * BK / 4); // 简化：每线程加载4个
    // 实际用向量化: 每线程加载 1 个 float4

    // A: 256 threads 加载 1024 floats → 每线程 4 floats (1 float4)
    // 将 BM x BK 视为 (BM) 行, 每行 BK=8 floats = 2个float4
    // 256 threads: 每行需要 8/4=2 个线程, 共 128/2=64 行/次, 但只有256线程
    // 实际: 128*8=1024, 1024/256=4 floats/thread = 1 float4/thread ✓

    const int a_tile_rows_per_iter = BM;          // 128
    const int a_float4_per_row    = BK / 4;       // 2
    const int a_total_float4      = a_tile_rows_per_iter * a_float4_per_row; // 256
    // 恰好 256 个 float4，每线程 1 个 ✓

    const int b_tile_rows_per_iter = BK;          // 8
    const int b_float4_per_row     = BN / 4;      // 32
    const int b_total_float4       = b_tile_rows_per_iter * b_float4_per_row; // 256
    // 恰好 256 个 float4 ✓

    // ---- 预计算 A/B 全局指针 ----
    const float* A_base = A + block_row_start * K;
    const float* B_base = B + block_col_start;

    // ==== 加载第 0 个 K-tile 到 buffer 0 ====
    {
        // Load A tile
        int a_idx = tid;  // 0..255, 对应第 a_idx 个 float4
        int a_row = a_idx / a_float4_per_row;       // 0..127
        int a_col4 = (a_idx % a_float4_per_row) * 4; // 0 or 4

        if (block_row_start + a_row < M) {
            float4 tmp = *reinterpret_cast<const float4*>(A_base + a_row * K + a_col4);
            // 转置存入shared: As[col][row] 方便后续按行读取
            As[0][a_col4 + 0 * 1 + a_row * 0] ; // 直接存 row-major: As[row * BK + col]
            As[0][a_row * BK + a_col4 + 0] = tmp.x;
            As[0][a_row * BK + a_col4 + 1] = tmp.y;
            As[0][a_row * BK + a_col4 + 2] = tmp.z;
            As[0][a_row * BK + a_col4 + 3] = tmp.w;
        } else {
            As[0][a_row * BK + a_col4 + 0] = 0.f;
            As[0][a_row * BK + a_col4 + 1] = 0.f;
            As[0][a_row * BK + a_col4 + 2] = 0.f;
            As[0][a_row * BK + a_col4 + 3] = 0.f;
        }

        // Load B tile
        int b_idx = tid;
        int b_row = b_idx / b_float4_per_row;       // 0..7
        int b_col4 = (b_idx % b_float4_per_row) * 4; // 0,4,...,124

        if (b_row < K) {
            float4 tmp = *reinterpret_cast<const float4*>(B_base + b_row * N + b_col4);
            Bs[0][b_row * BN + b_col4 + 0] = tmp.x;
            Bs[0][b_row * BN + b_col4 + 1] = tmp.y;
            Bs[0][b_row * BN + b_col4 + 2] = tmp.z;
            Bs[0][b_row * BN + b_col4 + 3] = tmp.w;
        } else {
            Bs[0][b_row * BN + b_col4 + 0] = 0.f;
            Bs[0][b_row * BN + b_col4 + 1] = 0.f;
            Bs[0][b_row * BN + b_col4 + 2] = 0.f;
            Bs[0][b_row * BN + b_col4 + 3] = 0.f;
        }
    }
    __syncthreads();

    // ==== 主循环: 遍历 K 维度 ====
    const int num_k_tiles = (K + BK - 1) / BK;

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        int buf_cur  = kt & 1;
        int buf_next = 1 - buf_cur;

        // ---- 预取下一个 K-tile (异步) ----
        if (kt + 1 < num_k_tiles) {
            int k_offset = (kt + 1) * BK;

            // Prefetch A
            int a_row  = tid / a_float4_per_row;
            int a_col4 = (tid % a_float4_per_row) * 4;
            int global_a_row = block_row_start + a_row;
            int global_a_col = k_offset + a_col4;

            if (global_a_row < M && global_a_col + 3 < K) {
                float4 tmp = *reinterpret_cast<const float4*>(A + global_a_row * K + global_a_col);
                As[buf_next][a_row * BK + a_col4 + 0] = tmp.x;
                As[buf_next][a_row * BK + a_col4 + 1] = tmp.y;
                As[buf_next][a_row * BK + a_col4 + 2] = tmp.z;
                As[buf_next][a_row * BK + a_col4 + 3] = tmp.w;
            } else {
                for (int j = 0; j < 4; ++j)
                    As[buf_next][a_row * BK + a_col4 + j] = 0.f;
            }

            // Prefetch B
            int b_row  = tid / b_float4_per_row;
            int b_col4 = (tid % b_float4_per_row) * 4;
            int global_b_row = k_offset + b_row;
            int global_b_col = block_col_start + b_col4;

            if (global_b_row < K && global_b_col + 3 < N) {
                float4 tmp = *reinterpret_cast<const float4*>(B + global_b_row * N + global_b_col);
                Bs[buf_next][b_row * BN + b_col4 + 0] = tmp.x;
                Bs[buf_next][b_row * BN + b_col4 + 1] = tmp.y;
                Bs[buf_next][b_row * BN + b_col4 + 2] = tmp.z;
                Bs[buf_next][b_row * BN + b_col4 + 3] = tmp.w;
            } else {
                for (int j = 0; j < 4; ++j)
                    Bs[buf_next][b_row * BN + b_col4 + j] = 0.f;
            }
        }

        // ---- 计算当前 K-tile ----
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // 加载 A 的一列到寄存器 (TM 个元素)
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
                reg_A[tm] = As[buf_cur][(thread_row * TM + tm) * BK + k];
            }
            // 加载 B 的一行到寄存器 (TN 个元素)
            #pragma unroll
            for (int tn = 0; tn < TN; ++tn) {
                reg_B[tn] = Bs[buf_cur][k * BN + thread_col * TN + tn];
            }
            // 外积累加
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
                #pragma unroll
                for (int tn = 0; tn < TN; ++tn) {
                    reg_C[tm][tn] += reg_A[tm] * reg_B[tn];
                }
            }
        }

        __syncthreads();  // 确保预取完成后再切换buffer
    }

    // ==== 写回结果 ====
    #pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
        int global_row = block_row_start + thread_row * TM + tm;
        if (global_row >= M) continue;

        // 向量化写回: TN=8 → 2个float4
        #pragma unroll
        for (int tn = 0; tn < TN; tn += 4) {
            int global_col = block_col_start + thread_col * TN + tn;
            if (global_col + 3 < N) {
                float4 out;
                out.x = reg_C[tm][tn + 0];
                out.y = reg_C[tm][tn + 1];
                out.z = reg_C[tm][tn + 2];
                out.w = reg_C[tm][tn + 3];
                *reinterpret_cast<float4*>(&C[global_row * N + global_col]) = out;
            } else {
                for (int j = 0; j < 4 && global_col + j < N; ++j)
                    C[global_row * N + global_col + j] = reg_C[tm][tn + j];
            }
        }
    }
}

// ============================================================
// Host 启动函数
// ============================================================
void launch_sgemm_optimized(float* A, float* B, float* C,
                            int M, int K, int N, cudaStream_t stream)
{
    dim3 block(BN / TN, BM / TM);  // (16, 16) = 256 threads
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm_optimized<<<grid, block, 0, stream>>>(A, B, C, M, K, N);
}