#include "kernels/vector_sum.cuh"

#include "cuda_utils.h"

// __global__ void vector_sum(double* arry, double* sum, int n) {
//     int tid = threadIdx.x;
//     int bid = blockIdx.x;
//     int id = tid + blockDim.x * blockIdx.x;
//     if (id >= n) return;

//     // 计算修改的起始block地址
//     double* x = arry + bid * blockDim.x;

//     // 开始归约操作，每次规模减半
//     for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
//         if (tid < offset) {
//             x[tid] += x[tid + offset];
//         }
//         // 同步，要求等齐所有的计算才开始下一轮规约
//         __syncthreads();
//     }

//     // 开始将x[0]的结果加入到sum中，会有竞争，使用原子操作
//     if (tid == 0) {
//         sum[bid] = x[tid];
//     }
// }

// __global__ void vector_sum(double* arry, double* sum, int n) {
//     int tid = threadIdx.x;
//     int bid = blockIdx.x;
//     int id = tid + blockDim.x * blockIdx.x;
//     if (id >= n) return;

//     // 计算修改的起始block地址
//     double* x = arry + bid * blockDim.x;

//     // 开始归约操作，每次规模减半
//     for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
//         if (tid < offset) {
//             x[tid] += x[tid + offset];
//         }
//         // 同步，要求等齐所有的计算才开始下一轮规约
//         __syncthreads();
//     }

//     // 开始将x[0]的结果加入到sum中，会有竞争，使用原子操作
//     if (tid == 0) {
//         atomicAdd(sum, x[tid]);
//     }
// }

void __global__ vector_sum(double *d_x, double *d_y, int N)
{
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ double s_y[];
    s_y[tid] = (idx < N) ? d_x[idx] : 0.0;
    //  必须完成初始化才能够继续进行计算
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {

        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        d_y[blockIdx.x] = s_y[0];
    }
}

double lunch_vector_sum(double* arry, int n, cudaStream_t stream) {
    dim3 block(128);
    dim3 grid((n + block.x - 1) / block.x);

    CudaMallocGuard<double> d_sum(grid.x);
    cudaMemset(d_sum.get(), 0, d_sum.bytes());
    const int mem = grid.x * sizeof(double);

    CudaEvent start, end;
    cudaEventRecord(start.get(), stream);
    vector_sum<<<grid, block, mem, stream>>>(arry, d_sum.get(), n);
    cudaEventRecord(end.get(), stream);
    cudaEventSynchronize(end.get());

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start.get(), end.get())); 
    std::cout << "spends miliseonds: " << ms << " ms"  << std::endl;

    CudaMallocHostGuard<double> h_sum(grid.x);

    cudaMemcpy(h_sum.get(), d_sum.get(), d_sum.bytes(), cudaMemcpyDeviceToHost);

    double sum = 0;
    for (int i = 0; i < h_sum.size(); ++i) sum += h_sum[i];

    return sum;
}