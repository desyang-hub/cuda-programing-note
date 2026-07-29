#include <cuda_runtime.h>
#include <iostream>

#include "cuda_utils.h"
#include "kernels/vecAdd.cuh"
#include <memory>
#include <thread>

void cuda_event_base() {
    cudaEvent_t event{};
    cudaEventCreate(&event);

    cudaEventDestroy(event);
    
    std::cout << "cuda_event_base finish!" << std::endl;
}

void cuda_event_timer() {
    CudaStream stream;
    CudaEvent start;
    CudaEvent stop;

    int N = 1 << 20;
    int TOTAL_BYTES = N * sizeof(float);

    CudaMallocGuard gA(TOTAL_BYTES);
    CudaMallocGuard gB(TOTAL_BYTES);
    CudaMallocGuard gC(TOTAL_BYTES);

    // 记录事件开始
    cudaEventRecord(start.get(), stream.get());

    std::unique_ptr<float[]> hC = std::unique_ptr<float[]>(new float[N]);

    lunchVecAdd(gA.get(), gB.get(), gC.get(), N, stream.get());

    // 记录事件结束点
    cudaEventRecord(stop.get(), stream.get());

    // 进行同步
    cudaStreamSynchronize(stream.get());

    // 事件耗时
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start.get(), stop.get());

    std::cout << "elapsedTime: " << elapsedTime << std::endl;

    cudaEventRecord(stop.get(), stream.get());

    cudaMemcpy(hC.get(), gC.get(), TOTAL_BYTES, cudaMemcpyDeviceToHost);

    // cudaStreamSynchronize(stream.get()); // 阻塞直到stream为空
    // cudaEventSynchronize(stop.get()); // 以阻塞方式返回结果

    while (cudaSuccess != cudaEventQuery(stop.get())) { // cudaEvnetQuery 以非阻塞的方式返回结果
        std::cout << "waiting!" << std::endl;
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }

    cudaEventElapsedTime(&elapsedTime, start.get(), stop.get());

    std::cout << "D2H Time: " << elapsedTime << std::endl;
}

int main(int argc, char const *argv[])
{

    cuda_event_base();
    cuda_event_timer();


    return 0;
}
