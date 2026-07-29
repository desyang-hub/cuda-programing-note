#pragma once

#include <cuda_runtime.h>
#include "macor.h"

class CudaMallocGuard {
private:
    void* gpu_ptr_;
public:
    CudaMallocGuard(size_t bytes_size) : gpu_ptr_{} {
        CUDA_CHECK(cudaMalloc(&gpu_ptr_, bytes_size));
    }
    ~CudaMallocGuard() {
        CUDA_CHECK(cudaFree(gpu_ptr_));
    }

    template<class T = float>
    T* get() {
        return reinterpret_cast<T*>(gpu_ptr_);
    }
};