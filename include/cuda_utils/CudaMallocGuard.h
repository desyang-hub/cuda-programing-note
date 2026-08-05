#pragma once

#include <cuda_runtime.h>
#include <memory>
#include "macor.h"
#include "noncopyable.h"

template<class T>
class CudaMallocGuard : public noncopyable {
    struct CudaDeleter {
        void operator()(T* p) const {
            if (p) {
                CUDA_CHECK(cudaFree(p));
            }
        }
    };
    
private:
    std::unique_ptr<T, CudaDeleter> gpu_ptr_{nullptr};
    size_t elements_size_{0};
    
public:
    explicit CudaMallocGuard(size_t elements_size) 
        : gpu_ptr_(nullptr), elements_size_(elements_size) {
        T* ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&ptr, elements_size * sizeof(T)));
        gpu_ptr_.reset(ptr);
    }
    
    // 移动构造和移动赋值自动生成（unique_ptr 支持移动）
    // 拷贝构造和拷贝赋值被 noncopyable 禁用
    
    T* get() const { 
        return gpu_ptr_.get(); 
    }
    
    size_t size() const { 
        return elements_size_; 
    }

    size_t bytes() const {
        return elements_size * sizeof(T);
    }
    
    // 获取设备指针（用于 kernel 调用）
    operator T* () const {  // 可选：隐式转换为设备指针
        return gpu_ptr_.get();
    }
};