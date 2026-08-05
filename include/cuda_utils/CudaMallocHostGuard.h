#pragma once

#include <cuda_runtime.h>
#include <memory>
#include "macor.h"
#include "noncopyable.h"

template<class T>
class CudaMallocHostGuard : public noncopyable {
    struct CudaHostDeleter {
        void operator()(T* p) const {
            if (p) {
                CUDA_CHECK(cudaFreeHost(p));
            }
        }
    };
    
private:
    std::unique_ptr<T, CudaHostDeleter> host_ptr_{nullptr};
    size_t elements_size_{0};
    
public:
    explicit CudaMallocHostGuard(size_t elements_size) 
        : host_ptr_(nullptr), elements_size_(elements_size) {
        T* ptr = nullptr;
        CUDA_CHECK(cudaMallocHost(&ptr, elements_size * sizeof(T)));
        host_ptr_.reset(ptr);
    }
    
    // 移动语义自动支持
    // 拷贝语义被禁用（noncopyable）
    
    // ==================== 基础访问 ====================
    T* get() const {
        return host_ptr_.get();
    }
    
    T& operator[](size_t i) {
        return host_ptr_.get()[i];
    }
    
    const T& operator[](size_t i) const {
        return host_ptr_.get()[i];
    }
    
    size_t size() const {
        return elements_size_;
    }
    
    size_t bytes() const {
        return elements_size_ * sizeof(T);
    }
    
    // ==================== 同步拷贝 ====================
    // 拷贝到设备内存（同步）
    void copy_to_device(void* device_ptr, cudaMemcpyKind kind = cudaMemcpyHostToDevice) const {
        CUDA_CHECK(cudaMemcpy(device_ptr, host_ptr_.get(), bytes(), kind));
    }
    
    // 从设备内存拷贝（同步）
    void copy_from_device(const void* device_ptr, cudaMemcpyKind kind = cudaMemcpyDeviceToHost) {
        CUDA_CHECK(cudaMemcpy(host_ptr_.get(), device_ptr, bytes(), kind));
    }
    
    // ==================== 异步拷贝（带 Stream） ====================
    // 异步拷贝到设备内存
    void copy_to_device_async(void* device_ptr, cudaStream_t stream = 0) const {
        CUDA_CHECK(cudaMemcpyAsync(device_ptr, host_ptr_.get(), bytes(), 
                                   cudaMemcpyHostToDevice, stream));
    }
    
    // 异步从设备内存拷贝
    void copy_from_device_async(const void* device_ptr, cudaStream_t stream = 0) {
        CUDA_CHECK(cudaMemcpyAsync(host_ptr_.get(), device_ptr, bytes(), 
                                   cudaMemcpyDeviceToHost, stream));
    }
    
    // 异步拷贝（自定义方向）
    void copy_async(void* dst, const void* src, cudaMemcpyKind kind, cudaStream_t stream = 0) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes(), kind, stream));
    }
    
    // ==================== 异步填充 ====================
    // 异步设置内存为特定值（需要设备指针或托管内存）
    void set_async(int value, cudaStream_t stream = 0) {
        CUDA_CHECK(cudaMemsetAsync(host_ptr_.get(), value, bytes(), stream));
    }
    
    // ==================== 带事件同步 ====================
    // 异步拷贝并记录事件
    void copy_to_device_async_with_event(void* device_ptr, 
                                         cudaEvent_t event, 
                                         cudaStream_t stream = 0) {
        copy_to_device_async(device_ptr, stream);
        CUDA_CHECK(cudaEventRecord(event, stream));
    }
    
    // ==================== 实用工具 ====================
    // 零初始化
    void set_zero() {
        CUDA_CHECK(cudaMemset(host_ptr_.get(), 0, bytes()));
    }
    
    // 检查是否有效
    bool is_valid() const {
        return host_ptr_.get() != nullptr;
    }
    
    // 重置大小
    void reset(size_t new_size) {
        T* ptr = nullptr;
        CUDA_CHECK(cudaMallocHost(&ptr, new_size * sizeof(T)));
        host_ptr_.reset(ptr);
        elements_size_ = new_size;
    }
};