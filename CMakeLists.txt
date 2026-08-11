cmake_minimum_required(VERSION 3.18)
project(CudaCmakeDemo LANGUAGES CXX CUDA)

# ===================== 全局设置 =====================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 86)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

# 自动检测本机 GPU 架构（仅在本机编译时有效）
# 如果交叉编译或打包分发，请显式指定如: set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 90)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()


find_package(CUDAToolkit REQUIRED)

# ===================== 构建链接库 =====================
set(CULIB cutool)
add_library(${CULIB}
    src/kernels/vector_add.cu
    src/kernels/vecAdd.cu
    src/kernels/err_kernel.cu
    src/kernels/vector_sum.cu
    src/gemm/sgemm_cpu.cu
    src/gemm/sgemm_gpu.cu
    src/gemm/sgemm_gpu_v1.cu
    src/gemm/sgemm_gpu_v2.cu
    src/gemm/sgemm_gpu_v3.cu
    src/gemm/sgemm_gpu_ai.cu
    src/gemm/sgemm_gpu_coalescing.cu
    src/gemm/sgemm_gpu_cache.cu
    src/gemm1/gemm_cpu.cu
    src/gemm1/gemm_naive.cu
    src/gemm1/gemm_coalescing.cu
)

target_include_directories(${CULIB} PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(${CULIB} PUBLIC CUDA::cublas CUDA::cudart)


# ===================== 构建目标 =====================
add_executable(cuda_demo
    src/main.cpp
)

target_link_libraries(cuda_demo
    PRIVATE ${CULIB}
)

# ===================== 编译选项 =====================
target_compile_options(cuda_demo PRIVATE
    # CUDA 专用选项
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --extended-lambda
    >
    # Debug 模式：启用设备调试符号 + 行号信息
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:
        -G -lineinfo
    >
    # Release 模式：快速数学优化
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:
        --use_fast_math -O3
    >
)

# ===================== 可选：链接 CUDA 库 =====================
# find_package(CUDAToolkit REQUIRED)
# target_link_libraries(cuda_demo PRIVATE CUDA::cublas CUDA::cudart)


add_subdirectory(examples)