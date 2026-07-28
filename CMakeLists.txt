cmake_minimum_required(VERSION 3.18)
project(CudaCmakeDemo LANGUAGES CXX CUDA)

# ===================== 全局设置 =====================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

# 自动检测本机 GPU 架构（仅在本机编译时有效）
# 如果交叉编译或打包分发，请显式指定如: set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 90)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# ===================== 构建目标 =====================
add_executable(cuda_demo
    src/main.cpp
    src/kernels/vector_add.cu
)

target_include_directories(cuda_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

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