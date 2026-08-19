find_package(CUDAToolkit REQUIRED)

add_executable(cuda_event_guild cuda_event_guild.cc)
target_link_libraries(cuda_event_guild PRIVATE ${CULIB})

add_executable(reduction_vec_sum_test reduction_vec_sum_test.cc)
target_link_libraries(reduction_vec_sum_test PRIVATE ${CULIB})

add_executable(gemm_cpu_test gemm_cpu_test.cc)
target_link_libraries(gemm_cpu_test PRIVATE ${CULIB})

add_executable(gemm_gpu_test gemm_gpu_test.cc)
target_link_libraries(gemm_gpu_test PRIVATE ${CULIB})

add_executable(strict_test strict_test.cc)
target_link_libraries(strict_test PRIVATE ${CULIB})

add_executable(gemm1_test gemm1_test.cc)
target_link_libraries(gemm1_test PRIVATE ${CULIB})

add_executable(ncu_test ncu_test.cpp)
target_link_libraries(ncu_test PRIVATE ${CULIB})