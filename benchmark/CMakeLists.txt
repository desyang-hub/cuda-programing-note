add_executable(bench
    benchmark.cu
)

target_link_libraries(bench
    PRIVATE ${CULIB}
)

# ===================== 编译选项 =====================
# target_compile_options(bench PRIVATE
#     # CUDA 专用选项
#     $<$<COMPILE_LANGUAGE:CUDA>:
#         --expt-relaxed-constexpr
#         --extended-lambda
#     >
#     # Debug 模式：启用设备调试符号 + 行号信息
#     $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:
#         -G -lineinfo
#     >
#     # Release 模式：快速数学优化
#     $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:
#         --use_fast_math -O3
#     >
# )