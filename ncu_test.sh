ncu --metrics dram__bytes.sum,smsp__sass_thread_inst_executed_op_fma_pred_on.sum \
    -o ncu_profile ./bin/gemm_gpu_test

ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed, \
    l1tex__throughput.avg.pct_of_peak_sustained_elapsed, \
    smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_active \
    ./bin/gemm_gpu_test


# 无 Tiling 版本
ncu --metrics dram__read_transactions.sum,smem__read_transactions.sum ./bin/gemm_gpu_test

# 有 Tiling 版本
ncu --metrics dram__read_transactions.sum,smem__read_transactions.sum ./bin/gemm_gpu_test

ncu --metrics l1tex__throughput.avg.pct_of_peak_sustained_elapsed ./bin/gemm_gpu_test


ncu --metrics smsp__warps_active.avg.per_cycle_active,smsp__warps_active.max.per_cycle_active,inst_executed.avg.per_cycle_elapsed,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__throughput.avg.pct_of_peak_sustained_elapsed,smsp__issue_active.avg.per_cycle_active,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,sm__warps_active.avg.per_cycle_active --target-processes all --kernel-name "sgemm_sharedMemTMTN" --print-summary per-kernel ./bin/ncu_test


# 基础：只看 bank conflict 相关指标
ncu --target-processes all \
    --kernel-name sgemm_sharedMemTMTN \
    --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
smsp__warps_active.avg.per_cycle_active \
    --set full \
    ./sgemm_test