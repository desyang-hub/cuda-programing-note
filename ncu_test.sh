ncu --metrics dram__bytes.sum,smsp__sass_thread_inst_executed_op_fma_pred_on.sum \
    -o ncu_profile ./bin/gemm_gpu_test

ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_active \
./bin/gemm_gpu_test


# 无 Tiling 版本
ncu --metrics dram__read_transactions.sum,smem__read_transactions.sum ./bin/gemm_gpu_test

# 有 Tiling 版本
ncu --metrics dram__read_transactions.sum,smem__read_transactions.sum ./bin/gemm_gpu_test

ncu --metrics l1tex__throughput.avg.pct_of_peak_sustained_elapsed ./bin/gemm_gpu_test