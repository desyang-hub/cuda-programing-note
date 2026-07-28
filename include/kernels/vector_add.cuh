#pragma once

/**
 * @brief 启动向量加法 kernel
 * @param d_a  设备端输入数组 A
 * @param d_b  设备端输入数组 B
 * @param d_c  设备端输出数组 C
 * @param n    元素数量
 */
void launchVectorAdd(const float* d_a, const float* d_b, float* d_c, int n);