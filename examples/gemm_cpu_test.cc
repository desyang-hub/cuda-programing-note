#include "gemm/gemm.cuh"
#include <iostream>
#include <vector>

int main(int argc, char const *argv[])
{
    
    int M = 256;
    int K = 128;
    int N = 1000;

    std::vector<float> a(M * K, 1);
    std::vector<float> b(K * N, 2);
    std::vector<float> c(M * N);
    


    lunch_sgemm_cpu(a.data(), b.data(), c.data(), M, K, N);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (c[OFFSET(i, j, N)] != 2 * K) {
                std::cout << "failed !" << std::endl;
                return -1;
            }
        }
    }

    std::cout << "success" << std::endl;

    return 0;
}
