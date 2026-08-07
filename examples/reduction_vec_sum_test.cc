#include "kernels/vector_sum.cuh"
#include "cuda_utils.h"

double init(CudaMallocHostGuard<double>& h_array) {
    double sum = 0;
    for (int i = 0; i < h_array.size(); ++i) {
        sum += i;
        h_array[i] = i;
    }
    return sum;
}

int main(int argc, char const *argv[])
{
    int N = 10000000;
    CudaMallocHostGuard<double> h_array(N);
    CudaMallocGuard<double> d_array(N);

    double gt_sum = init(h_array);

    h_array.copy_to_device(d_array.get());

    // 调用kernel

    double sum;

    for (int i = 0; i < 20; ++i) {
        sum = lunch_vector_sum(d_array.get(), N);
    }

    std::cout << sum << " " << gt_sum << std::endl;

    if (sum == gt_sum) {
        std::cout << "test pass !" << std::endl;
    } else {
        std::cout << "test failed !" << std::endl;
    }
    return 0;
}
