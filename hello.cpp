#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "stdio.h"
#include "omp.h"

void cuda_hello(sycl::stream stream_ct1){
    stream_ct1 << "Hello World from GPU!\n";
}

int main() {
    int tid, nthreads;

    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        sycl::stream stream_ct1(64 * 1024, 80, cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
            [=](sycl::nd_item<3> item_ct1) {
                cuda_hello(stream_ct1);
            });
    });

#pragma omp parallel private(tid, nthreads)
    {
        tid = omp_get_thread_num();
        nthreads = omp_get_num_threads();

        printf("Greetings from thread %i of %i\n", tid, nthreads);
    }

    dpct::get_current_device().queues_wait_and_throw();
    return 0;
}