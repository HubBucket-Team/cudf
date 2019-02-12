#include "make_indices.cuh"

#include "../../src/rmm/thrust_rmm_allocator.h"
#include <thrust/sequence.h>

std::size_t *make_indices(cudaStream_t         cudaStream,
                          const std::ptrdiff_t length) {
    std::size_t *     devPtr;
    const std::size_t size      = length * sizeof(const std::size_t);
    cudaError_t       cudaError = cudaMalloc(&devPtr, size);
    if (cudaSuccess != cudaError) { return nullptr; }
    thrust::sequence(rmm::exec_policy(cudaStream), devPtr, devPtr + length, 0);
    return devPtr;
}
