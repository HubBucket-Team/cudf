#include "make_indices.cuh"

#include "../../src/rmm/thrust_rmm_allocator.h"
#include <thrust/sequence.h>

std::size_t *make_indices(cudaStream_t         cudaStream,
                          const std::ptrdiff_t length) {
    std::size_t *     devPtr;
    const std::size_t size = length * sizeof(const std::size_t);

    rmmError_t rmmStatus =
        RMM_ALLOC(reinterpret_cast<void **>(&devPtr), size, cudaStream);

    if (RMM_SUCCESS != rmmStatus) { return nullptr; }

    thrust::sequence(rmm::exec_policy(cudaStream), devPtr, devPtr + length, 0);

    return devPtr;
}
