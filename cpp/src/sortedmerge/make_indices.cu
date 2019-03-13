#include <cudf.h>
#include "rmm/thrust_rmm_allocator.h"
#include "make_indices.cuh"
#include <thrust/sequence.h>

gdf_size_type *make_indices(cudaStream_t         cudaStream,
                          const std::ptrdiff_t length) {
    gdf_size_type *     devPtr;
    const gdf_size_type size = length * sizeof(const gdf_size_type);

    rmmError_t rmmStatus =
        RMM_ALLOC(reinterpret_cast<void **>(&devPtr), size, cudaStream);

    if (RMM_SUCCESS != rmmStatus) { return nullptr; }

    thrust::sequence(rmm::exec_policy(cudaStream)->on(cudaStream), devPtr, devPtr + length, 0);

    return devPtr;
}
