#include <cuda_runtime.h>
#include <cudf.h>

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>

#include "soa_info.cuh"
#include "typed_sorted_merge.cuh"

gdf_error gdf_sorted_merge(gdf_column **     left_cols,
                           gdf_column **     right_cols,
                           const std::size_t ncols,
                           gdf_column *      sort_by_cols,
                           gdf_column *      asc_desc,
                           gdf_column **     output_cols) {
    const std::size_t left_size  = left_cols[0]->size;
    const std::size_t right_size = right_cols[0]->size;

    const std::size_t total_size = left_size + right_size;

    gdf_column sides{nullptr, nullptr, total_size, GDF_INT32, 0, {}, nullptr};
    gdf_column indices{nullptr, nullptr, total_size, GDF_INT32, 0, {}, nullptr};

    rmmError_t rmmStatus;

    rmmStatus =
        RMM_ALLOC(&sides.data, sizeof(std::int32_t) * total_size, nullptr);
    if (RMM_SUCCESS != rmmStatus) { return GDF_MEMORYMANAGER_ERROR; }

    rmmStatus =
        RMM_ALLOC(&indices.data, sizeof(std::int32_t) * total_size, nullptr);
    if (RMM_SUCCESS != rmmStatus) {
        RMM_FREE(sides.data, nullptr);
        return GDF_MEMORYMANAGER_ERROR;
    }

    gdf_error gdf_status = typed_sorted_merge(left_cols,
                                              right_cols,
                                              ncols,
                                              sort_by_cols,
                                              asc_desc,
                                              &sides,
                                              &indices,
                                              nullptr);
    if (GDF_SUCCESS != gdf_status) { return gdf_status; }

    auto output_zip_iterator = thrust::make_zip_iterator(
        thrust::make_tuple(static_cast<std::int64_t *>(sides.data),
                           static_cast<std::int64_t *>(indices.data)));

    INITIALIZE_D_VALUES(left);
    INITIALIZE_D_VALUES(right);
    INITIALIZE_D_VALUES(output);

    thrust::for_each_n(
        thrust::device,
        thrust::make_zip_iterator(thrust::make_tuple(
            thrust::make_counting_iterator(0), output_zip_iterator)),
        total_size,
        [=] __device__(
            thrust::tuple<int, thrust::tuple<int, std::size_t>> group_tuple) {
            thrust::tuple<int, std::size_t> output_tuple =
                thrust::get<1>(group_tuple);

            const std::size_t side = thrust::get<0>(output_tuple);
            const std::size_t pos  = thrust::get<1>(output_tuple);

            for (std::size_t i = 0; i < ncols; i++) {
                const std::int64_t *left =
                    reinterpret_cast<std::int64_t *>(left_d_cols_data[i]);
                const std::int64_t *right =
                    reinterpret_cast<std::int64_t *>(right_d_cols_data[i]);

                std::int64_t *output =
                    reinterpret_cast<std::int64_t *>(output_d_cols_data[i]);

                output[thrust::get<0>(group_tuple)] =
                    0 == side ? left[pos] : right[pos];
            }
        });

    RMM_FREE(sides.data, nullptr);
    RMM_FREE(indices.data, nullptr);

    return GDF_SUCCESS;
}
