#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/merge.h>
#include <thrust/sequence.h>

#include "../../src/rmm/thrust_rmm_allocator.h"
#include "../../src/sqls/sqls_rtti_comp.h"

#include "make_indices.cuh"
#include "pair_rtti.cuh"
#include "soa_info.cuh"
#include "typed_sorted_merge.cuh"

static void
alloc_filtered_d_cols(const gdf_size_type sort_by_ncols,
                      std::int64_t **&    out_filtered_left_d_cols_data,
                      std::int64_t **&    out_filtered_right_d_cols_data,
                      std::int32_t *&     out_filtered_left_d_col_types,
                      std::int32_t *&     out_filtered_right_d_col_types,
                      cudaStream_t        cudaStream) {
    std::int64_t **filtered_left_d_cols_data;
    std::int64_t **filtered_right_d_cols_data;

    std::int32_t *filtered_left_d_col_types;
    std::int32_t *filtered_right_d_col_types;

    RMM_ALLOC(reinterpret_cast<void **>(&filtered_left_d_cols_data),
              sizeof(std::int64_t) * sort_by_ncols,
              cudaStream);
    RMM_ALLOC(reinterpret_cast<void **>(&filtered_right_d_cols_data),
              sizeof(std::int64_t) * sort_by_ncols,
              cudaStream);
    RMM_ALLOC(reinterpret_cast<void **>(&filtered_left_d_col_types),
              sizeof(std::int32_t) * sort_by_ncols,
              cudaStream);
    RMM_ALLOC(reinterpret_cast<void **>(&filtered_right_d_col_types),
              sizeof(std::int32_t) * sort_by_ncols,
              cudaStream);

    out_filtered_left_d_cols_data  = filtered_left_d_cols_data;
    out_filtered_right_d_cols_data = filtered_right_d_cols_data;
    out_filtered_left_d_col_types  = filtered_left_d_col_types;
    out_filtered_right_d_col_types = filtered_right_d_col_types;
}

enum side_value { LEFT_SIDE_VALUE = 0, RIGHT_SIDE_VALUE };

gdf_error typed_sorted_merge(gdf_column **     left_cols,
                             gdf_column **     right_cols,
                             const std::size_t ncols,
                             gdf_column *      sort_by_cols,
                             gdf_column *      asc_desc,
                             gdf_column *      output_sides,
                             gdf_column *      output_indices,
                             cudaStream_t      cudaStream) {
    GDF_REQUIRE((nullptr != left_cols && nullptr != right_cols),
                GDF_DATASET_EMPTY);

    GDF_REQUIRE(output_sides->dtype == GDF_INT32, GDF_UNSUPPORTED_DTYPE);
    GDF_REQUIRE(output_indices->dtype == GDF_INT32, GDF_UNSUPPORTED_DTYPE);

    const std::size_t total_size = left_cols[0]->size + right_cols[0]->size;
    GDF_REQUIRE(output_sides->size >= total_size, GDF_COLUMN_SIZE_MISMATCH);
    GDF_REQUIRE(output_indices->size >= total_size, GDF_COLUMN_SIZE_MISMATCH);

    GDF_REQUIRE(sort_by_cols->dtype == GDF_INT32, GDF_UNSUPPORTED_DTYPE);
    GDF_REQUIRE(sort_by_cols->size <= ncols, GDF_COLUMN_SIZE_TOO_BIG);

    INITIALIZE_D_VALUES(left);
    INITIALIZE_D_VALUES(right);

    gdf_size_type sort_by_ncols = sort_by_cols->size;

    std::size_t *left_indices  = make_indices(cudaStream, 8);
    std::size_t *right_indices = make_indices(cudaStream, 8);

    const thrust::constant_iterator<int> left_side =
        thrust::make_constant_iterator(static_cast<int>(LEFT_SIDE_VALUE));
    const thrust::constant_iterator<int> right_side =
        thrust::make_constant_iterator(static_cast<int>(RIGHT_SIDE_VALUE));

    thrust::zip_iterator<
        thrust::tuple<thrust::constant_iterator<int>, std::size_t *>>
        left_zip_iterator = thrust::make_zip_iterator(
            thrust::make_tuple(left_side, left_indices));
    thrust::zip_iterator<
        thrust::tuple<thrust::constant_iterator<int>, std::size_t *>>
        right_zip_iterator = thrust::make_zip_iterator(
            thrust::make_tuple(right_side, right_indices));

    auto output_zip_iterator = thrust::make_zip_iterator(
        thrust::make_tuple(static_cast<std::size_t *>(output_sides->data),
                           static_cast<std::size_t *>(output_indices->data)));

    std::int64_t **filtered_left_d_cols_data;
    std::int64_t **filtered_right_d_cols_data;
    std::int32_t * filtered_left_d_col_types;
    std::int32_t * filtered_right_d_col_types;
    alloc_filtered_d_cols(sort_by_ncols,
                          filtered_left_d_cols_data,
                          filtered_right_d_cols_data,
                          filtered_left_d_col_types,
                          filtered_right_d_col_types,
                          cudaStream);

    // filter left and right cols for sorting
    std::int32_t *sort_by_d_cols_data =
        reinterpret_cast<std::int32_t *>(sort_by_cols->data);
    thrust::for_each_n(
        thrust::device,
        thrust::make_counting_iterator(0),
        sort_by_ncols,
        [=] __device__(const int n) {
            const std::int32_t n_col = sort_by_d_cols_data[n];

            std::int64_t *left_data =
                reinterpret_cast<std::int64_t *>(left_d_cols_data[n_col]);

            std::int64_t *right_data =
                reinterpret_cast<std::int64_t *>(right_d_cols_data[n_col]);

            filtered_left_d_cols_data[n]  = left_data;
            filtered_right_d_cols_data[n] = right_data;

            filtered_left_d_col_types[n]  = left_d_col_types[n_col];
            filtered_right_d_col_types[n] = right_d_col_types[n_col];
        });

    PairRTTI<std::size_t> comp(
        {
            reinterpret_cast<void **>(filtered_left_d_cols_data),
            filtered_left_d_col_types,
        },
        {
            reinterpret_cast<void **>(filtered_right_d_cols_data),
            filtered_right_d_col_types,
        },
        sort_by_ncols);

    thrust::merge(thrust::device,
                  left_zip_iterator,
                  left_zip_iterator + 4,
                  right_zip_iterator,
                  right_zip_iterator + 2,
                  output_zip_iterator,
                  [=] __device__(thrust::tuple<int, std::size_t> left_tuple,
                                 thrust::tuple<int, std::size_t> right_tuple) {
                      const std::size_t left_row  = thrust::get<1>(left_tuple);
                      const std::size_t right_row = thrust::get<1>(right_tuple);

                      return comp.asc_desc_comparison(left_row, right_row);
                  });

    cudaFree(left_indices);
    cudaFree(right_indices);
    cudaFree(filtered_left_d_cols_data);
    cudaFree(filtered_right_d_cols_data);
    cudaFree(filtered_left_d_col_types);
    cudaFree(filtered_right_d_col_types);

    return GDF_SUCCESS;
}
