#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/merge.h>
#include <thrust/sequence.h>
#include <cudf.h>

#include "rmm/thrust_rmm_allocator.h"
#include "sqls/sqls_rtti_comp.h"

#include "alloc_filtered_cols.cuh"
#include "make_indices.cuh"
#include "pair_rtti.cuh"
#include "soa_info.cuh"
#include "typed_sorted_merge.cuh"

enum side_value { LEFT_SIDE_VALUE = 0, RIGHT_SIDE_VALUE };

gdf_error typed_sorted_merge(gdf_column **     left_cols,
                             gdf_column **     right_cols,
                             const gdf_size_type ncols,
                             gdf_column *      sort_by_cols,
                             gdf_column *      asc_desc,
                             gdf_column *      output_sides,
                             gdf_column *      output_indices,
                             cudaStream_t      cudaStream) {
    GDF_REQUIRE((nullptr != left_cols && nullptr != right_cols),
                GDF_DATASET_EMPTY);

    GDF_REQUIRE(nullptr != asc_desc, GDF_DATASET_EMPTY);
    GDF_REQUIRE(asc_desc || asc_desc->dtype == GDF_INT8, GDF_UNSUPPORTED_DTYPE);

    GDF_REQUIRE(output_sides->dtype == GDF_INT32, GDF_UNSUPPORTED_DTYPE);
    GDF_REQUIRE(output_indices->dtype == GDF_INT32, GDF_UNSUPPORTED_DTYPE);

    const gdf_size_type left_size  = left_cols[0]->size;
    const gdf_size_type right_size = right_cols[0]->size;

    const gdf_size_type total_size = left_size + right_size;
    GDF_REQUIRE(output_sides->size >= total_size, GDF_COLUMN_SIZE_MISMATCH);
    GDF_REQUIRE(output_indices->size >= total_size, GDF_COLUMN_SIZE_MISMATCH);

    GDF_REQUIRE(sort_by_cols->dtype == GDF_INT32, GDF_UNSUPPORTED_DTYPE);
    GDF_REQUIRE(sort_by_cols->size <= ncols, GDF_COLUMN_SIZE_TOO_BIG);

    // TODO: Get as gdf_sorted_merge parameters
    INITIALIZE_D_VALUES(left);
    INITIALIZE_D_VALUES(right);

    gdf_size_type sort_by_ncols = sort_by_cols->size;

    gdf_size_type *left_indices = make_indices(cudaStream, left_size);
    if (left_indices == nullptr) { return GDF_MEMORYMANAGER_ERROR; }

    gdf_size_type *right_indices = make_indices(cudaStream, right_size);
    if (left_indices == nullptr) {
        RMM_FREE(left_indices, cudaStream);
        return GDF_MEMORYMANAGER_ERROR;
    }

    const thrust::constant_iterator<int> left_side =
        thrust::make_constant_iterator(static_cast<int>(LEFT_SIDE_VALUE));
    const thrust::constant_iterator<int> right_side =
        thrust::make_constant_iterator(static_cast<int>(RIGHT_SIDE_VALUE));

    thrust::zip_iterator<
        thrust::tuple<thrust::constant_iterator<int>, gdf_size_type *>>
        left_zip_iterator = thrust::make_zip_iterator(
            thrust::make_tuple(left_side, left_indices));
    thrust::zip_iterator<
        thrust::tuple<thrust::constant_iterator<int>, gdf_size_type *>>
        right_zip_iterator = thrust::make_zip_iterator(
            thrust::make_tuple(right_side, right_indices));

    auto output_zip_iterator = thrust::make_zip_iterator(
        thrust::make_tuple(static_cast<std::int32_t *>(output_sides->data),
                           static_cast<std::int32_t *>(output_indices->data)));

    void **          filtered_left_d_cols_data    = nullptr;
    void **          filtered_right_d_cols_data   = nullptr;
    gdf_valid_type **filtered_left_d_valids_data  = nullptr;
    gdf_valid_type **filtered_right_d_valids_data = nullptr;
    std::int32_t *   filtered_left_d_col_types    = nullptr;
    std::int32_t *   filtered_right_d_col_types   = nullptr;
    gdf_error        gdf_status = alloc_filtered_d_cols(sort_by_ncols,
                                                 filtered_left_d_cols_data,
                                                 filtered_right_d_cols_data,
                                                 filtered_left_d_valids_data,
                                                 filtered_right_d_valids_data,
                                                 filtered_left_d_col_types,
                                                 filtered_right_d_col_types,
                                                 cudaStream);
    if (GDF_SUCCESS != gdf_status) {
        RMM_FREE(left_indices, cudaStream);
        RMM_FREE(right_indices, cudaStream);
        return gdf_status;
    }

    // filter left and right cols for sorting
    std::int32_t *sort_by_d_cols_data =
        reinterpret_cast<std::int32_t *>(sort_by_cols->data);
    thrust::for_each_n(
        rmm::exec_policy(cudaStream)->on(cudaStream),
        thrust::make_counting_iterator(0),
        sort_by_ncols,
        [=] __device__(const int n) {
            const std::int32_t n_col = sort_by_d_cols_data[n];

            void *const left_data  = left_d_cols_data[n_col];
            void *const right_data = right_d_cols_data[n_col];

            gdf_valid_type *const left_valids  = left_d_valids_data[n_col];
            gdf_valid_type *const right_valids = right_d_valids_data[n_col];

            const std::int32_t left_types  = left_d_col_types[n_col];
            const std::int32_t right_types = right_d_col_types[n_col];

            filtered_left_d_cols_data[n]  = left_data;
            filtered_right_d_cols_data[n] = right_data;

            filtered_left_d_valids_data[n]  = left_valids;
            filtered_right_d_valids_data[n] = right_valids;

            filtered_left_d_col_types[n]  = left_types;
            filtered_right_d_col_types[n] = right_types;
        });

    PairRTTI<gdf_size_type> comp(
        {
            filtered_left_d_cols_data,
            filtered_left_d_valids_data,
            filtered_left_d_col_types,
        },
        {
            filtered_right_d_cols_data,
            filtered_right_d_valids_data,
            filtered_right_d_col_types,
        },
        sort_by_ncols,
        static_cast<const std::int8_t *>(asc_desc->data));

    thrust::merge(rmm::exec_policy(cudaStream)->on(cudaStream),
                  left_zip_iterator,
                  left_zip_iterator + left_size,
                  right_zip_iterator,
                  right_zip_iterator + right_size,
                  output_zip_iterator,
                  [=] __device__(thrust::tuple<int, gdf_size_type> left_tuple,
                                 thrust::tuple<int, gdf_size_type> right_tuple) {
                      const gdf_size_type left_row  = thrust::get<1>(left_tuple);
                      const gdf_size_type right_row = thrust::get<1>(right_tuple);

                      return comp.asc_desc_comparison(left_row, right_row);
                  });

    RMM_FREE(left_indices, cudaStream);
    RMM_FREE(right_indices, cudaStream);
    RMM_FREE(filtered_left_d_cols_data, cudaStream);
    RMM_FREE(filtered_right_d_cols_data, cudaStream);
    RMM_FREE(filtered_left_d_col_types, cudaStream);
    RMM_FREE(filtered_right_d_col_types, cudaStream);

    return GDF_SUCCESS;
}
