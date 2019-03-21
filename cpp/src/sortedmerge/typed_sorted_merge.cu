#include <cudf.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/merge.h>
#include <thrust/sequence.h>

#include "rmm/thrust_rmm_allocator.h"
#include "sqls/sqls_rtti_comp.h"

#include "alloc_filtered_cols.cuh"
#include "pair_rtti.cuh"
#include "soa_info.cuh"
#include "typed_sorted_merge.cuh"

enum side_value { LEFT_SIDE_VALUE = 0, RIGHT_SIDE_VALUE };

class ColumnSparsedRTTI {
public:
    explicit ColumnSparsedRTTI(const gdf_size_type sparsedIndex,
                               const gdf_size_type columnIndex,
                               void *const         data)
        : sparsedIndex_{sparsedIndex}, columnIndex_{columnIndex}, data_{data} {}

    __device__ explicit ColumnSparsedRTTI()
        : sparsedIndex_{-1}, columnIndex_{-1}, data_{nullptr} {}

    __device__ bool operator<(const ColumnSparsedRTTI &other) const {
        const std::int64_t *thisData = static_cast<std::int64_t *>(data_);
        const std::int64_t *otherData =
            static_cast<std::int64_t *>(other.data_);

        const std::int64_t left_value  = thisData[columnIndex_];
        const std::int64_t right_value = otherData[other.columnIndex_];

        return left_value > right_value;
    }

    __device__ const thrust::tuple<std::int32_t, std::int32_t> ToTuple() const
        noexcept {
        return thrust::make_tuple(static_cast<std::int32_t>(sparsedIndex_),
                                  static_cast<std::int32_t>(columnIndex_));
    }

private:
    gdf_size_type sparsedIndex_;
    gdf_size_type columnIndex_;
    void *        data_;
};

gdf_error
typed_sorted_merge(gdf_column **       left_cols,
                   gdf_column **       right_cols,
                   const gdf_size_type ncols,
                   gdf_column *        sort_by_cols,
                   gdf_column *        asc_desc,
                   gdf_column *        output_sides,
                   gdf_column *        output_indices,
                   cudaStream_t        cudaStream) {
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

    const thrust::constant_iterator<gdf_size_type> left_side =
        thrust::make_constant_iterator(
            static_cast<gdf_size_type>(LEFT_SIDE_VALUE));
    const thrust::constant_iterator<gdf_size_type> right_side =
        thrust::make_constant_iterator(
            static_cast<gdf_size_type>(RIGHT_SIDE_VALUE));

    const thrust::counting_iterator<gdf_size_type> left_indices =
        thrust::make_counting_iterator(0);
    const thrust::counting_iterator<gdf_size_type> right_indices =
        thrust::make_counting_iterator(0);

    const thrust::zip_iterator<
        thrust::tuple<thrust::constant_iterator<gdf_size_type>,
                      thrust::counting_iterator<gdf_size_type>>>
        left_begin_zip_iterator = thrust::make_zip_iterator(
            thrust::make_tuple(left_side, left_indices));
    const thrust::zip_iterator<
        thrust::tuple<thrust::constant_iterator<gdf_size_type>,
                      thrust::counting_iterator<gdf_size_type>>>
        right_begin_zip_iterator = thrust::make_zip_iterator(
            thrust::make_tuple(right_side, right_indices));

    const thrust::zip_iterator<
        thrust::tuple<thrust::constant_iterator<gdf_size_type>,
                      thrust::counting_iterator<gdf_size_type>>>
        left_end_zip_iterator = thrust::make_zip_iterator(thrust::make_tuple(
            left_side + left_size, left_indices + left_size));
    const thrust::zip_iterator<
        thrust::tuple<thrust::constant_iterator<gdf_size_type>,
                      thrust::counting_iterator<gdf_size_type>>>
        right_end_zip_iterator = thrust::make_zip_iterator(thrust::make_tuple(
            right_side + right_size, right_indices + right_size));

    const thrust::zip_iterator<thrust::tuple<gdf_size_type *, gdf_size_type *>>
        output_zip_iterator = thrust::make_zip_iterator(thrust::make_tuple(
            static_cast<gdf_size_type *>(output_sides->data),
            static_cast<gdf_size_type *>(output_indices->data)));

    gdf_size_type sort_by_ncols = sort_by_cols->size;

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
    if (GDF_SUCCESS != gdf_status) { return gdf_status; }

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
                  left_begin_zip_iterator,
                  left_end_zip_iterator,
                  right_begin_zip_iterator,
                  right_end_zip_iterator,
                  output_zip_iterator,
                  [=] __device__(
                      thrust::tuple<gdf_size_type, gdf_size_type> left_tuple,
                      thrust::tuple<gdf_size_type, gdf_size_type> right_tuple) {
                      const gdf_size_type left_row = thrust::get<1>(left_tuple);
                      const gdf_size_type right_row =
                          thrust::get<1>(right_tuple);
                      return comp.asc_desc_comparison(right_row, left_row);
                  });

    thrust::device_vector<ColumnSparsedRTTI> leftColumnSparsed;
    leftColumnSparsed.reserve(left_size);
    for (gdf_size_type i = 0; i < left_size; i++) {
        leftColumnSparsed.push_back(
            ColumnSparsedRTTI{0, i, left_cols[0]->data});
    }

    thrust::device_vector<ColumnSparsedRTTI> rightColumnSparsed;
    rightColumnSparsed.reserve(right_size);
    for (gdf_size_type i = 0; i < right_size; i++) {
        rightColumnSparsed.push_back(
            ColumnSparsedRTTI{1, i, left_cols[0]->data});
    }

    thrust::device_vector<ColumnSparsedRTTI> outputColumnSparsed(total_size);

    thrust::merge(thrust::device,
                  leftColumnSparsed.begin(),
                  leftColumnSparsed.end(),
                  rightColumnSparsed.begin(),
                  rightColumnSparsed.end(),
                  outputColumnSparsed.begin(),
                  [=] __device__(const ColumnSparsedRTTI &left,
                                 const ColumnSparsedRTTI &right) {
                      return left < right;
                  });

    thrust::transform(
        thrust::device,
        outputColumnSparsed.begin(),
        outputColumnSparsed.end(),
        output_zip_iterator,
        [=] __device__(const ColumnSparsedRTTI &columnSparsedRTTI) {
            return columnSparsedRTTI.ToTuple();
        });

    RMM_FREE(filtered_left_d_cols_data, cudaStream);
    RMM_FREE(filtered_right_d_cols_data, cudaStream);
    RMM_FREE(filtered_left_d_col_types, cudaStream);
    RMM_FREE(filtered_right_d_col_types, cudaStream);

    return GDF_SUCCESS;
}
