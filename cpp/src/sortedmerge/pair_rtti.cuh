#pragma once

#include <cassert>
#include <cstdint>
#include <functional>

#include <cuda_runtime.h>
#include <cudf.h>

#include <utilities/cudf_utils.h>

template <class IndexT>
class PairRTTI {
public:
    class SideGroup;

    explicit PairRTTI(const SideGroup &   left_side_group,
                      const SideGroup &   right_side_group,
                      const gdf_size_type size);

    __device__ bool asc_desc_comparison(IndexT left_row,
                                        IndexT right_row) const {
        for (gdf_size_type i = 0; i < size_; i++) {
            const bool left_is_valid =
                gdf_is_valid(left_side_group_.valids[i], left_row);
            const bool right_is_valid =
                gdf_is_valid(right_side_group_.valids[i], right_row);

            if (!left_is_valid || !right_is_valid) { return true; }

            const gdf_dtype left_dtype =
                static_cast<gdf_dtype>(left_side_group_.types[i]);
            const gdf_dtype right_dtype =
                static_cast<gdf_dtype>(right_side_group_.types[i]);

            const void *const left_col  = left_side_group_.cols[i];
            const void *const right_col = right_side_group_.cols[i];

            // TODO: From sorted_merge function we can create a column wrapper
            //       class with type info instead of use soa_col_info. Thus, we
            //       can use the compiler type checking.
#define RIGHT_CASE(DTYPE, LEFT_CTYPE, RIGHT_CTYPE)                             \
    case DTYPE:                                                                \
        do {                                                                   \
            const LEFT_CTYPE left_value =                                      \
                reinterpret_cast<const LEFT_CTYPE *>(left_col)[i];             \
            const RIGHT_CTYPE right_value =                                    \
                reinterpret_cast<const RIGHT_CTYPE *>(right_col)[i];           \
            if (left_value < right_value) {                                    \
                return false;                                                  \
            } else {                                                           \
                continue;                                                      \
            }                                                                  \
        } while (0)


#define LEFT_CASE(DTYPE, LEFT_CTYPE)                                           \
    case DTYPE:                                                                \
        switch (right_dtype) {                                                 \
            RIGHT_CASE(GDF_INT8, LEFT_CTYPE, std::int8_t);                     \
            RIGHT_CASE(GDF_INT16, LEFT_CTYPE, std::int16_t);                   \
            RIGHT_CASE(GDF_INT32, LEFT_CTYPE, std::int32_t);                   \
            RIGHT_CASE(GDF_INT64, LEFT_CTYPE, std::int64_t);                   \
            RIGHT_CASE(GDF_FLOAT32, LEFT_CTYPE, float);                        \
            RIGHT_CASE(GDF_FLOAT64, LEFT_CTYPE, double);                       \
            RIGHT_CASE(GDF_DATE32, LEFT_CTYPE, std::int32_t);                  \
            RIGHT_CASE(GDF_DATE64, LEFT_CTYPE, std::int64_t);                  \
        }

            switch (left_dtype) {
                LEFT_CASE(GDF_INT8, std::int8_t);
                LEFT_CASE(GDF_INT16, std::int16_t);
                LEFT_CASE(GDF_INT32, std::int32_t);
                LEFT_CASE(GDF_INT64, std::int64_t);
                LEFT_CASE(GDF_FLOAT32, float);
                LEFT_CASE(GDF_FLOAT64, double);
                LEFT_CASE(GDF_DATE32, std::int32_t);
                LEFT_CASE(GDF_DATE64, std::int64_t);
            }
        }

        return true;
    }

private:
    const SideGroup     left_side_group_;
    const SideGroup     right_side_group_;
    const gdf_size_type size_;
};

template <class IndexT>
class PairRTTI<IndexT>::SideGroup {
public:
    const void *const *const           cols;
    const gdf_valid_type *const *const valids;
    const int *const                   types;
};
