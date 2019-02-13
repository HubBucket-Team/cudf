#pragma once

#include <cstdint>

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
            const bool left_valid =
                gdf_is_valid(left_side_group_.valids[i], left_row);
            const bool right_valid =
                gdf_is_valid(right_side_group_.valids[i], right_row);

            if (!left_valid || !right_valid) { return true; }

            const std::int64_t left_value =
                reinterpret_cast<const std::int64_t *>(
                    left_side_group_.cols[i])[left_row];
            const std::int64_t right_value =
                reinterpret_cast<const std::int64_t *>(
                    right_side_group_.cols[i])[right_row];

            if (left_value < right_value) { return false; }
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
