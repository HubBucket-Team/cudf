#pragma once

#include <cstdint>

#include <cudf.h>

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
    const void *const *const cols;
    const int *const         types;
};
