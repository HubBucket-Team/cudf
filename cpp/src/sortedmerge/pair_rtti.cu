#include "pair_rtti.cuh"

template <class IndexT>
PairRTTI<IndexT>::PairRTTI(const SideGroup &   left_side_group,
                           const SideGroup &   right_side_group,
                           const gdf_size_type size)
    : left_side_group_{left_side_group},
      right_side_group_{right_side_group}, size_{size} {}

template class PairRTTI<std::size_t>;
