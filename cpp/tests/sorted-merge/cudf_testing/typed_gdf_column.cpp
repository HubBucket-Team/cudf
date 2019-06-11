#include "typed_gdf_column.hpp"

#include <thrust/execution_policy.h>
#include <thrust/logical.h>

namespace cudf {
namespace testing {

template <class DType>
bool
TypedGdfColumn<DType>::operator==(const TypedGdfColumn<DType> &other) const
    noexcept {
    const GdfColumn &content = static_cast<const GdfColumn &>(other);

    gdf_column c_this  = *this;
    gdf_column c_other = content;

    if (c_this.dtype != c_other.dtype) { return false; }

    if (c_this.size != c_other.size) { return false; }

    assert(c_this.data != nullptr);
    assert(c_other.data != nullptr);

    assert(c_this.size > 0);
    assert(c_other.size > 0);

    const std::ptrdiff_t length = static_cast<std::ptrdiff_t>(c_this.size);

    using ctype = typename DType::ctype;

    const ctype *this_data  = static_cast<ctype *>(c_this.data);
    const ctype *other_data = static_cast<ctype *>(c_other.data);

    const bool datasAreNotEquals = !thrust::all_of(thrust::device,
                                                   this_data + length,
                                                   other_data,
                                                   other_data + length,
                                                   thrust::identity<ctype>{});

    if (datasAreNotEquals) { return false; }

    // TODO: Check valid

    return true;
}

}  // namespace testing
}  // namespace cudf
