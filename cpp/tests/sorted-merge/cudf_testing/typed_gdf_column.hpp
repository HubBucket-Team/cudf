#ifndef CUDF_TESTING_TYPED_GDF_COLUMN_HPP_
#define CUDF_TESTING_TYPED_GDF_COLUMN_HPP_

#include "gdf_column.hpp"

namespace cudf {
namespace testing {

template <class DType>
class TypedGdfColumn : public GdfColumn {
public:
    bool operator==(const TypedGdfColumn<DType> &) const noexcept;

private:
    TypedGdfColumn()                       = delete;
    TypedGdfColumn(const TypedGdfColumn &) = delete;
    void operator=(const TypedGdfColumn &) = delete;
};

template <class DType>
inline std::ostream &
operator<<(std::ostream &os, const TypedGdfColumn<DType> &typedGdfColumn) {
    return os << static_cast<const GdfColumn &>(typedGdfColumn);
}

template <class DType>
inline std::ostream &
operator<<(std::ostream &                                os,
           const std::shared_ptr<TypedGdfColumn<DType>> &typedGdfColumn) {
    return os << *typedGdfColumn;
}

}  // namespace testing
}  // namespace cudf

#endif
