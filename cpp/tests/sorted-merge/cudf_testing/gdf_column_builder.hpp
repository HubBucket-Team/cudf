#ifndef CUDF_TESTING_GDF_COLUMN_BUILDER_HPP_
#define CUDF_TESTING_GDF_COLUMN_BUILDER_HPP_

#include "gdf_column.hpp"

namespace cudf {
namespace testing {

template <class DType>
class GdfColumnBuilder {
public:
    virtual ~GdfColumnBuilder();

    virtual std::shared_ptr<GdfColumn> Build() const noexcept = 0;

    using ctype     = typename DType::ctype;
    using reference = GdfColumnBuilder<DType> &;

    virtual reference WithLength(const std::size_t length)          = 0;
    virtual reference SetData(const std::initializer_list<ctype> &) = 0;
    virtual reference SetValid(const std::initializer_list<bool> &) = 0;

    static std::unique_ptr<GdfColumnBuilder> Make();

protected:
    GdfColumnBuilder() = default;

private:
    GdfColumnBuilder(const GdfColumnBuilder &) = delete;
    void operator=(const GdfColumnBuilder &) = delete;
};

}  // namespace testing
}  // namespace cudf

#endif
