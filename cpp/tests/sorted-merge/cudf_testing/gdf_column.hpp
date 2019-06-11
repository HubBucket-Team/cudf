#ifndef CUDF_TESTING_GDF_COLUMN_HPP_
#define CUDF_TESTING_GDF_COLUMN_HPP_

#include <memory>
#include <ostream>

#include <cudf/types.h>

namespace cudf {
namespace testing {

class GdfColumn {
    friend std::ostream &operator<<(std::ostream &, const GdfColumn &);

public:
    // TODO: improve to remove no except
    virtual ~GdfColumn() noexcept(false);

    virtual operator const gdf_column &() const noexcept      = 0;
    virtual operator const gdf_column *const() const noexcept = 0;

    virtual bool operator==(const GdfColumn &) const noexcept = 0;

protected:
    inline GdfColumn() = default;

private:
    GdfColumn(const GdfColumn &) = delete;
    void operator=(const GdfColumn &) = delete;
};

inline std::ostream &
operator<<(std::ostream &os, const std::shared_ptr<GdfColumn> &gdfColumn) {
    return os << *gdfColumn;
}

}  // namespace testing
}  // namespace cudf

#endif
