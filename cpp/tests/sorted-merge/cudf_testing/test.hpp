#ifndef CUDF_TESTING_TEST_HPP_
#define CUDF_TESTING_TEST_HPP_

#include "gdf_column_builder.hpp"
#include "gdf_dtype.hpp"

#include <gtest/gtest.h>

namespace cudf {
namespace testing {

template <gdf_dtype... DT>
using DTypes = ::testing::Types<GdfDType<DT>...>;

class Test : public ::testing::Test {
protected:
    template <class DType>
    std::unique_ptr<GdfColumnBuilder<DType>> MakeGdfColumnBuilder();
};

using GdfNumberTypes =
    DTypes< GDF_FLOAT32, GDF_FLOAT64>;

}  // namespace testing
}  // namespace cudf

#define CUDF_TEST_NUMBERS(TypedTest)                                           \
    template <class GdfDType>                                                  \
    class TypedTest : public cudf::testing::Test {                             \
    protected:                                                                 \
        inline cudf::testing::GdfColumnBuilder<GdfDType> &GdfColumnBuilder() { \
            builders.push_back(std::move(MakeGdfColumnBuilder<GdfDType>()));   \
            return *builders.back();                                           \
        }                                                                      \
                                                                               \
    private:                                                                   \
        std::vector<                                                           \
            std::unique_ptr<cudf::testing::GdfColumnBuilder<GdfDType>>>        \
            builders;                                                          \
    };                                                                         \
    TYPED_TEST_CASE(TypedTest, cudf::testing::GdfNumberTypes)

#endif
