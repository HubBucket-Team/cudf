#include "test.hpp"

namespace cudf {
namespace testing {

template <class DType>
std::unique_ptr<GdfColumnBuilder<DType>>
Test::MakeGdfColumnBuilder() {
    return GdfColumnBuilder<DType>::Make();
}

#define TEST_FACTORY(DTYPE)                                     \
    template std::unique_ptr<GdfColumnBuilder<GdfDType<DTYPE>>> \
    Test::MakeGdfColumnBuilder<GdfDType<DTYPE>>()

TEST_FACTORY(GDF_INT8);
TEST_FACTORY(GDF_INT16);
TEST_FACTORY(GDF_INT32);
TEST_FACTORY(GDF_INT64);
TEST_FACTORY(GDF_FLOAT32);
TEST_FACTORY(GDF_FLOAT64);

}  // namespace testing
}  // namespace cudf
