#include "gdf_dtype.hpp"

namespace cudf {
namespace testing {

#define GDFDTYPE_EXPR_FACTORY(DTYPE)                  \
    constexpr gdf_dtype GdfDType<DTYPE>::dtype_value; \
    constexpr char      GdfDType<DTYPE>::name[]

GDFDTYPE_EXPR_FACTORY(GDF_INT8);
GDFDTYPE_EXPR_FACTORY(GDF_INT16);
GDFDTYPE_EXPR_FACTORY(GDF_INT32);
GDFDTYPE_EXPR_FACTORY(GDF_INT64);
GDFDTYPE_EXPR_FACTORY(GDF_FLOAT32);
GDFDTYPE_EXPR_FACTORY(GDF_FLOAT64);

}  // namespace testing
}  // namespace cudf
