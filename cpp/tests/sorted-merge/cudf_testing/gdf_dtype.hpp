#ifndef CUDF_TESTING_GDF_DTYPE_HPP_
#define CUDF_TESTING_GDF_DTYPE_HPP_

#include <cstdint>

#include <cudf/types.h>

namespace cudf {
namespace testing {

template <gdf_dtype DT>
class GdfDType {};

#define GDFDTYPE_FACTORY(DT, CT)                                  \
    template <>                                                   \
    class GdfDType<DT> {                                          \
    public:                                                       \
        using ctype                              = CT;            \
        static constexpr std::size_t size        = sizeof(ctype); \
        static constexpr gdf_dtype   dtype_value = DT;            \
        static constexpr char        name[]      = #DT;           \
    }

GDFDTYPE_FACTORY(GDF_INT8, std::int8_t);
GDFDTYPE_FACTORY(GDF_INT16, std::int16_t);
GDFDTYPE_FACTORY(GDF_INT32, std::int32_t);
GDFDTYPE_FACTORY(GDF_INT64, std::int64_t);
GDFDTYPE_FACTORY(GDF_FLOAT32, float);
GDFDTYPE_FACTORY(GDF_FLOAT64, double);

}  // namespace testing
}  // namespace cudf

#endif
