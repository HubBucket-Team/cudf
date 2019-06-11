#include "gdf_column_builder.hpp"

#include <iostream>

#include <cuda_runtime.h>

#include <bitmask/legacy_bitmask.hpp>
#include <utilities/bit_util.cuh>

#include "gdf_column-internal.hpp"

namespace cudf {
namespace testing {

namespace {
template <class DType>
class GdfColumnBuilderContent : public GdfColumnBuilder<DType> {
public:
    GdfColumnBuilderContent()
        : gdfcolumn_{nullptr,
                     nullptr,
                     -1,
                     DType::dtype_value,
                     -1,
                     {},
                     nullptr} {}

    std::shared_ptr<GdfColumn> Build() const noexcept final {
        return std::make_shared<internal::GdfColumnContent>(gdfcolumn_);
    }

    using ctype     = typename DType::ctype;
    using reference = GdfColumnBuilder<DType> &;

    reference WithLength(const std::size_t length) final {
        ctype *     data;
        cudaError_t cudaStatus = cudaMalloc(&data, length * DType::size);
        Check(cudaStatus, "Data cudaMalloc");

        gdfcolumn_.data = data;
        gdfcolumn_.size = static_cast<gdf_size_type>(length);

        return *this;
    }

    // PreCondition: Call #WithLength
    reference SetData(const std::initializer_list<ctype> &initialData) final {
        // TODO: Maybe it's better to create a flags about the previous
        // states.
        if (-1 == gdfcolumn_.size) {
            throw std::runtime_error(
                "You need to call `WithLength` before SetData");
        }

        const std::size_t length = static_cast<std::size_t>(gdfcolumn_.size);

        assert(length >= initialData.size());

        if (length != initialData.size()) {
            std::cout << "You are assigning less elements than gdf column "
                         "allocated length"
                      << std::endl;
        }

        cudaError_t cudaStatus = cudaMemcpy(gdfcolumn_.data,
                                            initialData.begin(),
                                            sizeof(ctype) * gdfcolumn_.size,
                                            cudaMemcpyHostToDevice);
        Check(cudaStatus, "Data cudaMemcpy");

        return *this;
    }

    // PreCondition: Call #SetData
    reference SetValid(const std::initializer_list<bool> &initialValid) final {
        if (nullptr == gdfcolumn_.data) {
            throw std::runtime_error(
                "You need to call `SetData` before SetValid");
        }

        cudaError_t cudaStatus;

        const std::size_t validSize =
            gdf_valid_allocation_size(initialValid.size());

        gdf_valid_type *valid;
        cudaStatus = cudaMalloc(&valid, validSize);
        Check(cudaStatus, "Valid cudaMalloc");

        gdf_valid_type *hostValid = new gdf_valid_type[validSize];
        memset(hostValid, 0, validSize);

        gdf_size_type null_count = 0;
        std::size_t   i          = 0;

        for (bool state : initialValid) {
            if (state) {
                gdf::util::turn_bit_on(hostValid, i);
                null_count++;
            }
            i++;
        }

        cudaStatus =
            cudaMemcpy(valid, hostValid, validSize, cudaMemcpyHostToDevice);
        Check(cudaStatus, "Valid cudaMemcpy");

        delete[] hostValid;

        gdfcolumn_.valid      = valid;
        gdfcolumn_.null_count = null_count;

        return *this;
    }

private:
    static void Check(const cudaError_t  cudaStatus,
                      const std::string &message) {
        if (cudaSuccess != cudaStatus) { throw std::runtime_error(message); }
    }

    gdf_column gdfcolumn_;
};
}  // namespace

template <class DType>
GdfColumnBuilder<DType>::~GdfColumnBuilder() = default;

template <class DType>
std::unique_ptr<GdfColumnBuilder<DType>>
GdfColumnBuilder<DType>::Make() {
    return std::make_unique<GdfColumnBuilderContent<DType>>();
}

template class GdfColumnBuilder<GdfDType<GDF_INT8>>;
template class GdfColumnBuilder<GdfDType<GDF_INT16>>;
template class GdfColumnBuilder<GdfDType<GDF_INT32>>;
template class GdfColumnBuilder<GdfDType<GDF_INT64>>;
template class GdfColumnBuilder<GdfDType<GDF_FLOAT32>>;
template class GdfColumnBuilder<GdfDType<GDF_FLOAT64>>;

}  // namespace testing
}  // namespace cudf
