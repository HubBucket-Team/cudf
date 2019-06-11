#include "gdf_column.hpp"

#include <cassert>
#include <cstring>
#include <unordered_map>

#include <cuda_runtime_api.h>

#include "gdf_dtype.hpp"

namespace cudf {
namespace testing {

namespace {
class GdfColumnContent : public GdfColumn {
public:
    explicit GdfColumnContent(void *          data       = nullptr,
                              gdf_valid_type *valid      = nullptr,
                              gdf_size_type   length     = 0,
                              const gdf_dtype dtype      = GDF_invalid,
                              gdf_size_type   null_count = 0)
        : gdfcolumn_{data, valid, length, dtype, null_count, {}, nullptr} {}

    explicit GdfColumnContent(const gdf_column &gdfcolumn)
        : gdfcolumn_{gdfcolumn} {}

    ~GdfColumnContent() noexcept(false) final {
        cudaError_t cudaStatus;

        cudaStatus = cudaFree(gdfcolumn_.data);
        if (cudaSuccess != cudaStatus) {
            throw std::runtime_error("GdfColumnContent cudaFree data");
        }

        cudaStatus = cudaFree(gdfcolumn_.valid);
        if (cudaSuccess != cudaStatus) {
            throw std::runtime_error("GdfColumnContent cudaFree valid");
        }
    }

    operator const gdf_column &() const noexcept final { return gdfcolumn_; }

    operator const gdf_column *const() const noexcept final {
        return &gdfcolumn_;
    };

    bool operator==(const GdfColumn &other) const noexcept final {
        const GdfColumnContent &content =
            static_cast<const GdfColumnContent &>(other);

        gdf_column c_this  = *this;
        gdf_column c_other = content;

        if (c_this.dtype != c_other.dtype) { return false; }

        if (c_this.size != c_other.size) { return false; }

        assert(c_this.data != nullptr);
        assert(c_other.data != nullptr);

        assert(c_this.size > 0);
        assert(c_other.size > 0);

        static std::
            unordered_map<gdf_dtype, std::size_t, std::hash<std::int32_t>>
                gdfSizeOf{{GDF_INT8, 1},
                          {GDF_INT16, 2},
                          {GDF_INT32, 4},
                          {GDF_INT64, 8},
                          {GDF_FLOAT32, 4},
                          {GDF_FLOAT64, 8}};

        const std::ptrdiff_t length =
            static_cast<std::ptrdiff_t>(c_this.size * gdfSizeOf[c_this.dtype]);

        std::uint8_t this_host[length];
        std::uint8_t other_host[length];

        cudaError_t cudaStatus;

        cudaStatus =
            cudaMemcpy(this_host, c_this.data, length, cudaMemcpyDeviceToHost);
        if (cudaSuccess != cudaStatus) {
            throw std::runtime_error("Equals cudaMemcpy for this");
        }

        cudaStatus = cudaMemcpy(
            other_host, c_other.data, length, cudaMemcpyDeviceToHost);
        if (cudaSuccess != cudaStatus) {
            throw std::runtime_error("Equals cudaMemcpy for other");
        }

        const bool datasAreNotEquals =
            std::memcmp(this_host, other_host, length);

        if (datasAreNotEquals) { return false; }

        // TODO: Check valid

        return true;
    }

    void WriteTo(std::ostream &os) const {
        switch (gdfcolumn_.dtype) {
#define WT_CASE(DTYPE) \
    case DTYPE: TypedWriteTo<GdfDType<DTYPE>>(os); break

            WT_CASE(GDF_INT8);
            WT_CASE(GDF_INT16);
            WT_CASE(GDF_INT32);
            WT_CASE(GDF_INT64);
            WT_CASE(GDF_FLOAT32);
            WT_CASE(GDF_FLOAT64);

#undef WT_CASE
            default: throw std::runtime_error("WriteTo not supported type");
        }
    }

private:
    template <class DType>
    void TypedWriteTo(std::ostream &os) const {
        typename DType::ctype *data_host =
            new typename DType::ctype[gdfcolumn_.size];

        cudaError_t cudaStatus =
            cudaMemcpy(data_host,
                       gdfcolumn_.data,
                       sizeof(typename DType::ctype) * gdfcolumn_.size,
                       cudaMemcpyDeviceToHost);

        if (cudaSuccess != cudaStatus) {
            throw std::runtime_error(
                "TypedWriteTo cudaMemcpy: " +
                std::string(cudaGetErrorName(cudaStatus)) +
                std::string(cudaGetErrorString(cudaStatus)));
        }

        os << "data =";
        for (std::size_t i = 0; i < static_cast<std::size_t>(gdfcolumn_.size);
             i++) {
            os << " " << +data_host[i];
        }
        os << std::endl;

        delete[] data_host;

        os << "size = " << gdfcolumn_.size << std::endl
           << "dtype = " << DType::name << " (as int = " << DType::dtype_value
           << ")" << std::endl;

        if (gdfcolumn_.valid) {
            os << "valid";
        } else {
            os << "wihtout valids";
        }
    }

    gdf_column gdfcolumn_;
};
}  // namespace

GdfColumn::~GdfColumn() noexcept(false) {}

std::ostream &
operator<<(std::ostream &os, const GdfColumn &gdfColumn) {
    const GdfColumnContent &content =
        static_cast<const GdfColumnContent &>(gdfColumn);
    content.WriteTo(os);
    return os;
};

}  // namespace testing
}  // namespace cudf
