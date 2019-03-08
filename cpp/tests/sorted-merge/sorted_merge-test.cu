#include <cassert>

#include <cuda_runtime.h>
#include <cudf.h>

#include <gtest/gtest.h>

template <gdf_dtype D>
class DTraits {};

#define DTRAITS_FACTORY(D, C)                                                  \
    template <>                                                                \
    class DTraits<D> {                                                         \
    public:                                                                    \
        using ctype                        = C;                                \
        static constexpr gdf_size_type size  = sizeof(ctype);                    \
        static constexpr gdf_dtype   dtype = D;                                \
    }

DTRAITS_FACTORY(GDF_INT8, std::int8_t);
DTRAITS_FACTORY(GDF_INT16, std::int16_t);
DTRAITS_FACTORY(GDF_INT32, std::int32_t);
DTRAITS_FACTORY(GDF_INT64, std::int64_t);
DTRAITS_FACTORY(GDF_FLOAT32, float);
DTRAITS_FACTORY(GDF_FLOAT64, double);
DTRAITS_FACTORY(GDF_DATE32, std::int32_t);
DTRAITS_FACTORY(GDF_DATE64, std::int64_t);

template <gdf_dtype D>
static gdf_column *MakeGdfColumn(
    const gdf_size_type                                       length,
    const std::initializer_list<typename DTraits<D>::ctype> initialData = {}) {
    assert(length >= initialData.size());

    cudaError_t cudaError;

    void *data;
    cudaError = cudaMalloc(&data, length * sizeof(DTraits<D>::size));
    if (cudaSuccess != cudaError) {
        throw std::runtime_error("cudaMalloc for data");
    }

    const gdf_size_type validLength =
        static_cast<gdf_size_type>(std::ceil(length / 8.0));
    gdf_valid_type *valid;
    cudaError = cudaMalloc(&valid, validLength);
    if (cudaSuccess != cudaError) {
        cudaFree(data);
        throw std::runtime_error("cudaMalloc for valid");
    }
    cudaError = cudaMemset(valid, 0, validLength);
    if (cudaSuccess != cudaError) {
        cudaFree(data);
        cudaFree(valid);
        throw std::runtime_error("cudaMemset for clean valid");
    }
    cudaError = cudaMemset(valid,
                           static_cast<std::uint8_t>(-1) >>
                               (sizeof(std::uint64_t) - initialData.size()),
                           1);
    if (cudaSuccess != cudaError) {
        cudaFree(data);
        cudaFree(valid);
        throw std::runtime_error("cudaMemset for valid");
    }

    cudaError = cudaMemcpy(data,
                           initialData.begin(),
                           initialData.size() * DTraits<D>::size,
                           cudaMemcpyHostToDevice);
    if (cudaSuccess != cudaError) {
        cudaFree(data);
        cudaFree(valid);
        throw std::runtime_error("cudaMemcpy initial data");
    }

    return new gdf_column{
        data, valid, length, D, length - initialData.size(), {}, nullptr};
}

template <class DTraits>
class SortedMergeTest : public testing::Test {
protected:
    static void ExpectEqColumns(const typename DTraits::ctype *expectedData,
                                gdf_column *                   outputColumn,
                                const gdf_size_type              outputLength,
                                const std::string &            message) {
        typename DTraits::ctype resultData[outputLength];
        cudaError_t             cudaError = cudaMemcpy(resultData,
                                           outputColumn->data,
                                           outputLength * DTraits::size,
                                           cudaMemcpyDeviceToHost);
        if (cudaSuccess != cudaError) { FAIL() << "cudaMempcy output column"; }

        for (gdf_size_type i = 0; i < 6; i++) {
            EXPECT_EQ(expectedData[i], resultData[i])
                << "i = " << i << " " << message;
        }
    }
};

using SortedMergerTypes = testing::Types<DTraits<GDF_INT8>,
                                         DTraits<GDF_INT16>,
                                         DTraits<GDF_INT32>,
                                         DTraits<GDF_INT64>,
                                         DTraits<GDF_FLOAT32>,
                                         DTraits<GDF_FLOAT64>,
                                         DTraits<GDF_DATE32>,
                                         DTraits<GDF_DATE64>>;
TYPED_TEST_CASE(SortedMergeTest, SortedMergerTypes);

TYPED_TEST(SortedMergeTest, MergeTwoSortedColumns) {
    gdf_column *leftColumn1 = MakeGdfColumn<TypeParam::dtype>(4, {0, 1, 2, 3});
    gdf_column *leftColumn2 = MakeGdfColumn<TypeParam::dtype>(4, {4, 5, 6, 7});

    gdf_column *rightColumn1 = MakeGdfColumn<TypeParam::dtype>(2, {1, 2});
    gdf_column *rightColumn2 = MakeGdfColumn<TypeParam::dtype>(2, {8, 9});

    const gdf_size_type outputLength = 16;
    gdf_column *outputColumn1 = MakeGdfColumn<TypeParam::dtype>(outputLength);
    gdf_column *outputColumn2 = MakeGdfColumn<TypeParam::dtype>(outputLength);

    gdf_column *leftColumns[]  = {leftColumn1, leftColumn2};
    gdf_column *rightColumns[] = {rightColumn1, rightColumn2};

    gdf_column *outputColumns[] = {outputColumn1, outputColumn2};

    gdf_column *orders =
        MakeGdfColumn<GDF_INT8>(2, {GDF_ORDER_ASC, GDF_ORDER_DESC});

    gdf_column *outputIndices = MakeGdfColumn<GDF_INT32>(1, {0});

    const gdf_size_type columnsLength = 2;
    gdf_error         gdfError      = gdf_sorted_merge(leftColumns,
                                          rightColumns,
                                          columnsLength,
                                          outputIndices,
                                          orders,
                                          outputColumns);

    EXPECT_EQ(GDF_SUCCESS, gdfError);

    const typename TypeParam::ctype expectedData1[] = {0, 1, 1, 2, 2, 3};
    const typename TypeParam::ctype expectedData2[] = {4, 5, 8, 9, 6, 7};

    this->ExpectEqColumns(
        expectedData1, outputColumn1, outputLength, "block 1");
    this->ExpectEqColumns(
        expectedData2, outputColumn2, outputLength, "block 2");
}
