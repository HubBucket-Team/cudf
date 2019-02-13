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
        using ctype                       = C;                                 \
        static constexpr std::size_t size = sizeof(ctype);                     \
    }

DTRAITS_FACTORY(GDF_INT8, std::int8_t);
DTRAITS_FACTORY(GDF_INT16, std::int16_t);
DTRAITS_FACTORY(GDF_INT32, std::int32_t);
DTRAITS_FACTORY(GDF_INT64, std::int64_t);
DTRAITS_FACTORY(GDF_FLOAT32, float);
DTRAITS_FACTORY(GDF_FLOAT64, double);

#undef DTRAITS_FACTORY


template <gdf_dtype D>
static gdf_column *MakeGdfColumn(
    const std::size_t                                       length,
    const std::initializer_list<typename DTraits<D>::ctype> initialData = {}) {
    assert(length >= initialData.size());

    cudaError_t cudaError;

    void *data;
    cudaError = cudaMalloc(&data, length * sizeof(DTraits<D>::size));
    if (cudaSuccess != cudaError) {
        throw std::runtime_error("cudaMalloc for data");
    }

    const std::size_t validLength =
        static_cast<std::size_t>(std::ceil(length / 8.0));
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
        throw std::runtime_error("cudaMemset for valid");
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

static void ExpectEqColumns(const std::int64_t *expectedData,
                            gdf_column *        outputColumn,
                            const std::size_t   outputLength,
                            const std::string & message) {
    std::int64_t resultData[outputLength];
    cudaError_t  cudaError = cudaMemcpy(resultData,
                                       outputColumn->data,
                                       outputLength * DTraits<GDF_INT64>::size,
                                       cudaMemcpyDeviceToHost);
    if (cudaSuccess != cudaError) { FAIL() << "cudaMempcy output column"; }

    for (std::size_t i = 0; i < 6; i++) {
        EXPECT_EQ(expectedData[i], resultData[i])
            << "i = " << i << " " << message;
    }
}

TEST(SortedMergeTest, MergeTwoSortedColumns) {
    gdf_column *leftColumn1 = MakeGdfColumn<GDF_INT64>(4, {0, 1, 2, 3});
    gdf_column *leftColumn2 = MakeGdfColumn<GDF_INT64>(4, {4, 5, 6, 7});

    gdf_column *rightColumn1 = MakeGdfColumn<GDF_INT64>(2, {1, 2});
    gdf_column *rightColumn2 = MakeGdfColumn<GDF_INT64>(2, {8, 9});

    const std::size_t outputLength  = 16;
    gdf_column *      outputColumn1 = MakeGdfColumn<GDF_INT64>(outputLength);
    gdf_column *      outputColumn2 = MakeGdfColumn<GDF_INT64>(outputLength);

    gdf_column *leftColumns[]  = {leftColumn1, leftColumn2};
    gdf_column *rightColumns[] = {rightColumn1, rightColumn2};

    gdf_column *outputColumns[] = {outputColumn1, outputColumn2};

    gdf_column *orders =
        MakeGdfColumn<GDF_INT64>(2, {GDF_ORDER_ASC, GDF_ORDER_DESC});

    gdf_column *outputIndices = MakeGdfColumn<GDF_INT32>(1, {0});

    const std::size_t columnsLength = 2;
    gdf_error         gdfError      = gdf_sorted_merge(leftColumns,
                                          rightColumns,
                                          columnsLength,
                                          outputIndices,
                                          orders,
                                          outputColumns);

    EXPECT_EQ(GDF_SUCCESS, gdfError);

    const std::int64_t expectedData1[] = {0, 1, 1, 2, 2, 3};
    const std::int64_t expectedData2[] = {4, 5, 8, 9, 6, 7};

    ExpectEqColumns(expectedData1, outputColumn1, outputLength, "block 1");
    ExpectEqColumns(expectedData2, outputColumn2, outputLength, "block 2");
}
