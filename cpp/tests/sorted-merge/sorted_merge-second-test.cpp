#include <cudf.h>
#include <rmm/rmm.hpp>

#include "cudf_testing/test.hpp"

CUDF_TEST_NUMBERS(SortedMergeSecondTest);

TYPED_TEST(SortedMergeSecondTest, WithTwoNumberShortColumns) {
    ASSERT_EQ(RMM_SUCCESS, rmmInitialize(nullptr));

    using cudf::testing::GdfColumnBuilder;
    using cudf::testing::GdfDType;

    // left
    auto leftColumn1 =
      GdfColumnBuilder<GdfDType<GDF_INT64>>::Make()
                           ->WithLength(6)
                           .SetData({0, 0, 0, 1, 1, 1})
                           .Build();

    auto leftColumn2 =
      this->GdfColumnBuilder()
                           .WithLength(6)
                           .SetData({
                               3305832.000000,
                               5744833.000000,
                               7382853.000000,
                               3287842.000000,
                               6416831.000000,
                               6705814.000000,
                           })
                           .Build();

    // right
    auto rightColumn1 = GdfColumnBuilder<GdfDType<GDF_INT64>>::Make()
                            ->WithLength(8)
                            .SetData({0, 0, 0, 1, 1, 1, 1, 2})
                            .Build();

    auto rightColumn2 = this->GdfColumnBuilder()
                            .WithLength(8)
                            .SetData({
                                 379185.000000,
                                 428785.000000,
                                7617827.000000,
                                 345786.000000,
                                 526583.000000,
                                7497812.000000,
                                9320801.000000,
                                -361886.000000,
                            })
                            .Build();

    // to gdf_column
    gdf_column left_c_1 = *leftColumn1;
    gdf_column left_c_2 = *leftColumn2;

    gdf_column right_c_1 = *rightColumn1;
    gdf_column right_c_2 = *rightColumn2;

    // format for sorted merger
    gdf_column *left[]  = {&left_c_1, &left_c_2};
    gdf_column *right[] = {&right_c_1, &right_c_2};

    // output
    auto outputColumn1 =
      GdfColumnBuilder<GdfDType<GDF_INT64>>::Make()
                           ->WithLength(14)
                           .Build();
    auto outputColumn2 = this->GdfColumnBuilder().WithLength(14).Build();

    gdf_column output_c_1 = *outputColumn1;
    gdf_column output_c_2 = *outputColumn2;

    gdf_column *output[] = {&output_c_1, &output_c_2};

    // paramters for sorted merger
    const gdf_size_type ncols = 2;

    auto sortByIndices = GdfColumnBuilder<GdfDType<GDF_INT32>>::Make()
                             ->WithLength(2)
                             .SetData({0, 1})
                             .Build();
    gdf_column sort_by_cols = *sortByIndices;

    auto ascDescForColumns = GdfColumnBuilder<GdfDType<GDF_INT8>>::Make()
                                 ->WithLength(2)
                                 .SetData({GDF_ORDER_ASC, GDF_ORDER_ASC})
                                 .Build();
    gdf_column asc_desc = *ascDescForColumns;

    // call sorted merger
    gdf_error gdf_status =
        gdf_sorted_merge(left, right, ncols, &sort_by_cols, &asc_desc, output);

    EXPECT_EQ(GDF_SUCCESS, gdf_status);

    // expected
    auto expectedColumn1 =
        this->GdfColumnBuilder()
            .WithLength(13)
            .SetData({1, 1, 1, 2, 3, 3, 5, 5, 7, 8, 9, 11, 13})
            .Build();

    auto expectedColumn2 =
        this->GdfColumnBuilder()
            .WithLength(13)
            .SetData({4, 5, 7, 9, 5, 11, 6, 13, 7, 13, 8, 9, 14})
            .Build();

    EXPECT_EQ(*expectedColumn1, *outputColumn1);
    EXPECT_EQ(*expectedColumn2, *outputColumn2);

    ASSERT_EQ(RMM_SUCCESS, rmmFinalize());
}
