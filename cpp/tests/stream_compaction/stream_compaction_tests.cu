/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stream_compaction.hpp>

#include <utilities/error_utils.hpp>

#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/cudf_test_fixtures.h>

struct ApplyBooleanMaskErrorTest : GdfTest {};

// Test ill-formed inputs

TEST_F(ApplyBooleanMaskErrorTest, NullPtrs)
{
  constexpr gdf_size_type column_size{1000};

  cudf::test::column_wrapper<int32_t> source{column_size};
  cudf::test::column_wrapper<gdf_bool> mask{column_size};
             
  CUDF_EXPECT_THROW_MESSAGE(cudf::apply_boolean_mask(nullptr, mask), 
                            "Null input");

  CUDF_EXPECT_THROW_MESSAGE(cudf::apply_boolean_mask(source, nullptr),
                            "Null boolean_mask");
}

TEST_F(ApplyBooleanMaskErrorTest, SizeMismatch)
{
  constexpr gdf_size_type column_size{1000};
  constexpr gdf_size_type mask_size{500};

  cudf::test::column_wrapper<int32_t> source{column_size};
  cudf::test::column_wrapper<gdf_bool> mask{mask_size};
             
  CUDF_EXPECT_THROW_MESSAGE(cudf::apply_boolean_mask(source, mask), 
                            "Column size mismatch");
}

TEST_F(ApplyBooleanMaskErrorTest, NonBooleanMask)
{
  constexpr gdf_size_type column_size{1000};

  cudf::test::column_wrapper<int32_t> source{column_size};
  cudf::test::column_wrapper<float> nonbool_mask{column_size};
             
  CUDF_EXPECT_THROW_MESSAGE(cudf::apply_boolean_mask(source, nonbool_mask), 
                            "Mask must be Boolean type");

  cudf::test::column_wrapper<cudf::bool8> bool_mask{column_size, true};
  EXPECT_NO_THROW(cudf::apply_boolean_mask(source, bool_mask));
}

template <typename T>
struct ApplyBooleanMaskTest : GdfTest {};

using test_types =
    ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double>;
TYPED_TEST_CASE(ApplyBooleanMaskTest, test_types);

// Test computation

/*
 * Runs apply_boolean_mask checking for errors, and compares the result column 
 * to the specified expected result column.
 */
template <typename T>
void BooleanMaskTest(cudf::test::column_wrapper<T> source,
                     cudf::test::column_wrapper<cudf::bool8> mask,
                     cudf::test::column_wrapper<T> expected)
{
  gdf_column result;
  EXPECT_NO_THROW(result = cudf::apply_boolean_mask(source, mask));

  EXPECT_TRUE(expected == result);

  if (!(expected == result)) {
    std::cout << "expected\n";
    expected.print();
    std::cout << expected.get()->null_count << "\n";
    std::cout << "result\n";
    print_gdf_column(&result);
    std::cout << result.null_count << "\n";
  }

  gdf_column_free(&result);
}

TYPED_TEST(ApplyBooleanMaskTest, Identity)
{
  constexpr gdf_size_type column_size{1000001};

  BooleanMaskTest<TypeParam>(
    cudf::test::column_wrapper<TypeParam>{column_size,                             
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return true; }},
    cudf::test::column_wrapper<cudf::bool8>{column_size,
      [](gdf_index_type row) { return cudf::bool8{true}; },
      [](gdf_index_type row) { return true; }},
    cudf::test::column_wrapper<TypeParam>{column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return true; }});
}

TYPED_TEST(ApplyBooleanMaskTest, MaskAllFalse)
{
  constexpr gdf_size_type column_size{1000};

  BooleanMaskTest<TypeParam>(
    cudf::test::column_wrapper<TypeParam>{column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return true; }},
    cudf::test::column_wrapper<cudf::bool8>{column_size,
      [](gdf_index_type row) { return cudf::bool8{false}; },
      [](gdf_index_type row) { return true; }},
    cudf::test::column_wrapper<TypeParam>{0, false});
}

TYPED_TEST(ApplyBooleanMaskTest, MaskAllNull)
{
  constexpr gdf_size_type column_size{1000};

  BooleanMaskTest<TypeParam>(
    cudf::test::column_wrapper<TypeParam>{column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return true; }},
    cudf::test::column_wrapper<cudf::bool8>{column_size, 
      [](gdf_index_type row) { return cudf::bool8{true}; },
      [](gdf_index_type row) { return false; }},
    cudf::test::column_wrapper<TypeParam>{0, false});
}

TYPED_TEST(ApplyBooleanMaskTest, MaskEvensFalse)
{
  constexpr gdf_size_type column_size{1000};

  BooleanMaskTest<TypeParam>(
    cudf::test::column_wrapper<TypeParam>{column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return true; }},
    cudf::test::column_wrapper<cudf::bool8>{column_size,
      [](gdf_index_type row) { return cudf::bool8{row % 2 == 1}; },
      [](gdf_index_type row) { return true; }},
    cudf::test::column_wrapper<TypeParam>{(column_size + 1) / 2,
      [](gdf_index_type row) { return 2 * row + 1;  },
      [](gdf_index_type row) { return true; }});
}

TYPED_TEST(ApplyBooleanMaskTest, MaskEvensNull)
{
  constexpr gdf_size_type column_size{1000};

  // mix it up a bit by setting the input odd values to be null
  // Since the bool mask has even values null, the output
  // vector should have all values nulled

  BooleanMaskTest<TypeParam>(
    cudf::test::column_wrapper<TypeParam>{column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return row % 2 == 0; }},
    cudf::test::column_wrapper<cudf::bool8>{column_size,
      [](gdf_index_type row) { return cudf::bool8{true}; },
      [](gdf_index_type row) { return row % 2 == 1; }},
    cudf::test::column_wrapper<TypeParam>{(column_size + 1) / 2,
      [](gdf_index_type row) { return 2 * row + 1;  },
      [](gdf_index_type row) { return false; }});
}

TYPED_TEST(ApplyBooleanMaskTest, NoNullMask)
{
  constexpr gdf_size_type column_size{1000};

  std::vector<TypeParam> source(column_size, TypeParam{0});
  std::vector<TypeParam> expected((column_size + 1) / 2, TypeParam{0});
  std::iota(source.begin(), source.end(), TypeParam{0});
  std::generate(expected.begin(), expected.end(), 
                [n = -1] () mutable { return n+=2; });

  BooleanMaskTest<TypeParam>(
    cudf::test::column_wrapper<TypeParam>{source},
    cudf::test::column_wrapper<cudf::bool8>{column_size,
      [](gdf_index_type row) { return cudf::bool8{true}; },
      [](gdf_index_type row) { return row % 2 == 1; }},
    cudf::test::column_wrapper<TypeParam> {expected});
}