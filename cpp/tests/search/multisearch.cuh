/*
 * Copyright 2019 BlazingDB, Inc.
 *     Copyright 2019 Eyal Rozenberg <eyalroz@blazingdb.com>
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

#ifndef CUDF_TESTS_MULTISEARCH_CUH_
#define CUDF_TESTS_MULTISEARCH_CUH_

#include <tests/utilities/cudf_test_fixtures.h> // for GdfTest
#include <tests/utilities/cudf_test_utils.cuh>
#include <tests/utilities/column_wrapper.cuh>
#include <utilities/fill.cuh>
#include <utilities/bit_util.cuh>
#include <utilities/type_dispatcher.hpp>
#include <utilities/miscellany.hpp>

#include <cudf.h>

#include <gtest/gtest.h>

#include <iostream>
#include <iomanip>
#include <tuple>

#define ASSERT_CUDF_SUCCESS(gdf_error_expression) \
{ \
    gdf_error _assert_cudf_success_eval_result;\
	ASSERT_NO_THROW(_assert_cudf_success_eval_result = gdf_error_expression); \
	const char* _assertion_failure_message = #gdf_error_expression; \
	ASSERT_EQ(_assert_cudf_success_eval_result, GDF_SUCCESS) << "Failing expression: " << _assertion_failure_message; \
}


// This should really be part of the cudf API (and type-erased)
template <typename E>
gdf_error fill(
    gdf_column&   column,
    E&&           value,
    bool          fill_with_nulls = false)
{
    cudf::util::cuda::scoped_stream stream;
    return cudf::fill(column, stream, std::forward<E>(value), fill_with_nulls);
}

template<typename E>
void expect_columns_are_equal(
    cudf::test::column_wrapper<E> const&  lhs,
    const std::string&        lhs_name,
    cudf::test::column_wrapper<E> const&  rhs,
    const std::string&        rhs_name,
    bool                      print_all_unequal_pairs = false)
{
    auto& lhs_gdf_column = *(lhs.get()); // TODO: Make this const
    auto& rhs_gdf_column = *(rhs.get()); // TODO: Make this const
    EXPECT_EQ(lhs_gdf_column.dtype, rhs_gdf_column.dtype);
    EXPECT_EQ(lhs_gdf_column.size, rhs_gdf_column.size);
    auto common_size = lhs_gdf_column.size;
    EXPECT_EQ(lhs_gdf_column.null_count, rhs_gdf_column.null_count);
    if (common_size == 0) { return; }
    auto non_nullable = (lhs_gdf_column.null_count == 0);

    auto lhs_on_host = lhs.to_host();
    auto rhs_on_host = rhs.to_host();

    const E* lhs_data_on_host  = &(std::get<0>(lhs_on_host)[0]);
    const E* rhs_data_on_host  = &(std::get<0>(rhs_on_host)[0]);

    const gdf_valid_type * lhs_valid_on_host = &(std::get<1>(lhs_on_host)[0]);
    const gdf_valid_type * rhs_valid_on_host = &(std::get<1>(rhs_on_host)[0]);

    auto max_name_length = std::max(lhs_name.length(), rhs_name.length());

    for(gdf_size_type i = 0; i < common_size; i++) {
        auto lhs_element_is_valid = non_nullable or gdf::util::bit_is_set<gdf_valid_type, gdf_size_type>(lhs_valid_on_host, i);
        auto rhs_element_is_valid = non_nullable or gdf::util::bit_is_set<gdf_valid_type, gdf_size_type>(rhs_valid_on_host, i);
        auto elements_are_equal =
            (not lhs_element_is_valid and not rhs_element_is_valid) or
            (lhs_element_is_valid == rhs_element_is_valid and lhs_data_on_host[i] == rhs_data_on_host[i]);
        EXPECT_TRUE(elements_are_equal)
            << std::left << std::setw(max_name_length) << lhs_name << std::right << '[' << i << "] = " << (lhs_element_is_valid ? std::to_string(lhs_data_on_host[i]) : "@") << '\n'
            << std::left << std::setw(max_name_length) << rhs_name << std::right << '[' << i << "] = " << (rhs_element_is_valid ? std::to_string(rhs_data_on_host[i]) : "@") ;
        if (not print_all_unequal_pairs and not elements_are_equal) { break; }
    }
}

template<typename E>
void expect_column(
    cudf::test::column_wrapper<E> const&  actual,
    cudf::test::column_wrapper<E> const&  expected,
    bool                                  print_all_unequal_pairs = false)
{
    return expect_columns_are_equal<E>(expected, "Expected", actual, "Actual", print_all_unequal_pairs);
}


template<typename Container>
auto make_validity_initializer(const Container& unpacked_validity_indicators) noexcept
{
    return [unpacked_validity_indicators](gdf_size_type pos) { return *(std::cbegin(unpacked_validity_indicators) + pos); };
}


enum : bool {
    find_first_greater = true,
    find_first_greater_or_equal = false
};

enum : bool {
    nulls_appear_before_values = true,
    nulls_appear_after_values = false
};

enum : bool {
    use_haystack_length_for_not_found = true,
    use_null_for_not_found = false
};


enum : bool {
    non_nullable = false,
    nullable = true
};

template <typename E>
void print(const cudf::test::column_wrapper<E>& wrapper, const std::string& title)
{
    std::cout << title << std::endl;
    wrapper.print();
    return;
}

#define self_titled_print(wrapper) print(wrapper, CUDF_STRINGIFY(wrapper))

#endif // CUDF_TESTS_MULTISEARCH_CUH_
