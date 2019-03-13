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

#include "column_utils.hpp"

namespace cudf {

gdf_error validate(const gdf_column& column)
{
    if (column.data == nullptr)                              { return GDF_INVALID_API_CALL; }
    if (column.dtype == GDF_invalid)                         { return GDF_DTYPE_MISMATCH;   }
    if (column.dtype >= N_GDF_TYPES)                         { return GDF_DTYPE_MISMATCH;   }
    if (not is_nullable(column) and (column.null_count > 0)) { return GDF_VALIDITY_MISSING; }
    return GDF_SUCCESS;
}

gdf_error validate(const gdf_column* column_ptr)
{
    if (column_ptr == nullptr) { return GDF_DATASET_EMPTY; }
    return validate(*column_ptr);
}

gdf_error validate(const gdf_column* const* column_sequence_ptr, gdf_num_columns_type num_columns)
{
    if (column_sequence_ptr == nullptr) { return GDF_DATASET_EMPTY; }
    for(gdf_column_index_type column_index = 0; column_index < num_columns; column_index++) {
        auto single_column_validation_result = validate(column_sequence_ptr[column_index]);
        if (single_column_validation_result != GDF_SUCCESS) { return single_column_validation_result; }
    }
    return GDF_SUCCESS;
}

bool have_matching_types(const gdf_column& validated_column_1, const gdf_column& validated_column_2)
{
    if (validated_column_1.dtype != validated_column_2.dtype) { return GDF_DTYPE_MISMATCH; }
    if (detail::logical_xor(is_nullable(validated_column_1), is_nullable(validated_column_2))) { return GDF_VALIDITY_MISSING; }
    return GDF_SUCCESS;
}

bool have_matching_types(const gdf_column* validated_column_ptr_1, const gdf_column* validated_column_ptr_2)
{
    return have_matching_types(validated_column_ptr_1, validated_column_ptr_2);
}

bool has_uniform_column_sizes(const gdf_column* const * validated_column_sequence, gdf_num_columns_type num_columns)
{
    if (num_columns == 0) { return true; }
    auto uniform_size = validated_column_sequence[0]->size;
    auto has_appropriate_size =
        [&uniform_size](const gdf_column* cp) { return cp->size == uniform_size; };
    return std::all_of(
        validated_column_sequence + 1,
        validated_column_sequence + num_columns,
        has_appropriate_size);
}


bool have_matching_types(
    const gdf_column* const* validated_column_sequence_ptr_1,
    const gdf_column* const* validated_column_sequence_ptr_2,
    gdf_num_columns_type num_columns)
{
    // I'd use all_of but that would require a zip iterator.
    for(gdf_column_index_type i = 0; i < num_columns; i++) {
        auto lhs = validated_column_sequence_ptr_1[i];
        auto rhs = validated_column_sequence_ptr_2[i];
        if (not have_matching_types(lhs, rhs)) { return false; }
    }
    return true;
}

} // namespace
