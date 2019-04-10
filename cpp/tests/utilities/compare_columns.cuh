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


#include <utilities/miscellany.hpp>
#include <utilities/type_dispatcher.hpp>
#include <utilities/column_utils.hpp>

namespace cudf {

namespace kernels {

template <typename T, typename Size>
__global__ void
compare(T* buffer, Size length, T value)
{
    Size pos = threadIdx.x + blockIdx.x * blockDim.x;
    if (pos < length) { buffer[pos] = value; }
}

} // namespace kernels


struct comparison_helper {
    template <typename T>
    bool operator()(const void* __restrict__ lhs, const void* __restrict__ rhs, gdf_size_type size)
    {
        return
            thrust::equal(
                thrust::cuda::par,
                reinterpret_cast<const T*>(lhs),
                reinterpret_cast<const T*>(lhs) + size,
                reinterpret_cast<const T*>(rhs)
            );
    }
};

// Notes:
// - Can be easily "upgraded" into a CUDF API function if you like, using
//   a short wrapper
// - The null count is set _prospectively, even though the API call might fail.
// - Perhaps there should really be a separate utility function for turning-on
//   long stretches of a packed bit sequence, or the entire sequence.
bool equal(
    gdf_column&   lhs,
    gdf_column&   rhs,
    cudaStream_t  stream,
    bool          byte_for_byte = false)
{
    if (left.size != right.size) { return false; }
    if (left.dtype != right.dtype) { return false; }
    if (left.null_count != right.null_count) { return false; }

    auto common_dtype = lhs.dtype;
    auto common_size = lhs.size;
    bool need_to_compare_time_unit =
        (common_dtype == GDF_TIMESTAMP) or byte_for_byte;
    if (need_to_compare_time_unit and left.dtype_info.time_unit != right.dtype_info.time_unit) { return false; }

    if (byte_for_byte) {
        if (logical_xor(left.data == nullptr, right.data == nullptr)) { return false; }
        if (logical_xor(left.valid == nullptr, right.valid == nullptr)) { return false; }


        if (lhs.data) {
            if (not (util::type_dispatcher(common_dtype, comparison_helper, lhs.data, rhs.data, common_size))) { return false; }
        }

        if (lhs.valid) {
            bitmask_size =  gdf::util::packed_bit_sequence_size_in_bytes<uint32_t, gdf_size_type>(column.size);
            if (not (util::type_dispatcher(gdf_valid_type, comparison_helper, lhs.valid, rhs.valid, bitmask_size))) { return false; }
        }
    }
    return true;
}

} // namespace cudf
