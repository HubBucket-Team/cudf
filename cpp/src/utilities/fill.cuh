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

namespace cudf {

namespace kernels {

template <typename T, typename Size>
__global__ void
memset(T* buffer, Size length, T value)
{
    Size pos = threadIdx.x + blockIdx.x * blockDim.x;
    if (pos < length) { buffer[pos] = value; }
}

} // namespace kernels



// Notes:
// - Can be easily "upgraded" into a CUDF API function if you like, using
//   a short wrapper
// - The null count is set _prospectively, even though the API call might fail.
// - Perhaps there should really be a separate utility function for turning-on
//   long stretches of a packed bit sequence, or the entire sequence.
template <typename E>
gdf_error fill(
    gdf_column&   column,
    cudaStream_t  stream,
    E             value,
    bool          fill_with_nulls = false)
{
    auto null_indicator_column_size =
        gdf::util::packed_bit_sequence_size_in_bytes<uint32_t, gdf_size_type>(column.size);
    if (fill_with_nulls) {
        if (not cudf::is_nullable(column)) { return GDF_VALIDITY_MISSING; }
        CUDA_TRY ( cudaMemsetAsync(column.valid, 0, null_indicator_column_size, stream) );
        column.null_count = column.size;
    }
    else {
        enum { threads_per_block = 256 }; // TODO: Magic number... :-(
        auto grid_config { cudf::util::form_naive_1d_grid(column.size, threads_per_block) };
        kernels::memset<E>
            <<<
                grid_config.num_threads_per_block,
                grid_config.num_blocks,
                cudf::util::cuda::no_dynamic_shared_memory,
                stream
            >>>
            (static_cast<E*>(column.data), column.size, value);
        CUDA_TRY ( cudaGetLastError() );
        if (cudf::is_nullable(column)) {
            CUDA_TRY ( cudaMemsetAsync(column.valid, ~0, null_indicator_column_size, stream) );
            column.null_count = 0;
        }
    }
    return GDF_SUCCESS;
}

} // namespace cudf
