
/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Alexander Ocsa <alexander@blazingdb.com>
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
#pragma once

#include <climits>
#include <cstdint>

namespace gdf {
namespace util {

static constexpr int ValidSize = 32;
using ValidType = uint32_t;


// Instead of this function, use gdf_get_num_chars_bitmask from gdf/utils.h
//__host__ __device__ __forceinline__
//  size_t
//  valid_size(size_t column_length)
//{
//  const size_t n_ints = (column_length / ValidSize) + ((column_length % ValidSize) ? 1 : 0);
//  return n_ints * sizeof(ValidType);
//}

// Instead of this function, use gdf_is_valid from gdf/utils.h
///__host__ __device__ __forceinline__ bool get_bit(const gdf_valid_type* const bits, size_t i)
///{
///  return  bits == nullptr? true :  bits[i >> size_t(3)] & (1 << (i & size_t(7)));
///}

__host__ __device__ __forceinline__ void turn_bit_on(uint8_t* const bits, size_t i)
{
  bits[i / 8] |= (1 << (i % 8));
}

__host__ __device__ __forceinline__ void turn_bit_off(uint8_t* const bits, size_t i)
{
  bits[i / 8] &= ~(1 << (i % 8));
}

__host__ __device__ __forceinline__ size_t last_byte_index(size_t column_size)
{
  return (column_size + 8 - 1) / 8;
}

/**
 * Checks if a bit is set in a sequence of bits in container types,
 * such that within each container the bits are ordered LSB to MSB
 *
 * @note this is endianness-neutral
 *
 * @param bits pointer to the beginning of the sequence of bits
 * @param bit_index index to check in the sequence
 * @return 0 if the bit is unset, 1 otherwise
 */
template <typename BitContainer, typename Size>
__host__ __device__ __forceinline__ auto bit_is_set(const BitContainer* bits, Size bit_index)
{
    enum : Size { bits_per_container = sizeof(BitContainer) * CHAR_BIT };
    auto container_index = bit_index / bits_per_container;
    auto intra_container_index = bit_index % bits_per_container;
    return (bits[container_index] >> intra_container_index) & 1;
}

template <typename BitContainer, typename Size>
__host__ __device__ __forceinline__ auto bit_is_set(const BitContainer& bit_container, Size bit_index)
{
    enum : Size { bits_per_container = sizeof(BitContainer) * CHAR_BIT };
    auto intra_container_index = bit_index % bits_per_container;
    return (bit_container >> intra_container_index) & 1;
}

static inline std::string chartobin(gdf_valid_type c, size_t size = 8)
{
  std::string bin;
  bin.resize(size);
  bin[0] = 0;
  size_t i;
  for (i = 0; i < size; i++) {
    bin[i] = (c % 2) + '0';
    c /= 2;
  }
  return bin;
}

static inline std::string gdf_valid_to_str(gdf_valid_type* valid, size_t column_size)
{
  size_t last_byte = gdf::util::last_byte_index(column_size);
  std::string response;
  for (size_t i = 0; i < last_byte; i++) {
    size_t n_bits = last_byte != i + 1 ? 8 : column_size - 8 * (last_byte - 1);
    auto result = chartobin(valid[i], n_bits);
    response += std::string(result);
  }
  return response;
}

} // namespace util
} // namespace gdf
