/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Eyal Rozenberg <eyalroz@blazingdb.com>
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

#ifndef UTIL_MISCELLANY_HPP_
#define UTIL_MISCELLANY_HPP_

// Code in this file should aventually be placed in finer-grained utility headers

#include <cstdlib> // for std::div

extern "C" {
#include <cudf.h>
}

#include <utilities/type_dispatcher.hpp>

#ifndef NVCC
#ifndef __host__

#define __host__
#define __device__
#define __forceinline__ inline

#endif
#endif

namespace cudf {

constexpr inline bool is_an_integer(gdf_dtype element_type)
{
    return
        element_type == GDF_INT8  or
        element_type == GDF_INT16 or
        element_type == GDF_INT32 or
        element_type == GDF_INT64;
}

constexpr bool is_nullable(const gdf_column& column)
{
    return column.valid != nullptr;
}

namespace detail {

struct size_of_helper {
    template <typename T>
    constexpr int operator()() const { return sizeof(T); }
};

}

std::size_t inline size_of(gdf_dtype element_type) {
    return type_dispatcher(element_type, detail::size_of_helper{});
}

inline std::size_t width(const gdf_column& col)
{
    return size_of(col.dtype);
}

inline std::size_t data_size_in_bytes(const gdf_column& col)
{
    return col.size * width(col);
}

namespace util {

namespace cuda {

enum : unsigned { no_dynamic_shared_memory = 0 };

struct scoped_stream {
    cudaStream_t stream_ { nullptr };

    scoped_stream() {
        cudaStreamCreate(&stream_);
        assert(stream_ != nullptr and "Failed creating a CUDA stream");
    }
    operator cudaStream_t() { return stream_; }
    ~scoped_stream() {
        if (not std::uncaught_exception()) {
            cudaStreamSynchronize(stream_);
        }
        cudaStreamDestroy(stream_);
    }
};

template <typename T>
cudaError_t copy_single_value(
    T&             destination,
    const T&       source,
    cudaStream_t   stream)
{
    static_assert(std::is_trivially_copyable<T>::value, "Invalid type specified - it must be trivially copyable");
    cudaMemcpyAsync(&destination, &source, sizeof(T), cudaMemcpyDefault, stream);
    if (cudaPeekAtLastError() != cudaSuccess) { return cudaGetLastError(); }
    cudaStreamSynchronize(stream);
    return cudaGetLastError();
}

} // namespace cuda

template <typename T>
inline constexpr bool is_power_of_2(T val) { return (val & (val-1)) == 0; } // Yes, this works

template <typename T>
inline constexpr T round_down_to_power_of_2(T val, T power_of_two)
{
    return val & ~(power_of_two - 1);
}

template <typename T>
inline constexpr T round_up_to_power_of_2(T val, T power_of_two)
{
    // TODO: Make sure this works for 0 like one might expect
    return round_down_to_power_of_2(val - 1, power_of_two) + power_of_two;
}


/**
* Divides the left-hand-side by the right-hand-side, rounding up
* to an integral multiple of the right-hand-side, e.g. (9,5) -> 2 , (10,5) -> 2, (11,5) -> 3.
*
* @param dividend the number to divide
* @param divisor the number of by which to divide
* @return The least integer multiple of {@link divisor} which is greater-or-equal to
* the non-integral division dividend/divisor.
*
* @note sensitive to overflow, i.e. if dividend > std::numeric_limits<S>::max() - divisor,
* the result will be incorrect
*/
template <typename S, typename T>
constexpr inline S div_rounding_up_unsafe(const S& dividend, const T& divisor) {
    return (dividend + divisor - 1) / divisor;
}

namespace detail {

template <typename I>
constexpr inline I div_rounding_up_safe(std::integral_constant<bool, false>, I dividend, I divisor)
{
    // TODO: This could probably be implemented faster
    return (dividend > divisor) ?
        1 + div_rounding_up_unsafe(dividend - divisor, divisor) :
        (dividend > 0);
}

template <typename I>
constexpr inline I div_rounding_up_safe(std::integral_constant<bool, true>, I dividend, I divisor)
{
#if cplusplus >= 201402L
        auto div_result = std::div(dividend, divisor);
        return div_result.quot + !(!div_result.rem);
#else
        // Hopefully the compiler will optimize the two calls away.
        return std::div(dividend, divisor).quot + !(!std::div(dividend, divisor).rem);
#endif
}


} // namespace detail

/**
* Divides the left-hand-side by the right-hand-side, rounding up
* to an integral multiple of the right-hand-side, e.g. (9,5) -> 2 , (10,5) -> 2, (11,5) -> 3.
*
* @param dividend the number to divide
* @param divisor the number of by which to divide
* @return The least integer multiple of {@link divisor} which is greater-or-equal to
* the non-integral division dividend/divisor.
*
* @note will not overflow, and may _or may not_ be slower than the intuitive
* approach of using (dividend + divisor - 1) / divisor
*/
template <typename I>
constexpr inline I div_rounding_up_safe(I dividend, I divisor)
{
    using i_is_a_signed_type = std::integral_constant<bool, std::is_signed<I>::value>;
    return detail::div_rounding_up_safe(i_is_a_signed_type{}, dividend, divisor);
}

template <typename T, template <typename S> class Trait>
using having_trait_t = typename std::enable_if_t<Trait<T>::value>;

// TODO: Use enable_if_T or having_trait_t to only allow the following
// to be instantiated for integral types I
template <typename I>
constexpr inline bool
is_a_power_of_two(I val)
    // util::having_trait_t<I, std::is_integral> val)
{
    return ((val - 1) & val) == 0;
}

/*
template <typename I>
constexpr inline I div_rounding_up_by_power_of_2(I dividend, I divisor)
{
    // TODO: assert divisor is indeed a power of 2
    div_rounding_down_by_power_of_2
        dividend && (divisor - 1);

        (dividend > divisor) ?
        1 + div_rounding_up_unsafe(dividend - divisor, divisor) :
        (dividend > 0);
}
*/




template <typename T>
constexpr inline std::size_t size_in_bits() { return sizeof(T) * CHAR_BIT; }

template <typename T>
constexpr inline std::size_t size_in_bits(const T&) { return size_in_bits<T>(); }

// TODO: Use enable_if_T or having_trait_t to only allow the following
// to be instantiated for integral types I
template <typename I>
constexpr inline I
clear_lower_bits_unsafe(
//    util::having_trait_t<I, std::is_integral>  val,
    I                                          val,
    unsigned                                   num_bits_to_clear)
{
    auto lower_bits_mask = I{1} << (num_bits_to_clear - 1);
    return val & ~lower_bits_mask;
}


// TODO: Use enable_if_T or having_trait_t to only allow the following
// to be instantiated for integral types I
template <typename I>
constexpr inline I
clear_lower_bits_safe(
//    util::having_trait_t<I, std::is_integral> val,
    I                                           val,
    unsigned                                    num_bits_to_clear)
{
    return (num_bits_to_clear > 0) ?
        clear_lower_bits_unsafe(val, num_bits_to_clear) : val;
}

// TODO: Use the cuda-api-wrappers library instead
inline constexpr auto form_naive_1d_grid(
    int overall_num_elements,
    int threads_per_block,
    int elements_per_thread = 1)
{
    struct one_dimensional_grid_params_t {
        int num_blocks;
        int threads_per_block;
    };
    auto num_blocks = util::div_rounding_up_safe(overall_num_elements, elements_per_thread * threads_per_block);
    return one_dimensional_grid_params_t { num_blocks, threads_per_block };
}


} // namespace util



} // namespace cudf


#ifndef NVCC
#ifndef __host__

#undef __host__
#undef __device__
#undef __forceinline__

#endif
#endif

#endif // UTIL_MISCELLANY_HPP_
