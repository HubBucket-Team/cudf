/**
 * This code in this this file is (C) Eyal Rozenberg and CWI Amsterdam
 * under terms of the BSD 3-Clause license. See:
 * https://github.com/eyalroz/libgiddy/blob/master/LICENSE
 *
 * Talk to Eyal <eyalroz@blazingdb.com> about sublicensing/relicensing
 */


#ifndef DEVICE_SIDE_UTILITIES_CUH_
#define DEVICE_SIDE_UTILITIES_CUH_

#include <device_functions.h>
#include <cuda_runtime.h>
#include "cuda_builtins.cuh"

#include <type_traits>

#define __fd__ __forceinline__ __device__

namespace lane {

__fd__ unsigned index() { return threadIdx.x % warp_size; }
__fd__ unsigned is_set_in_mask(lane_mask_t mask) { return  mask & (1 << lane::index()); }

} // namespace lane

namespace warp {

__fd__ unsigned index() { return threadIdx.x / warp_size; }
__fd__ unsigned global_index() { return (threadIdx.x + blockIdx.x * blockDim.x ) / warp_size; }

} // namespace warp

namespace block {

__fd__ bool is_first_in_grid() { return blockIdx.x == 0; }
__fd__ bool is_last_in_grid() { return blockIdx.x == gridDim.x - 1; }

} // namespace warp


template <typename Function, typename Size = std::size_t>
__fd__ void at_warp_stride(Size length, const Function& f)
{
    for(// _not_ the global thread index! - one element per warp
        Size pos = ::lane::index();
        pos < length;
        pos += warp_size)
    {
        f(pos);
    }
}

template <typename T, typename S>
__fd__ constexpr typename std::common_type<T,S>::type
round_down_to_power_of_2(const T& x, const S& power_of_2)
{
    using result_type = typename std::common_type<T,S>::type;
    return ((result_type) x) & ~(((result_type) power_of_2) - 1);
}

template <typename T, typename S>
__fd__ constexpr typename std::common_type<T,S>::type
round_up_to_power_of_2(T x, S power_of_2) {
    using result_type = typename std::common_type<T,S>::type;
    return round_down_to_power_of_2 ((result_type) x + (result_type) power_of_2 - 1, (result_type) power_of_2);
}

template <typename T>
__fd__ constexpr T round_up_to_full_warps(const T& x) {
    return round_up_to_power_of_2<T, unsigned>(x, warp_size);
}


template <typename Function, typename Size>
__fd__ void at_block_stride(
    const Function& f, Size length, unsigned serialization_factor)
{
    Size pos = threadIdx.x + serialization_factor * blockIdx.x * blockDim.x;
    auto threads_per_block = blockDim.x;
    if (pos + threads_per_block * (serialization_factor - 1) < length) {
        // The usual case, which occurs for most blocks in the grid
        #pragma unroll
        for(unsigned i = 0; i < serialization_factor; i++) {
            f(pos);
            pos += threads_per_block;
        }
    }
    else {
        // We're at the blocks at the end of the grid. In this case, we know we'll
        // be stopped not by getting to the serialization_factor+1'th iteration,
        // but by getting to the end of the range on which we work
        #pragma unroll
        for(; pos < length; pos += threads_per_block) { f(pos); }
    }
}


template <typename T>
__fd__ unsigned log2_of_power_of_2(T p)
{
    std::make_unsigned_t<std::remove_cv_t<T>> mask = p - 1; // Remember: 0 is _not_ a power of 2
    return  builtins::population_count(mask);
}

template <typename T, typename S>
__fd__ T div_by_power_of_2(const T dividend, const S divisor)
{
    return dividend >> log2_of_power_of_2(divisor);
}

/**
 * Why do you need this function, you ask? Wouldn't the compiler
 * just optimize the division for you if it's a power of 2? No! If
 * you know it's going to be a power of 2 but the type is merely I,
 * and the compiler can't know in advance what's it's going to run
 * with - it has to be conservative and use regular division
 *
 */
template <typename T, typename S>
__fd__ T
div_by_power_of_2_rounding_up(const T dividend, const S divisor)
{
    std::make_unsigned_t<std::remove_cv_t<S>> mask = divisor - 1; // Remember: 0 is _not_ a power of 2
    auto log_2_of_divisor = builtins::population_count(mask);
    auto correction_for_rounding_up = ((dividend & mask) + mask) >> log_2_of_divisor;
        // Will be either 0 or 1

    return (dividend >> log_2_of_divisor) + correction_for_rounding_up;
}


namespace warp {

__fd__ unsigned num_preceding_satisfying_lanes(int cond)
{
    lane_mask_t previous_lanes_mask { (lane_mask_t{1} << lane::index()) - 1};
    return builtins::population_count(
#if (__CUDACC_VER_MAJOR__ < 9)
        builtins::warp_ballot(cond) & previous_lanes_mask)
#else
        builtins::warp_ballot(previous_lanes_mask, cond)
#endif
    ) - !!cond;
}



template <typename Function>
__fd__ void execute_by_single_lane(
    Function f, unsigned designated_computing_lane = 0)
{
    if (lane::index() == designated_computing_lane) { f(); }
}

// Note: This ignores any coalescing concerns, and may thus be quite sub-optimal
template <typename T, typename Size>
__fd__ void naive_copy(
    T*        __restrict__  target,
    const T*  __restrict__  source,
    Size                    length)
{
    #pragma unroll
    for(Size pos = lane::index(); pos < length; pos += warp_size)
    {
        target[pos] = source[pos];
    }
}

template <typename T, typename Size>
__fd__ void fill_n(T* __restrict__ target, Size count, const T& __restrict__ value)
{
    T copy_of_value { value };
        // Just making sure the compiler doesn't get any weird ideas and optimizes
        // things the way we want them
    for(Size index = lane::index(); index < count; index += warp_size)
    {
        target[index] = copy_of_value;
    }
}

/**
 * @brief Packs a warp size's worth of bits into a single native word (which
 * is possible since a warp has 32 lanes and the native word in cuda has 32
 * bits).
 *
 * @todo Check if NVCC is able to optimize the conversion to bool away; if it
 * is, replace the `int` here with a `bool`
 *
 * @param[in] bit a value to pack. Typically 0 or 1, but any non-zero will be
 * interpreted as an 'on' bit
 * @return
 */
__fd__ unsigned pack_bit_lane_bits(int bit) { return builtins::warp_ballot(bit); }


} // namespace warp

namespace block {

template <typename Function>
__fd__ void execute_by_single_thread(
    Function f, unsigned designated_computing_thread = 0)
{
    if (threadIdx.x == designated_computing_thread) { f(); }
}

} // namespace block

#undef __fd__

#endif // DEVICE_SIDE_UTILITIES_CUH_
