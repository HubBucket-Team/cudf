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




#ifdef NVCC
#define __fhd__ __forceinline__ __host__ __device__
#define __hd__  __host__ __device__
#else
#define __fhd__ inline
#define __hd__
#endif

namespace cudf {

namespace util {

/**
 * @brief A simplistic, unsafe, index range, without all of the standard library's
 * container frills (for now).
 *
 * @note Boost has an irange class, but let's not go there. C++ itself might get it
 * in 2020, but we don't support that.
 */
template <typename I>
struct index_range {
    I start;  // first element which _may_ be the one we're after (i.e.
                          // the first greater or the first greater-or-equal); before
                          // this one, all elements are lower / lower-or-equal than our pivot

    I end;    // first element after which _cannot_ _impossible_ element index

    constexpr __fhd__ I     middle()      const { return start + (end - start) / 2; }
    constexpr __fhd__ I     last()        const { return end - 1; }
    constexpr __fhd__ I     length()      const { return end - start; } // TODO: What if end < start?
    constexpr __fhd__ bool  is_empty()    const { return length() <= 0; }
    constexpr __fhd__ bool  is_singular() const { return length() == 1; }

    constexpr __fhd__ void  drop_lower_half() { start = middle(); }
    constexpr __fhd__ void  drop_upper_half() { end   = middle(); }

    static index_range constexpr __fhd__ trivial_empty() { return { 0, 0 }; }
    static index_range constexpr __fhd__ singular(I index) { return { index, index+1 }; }
};

template <typename I>
constexpr __fhd__ index_range<I> lower_half(index_range<I> range)
{
    return { range.start, range.middle() };
}
template <typename I>

constexpr __fhd__ index_range<I> upper_half(index_range<I> range)
{
    return { range.middle(), range.end };
}

template <typename I>
constexpr __fhd__ index_range<I> strict_upper_half(index_range<I> range)
{
    return { range.middle() + 1, range.end };
}

template <typename I>
constexpr __fhd__ index_range<I> intersection(index_range<I> lhs, index_range<I> rhs)
{
    const auto& starts_first  { (lhs.start < rhs.start) ? lhs : rhs };
    const auto& starts_second { (lhs.start < rhs.start) ? rhs : lhs };

    if (starts_first.end <= starts_second.start) {
        return index_range<I>::trivial_empty();
    }
    return { starts_second.start, starts_first.end };
}

} // namespace util
} // namespace cudf



#undef __fhd__
#undef __fd__

