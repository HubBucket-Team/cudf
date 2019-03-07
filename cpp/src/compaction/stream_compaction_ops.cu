/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Felipe Aramburu <felipe@blazingdb.com>
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

#include <cudf.h>
#include <utilities/cudf_utils.h>
#include <utilities/error_utils.hpp>
#include <utilities/miscellany.hpp>

#include <rmm/thrust_rmm_allocator.h>


#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/iterator/counting_iterator.h>

#include <thrust/execution_policy.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/transform_iterator.h>
#include "utilities/error_utils.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <map>

namespace detail {

#if __cplusplus <= 201703L

// TODO: drop these when we switch to C++17

template <typename T>
constexpr T gcd(T m, T n)
{
    // Note: Assuming m and n are positive
    return (n == 0) ? m : gcd(n, m % n);
}

template <typename T>
constexpr T lcm(T m, T n)
{
    // Note: Assuming m and n are positive
    return (m / gcd(m, n)) * n;
}
#else

template <typename T>
using lcm = std::lcm;

#endif


} // namespace detail


gdf_size_type get_number_of_bytes_for_valid (gdf_size_type column_size) {
    static constexpr const auto allocation_quantum {
        detail::lcm(sizeof(uint32_t), sizeof(gdf_valid_type))
    };
    return cudf::util::div_rounding_up_safe<gdf_size_type>(column_size, allocation_quantum);
}


struct shift_left: public thrust::unary_function<gdf_valid_type,gdf_valid_type>
{

	gdf_valid_type num_bits;
	shift_left(gdf_valid_type num_bits): num_bits(num_bits){

	}

  __host__ __device__
  gdf_valid_type operator()(gdf_valid_type x) const
  {
    return x << num_bits;
  }
};

struct shift_right: public thrust::unary_function<gdf_valid_type,gdf_valid_type>
{

	gdf_valid_type num_bits;
	bool not_too_many;
	shift_right(gdf_valid_type num_bits, bool not_too_many)
		: num_bits(num_bits), not_too_many(not_too_many){

	}

  __host__ __device__
  gdf_valid_type operator()(gdf_valid_type x) const
  {
	    //if you want to force the shift to be fill bits with 0 you need to use an unsigned type
	  /*if (not_too_many) { // is the last 
		return  x; 
	  }*/
	  return *((unsigned char *) &x) >> num_bits;

  }
};
 
struct bit_or: public thrust::unary_function<thrust::tuple<gdf_valid_type,gdf_valid_type>,gdf_valid_type>
{
	 

	__host__ __device__
	gdf_valid_type operator()(thrust::tuple<gdf_valid_type,gdf_valid_type> x) const
	{
		return thrust::get<0>(x) | thrust::get<1>(x);
	}
};
 
std::map<gdf_dtype, int16_t> column_type_width = {{GDF_INT8, sizeof(int8_t)}, {GDF_INT16, sizeof(int16_t)},{GDF_INT32, sizeof(int32_t)}, {GDF_INT64, sizeof(int64_t)},
		{GDF_FLOAT32, sizeof(float)}, {GDF_FLOAT64, sizeof(double)} };


size_t  get_last_byte_length(size_t column_size) {
    size_t n_bytes = get_number_of_bytes_for_valid(column_size);
    size_t length = column_size - GDF_VALID_BITSIZE * (n_bytes - 1);
    if (n_bytes == 1 ) {
        length = column_size;
    }
    return  length;
}

size_t  get_right_byte_length(size_t column_size, size_t iter, size_t left_length) {
    size_t n_bytes = get_number_of_bytes_for_valid(column_size);
    size_t length = column_size - GDF_VALID_BITSIZE * (n_bytes - 1);
    if (iter == n_bytes - 1) { // the last one
        if (left_length + length > GDF_VALID_BITSIZE) {
            length = GDF_VALID_BITSIZE - left_length;
        }
    }
    else {
        length = GDF_VALID_BITSIZE - left_length;
    }
    return length;
}
 

 bool last_with_too_many_bits(size_t column_size, size_t iter, size_t left_length) {
    size_t n_bytes = get_number_of_bytes_for_valid(column_size);
    size_t length = column_size - GDF_VALID_BITSIZE * (n_bytes - 1);
    if (iter == n_bytes) { // the last one
        // the last one has to many bits
        if (left_length + length > GDF_VALID_BITSIZE) {
            return true;
        }
    }
    return false;
}


 gdf_valid_type concat_bins (gdf_valid_type A, gdf_valid_type B, int len_a, int len_b, bool has_next, size_t right_length){
    A = A << len_b;
    if (!has_next) {
        B = B << len_a;
        B = B >> len_a;
    } else {
        B = B >> right_length - len_b;
    }
    return  (A | B);
}

gdf_error gdf_concat(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
{
	GDF_REQUIRE( (lhs->dtype == output->dtype ) && ( rhs->dtype == output->dtype), GDF_VALIDITY_MISSING);
	GDF_REQUIRE(output->size == lhs->size + rhs->size, GDF_COLUMN_SIZE_MISMATCH);
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	int type_width = column_type_width[ lhs->dtype ];

	cudaMemcpyAsync(output->data, lhs->data, type_width * lhs->size, cudaMemcpyDeviceToDevice, stream);
	cudaMemcpyAsync( (void *)( (int8_t*) (output->data) + type_width * lhs->size), rhs->data, type_width * rhs->size, cudaMemcpyDeviceToDevice, stream);
	
	int left_num_chars = get_number_of_bytes_for_valid(lhs->size);
	int right_num_chars = get_number_of_bytes_for_valid(rhs->size);
  	int output_num_chars = get_number_of_bytes_for_valid(output->size); 
					
	thrust::device_ptr<gdf_valid_type> left_device_bits = thrust::device_pointer_cast((gdf_valid_type *)lhs->valid);
	thrust::device_ptr<gdf_valid_type> right_device_bits = thrust::device_pointer_cast((gdf_valid_type *)rhs->valid);
	thrust::device_ptr<gdf_valid_type> output_device_bits = thrust::device_pointer_cast((gdf_valid_type *)output->valid);

	thrust::copy(left_device_bits, left_device_bits + left_num_chars, output_device_bits);
	
	gdf_valid_type shift_bits = (GDF_VALID_BITSIZE - (lhs->size % GDF_VALID_BITSIZE));
	if(shift_bits == 8){
		shift_bits = 0;
	}
	if (right_num_chars > 0) {
		size_t prev_len = get_last_byte_length(lhs->size);

		// copy all the rnbytes bytes  from right column
		if (shift_bits == 0) { 
			thrust::copy(right_device_bits, right_device_bits + right_num_chars, output_device_bits + left_num_chars);
		}
		else { 
			thrust::host_vector<gdf_valid_type> last_byte (2);
			thrust::copy (left_device_bits + left_num_chars - 1, left_device_bits + left_num_chars, last_byte.begin());
			thrust::copy (right_device_bits, right_device_bits + 1, last_byte.begin() + 1);
			        
			size_t curr_len = get_right_byte_length(rhs->size, 0, prev_len);

			if (1 != right_num_chars) {
				last_byte[1] = last_byte[1] >> prev_len;
			}
			auto flag = last_with_too_many_bits(rhs->size, 0 + 1, prev_len);
			size_t last_right_byte_length = rhs->size - GDF_VALID_BITSIZE * (right_num_chars - 1);
			last_byte[0] = concat_bins(last_byte[0], last_byte[1], prev_len, curr_len, flag, last_right_byte_length);

			thrust::copy( last_byte.begin(), last_byte.begin() + 1, output_device_bits + left_num_chars - 1);
			
			if(right_num_chars > 1)  {
				using first_iterator_type = thrust::transform_iterator<shift_left,rmm::device_vector<gdf_valid_type>::iterator>;
				using second_iterator_type = thrust::transform_iterator<shift_right,rmm::device_vector<gdf_valid_type>::iterator>;
				using offset_tuple = thrust::tuple<first_iterator_type, second_iterator_type>;
				using zipped_offset = thrust::zip_iterator<offset_tuple>;

				auto too_many_bits = last_with_too_many_bits(rhs->size, right_num_chars, prev_len);
				size_t last_byte_length = get_last_byte_length(rhs->size);

				if (last_byte_length >= (GDF_VALID_BITSIZE - shift_bits)) { //  
					thrust::host_vector<gdf_valid_type> last_byte (right_device_bits + right_num_chars - 1, right_device_bits + right_num_chars);
					last_byte[0] = last_byte[0] << GDF_VALID_BITSIZE - last_byte_length;
					thrust::copy( last_byte.begin(), last_byte.begin() + 1, right_device_bits + right_num_chars - 1);
				}
				
				zipped_offset  zipped_offset_iter(
						thrust::make_tuple(
								thrust::make_transform_iterator<shift_left, rmm::device_vector<gdf_valid_type>::iterator >(
										right_device_bits,
										shift_left(shift_bits)),
								
								thrust::make_transform_iterator<shift_right, rmm::device_vector<gdf_valid_type>::iterator >(
										right_device_bits + 1,
										shift_right(GDF_VALID_BITSIZE - shift_bits, !too_many_bits))
						)	
				);
				//so what this does is give you an iterator which gives you a tuple where you have your char, and the char after you, so you can get the last bits!
				using transformed_or = thrust::transform_iterator<bit_or, zipped_offset>;
				//now we want to make a transform iterator that ands these values together
				transformed_or ored_offset_iter =
						thrust::make_transform_iterator<bit_or,zipped_offset> (
								zipped_offset_iter,
								bit_or()
						);
				//because one of the iterators is + 1 we dont want to read the last char here since it could be past the end of our allocation
				thrust::copy( ored_offset_iter, ored_offset_iter + right_num_chars - 1, output_device_bits + left_num_chars);

				thrust::host_vector<gdf_valid_type> last_byte (right_device_bits + right_num_chars - 1, right_device_bits + right_num_chars);
				last_byte[0] = last_byte[0] >> GDF_VALID_BITSIZE - last_byte_length;
				thrust::copy( last_byte.begin(), last_byte.begin() + 1, right_device_bits + right_num_chars - 1);

				if ( !too_many_bits ) {
					thrust::host_vector<gdf_valid_type> last_byte (2);
					thrust::copy (right_device_bits + right_num_chars - 2, right_device_bits + right_num_chars - 1, last_byte.begin());
					thrust::copy (right_device_bits + right_num_chars - 1, right_device_bits + right_num_chars, last_byte.begin() + 1);
					last_byte[0] = last_byte[0] << last_byte_length | last_byte[1];
					thrust::copy( last_byte.begin(), last_byte.begin() + 1, output_device_bits + output_num_chars - 1);
				} 
			}
		}
		if( last_with_too_many_bits(rhs->size, right_num_chars, prev_len)){
			thrust::host_vector<gdf_valid_type> last_byte (right_device_bits + right_num_chars - 1, right_device_bits + right_num_chars);
			size_t prev_len = get_last_byte_length(lhs->size);
			size_t curr_len = get_right_byte_length(rhs->size, right_num_chars - 1,  prev_len);
			last_byte[0] = last_byte[0] << curr_len;
			last_byte[0] = last_byte[0] >> curr_len;
			thrust::copy( last_byte.begin(), last_byte.begin() + 1, output_device_bits + output_num_chars - 1);
		}
	}

	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);
	return GDF_SUCCESS;
}


