/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#ifndef GROUPBY_SORT_COMPUTE_API_H
#define GROUPBY_SORT_COMPUTE_API_H

#include <cuda_runtime.h>
#include <limits>
#include <memory>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/copy.h>

#include "rmm/thrust_rmm_allocator.h"
#include "dataframe/cudf_table.cuh"


using IndexT = int;
template <typename T>
using Vector = thrust::device_vector<T, rmm_allocator<T>>;

/* --------------------------------------------------------------------------*/
/** 
* @Synopsis Performs the groupby operation for an arbtirary number of groupby columns and
* and a single aggregation column.
* 
* @Param[in] groupby_input_table The set of columns to groupby
* @Param[in] in_aggregation_column The column to perform the aggregation on. These act as the hash table values
* @Param[out] groupby_output_table Preallocated buffer(s) for the groupby column result. This will hold a single
* entry for every unique row in the input table.
* @Param[out] out_aggregation_column Preallocated output buffer for the resultant aggregation column that 
*                                     corresponds to the out_groupby_column where entry 'i' is the aggregation 
*                                     for the group out_groupby_column[i] 
* @Param out_size The size of the output
* @Param aggregation_op The aggregation operation to perform 
* @Param sort_result Flag to optionally sort the output table
* 
* @Returns   
*/
/* ----------------------------------------------------------------------------*/
template< typename aggregation_type,
          typename size_type,
          typename aggregation_operation>
gdf_error GroupbySort(size_type num_groupby_cols,
                        int32_t* d_sorted_indices,
                        gdf_column* in_groupby_columns[],       
                        const aggregation_type * const in_aggregation_column,
                        gdf_column* out_groupby_columns[],
                        aggregation_type * out_aggregation_column,
                        size_type * out_size,
                        aggregation_operation aggregation_op,
                        gdf_context* ctxt)
{
  int32_t nrows = in_groupby_columns[0]->size;
  Vector<void*> d_cols(num_groupby_cols);    
  Vector<int> d_types(num_groupby_cols, 0); 
  void** d_col_data = d_cols.data().get();
  int* d_col_types = d_types.data().get();
  bool nulls_are_smallest = ctxt->flag_nulls_sort_behavior == 1;
  soa_col_info(in_groupby_columns, num_groupby_cols, d_col_data, nullptr, d_col_types);
  LesserRTTI<int32_t> f(d_col_data, nullptr, d_col_types, nullptr, num_groupby_cols, nulls_are_smallest);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  auto exec = rmm::exec_policy(stream)->on(stream);

  auto agg_col_iter = thrust::make_permutation_iterator(in_aggregation_column, d_sorted_indices);

  auto ret =
        thrust::reduce_by_key(exec,
                              d_sorted_indices, d_sorted_indices+nrows, 
                              agg_col_iter,  
                              d_sorted_indices, 
                              out_aggregation_column,  
                              [f] __device__(IndexT key1, IndexT key2) {
                                  return f.equal(key1, key2);
                              },
                              aggregation_op);
  size_type new_size = thrust::distance(out_aggregation_column, ret.second);
  *out_size = new_size;

  // run gather operation to establish new order
  std::unique_ptr< gdf_table<int32_t> > table_in{new gdf_table<int32_t>{num_groupby_cols, in_groupby_columns}};
  std::unique_ptr< gdf_table<int32_t> > table_out{new gdf_table<int32_t>{num_groupby_cols, out_groupby_columns}};
  auto status = table_in->gather<int32_t>(d_sorted_indices, *table_out.get());
  if (status != GDF_SUCCESS)
      return status;

	for (int i = 0; i < num_groupby_cols; i++) {
		out_groupby_columns[i]->size = new_size;
	}

  return GDF_SUCCESS;
}

template< typename aggregation_type,
          typename size_type,
          typename aggregation_operation>
gdf_error GroupbySortFirstOp(size_type num_groupby_cols,
                        int32_t* d_sorted_indices,
                        gdf_column* in_groupby_columns[],       
                        const aggregation_type * const in_aggregation_column,
                        gdf_column* out_groupby_columns[],
                        aggregation_type * out_aggregation_column,
                        size_type * out_size,
                        aggregation_operation aggregation_op,
                        gdf_context* ctxt)
{
  int32_t nrows = in_groupby_columns[0]->size;
  Vector<void*> d_cols(num_groupby_cols);    
  Vector<int> d_types(num_groupby_cols, 0); 
  void** d_col_data = d_cols.data().get();
  int* d_col_types = d_types.data().get();
  bool nulls_are_smallest = ctxt->flag_nulls_sort_behavior == 1;
  soa_col_info(in_groupby_columns, num_groupby_cols, d_col_data, nullptr, d_col_types);
  LesserRTTI<int32_t> f(d_col_data, nullptr, d_col_types, nullptr, num_groupby_cols, nulls_are_smallest);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  auto exec = rmm::exec_policy(stream)->on(stream);

  auto agg_col_iter = thrust::make_permutation_iterator(in_aggregation_column, d_sorted_indices);

  auto ret =
        thrust::unique_by_key_copy(exec,
                                    d_sorted_indices, d_sorted_indices+nrows, 
                                    agg_col_iter,  
                                    d_sorted_indices, 
                                    out_aggregation_column,  
                                    [f] __device__(IndexT key1, IndexT key2) {
                                        return f.equal(key1, key2);
                                    });
  size_type new_size = thrust::distance(out_aggregation_column, ret.second);
  *out_size = new_size;

  // run gather operation to establish new order
  std::unique_ptr< gdf_table<int32_t> > table_in{new gdf_table<int32_t>{num_groupby_cols, in_groupby_columns}};
  std::unique_ptr< gdf_table<int32_t> > table_out{new gdf_table<int32_t>{num_groupby_cols, out_groupby_columns}};
  auto status = table_in->gather<int32_t>(d_sorted_indices, *table_out.get());
  if (status != GDF_SUCCESS)
      return status;

	for (int i = 0; i < num_groupby_cols; i++) {
		out_groupby_columns[i]->size = new_size;
	}

  return GDF_SUCCESS;
}


template< typename aggregation_type,
          typename size_type,
          typename aggregation_operation>
gdf_error GroupbySortCountDistinct(size_type num_groupby_cols,
                        int32_t* d_sorted_indices,
                        gdf_column* in_groupby_columns[],
                        gdf_column* in_groupby_columns_with_agg[],       
                        const aggregation_type * const in_aggregation_column,
                        gdf_column* out_groupby_columns[],
                        aggregation_type * out_aggregation_column,
                        size_type * out_size,
                        aggregation_operation aggregation_op,
                        gdf_context* ctxt)
{
  int32_t nrows = in_groupby_columns[0]->size;
  Vector<void*> d_cols(num_groupby_cols + 1);    
  Vector<int> d_types(num_groupby_cols + 1, 0); 
  void** d_col_data = d_cols.data().get();
  int* d_col_types = d_types.data().get();
  bool nulls_are_smallest = ctxt->flag_nulls_sort_behavior == 1;

  soa_col_info(in_groupby_columns_with_agg, num_groupby_cols + 1, d_col_data, nullptr, d_col_types);
  LesserRTTI<int32_t> f(d_col_data, nullptr, d_col_types, nullptr, num_groupby_cols + 1, nulls_are_smallest);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  auto exec = rmm::exec_policy(stream)->on(stream);

  auto ret =
        thrust::reduce_by_key(exec,
                              d_sorted_indices, d_sorted_indices+nrows, 
                              thrust::make_constant_iterator<int32_t>(1),  
                              d_sorted_indices, 
                              out_aggregation_column,  
                              [f] __device__(IndexT key1, IndexT key2) {
                                  return f.equal(key1, key2);
                              }
                              );
  size_type new_size = thrust::distance(out_aggregation_column, ret.second);
  *out_size = new_size;
 
  std::unique_ptr< gdf_table<int32_t> > table_in{new gdf_table<int32_t>{num_groupby_cols, in_groupby_columns}};
  std::unique_ptr< gdf_table<int32_t> > table_out{new gdf_table<int32_t>{num_groupby_cols, out_groupby_columns}};
  auto status = table_in->gather<int32_t>(d_sorted_indices, *table_out.get());
  if (status != GDF_SUCCESS)
      return status;

	for (int i = 0; i < num_groupby_cols; i++) {
		out_groupby_columns[i]->size = new_size;
	}

  return GDF_SUCCESS;
}

#endif
