#include <cassert>
#include "cudf.h"


#include <iostream>
#include <cassert>
#include <iterator>

#include <thrust/tuple.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/unique.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/distance.h>
#include <thrust/advance.h>
#include <thrust/gather.h>

#include "new_groupby.hpp"

#include "rmm/thrust_rmm_allocator.h"

#include "utilities/nvtx/nvtx_utils.h"
#include "utilities/error_utils.h"
#include "aggregation_operations.hpp"
 

namespace without_agg{
   

void soa_col_info(gdf_column* cols, size_t ncols, void** d_cols, int* d_types)
{
  std::vector<void*> v_cols(ncols,nullptr);
  std::vector<int>   v_types(ncols, 0);
  for(size_t i=0;i<ncols;++i)
    {
      v_cols[i] = cols[i].data;
      v_types[i] = cols[i].dtype;
    }

  void** h_cols = &v_cols[0];
  int* h_types = &v_types[0];
  cudaMemcpy(d_cols, h_cols, ncols*sizeof(void*), cudaMemcpyHostToDevice);//TODO: add streams
  cudaMemcpy(d_types, h_types, ncols*sizeof(int), cudaMemcpyHostToDevice);//TODO: add streams
}



//###########################################################################
//#                          Multi-column ORDER-BY:                         #
//###########################################################################
//Version with array of columns,
//using type erasure and RTTI at
//comparison operator level;
//
//args:
//Input:
// nrows    = # rows;
// ncols    = # columns;
// d_cols   = device array to ncols type erased columns;
// d_gdf_t  = device array to runtime column types;
// stream   = cudaStream to work in;
//
//Output:
// d_indx   = vector of indices re-ordered after sorting;
//
template<typename IndexT>
void multi_col_order_by(size_t nrows,
			size_t ncols,
			void* const* d_cols,
			int* const  d_gdf_t,
			IndexT*      d_indx,
			cudaStream_t stream = NULL)
{
  LesserRTTI<IndexT> f(d_cols, d_gdf_t, ncols);

  rmm_temp_allocator allocator(stream);
  thrust::sequence(thrust::cuda::par(allocator).on(stream), d_indx, d_indx+nrows, 0);//cannot use counting_iterator
  //                                          2 reasons:
  //(1.) need to return a container result;
  //(2.) that container must be mutable;
  
  thrust::sort(thrust::cuda::par(allocator).on(stream),
               d_indx, d_indx+nrows,
               [f] __device__ (IndexT i1, IndexT i2) {
                 return f.less(i1, i2);
               });
}


template<typename ValsT,
         typename IndexT,
         typename Reducer>
size_t multi_col_group_by_sort(size_t         nrows,
                               size_t         ncols,
                               void* const*   d_cols,
                               int* const     d_gdf_t,
                               const ValsT*   ptr_d_agg,
                               Reducer        fctr,
                               IndexT*        ptr_d_indx,
                               ValsT*         ptr_d_agg_p,
                               IndexT*        ptr_d_kout,
                               ValsT*         ptr_d_vout,
                               bool           sorted = false,
                               cudaStream_t   stream = NULL)
{
  if( !sorted )
    multi_col_order_by(nrows, ncols, d_cols, d_gdf_t, ptr_d_indx, stream);

  rmm_temp_allocator allocator(stream);
  
  thrust::gather(thrust::cuda::par(allocator).on(stream),
                 ptr_d_indx, ptr_d_indx + nrows,  //map[i]
  		 ptr_d_agg,                    //source[i]
  		 ptr_d_agg_p);                 //source[map[i]]

  LesserRTTI<IndexT> f(d_cols, d_gdf_t, ncols);
  
  thrust::pair<IndexT*, ValsT*> ret =
    thrust::reduce_by_key(thrust::cuda::par(allocator).on(stream),
                          ptr_d_indx, ptr_d_indx + nrows,
                          ptr_d_agg_p,
                          ptr_d_kout,
                          ptr_d_vout,
                          [f] __device__(IndexT key1, IndexT key2) {
                              return f.equal(key1, key2);
                          },
                          fctr);

  size_t new_sz = thrust::distance(ptr_d_vout, ret.second);
  return new_sz;
}


template<typename ValsT,
	       typename IndexT>
size_t multi_col_group_by_sum_sort(size_t         nrows,
                                   size_t         ncols,
                                   void* const*   d_cols,
                                   int* const     d_gdf_t,
                                   const ValsT*   ptr_d_agg,
                                   IndexT*        ptr_d_indx,
                                   ValsT*         ptr_d_agg_p,
                                   IndexT*        ptr_d_kout,
                                   ValsT*         ptr_d_vout,
                                   bool           sorted = false,
                                   cudaStream_t   stream = NULL)
{
  auto lamb = [] __device__ (ValsT x, ValsT y) {
		return x+y;
  };

  using ReducerT = decltype(lamb);

  return multi_col_group_by_sort(nrows,
                                 ncols,
                                 d_cols,
                                 d_gdf_t,
                                 ptr_d_agg,
                                 lamb,
                                 ptr_d_indx,
                                 ptr_d_agg_p,
                                 ptr_d_kout,
                                 ptr_d_vout,
                                 sorted,
                                 stream);
}



//apparent duplication of info between
//gdf_column array and two arrays:
//           d_cols = data slice of gdf_column array;
//           d_types = dtype slice of gdf_column array;
//but it's necessary because the gdf_column array is host
//(even though its data slice is on device)
//
gdf_error gdf_group_by_sum(size_t nrows,     //in: # rows
                           gdf_column* cols, //in: host-side array of gdf_columns
                           size_t ncols,     //in: # cols
                           int flag_sorted,  //in: flag specififying if rows are pre-sorted (1) or not (0)
                           gdf_column& agg_in,//in: column to aggregate
                           void** d_cols,    //out: pre-allocated device-side array to be filled with gdf_column::data for each column; slicing of gdf_column array (host)
                           int* d_types,     //out: pre-allocated device-side array to be filled with gdf_colum::dtype for each column; slicing of gdf_column array (host)
                           IndexT* d_indx,      //out: device-side array of row indices after sorting
                           gdf_column& agg_p, //out: reordering of d_agg after sorting; requires shallow (trivial) copy-construction (see static_assert below);
                           IndexT* d_kout,      //out: device-side array of rows after group-by
                           gdf_column& c_vout,//out: aggregated column; requires shallow (trivial) copy-construction (see static_assert below);
                           size_t* new_sz)   //out: host-side # rows of d_count
{
  //not supported by g++-4.8:
  //
  //static_assert(std::is_trivially_copy_constructible<gdf_column>::value,
  //		"error: gdf_column must have shallow copy constructor; otherwise cannot pass output by copy.");

#ifdef DEBUG_
  run_echo(nrows,     //in: # rows
           cols, //in: host-side array of gdf_columns
           ncols,     //in: # cols
           flag_sorted,  //in: flag specififying if rows are pre-sorted (1) or not (0)
           agg_in);//in: column to aggregate
#endif

  assert( agg_in.dtype == agg_p.dtype );
  assert( agg_in.dtype == c_vout.dtype );
  
  //copy H-D:
  //
  soa_col_info(cols, ncols, d_cols, d_types);

  // use type dispatcher 

    // switch( agg_in.dtype )
    // {
    // case GDF_INT8:
    //   {
        // using T = char;

        // T* d_agg   = static_cast<T*>(agg_in.data);
        // T* d_agg_p = static_cast<T*>(agg_p.data);
        // T* d_vout  = static_cast<T*>(c_vout.data);
        // *new_sz = multi_col_group_by_sum_sort(nrows,
        //                                       ncols,
        //                                       d_cols,
        //                                       d_types,
        //                                       d_agg,
        //                                       d_indx,
        //                                       d_agg_p,
        //                                       d_kout,
        //                                       d_vout,
        //                                       flag_sorted);
	
  return GDF_SUCCESS;
}


gdf_error gdf_group_by_single(int ncols,                    // # columns
                              gdf_column** cols,            //input cols
                              gdf_column* col_agg,          //column to aggregate on
                              gdf_column* out_col_indices,  //if not null return indices of re-ordered rows
                              gdf_column** out_col_values,  //if not null return the grouped-by columns
                                                            //(multi-gather based on indices, which are needed anyway)
                              gdf_column* out_col_agg,      //aggregation result
                              gdf_context* ctxt,            //struct with additional info: bool is_sorted, flag_sort_or_hash, bool flag_count_distinct
                              gdf_agg_op op)                //aggregation operation
{
  CUDA_TRY(cudaDeviceSynchronize());
  
  if((0 == ncols)
     || (nullptr == cols)
     || (nullptr == col_agg)
     || (nullptr == out_col_agg)
     || (nullptr == ctxt))
  {
    return GDF_DATASET_EMPTY;
  }
  for (int i = 0; i < ncols; ++i) {
  	GDF_REQUIRE(!cols[i]->valid || !cols[i]->null_count, GDF_VALIDITY_UNSUPPORTED);
  }
  GDF_REQUIRE(!col_agg->valid || !col_agg->null_count, GDF_VALIDITY_UNSUPPORTED);

  // If there are no rows in the input, set the output rows to 0 
  // and return immediately with success
  if( (0 == cols[0]->size )
      || (0 == col_agg->size))
  {
    if( (nullptr != out_col_agg) ){
      out_col_agg->size = 0;
    }
    if(nullptr != out_col_indices ) {
        out_col_indices->size = 0;
    }

    for(int col = 0; col < ncols; ++col){
      if(nullptr != out_col_values){
        if( nullptr != out_col_values[col] ){
          out_col_values[col]->size = 0;
        }
      }
    }
    return GDF_SUCCESS;
  }

  gdf_error gdf_error_code{GDF_SUCCESS};
  
  PUSH_RANGE("LIBGDF_GROUPBY", GROUPBY_COLOR);
  
  if( ctxt->flag_method == GDF_SORT )
    {
      std::vector<gdf_column> v_cols(ncols);
      for(auto i = 0; i < ncols; ++i)
        {
          v_cols[i] = *(cols[i]);
        }
      
      gdf_column* h_columns = &v_cols[0];
      size_t nrows = h_columns[0].size;

      size_t n_group = 0;

      Vector<IndexT> d_indx;//allocate only if necessary (see below)
      Vector<void*> d_cols(ncols, nullptr);
      Vector<int>   d_types(ncols, 0);
  
      void** d_col_data = d_cols.data().get();
      int* d_col_types = d_types.data().get();

      IndexT* ptr_d_indx = nullptr;
      if( out_col_indices )
        ptr_d_indx = static_cast<IndexT*>(out_col_indices->data);
      else
        {
          d_indx.resize(nrows);
          ptr_d_indx = d_indx.data().get();
        }

      Vector<IndexT> d_sort(nrows, 0);
      IndexT* ptr_d_sort = d_sort.data().get();
      
      gdf_column c_agg_p;
      c_agg_p.dtype = col_agg->dtype;
      c_agg_p.size = nrows;
      Vector<char> d_agg_p(nrows * dtype_size(c_agg_p.dtype));//purpose: avoids a switch-case on type;
      c_agg_p.data = d_agg_p.data().get();

      switch( op )
        {
        case GDF_SUM:
          gdf_group_by_sum(nrows,
                           h_columns,
                           static_cast<size_t>(ncols),
                           ctxt->flag_sorted,
                           *col_agg,
                           d_col_data, //allocated
                           d_col_types,//allocated
                           ptr_d_sort, //allocated
                           c_agg_p,    //allocated
                           ptr_d_indx, //allocated (or, passed in)
                           *out_col_agg,
                           &n_group);
          break;
          
        // case GDF_MIN:
        //   gdf_group_by_min(nrows,
        //                    h_columns,
        //                    static_cast<size_t>(ncols),
        //                    ctxt->flag_sorted,
        //                    *col_agg,
        //                    d_col_data, //allocated
        //                    d_col_types,//allocated
        //                    ptr_d_sort, //allocated
        //                    c_agg_p,    //allocated
        //                    ptr_d_indx, //allocated (or, passed in)
        //                    *out_col_agg,
        //                    &n_group);
        //   break;

        // case GDF_MAX:
        //   gdf_group_by_max(nrows,
        //                    h_columns,
        //                    static_cast<size_t>(ncols),
        //                    ctxt->flag_sorted,
        //                    *col_agg,
        //                    d_col_data, //allocated
        //                    d_col_types,//allocated
        //                    ptr_d_sort, //allocated
        //                    c_agg_p,    //allocated
        //                    ptr_d_indx, //allocated (or, passed in)
        //                    *out_col_agg,
        //                    &n_group);
        //   break;

        // case GDF_AVG:
        //   {
        //     Vector<IndexT> d_cout(nrows, 0);
        //     IndexT* ptr_d_cout = d_cout.data().get();
            
        //     gdf_group_by_avg(nrows,
        //                      h_columns,
        //                      static_cast<size_t>(ncols),
        //                      ctxt->flag_sorted,
        //                      *col_agg,
        //                      d_col_data, //allocated
        //                      d_col_types,//allocated
        //                      ptr_d_sort, //allocated
        //                      ptr_d_cout, //allocated
        //                      c_agg_p,    //allocated
        //                      ptr_d_indx, //allocated (or, passed in)
        //                      *out_col_agg,
        //                      &n_group);
        //   }
        //   break;
        // case GDF_COUNT_DISTINCT:
        //   {
        //     assert( out_col_agg );
        //     assert( out_col_agg->size >= 1);

        //     gdf_group_by_count(nrows,
        //                        h_columns,
        //                        static_cast<size_t>(ncols),
        //                        ctxt->flag_sorted,
        //                        d_col_data, //allocated
        //                        d_col_types,//allocated
        //                        ptr_d_sort, //allocated
        //                        ptr_d_indx, //allocated (or, passed in)
        //                        *out_col_agg, //passed in
        //                        &n_group,
        //                        true);
            
        //   }
        //   break;
        // case GDF_COUNT:
        //   {
        //     assert( out_col_agg );

        //     gdf_group_by_count(nrows,
        //                        h_columns,
        //                        static_cast<size_t>(ncols),
        //                        ctxt->flag_sorted,
        //                        d_col_data, //allocated
        //                        d_col_types,//allocated
        //                        ptr_d_sort, //allocated
        //                        ptr_d_indx, //allocated (or, passed in)
        //                        *out_col_agg, //passed in
        //                        &n_group);
            
        //   }
        //   break;
        default: // To eliminate error for unhandled enumerant N_GDF_AGG_OPS
          gdf_error_code = GDF_INVALID_API_CALL;
        }

      if( out_col_values )
        {
          // todo: what the hell is this
          // multi_gather_host(ncols, cols, out_col_values, ptr_d_indx, n_group);
        }

      out_col_agg->size = n_group;
      if( out_col_indices )
        out_col_indices->size = n_group;

      //TODO: out_<col>->valid = ?????
    }
   
  POP_RANGE();
  
  return gdf_error_code;
} 


gdf_error gdf_group_by_sum(int ncols,                    // # columns
                           gdf_column** cols,            //input cols
                           gdf_column* col_agg,          //column to aggregate on
                           gdf_column* out_col_indices,  //if not null return indices of re-ordered rows
                           gdf_column** out_col_values,  //if not null return the grouped-by columns
                                                         //(multi-gather based on indices, which are needed anyway)
                           gdf_column* out_col_agg,      //aggregation result
                           gdf_context* ctxt)            //struct with additional info: bool is_sorted, flag_sort_or_hash, bool flag_count_distinct
{  
  return gdf_group_by_single(ncols, cols, col_agg, out_col_indices, out_col_values, out_col_agg, ctxt, GDF_SUM);
}


} 
/* WSM NOTE: For pandas like group by version. Do a less than operator where any null returns false. 
Then to calculate where the nulls start to ignore them, OR all the valids then do a null count*/
