#ifndef SAMPLE_GENERATOR_HPP
#define SAMPLE_GENERATOR_HPP

namespace cudf {
    
// Forward declaration
struct table;

namespace detail {

gdf_error gather_random_samples(table const* source_table,
                          table* sampled_data,
                          gdf_size_type num_samples,
                          gdf_index_type * temp_random_indeces = nullptr);

}  // namespace detail
} // namespace cudf

#endif
