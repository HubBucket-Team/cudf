#include "random_generator.cuh"

namespace cudf {
namespace detail {

gdf_error gather_random_samples(table const* source_table,
                          table* sampled_data,
                          gdf_size_type num_samples,
                          gdf_index_type * temp_random_indices) {

    GDF_REQUIRE(source_table->num_columns() > 0), GDF_DATASET_EMPTY);
    GDF_REQUIRE(source_table->num_columns() == sampled_data->num_columns()), GDF_COLUMN_SIZE_MISMATCH);
    GDF_REQUIRE(sampled_data->num_columns(0)->size >= num_samples), GDF_COLUMN_SIZE_MISMATCH);
        
    if (num_samples <= 0) {
        return GDF_SUCCESS;
    }
    rmm::device_vector<gdf_index_type> temp_indices_vect;
    if (temp_random_indices == nullptr){
        temp_indices_vect.resize(num_samples);
        temp_random_indices = temp_indices_vect.data().get();
    }
    
    gdf_error status = generate_random_vector<gdf_index_type>(temp_random_indices, 0, sampled_data->num_columns(0)->size, num_samples);

    return gather(source_table, temp_random_indices, sampled_data);   
}

}  // namespace detail

gdf_error gather_random_samples(table const* source_table,
                          table* sampled_data,
                          gdf_size_type num_samples) {
  return detail::gather_random_samples(source_table, sampled_data, num_samples, nullptr);
}

} // namespace cudf

