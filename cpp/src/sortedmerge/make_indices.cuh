#pragma once

#include <cuda_runtime.h>

#include <bits/c++config.h>

gdf_size_type *make_indices(cudaStream_t         cudaStreaam,
                          const std::ptrdiff_t length);
