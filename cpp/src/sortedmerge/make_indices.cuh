#pragma once

#include <cuda_runtime.h>

#include <bits/c++config.h>

std::size_t *make_indices(cudaStream_t         cudaStreaam,
                          const std::ptrdiff_t length);
