cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(CUDF_BENCHMARKS LANGUAGES C CXX CUDA)

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureBenchmark CMAKE_BENCHMARK_NAME CMAKE_BENCHMARK_SRC)
    add_executable(${CMAKE_BENCHMARK_NAME} ${CMAKE_BENCHMARK_SRC})
    set_target_properties(${CMAKE_BENCHMARK_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${CMAKE_BENCHMARK_NAME} benchmark benchmark_main pthread cudf)
    set_target_properties(${CMAKE_BENCHMARK_NAME} PROPERTIES
                            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gbenchmarks")
endfunction(ConfigureBenchmark)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${GBENCHMARK_INCLUDE_DIR}"
                    "${ARROW_INCLUDE_DIR}"
                    "${FLATBUFFERS_INCLUDE_DIR}"
                    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
                    "${CMAKE_BINARY_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_SOURCE_DIR}"
                    "${CMAKE_SOURCE_DIR}/src"
                    "${CMAKE_SOURCE_DIR}/thirdparty/cub"
                    "${CMAKE_SOURCE_DIR}/thirdparty/moderngpu/src"
                    "${CMAKE_SOURCE_DIR}/thirdparty/cnmem/include")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the link directories for nvcc
                 "${CMAKE_BINARY_DIR}/lib"
                 "${FLATBUFFERS_LIBRARY_DIR}"
                 "${GBENCHMARK_LIBRARY_DIR}")

###################################################################################################
### benchmark sources #############################################################################
###################################################################################################

###################################################################################################
# - replace benchmark -----------------------------------------------------------------------------

set(REPLACE_BENCHMARK_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/replace/replace_benchmark.cu")

ConfigureBenchmark(REPLACE_BENCHMARK "${REPLACE_BENCHMARK_SRC}")

###################################################################################################
