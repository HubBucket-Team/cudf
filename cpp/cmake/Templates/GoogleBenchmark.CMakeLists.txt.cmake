cmake_minimum_required(VERSION 3.12)

include(ExternalProject)

ExternalProject_Add(GoogleBenchmark
                    GIT_REPOSITORY    https://github.com/google/benchmark.git
                    GIT_TAG           v1.4.1
                    SOURCE_DIR        "${GBENCHMARK_ROOT}/googlebenchmark"
                    BINARY_DIR        "${GBENCHMARK_ROOT}/build"
                    INSTALL_DIR		  "${GBENCHMARK_ROOT}/install"
                    CMAKE_ARGS        ${GBENCHMARK_CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${GBENCHMARK_ROOT}/install)








