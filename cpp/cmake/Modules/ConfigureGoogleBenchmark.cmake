set(GBENCHMARK_ROOT "${CMAKE_BINARY_DIR}/googlebenchmark")

set(GBENCHMARK_CMAKE_ARGS " -DCMAKE_BUILD_TYPE=RELEASE" 
                          " -DBENCHMARK_ENABLE_GTEST_TESTS=OFF"
                          " -DCMAKE_C_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0"      # enable old ABI for C/C++
                          " -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0")   # enable old ABI for C/C++


configure_file("${CMAKE_SOURCE_DIR}/cmake/Templates/GoogleBenchmark.CMakeLists.txt.cmake"
               "${GBENCHMARK_ROOT}/CMakeLists.txt")

file(MAKE_DIRECTORY "${GBENCHMARK_ROOT}/build")
file(MAKE_DIRECTORY "${GBENCHMARK_ROOT}/install")

execute_process(COMMAND ${CMAKE_COMMAND} -G ${CMAKE_GENERATOR} .
                RESULT_VARIABLE GBENCHMARK_CONFIG
                WORKING_DIRECTORY ${GBENCHMARK_ROOT})

if(GBENCHMARK_CONFIG)
    message(FATAL_ERROR "Configuring GoogleBenchmark failed: " ${GBENCHMARK_CONFIG})
endif(GBENCHMARK_CONFIG)

# Parallel builds cause Travis to run out of memory
unset(PARALLEL_BUILD)
if($ENV{TRAVIS})
    if(NOT DEFINED ENV{CMAKE_BUILD_PARALLEL_LEVEL})
        message(STATUS "Disabling Parallel CMake build on Travis")
    else()
        set(PARALLEL_BUILD --parallel)
        message(STATUS "Using $ENV{CMAKE_BUILD_PARALLEL_LEVEL} build jobs on Travis")
    endif(NOT DEFINED ENV{CMAKE_BUILD_PARALLEL_LEVEL})
else()
    set(PARALLEL_BUILD --parallel)
    message("STATUS Enabling Parallel CMake build")
endif($ENV{TRAVIS})

execute_process(COMMAND ${CMAKE_COMMAND} --build ${PARALLEL_BUILD} ..
                RESULT_VARIABLE GBENCHMARK_BUILD
                WORKING_DIRECTORY ${GBENCHMARK_ROOT}/build)

if(GBENCHMARK_BUILD)
    message(FATAL_ERROR "Building GoogleBenchmark failed: " ${GBENCHMARK_BUILD})
endif(GBENCHMARK_BUILD)

message(STATUS "GoogleBenchmark installed here: " ${GBENCHMARK_ROOT}/install)
set(GBENCHMARK_INCLUDE_DIR "${GBENCHMARK_ROOT}/install/include")
set(GBENCHMARK_LIBRARY_DIR "${GBENCHMARK_ROOT}/install/lib")
set(GBENCHMARK_FOUND TRUE)

