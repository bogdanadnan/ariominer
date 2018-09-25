cmake_minimum_required (VERSION 2.8)
project (ArioMiner)

include (TargetArch.cmake)

target_architecture (ARCH)
MESSAGE( STATUS "Target architecture is: " ${ARCH} )

set(SOURCE
        hash/cpu/cpu_hasher.cpp hash/cpu/cpu_hasher.h
        hash/hasher.cpp hash/hasher.h
        http/http.cpp http/http.h
        http/client.cpp http/client.h
        http/server.cpp http/server.h
        http/simplejson/json.h
        miner/miner.cpp miner/miner.h
        proxy/proxy.cpp proxy/proxy.h
        app/main.cpp
        app/arguments.cpp app/arguments.h common/common.h
        crypt/sha512.cpp crypt/sha512.h
        http/http_parser/http_parser.c http/http_parser/http_parser.h
        miner/mini-gmp/mini-gmp.h miner/mini-gmp/mini-gmp.c
        common/common.cpp
        hash/argon2/argon2.cpp hash/argon2/argon2.h
        hash/argon2/defs.h
        hash/argon2/argon2profile_4_4_16384.c hash/argon2/argon2profile_1_1_524288.c
        hash/argon2/blake2/blake2b.c hash/argon2/blake2/blake2.h
        hash/gpu/opencl/opencl_hasher.cpp hash/gpu/opencl/opencl_hasher.h
        hash/gpu/opencl/opencl_kernel.cpp hash/gpu/opencl/opencl_kernel.h
        hash/gpu/cuda/cuda_hasher.cpp hash/gpu/cuda/cuda_hasher.h
        crypt/base64.cpp crypt/base64.h crypt/random_generator.cpp crypt/random_generator.h
        autotune/autotune.cpp autotune/autotune.h
        app/runner.h
        hash/argon2/blake2/blake2b-round.h
        hash/argon2/blake2/blake2b-load-sse41.h
        hash/argon2/blake2/blake2b-load-sse2.h
        hash/argon2/blake2/blake2-config.h)

set(CUDA_KERNEL
        hash/gpu/cuda/cuda_kernel.cu)

set(ARGON2_FILL_BLOCKS_SRC
        hash/cpu/argon2_opt/implementation.c
        hash/cpu/argon2_opt/blamka-round-opt.h
        hash/cpu/argon2_opt/blamka-round-ref.h
        hash/argon2/defs.h
        hash/argon2/blake2/blake2-impl.h)

set (CMAKE_CXX_STANDARD 11)
set (ArioMiner_VERSION_MAJOR 0)
set (ArioMiner_VERSION_MINOR 2)
set (ArioMiner_VERSION_REVISION 0)

set (CMAKE_MACOSX_RPATH 0)

configure_file (
        "${PROJECT_SOURCE_DIR}/config.h.in"
        "${PROJECT_BINARY_DIR}/config.h"
)

include_directories("${PROJECT_BINARY_DIR}")
include_directories(hash/cpu/cpu_features/include)

find_package (Threads)

add_subdirectory(hash/cpu/cpu_features)


if(NOT WITHOUT_CUDA)
    add_definitions(-DWITH_CUDA)
    find_package(CUDA REQUIRED)
    set(
            CUDA_NVCC_FLAGS
            ${CUDA_NVCC_FLAGS};
#            -g -G
            -O3
    )
    cuda_add_executable(ariominer ${SOURCE} ${CUDA_KERNEL})
else()
    add_executable(ariominer ${SOURCE})
endif()

if(NOT WITHOUT_OPENCL)
    add_definitions(-DWITH_OPENCL)
    find_package(OpenCL REQUIRED)
    include_directories(${OpenCL_INCLUDE_DIR})
    target_link_libraries(ariominer ${OpenCL_LIBRARY})
endif()

target_link_libraries(ariominer cpu_features ${CMAKE_THREAD_LIBS_INIT})
target_compile_options(ariominer PRIVATE -O3 -mtune=native -march=native)

add_library(argon2_fill_blocks_REF SHARED ${ARGON2_FILL_BLOCKS_SRC})
set_target_properties(argon2_fill_blocks_REF
        PROPERTIES
        PREFIX ""
        SUFFIX ".opt"
        LIBRARY_OUTPUT_DIRECTORY modules
        )
target_compile_options(argon2_fill_blocks_REF PRIVATE -DBUILD_REF -O3)
add_dependencies(ariominer argon2_fill_blocks_REF)

if(ARCH STREQUAL "x86_64")
    add_library(argon2_fill_blocks_SSE2 SHARED ${ARGON2_FILL_BLOCKS_SRC})
    add_library(argon2_fill_blocks_SSSE3 SHARED ${ARGON2_FILL_BLOCKS_SRC})
    add_library(argon2_fill_blocks_AVX SHARED ${ARGON2_FILL_BLOCKS_SRC})
    add_library(argon2_fill_blocks_AVX2 SHARED ${ARGON2_FILL_BLOCKS_SRC})
    add_library(argon2_fill_blocks_AVX512F SHARED ${ARGON2_FILL_BLOCKS_SRC})
    set_target_properties(argon2_fill_blocks_REF argon2_fill_blocks_SSE2 argon2_fill_blocks_SSSE3 argon2_fill_blocks_AVX argon2_fill_blocks_AVX2 argon2_fill_blocks_AVX512F
            PROPERTIES
            PREFIX ""
            SUFFIX ".opt"
            LIBRARY_OUTPUT_DIRECTORY modules
            )
    target_compile_options(argon2_fill_blocks_SSE2 PRIVATE -msse2 -O3)
    target_compile_options(argon2_fill_blocks_SSSE3 PRIVATE -mssse3 -O3)
    target_compile_options(argon2_fill_blocks_AVX PRIVATE -mavx -O3)
    target_compile_options(argon2_fill_blocks_AVX2 PRIVATE -mavx2 -O3)
    target_compile_options(argon2_fill_blocks_AVX512F PRIVATE -mavx512f -O3)
    add_dependencies(ariominer argon2_fill_blocks_SSE2 argon2_fill_blocks_SSSE3 argon2_fill_blocks_AVX argon2_fill_blocks_AVX2 argon2_fill_blocks_AVX512F)
endif(ARCH STREQUAL "x86_64")

if(ARCH STREQUAL "arm")
    add_library(argon2_fill_blocks_NEON SHARED ${ARGON2_FILL_BLOCKS_SRC})
    set_target_properties(argon2_fill_blocks_NEON
            PROPERTIES
            PREFIX ""
            SUFFIX ".opt"
            LIBRARY_OUTPUT_DIRECTORY modules
            )
    target_compile_options(argon2_fill_blocks_NEON PRIVATE -D__NEON__ -mfpu=neon -funsafe-math-optimizations -O3)
endif(ARCH STREQUAL "arm")
