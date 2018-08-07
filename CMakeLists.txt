cmake_minimum_required (VERSION 2.6)
project (ArioMiner)
set(SOURCE
        hash/cpu/cpu_hasher.cpp hash/cpu/cpu_hasher.h
        hash/gpu/gpu_hasher.cpp hash/gpu/gpu_hasher.h
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
        http/mongoose/mongoose.c
        http/mongoose/mongoose.h
        miner/mini-gmp/mini-gmp.h miner/mini-gmp/mini-gmp.c
        common/common.cpp
        hash/argon2/argon2.cpp hash/argon2/argon2.h
        hash/argon2/defs.h hash/argon2/adresses.c
        hash/argon2/blake2/blake2b.c hash/argon2/blake2/blake2.h
        hash/gpu/OpenCL_Cpp/cl.hpp
        hash/gpu/opencl_kernel.cpp hash/gpu/opencl_kernel.h)

set(ARGON2_FILL_BLOCKS_SRC
        hash/cpu/argon2_opt/implementation.c
        hash/cpu/argon2_opt/blamka-round-opt.h
        hash/cpu/argon2_opt/blamka-round-ref.h
        hash/argon2/defs.h
        hash/argon2/blake2/blake2-impl.h
        hash/argon2/adresses.c)

set (CMAKE_CXX_STANDARD 11)
set (ArioMiner_VERSION_MAJOR 0)
set (ArioMiner_VERSION_MINOR 1)
set (CMAKE_MACOSX_RPATH 0)

configure_file (
        "${PROJECT_SOURCE_DIR}/config.h.in"
        "${PROJECT_BINARY_DIR}/config.h"
)

include_directories("${PROJECT_BINARY_DIR}")
include_directories(hash/cpu/cpu_features/include)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

find_package (Threads)

add_subdirectory(hash/cpu/cpu_features)

add_executable(ariominer ${SOURCE})
add_library(argon2_fill_blocks_REF SHARED ${ARGON2_FILL_BLOCKS_SRC})
add_library(argon2_fill_blocks_SSE2 SHARED ${ARGON2_FILL_BLOCKS_SRC})
add_library(argon2_fill_blocks_SSSE3 SHARED ${ARGON2_FILL_BLOCKS_SRC})
add_library(argon2_fill_blocks_AVX2 SHARED ${ARGON2_FILL_BLOCKS_SRC})
add_library(argon2_fill_blocks_AVX512F SHARED ${ARGON2_FILL_BLOCKS_SRC})

add_dependencies(ariominer argon2_fill_blocks_REF argon2_fill_blocks_SSE2 argon2_fill_blocks_SSSE3 argon2_fill_blocks_AVX2 argon2_fill_blocks_AVX512F)

set_target_properties(argon2_fill_blocks_REF argon2_fill_blocks_SSE2 argon2_fill_blocks_SSSE3 argon2_fill_blocks_AVX2 argon2_fill_blocks_AVX512F
        PROPERTIES
        PREFIX ""
        SUFFIX ".opt"
        LIBRARY_OUTPUT_DIRECTORY modules
        )

target_compile_options(argon2_fill_blocks_REF PRIVATE -DBUILD_REF -O3)
target_compile_options(argon2_fill_blocks_SSE2 PRIVATE -msse2 -O3)
target_compile_options(argon2_fill_blocks_SSSE3 PRIVATE -mssse3 -O3)
target_compile_options(argon2_fill_blocks_AVX2 PRIVATE -mavx2 -O3)
target_compile_options(argon2_fill_blocks_AVX512F PRIVATE -mavx512f -O3)
target_compile_options(ariominer PRIVATE -O3)

target_link_libraries(ariominer cpu_features ${OpenCL_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})
