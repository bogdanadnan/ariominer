cmake_minimum_required (VERSION 2.6)
project (ArioMiner)

include (TargetArch.cmake)

target_architecture (ARCH)
MESSAGE( STATUS "Target architecture is: " ${ARCH} )

set(CMAKE_EXE_LINKER_FLAGS "-static")

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
        crypt/base64.cpp crypt/base64.h crypt/random_generator.cpp crypt/random_generator.h app/runner.h)

set(ARGON2_FILL_BLOCKS_SRC
        hash/cpu/argon2_opt/implementation.c
        hash/cpu/argon2_opt/blamka-round-opt.h
        hash/cpu/argon2_opt/blamka-round-ref.h
        hash/argon2/defs.h
        hash/argon2/blake2/blake2-impl.h)

set (CMAKE_CXX_STANDARD 11)
set (ArioMiner_VERSION_MAJOR 0)
set (ArioMiner_VERSION_MINOR 1)
set (ArioMiner_VERSION_REVISION 2)

set (CMAKE_MACOSX_RPATH 0)

configure_file (
        "${PROJECT_SOURCE_DIR}/config.h.in"
        "${PROJECT_BINARY_DIR}/config.h"
)

include_directories("${PROJECT_BINARY_DIR}")
include_directories(hash/cpu/cpu_features/include)

find_package (Threads)

add_subdirectory(hash/cpu/cpu_features)

add_executable(ariominer ${SOURCE})

add_library(argon2_fill_blocks_REF STATIC ${ARGON2_FILL_BLOCKS_SRC})
target_compile_options(argon2_fill_blocks_REF PRIVATE -DBUILD_REF -O3 -DOPT_NAME=REF)
add_dependencies(ariominer argon2_fill_blocks_REF)

if(ARCH STREQUAL "x86_64")
    add_library(argon2_fill_blocks_SSE2 STATIC ${ARGON2_FILL_BLOCKS_SRC})
    add_library(argon2_fill_blocks_SSSE3 STATIC ${ARGON2_FILL_BLOCKS_SRC})
    add_library(argon2_fill_blocks_AVX2 STATIC ${ARGON2_FILL_BLOCKS_SRC})
    add_library(argon2_fill_blocks_AVX512F STATIC ${ARGON2_FILL_BLOCKS_SRC})
    target_compile_options(argon2_fill_blocks_SSE2 PRIVATE -msse2 -O3 -DOPT_NAME=SSE2)
    target_compile_options(argon2_fill_blocks_SSSE3 PRIVATE -mssse3 -O3 -DOPT_NAME=SSSE3)
    target_compile_options(argon2_fill_blocks_AVX2 PRIVATE -mavx2 -O3 -DOPT_NAME=AVX2)
    target_compile_options(argon2_fill_blocks_AVX512F PRIVATE -mavx512f -O3 -DOPT_NAME=AVX512F)
    add_dependencies(ariominer argon2_fill_blocks_SSE2 argon2_fill_blocks_SSSE3 argon2_fill_blocks_AVX2 argon2_fill_blocks_AVX512F)

    target_link_libraries(ariominer cpu_features argon2_fill_blocks_REF argon2_fill_blocks_SSE2 argon2_fill_blocks_SSSE3 argon2_fill_blocks_AVX2 argon2_fill_blocks_AVX512F ${CMAKE_THREAD_LIBS_INIT})
endif(ARCH STREQUAL "x86_64")

if(ARCH STREQUAL "arm")
    add_library(argon2_fill_blocks_NEON STATIC ${ARGON2_FILL_BLOCKS_SRC})
    target_compile_options(argon2_fill_blocks_NEON PRIVATE -D__NEON__ -mfpu=neon -funsafe-math-optimizations -O3 -DOPT_NAME=NEON)
    target_link_libraries(ariominer cpu_features argon2_fill_blocks_REF argon2_fill_blocks_NEON ${CMAKE_THREAD_LIBS_INIT})
endif(ARCH STREQUAL "arm")
