cmake_minimum_required (VERSION 2.8)
project (ArioMiner)

include (TargetArch.cmake)

target_architecture (ARCH)
MESSAGE( STATUS "Target architecture is: " ${ARCH} )

SET(CMAKE_SKIP_BUILD_RPATH FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
SET(CMAKE_INSTALL_RPATH "./")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(SOURCE
        hash/hasher.h
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
        crypt/base64.cpp crypt/base64.h crypt/random_generator.cpp crypt/random_generator.h
        autotune/autotune.cpp autotune/autotune.h
        app/runner.h
        hash/argon2/blake2/blake2b-round.h
        hash/argon2/blake2/blake2b-load-sse41.h
        hash/argon2/blake2/blake2b-load-sse2.h
        hash/argon2/blake2/blake2-config.h
        )
set(SOURCE_HASHER hash/hasher.cpp hash/hasher.h)
set(SOURCE_CPU_HASHER hash/cpu/cpu_hasher.cpp hash/cpu/cpu_hasher.h)
set(SOURCE_OPENCL_HASHER hash/gpu/opencl/opencl_hasher.cpp hash/gpu/opencl/opencl_hasher.h
        hash/gpu/opencl/opencl_kernel.cpp hash/gpu/opencl/opencl_kernel.h)
set(SOURCE_CUDA_HASHER hash/gpu/cuda/cuda_hasher.cpp hash/gpu/cuda/cuda_hasher.h
        hash/gpu/cuda/cuda_kernel.cu)
set(SOURCE_AMDGCN_HASHER hash/gpu/amdgcn/amdgcn_hasher.cpp hash/gpu/amdgcn/amdgcn_hasher.h
        hash/gpu/amdgcn/amdgcn_kernel.cpp hash/gpu/amdgcn/amdgcn_kernel.h)
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

find_package(Threads)

add_subdirectory(hash/cpu/cpu_features)
set_property(TARGET cpu_features PROPERTY POSITION_INDEPENDENT_CODE ON)

add_executable(ariominer ${SOURCE})
add_library(hasher SHARED ${SOURCE_HASHER})
target_compile_definitions(hasher PRIVATE EXPORT_SYMBOLS=1)
target_link_libraries(hasher ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT})

add_library(cpu_hasher MODULE ${SOURCE_CPU_HASHER})
set_target_properties(cpu_hasher
        PROPERTIES
        PREFIX ""
        SUFFIX ".hsh"
        LIBRARY_OUTPUT_DIRECTORY modules
        )
target_compile_definitions(cpu_hasher PRIVATE EXPORT_SYMBOLS=1)
target_link_libraries(cpu_hasher hasher cpu_features)
add_dependencies(ariominer cpu_hasher)

if(NOT WITHOUT_CUDA)
    add_definitions(-DWITH_CUDA)
    find_package(CUDA REQUIRED)
    set(
            CUDA_NVCC_FLAGS
            ${CUDA_NVCC_FLAGS};
            -O3
    )
    cuda_add_library(cuda_hasher MODULE ${SOURCE_CUDA_HASHER})
    set_target_properties(cuda_hasher
            PROPERTIES
            PREFIX ""
            SUFFIX ".hsh"
            LIBRARY_OUTPUT_DIRECTORY modules
            )
    target_compile_definitions(cuda_hasher PRIVATE EXPORT_SYMBOLS=1)
    target_link_libraries(cuda_hasher hasher)
    add_dependencies(ariominer cuda_hasher)
endif()

if(NOT WITHOUT_OPENCL)
    add_definitions(-DWITH_OPENCL)
    find_package(OpenCL REQUIRED)
    include_directories(${OpenCL_INCLUDE_DIR})
    add_library(opencl_hasher MODULE ${SOURCE_OPENCL_HASHER})
    set_target_properties(opencl_hasher
            PROPERTIES
            PREFIX ""
            SUFFIX ".hsh"
            LIBRARY_OUTPUT_DIRECTORY modules
            )
    target_compile_definitions(opencl_hasher PRIVATE EXPORT_SYMBOLS=1)
    target_link_libraries(opencl_hasher hasher ${OpenCL_LIBRARY})
    add_dependencies(ariominer opencl_hasher)

    if(NOT WITHOUT_AMDGCN)
        add_definitions(-DWITH_AMDGCN)
        list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/hash/gpu/amdgcn/CLRX/cmake")
        add_subdirectory("hash/gpu/amdgcn/CLRX")
        include_directories("hash/gpu/amdgcn/CLRX")
        include_directories("${PROJECT_BINARY_DIR}/hash/gpu/amdgcn/CLRX/")

        add_library(amdgcn_hasher MODULE ${SOURCE_AMDGCN_HASHER})
        set_target_properties(amdgcn_hasher
                PROPERTIES
                PREFIX ""
                SUFFIX ".hsh"
                LIBRARY_OUTPUT_DIRECTORY modules
                )
        target_compile_definitions(amdgcn_hasher PRIVATE EXPORT_SYMBOLS=1)
	set_property(TARGET CLRXCLHelperStatic PROPERTY POSITION_INDEPENDENT_CODE ON)
	set_property(TARGET CLRXAmdAsmStatic PROPERTY POSITION_INDEPENDENT_CODE ON)
	set_property(TARGET CLRXAmdBinStatic PROPERTY POSITION_INDEPENDENT_CODE ON)
	set_property(TARGET CLRXUtilsStatic PROPERTY POSITION_INDEPENDENT_CODE ON)
        target_link_libraries(amdgcn_hasher hasher CLRXCLHelperStatic CLRXAmdAsmStatic
                CLRXAmdBinStatic CLRXUtilsStatic ${OpenCL_LIBRARY})
        add_dependencies(ariominer amdgcn_hasher)
    endif()
endif()

target_link_libraries(ariominer hasher)

add_library(argon2_fill_blocks_REF MODULE ${ARGON2_FILL_BLOCKS_SRC})
set_target_properties(argon2_fill_blocks_REF
        PROPERTIES
        PREFIX ""
        SUFFIX ".opt"
        LIBRARY_OUTPUT_DIRECTORY modules
        )
target_compile_definitions(argon2_fill_blocks_REF PRIVATE EXPORT_SYMBOLS=1 BUILD_REF=1)
add_dependencies(ariominer argon2_fill_blocks_REF)

if(ARCH STREQUAL "x86_64")
    add_library(argon2_fill_blocks_SSE2 MODULE ${ARGON2_FILL_BLOCKS_SRC})
    add_library(argon2_fill_blocks_SSSE3 MODULE ${ARGON2_FILL_BLOCKS_SRC})
    add_library(argon2_fill_blocks_AVX MODULE ${ARGON2_FILL_BLOCKS_SRC})
    add_library(argon2_fill_blocks_AVX2 MODULE ${ARGON2_FILL_BLOCKS_SRC})
    add_library(argon2_fill_blocks_AVX512F MODULE ${ARGON2_FILL_BLOCKS_SRC})
    set_target_properties(argon2_fill_blocks_REF argon2_fill_blocks_SSE2 argon2_fill_blocks_SSSE3 argon2_fill_blocks_AVX argon2_fill_blocks_AVX2 argon2_fill_blocks_AVX512F
            PROPERTIES
            PREFIX ""
            SUFFIX ".opt"
            LIBRARY_OUTPUT_DIRECTORY modules
            )
    target_compile_options(argon2_fill_blocks_SSE2 PRIVATE -msse2)
    target_compile_options(argon2_fill_blocks_SSSE3 PRIVATE -mssse3)
    target_compile_options(argon2_fill_blocks_AVX PRIVATE -mavx)
    target_compile_options(argon2_fill_blocks_AVX2 PRIVATE -mavx2)
    target_compile_options(argon2_fill_blocks_AVX512F PRIVATE -mavx512f)
    target_compile_definitions(argon2_fill_blocks_SSE2 PRIVATE EXPORT_SYMBOLS=1)
    target_compile_definitions(argon2_fill_blocks_SSSE3 PRIVATE EXPORT_SYMBOLS=1)
    target_compile_definitions(argon2_fill_blocks_AVX PRIVATE EXPORT_SYMBOLS=1)
    target_compile_definitions(argon2_fill_blocks_AVX2 PRIVATE EXPORT_SYMBOLS=1)
    target_compile_definitions(argon2_fill_blocks_AVX512F PRIVATE EXPORT_SYMBOLS=1)
    add_dependencies(ariominer argon2_fill_blocks_SSE2 argon2_fill_blocks_SSSE3 argon2_fill_blocks_AVX argon2_fill_blocks_AVX2 argon2_fill_blocks_AVX512F)
endif(ARCH STREQUAL "x86_64")

if(ARCH STREQUAL "arm")
    add_library(argon2_fill_blocks_NEON MODULE ${ARGON2_FILL_BLOCKS_SRC})
    set_target_properties(argon2_fill_blocks_NEON
            PROPERTIES
            PREFIX ""
            SUFFIX ".opt"
            LIBRARY_OUTPUT_DIRECTORY modules
            )
    target_compile_options(argon2_fill_blocks_NEON PRIVATE -D__NEON__ -mfpu=neon -funsafe-math-optimizations)
    target_compile_definitions(argon2_fill_blocks_NEON PRIVATE EXPORT_SYMBOLS=1)
    add_dependencies(ariominer argon2_fill_blocks_NEON)
endif(ARCH STREQUAL "arm")
