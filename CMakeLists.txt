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
        http/http.cpp http/http.h
        http/client.cpp http/client.h
        http/civetweb/civetweb.c http/civetweb/CivetServer.cpp
        http/simplejson/json.h
        miner/miner.cpp miner/miner.h
        proxy/proxy.cpp proxy/proxy.h
        app/main.cpp
        http/http_parser/http_parser.c http/http_parser/http_parser.h
        miner/mini-gmp/mini-gmp.h miner/mini-gmp/mini-gmp.c
        autotune/autotune.cpp autotune/autotune.h
        app/runner.h
        miner/miner_api.cpp miner/miner_api.h http/pool_settings_provider.cpp http/pool_settings_provider.h http/simplejson/json.cpp proxy/proxy_server.cpp proxy/proxy_server.h http/node_api.cpp http/node_api.h)
set(SOURCE_COMMON app/arguments.cpp app/arguments.h common/common.h common/common.cpp common/dllimport.h common/dllexport.h
        crypt/sha512.cpp crypt/sha512.h crypt/base64.cpp crypt/base64.h crypt/random_generator.cpp crypt/random_generator.h
        common/cfgpath.h)
set(SOURCE_HASHER hash/hasher.cpp hash/hasher.h hash/argon2/argon2.cpp hash/argon2/argon2.h
        hash/argon2/defs.h hash/argon2/blake2/blake2b.c hash/argon2/blake2/blake2.h hash/argon2/blake2/blake2b-round.h
        hash/argon2/blake2/blake2b-load-sse41.h hash/argon2/blake2/blake2b-load-sse2.h hash/argon2/blake2/blake2-config.h
        hash/argon2/argon2profile_4_4_16384.c hash/argon2/argon2profile_1_1_524288.c)
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
if(WIN32)
    include_directories(win64/include)
    list(APPEND SOURCE_COMMON win64/src/dlfcn.c win64/src/win64.c)
endif()

find_package(Threads)

add_subdirectory(hash/cpu/cpu_features)
set_property(TARGET cpu_features PROPERTY POSITION_INDEPENDENT_CODE ON)

add_executable(ariominer ${SOURCE})
add_library(common SHARED ${SOURCE_COMMON})
add_library(hasher SHARED ${SOURCE_HASHER})
target_link_libraries(hasher common ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT})
if(WIN32)
    target_link_libraries(common Psapi.lib)
endif()
add_library(cpu_hasher MODULE ${SOURCE_CPU_HASHER})
set_target_properties(cpu_hasher
        PROPERTIES
        PREFIX ""
        SUFFIX ".hsh"
        LIBRARY_OUTPUT_DIRECTORY modules
        )
target_link_libraries(cpu_hasher hasher cpu_features)
add_dependencies(ariominer cpu_hasher)

if(NOT WITHOUT_CUDA)
    add_definitions(-DWITH_CUDA)
    find_package(CUDA REQUIRED)
    set(
            CUDA_NVCC_FLAGS
            ${CUDA_NVCC_FLAGS};
            -O3 -arch=compute_35
    )
    cuda_add_library(cuda_hasher MODULE ${SOURCE_CUDA_HASHER})
    set_target_properties(cuda_hasher
            PROPERTIES
            PREFIX ""
            SUFFIX ".hsh"
            LIBRARY_OUTPUT_DIRECTORY modules
            )
    target_link_libraries(cuda_hasher hasher)
    if(NOT WIN32)
        add_definitions(-DPARALLEL_CUDA)
    endif()
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
        target_link_libraries(amdgcn_hasher hasher CLRXCLHelperStatic CLRXAmdAsmStatic
                CLRXAmdBinStatic CLRXUtilsStatic ${OpenCL_LIBRARY})
        add_dependencies(ariominer amdgcn_hasher)
    endif()
endif()

target_link_libraries(ariominer common hasher)
if(WIN32)
    target_link_libraries(ariominer ws2_32.lib)
endif()
if(APPLE)
    set_target_properties(ariominer PROPERTIES LINK_FLAGS "-framework CoreServices")
endif()

add_library(argon2_fill_blocks_REF MODULE ${ARGON2_FILL_BLOCKS_SRC})
set_target_properties(argon2_fill_blocks_REF
        PROPERTIES
        PREFIX ""
        SUFFIX ".opt"
        LIBRARY_OUTPUT_DIRECTORY modules
        )
target_compile_definitions(argon2_fill_blocks_REF PRIVATE BUILD_REF=1)
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
    add_dependencies(ariominer argon2_fill_blocks_SSE2 argon2_fill_blocks_SSSE3 argon2_fill_blocks_AVX argon2_fill_blocks_AVX2 argon2_fill_blocks_AVX512F)
endif(ARCH STREQUAL "x86_64")

if(ARCH STREQUAL "arm" OR ARCH STREQUAL "aarch64")
    add_library(argon2_fill_blocks_NEON MODULE ${ARGON2_FILL_BLOCKS_SRC})
    set_target_properties(argon2_fill_blocks_NEON
            PROPERTIES
            PREFIX ""
            SUFFIX ".opt"
            LIBRARY_OUTPUT_DIRECTORY modules
            )
    target_compile_options(common PRIVATE -D__NEON__)
    if(ARCH STREQUAL "arm")
       target_compile_options(argon2_fill_blocks_NEON PRIVATE -D__NEON__ -mfpu=neon -funsafe-math-optimizations)
    else()
       target_compile_options(argon2_fill_blocks_NEON PRIVATE -D__NEON__)
    endif(ARCH STREQUAL "arm")

    add_dependencies(ariominer argon2_fill_blocks_NEON)
endif(ARCH STREQUAL "arm" OR ARCH STREQUAL "aarch64")

if(WIN32)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /D_CRT_SECURE_NO_WARNINGS=1")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /D_CRT_SECURE_NO_WARNINGS=1")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /D_CRT_SECURE_NO_WARNINGS=1")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /D_CRT_SECURE_NO_WARNINGS=1")
endif()

add_custom_target(copy-reporting-files ALL
        COMMAND cmake -E copy_directory "${CMAKE_SOURCE_DIR}/proxy/reporting/dist" "${CMAKE_BINARY_DIR}/reporting")

add_dependencies(ariominer copy-reporting-files)
