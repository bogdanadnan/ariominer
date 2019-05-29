####
#  CLRadeonExtender - Unofficial OpenCL Radeon Extensions Library
#  Copyright (C) 2014-2018 Mateusz Szpakowski
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  This library is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public
#  License along with this library; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
####

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.1)

SET(LIBAMDBINSRC
        AmdBinaries.cpp
        AmdBinGen.cpp
        AmdCL2Binaries.cpp
        AmdCL2BinGen.cpp
        ElfBinaries.cpp
        GalliumBinaries.cpp
        ROCmBinaries.cpp)

ADD_LIBRARY(CLRXAmdBinStatic STATIC ${LIBAMDBINSRC})
SET_TARGET_OUTNAME(CLRXAmdBinStatic CLRXAmdBin)
SET_PROPERTY(TARGET CLRXAmdBinStatic PROPERTY POSITION_INDEPENDENT_CODE ON)

if(WIN32)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /D_CRT_SECURE_NO_WARNINGS=1 /MD")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /D_CRT_SECURE_NO_WARNINGS=1 /MDd")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /D_CRT_SECURE_NO_WARNINGS=1 /MD")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /D_CRT_SECURE_NO_WARNINGS=1 /MDd")
endif()
