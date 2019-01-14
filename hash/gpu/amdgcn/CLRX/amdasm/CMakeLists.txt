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

SET(LIBAMDASMSRC 
        AsmAmdCL2Format.cpp
        AsmAmdFormat.cpp
        AsmExpression.cpp
        AsmFormats.cpp
        AsmGalliumFormat.cpp
        AsmPseudoOps.cpp
        AsmPseudoOpsCode1.cpp
        AsmROCmFormat.cpp
        AsmRegAlloc.cpp
        AsmRegAllocLive.cpp
        AsmRegAllocSSAData.cpp
        AsmSource.cpp
        AsmWait.cpp
        Assembler.cpp
        Disassembler.cpp
        DisasmAmd.cpp
        DisasmAmdCL2.cpp
        DisasmGallium.cpp
        DisasmROCm.cpp
        GCNAsmEncode1.cpp
        GCNAsmEncode2.cpp
        GCNAsmHelpers.cpp
        GCNAssembler.cpp
        GCNDisasm.cpp
        GCNDisasmDecode.cpp
        GCNInstructions.cpp)

ADD_LIBRARY(CLRXAmdAsmStatic STATIC ${LIBAMDASMSRC})
SET_TARGET_OUTNAME(CLRXAmdAsmStatic CLRXAmdAsm)
SET_PROPERTY(TARGET CLRXAmdAsmStatic PROPERTY POSITION_INDEPENDENT_CODE ON)

if(WIN32)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /D_CRT_SECURE_NO_WARNINGS=1 /MD")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /D_CRT_SECURE_NO_WARNINGS=1 /MDd")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /D_CRT_SECURE_NO_WARNINGS=1 /MD")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /D_CRT_SECURE_NO_WARNINGS=1 /MDd")
endif()
