//
// Created by Haifa Bogdan Adnan on 07.10.2018.
//

#include "../../../common/common.h"

#include "amdgcn_kernel.h"

string amdgcn_kernel = R"ffDXD(
.ifarch GCN1.4
    HASH = %s10
    # helper macros for integer add/sub instructions
    .macro vadd_u32 dest, cdest, src0, src1, mods:vararg
        v_add_co_u32 \dest, \cdest, \src0, \src1 \mods
    .endm
    .macro vaddc_u32 dest, cdest, src0, src1, csrc, mods:vararg
        v_addc_co_u32 \dest, \cdest, \src0, \src1, \csrc \mods
    .endm
    .macro vsub_u32 dest, cdest, src0, src1, mods:vararg
        v_sub_co_u32 \dest, \cdest, \src0, \src1 \mods
    .endm
.else
    HASH = %s8
    # helper macros for integer add/sub instructions
    .macro vadd_u32 dest, cdest, src0, src1, mods:vararg
        v_add_u32 \dest, \cdest, \src0, \src1 \mods
    .endm
    .macro vaddc_u32 dest, cdest, src0, src1, csrc, mods:vararg
        v_addc_u32 \dest, \cdest, \src0, \src1, \csrc \mods
    .endm
    .macro vsub_u32 dest, cdest, src0, src1, mods:vararg
        v_sub_u32 \dest, \cdest, \src0, \src1 \mods
    .endm
.endif
.globaldata     # const data (local index table - 512 bytes/cross lane operation indexes - 256 bytes/gblocks segments - 256 bytes)
local_index:
    .byte 0x00, 0x04, 0x08, 0x0C, 0x01, 0x05, 0x09, 0x0D, 0x02, 0x06, 0x0A, 0x0E, 0x03, 0x07, 0x0B, 0x0F
    .byte 0x10, 0x14, 0x18, 0x1C, 0x11, 0x15, 0x19, 0x1D, 0x12, 0x16, 0x1A, 0x1E, 0x13, 0x17, 0x1B, 0x1F
    .byte 0x20, 0x24, 0x28, 0x2C, 0x21, 0x25, 0x29, 0x2D, 0x22, 0x26, 0x2A, 0x2E, 0x23, 0x27, 0x2B, 0x2F
    .byte 0x30, 0x34, 0x38, 0x3C, 0x31, 0x35, 0x39, 0x3D, 0x32, 0x36, 0x3A, 0x3E, 0x33, 0x37, 0x3B, 0x3F
    .byte 0x40, 0x44, 0x48, 0x4C, 0x41, 0x45, 0x49, 0x4D, 0x42, 0x46, 0x4A, 0x4E, 0x43, 0x47, 0x4B, 0x4F
    .byte 0x50, 0x54, 0x58, 0x5C, 0x51, 0x55, 0x59, 0x5D, 0x52, 0x56, 0x5A, 0x5E, 0x53, 0x57, 0x5B, 0x5F
    .byte 0x60, 0x64, 0x68, 0x6C, 0x61, 0x65, 0x69, 0x6D, 0x62, 0x66, 0x6A, 0x6E, 0x63, 0x67, 0x6B, 0x6F
    .byte 0x70, 0x74, 0x78, 0x7C, 0x71, 0x75, 0x79, 0x7D, 0x72, 0x76, 0x7A, 0x7E, 0x73, 0x77, 0x7B, 0x7F
    .byte 0x00, 0x05, 0x0A, 0x0F, 0x01, 0x06, 0x0B, 0x0C, 0x02, 0x07, 0x08, 0x0D, 0x03, 0x04, 0x09, 0x0E
    .byte 0x10, 0x15, 0x1A, 0x1F, 0x11, 0x16, 0x1B, 0x1C, 0x12, 0x17, 0x18, 0x1D, 0x13, 0x14, 0x19, 0x1E
    .byte 0x20, 0x25, 0x2A, 0x2F, 0x21, 0x26, 0x2B, 0x2C, 0x22, 0x27, 0x28, 0x2D, 0x23, 0x24, 0x29, 0x2E
    .byte 0x30, 0x35, 0x3A, 0x3F, 0x31, 0x36, 0x3B, 0x3C, 0x32, 0x37, 0x38, 0x3D, 0x33, 0x34, 0x39, 0x3E
    .byte 0x40, 0x45, 0x4A, 0x4F, 0x41, 0x46, 0x4B, 0x4C, 0x42, 0x47, 0x48, 0x4D, 0x43, 0x44, 0x49, 0x4E
    .byte 0x50, 0x55, 0x5A, 0x5F, 0x51, 0x56, 0x5B, 0x5C, 0x52, 0x57, 0x58, 0x5D, 0x53, 0x54, 0x59, 0x5E
    .byte 0x60, 0x65, 0x6A, 0x6F, 0x61, 0x66, 0x6B, 0x6C, 0x62, 0x67, 0x68, 0x6D, 0x63, 0x64, 0x69, 0x6E
    .byte 0x70, 0x75, 0x7A, 0x7F, 0x71, 0x76, 0x7B, 0x7C, 0x72, 0x77, 0x78, 0x7D, 0x73, 0x74, 0x79, 0x7E
    .byte 0x00, 0x20, 0x40, 0x60, 0x01, 0x21, 0x41, 0x61, 0x02, 0x22, 0x42, 0x62, 0x03, 0x23, 0x43, 0x63
    .byte 0x04, 0x24, 0x44, 0x64, 0x05, 0x25, 0x45, 0x65, 0x06, 0x26, 0x46, 0x66, 0x07, 0x27, 0x47, 0x67
    .byte 0x08, 0x28, 0x48, 0x68, 0x09, 0x29, 0x49, 0x69, 0x0A, 0x2A, 0x4A, 0x6A, 0x0B, 0x2B, 0x4B, 0x6B
    .byte 0x0C, 0x2C, 0x4C, 0x6C, 0x0D, 0x2D, 0x4D, 0x6D, 0x0E, 0x2E, 0x4E, 0x6E, 0x0F, 0x2F, 0x4F, 0x6F
    .byte 0x10, 0x30, 0x50, 0x70, 0x11, 0x31, 0x51, 0x71, 0x12, 0x32, 0x52, 0x72, 0x13, 0x33, 0x53, 0x73
    .byte 0x14, 0x34, 0x54, 0x74, 0x15, 0x35, 0x55, 0x75, 0x16, 0x36, 0x56, 0x76, 0x17, 0x37, 0x57, 0x77
    .byte 0x18, 0x38, 0x58, 0x78, 0x19, 0x39, 0x59, 0x79, 0x1A, 0x3A, 0x5A, 0x7A, 0x1B, 0x3B, 0x5B, 0x7B
    .byte 0x1C, 0x3C, 0x5C, 0x7C, 0x1D, 0x3D, 0x5D, 0x7D, 0x1E, 0x3E, 0x5E, 0x7E, 0x1F, 0x3F, 0x5F, 0x7F
    .byte 0x00, 0x21, 0x50, 0x71, 0x01, 0x30, 0x51, 0x60, 0x02, 0x23, 0x52, 0x73, 0x03, 0x32, 0x53, 0x62
    .byte 0x04, 0x25, 0x54, 0x75, 0x05, 0x34, 0x55, 0x64, 0x06, 0x27, 0x56, 0x77, 0x07, 0x36, 0x57, 0x66
    .byte 0x08, 0x29, 0x58, 0x79, 0x09, 0x38, 0x59, 0x68, 0x0A, 0x2B, 0x5A, 0x7B, 0x0B, 0x3A, 0x5B, 0x6A
    .byte 0x0C, 0x2D, 0x5C, 0x7D, 0x0D, 0x3C, 0x5D, 0x6C, 0x0E, 0x2F, 0x5E, 0x7F, 0x0F, 0x3E, 0x5F, 0x6E
    .byte 0x10, 0x31, 0x40, 0x61, 0x11, 0x20, 0x41, 0x70, 0x12, 0x33, 0x42, 0x63, 0x13, 0x22, 0x43, 0x72
    .byte 0x14, 0x35, 0x44, 0x65, 0x15, 0x24, 0x45, 0x74, 0x16, 0x37, 0x46, 0x67, 0x17, 0x26, 0x47, 0x76
    .byte 0x18, 0x39, 0x48, 0x69, 0x19, 0x28, 0x49, 0x78, 0x1A, 0x3B, 0x4A, 0x6B, 0x1B, 0x2A, 0x4B, 0x7A
    .byte 0x1C, 0x3D, 0x4C, 0x6D, 0x1D, 0x2C, 0x4D, 0x7C, 0x1E, 0x3F, 0x4E, 0x6F, 0x1F, 0x2E, 0x4F, 0x7E
    .byte 0x00, 0x01, 0x02, 0x03, 0x01, 0x02, 0x03, 0x00, 0x02, 0x03, 0x00, 0x01, 0x03, 0x00, 0x01, 0x02
    .byte 0x04, 0x05, 0x06, 0x07, 0x05, 0x06, 0x07, 0x04, 0x06, 0x07, 0x04, 0x05, 0x07, 0x04, 0x05, 0x06
    .byte 0x08, 0x09, 0x0A, 0x0B, 0x09, 0x0A, 0x0B, 0x08, 0x0A, 0x0B, 0x08, 0x09, 0x0B, 0x08, 0x09, 0x0A
    .byte 0x0C, 0x0D, 0x0E, 0x0F, 0x0D, 0x0E, 0x0F, 0x0C, 0x0E, 0x0F, 0x0C, 0x0D, 0x0F, 0x0C, 0x0D, 0x0E
    .byte 0x10, 0x11, 0x12, 0x13, 0x11, 0x12, 0x13, 0x10, 0x12, 0x13, 0x10, 0x11, 0x13, 0x10, 0x11, 0x12
    .byte 0x14, 0x15, 0x16, 0x17, 0x15, 0x16, 0x17, 0x14, 0x16, 0x17, 0x14, 0x15, 0x17, 0x14, 0x15, 0x16
    .byte 0x18, 0x19, 0x1A, 0x1B, 0x19, 0x1A, 0x1B, 0x18, 0x1A, 0x1B, 0x18, 0x19, 0x1B, 0x18, 0x19, 0x1A
    .byte 0x1C, 0x1D, 0x1E, 0x1F, 0x1D, 0x1E, 0x1F, 0x1C, 0x1E, 0x1F, 0x1C, 0x1D, 0x1F, 0x1C, 0x1D, 0x1E
    .byte 0x00, 0x01, 0x10, 0x11, 0x01, 0x10, 0x11, 0x00, 0x02, 0x03, 0x12, 0x13, 0x03, 0x12, 0x13, 0x02
    .byte 0x04, 0x05, 0x14, 0x15, 0x05, 0x14, 0x15, 0x04, 0x06, 0x07, 0x16, 0x17, 0x07, 0x16, 0x17, 0x06
    .byte 0x08, 0x09, 0x18, 0x19, 0x09, 0x18, 0x19, 0x08, 0x0A, 0x0B, 0x1A, 0x1B, 0x0B, 0x1A, 0x1B, 0x0A
    .byte 0x0C, 0x0D, 0x1C, 0x1D, 0x0D, 0x1C, 0x1D, 0x0C, 0x0E, 0x0F, 0x1E, 0x1F, 0x0F, 0x1E, 0x1F, 0x0E
    .byte 0x10, 0x11, 0x00, 0x01, 0x11, 0x00, 0x01, 0x10, 0x12, 0x13, 0x02, 0x03, 0x13, 0x02, 0x03, 0x12
    .byte 0x14, 0x15, 0x04, 0x05, 0x15, 0x04, 0x05, 0x14, 0x16, 0x17, 0x06, 0x07, 0x17, 0x06, 0x07, 0x16
    .byte 0x18, 0x19, 0x08, 0x09, 0x19, 0x08, 0x09, 0x18, 0x1A, 0x1B, 0x0A, 0x0B, 0x1B, 0x0A, 0x0B, 0x1A
    .byte 0x1C, 0x1D, 0x0C, 0x0D, 0x1D, 0x0C, 0x0D, 0x1C, 0x1E, 0x1F, 0x0E, 0x0F, 0x1F, 0x0E, 0x0F, 0x1E
    .byte 0x00, 0x00, 0x01, 0x00, 0xf8, 0x0f, 0xff, 0x03, 0xf8, 0x1f, 0xfb, 0x13, 0xf8, 0x2f, 0xfd, 0x23
    .byte 0xf8, 0x3f, 0xff, 0x33, 0xf8, 0x4f, 0xff, 0x03, 0xf8, 0x5f, 0xfb, 0x13, 0xf8, 0x6f, 0xfd, 0x23
    .byte 0xf8, 0x7f, 0xff, 0x33, 0xf8, 0x8f, 0xff, 0x03, 0xf8, 0x9f, 0xfb, 0x13, 0xf8, 0xaf, 0xfd, 0x23
    .byte 0xf8, 0xbf, 0xff, 0x33, 0xf8, 0xcf, 0xff, 0x03, 0xf8, 0xdf, 0xfb, 0x13, 0xf8, 0xef, 0xfd, 0x23
    .byte 0xfe, 0x03, 0x01, 0x10, 0xf8, 0x13, 0xfd, 0x07, 0xf8, 0x23, 0xfb, 0x17, 0xf8, 0x33, 0xfd, 0x27
    .byte 0xf8, 0x43, 0xff, 0x37, 0xf8, 0x53, 0xfd, 0x07, 0xf8, 0x63, 0xfb, 0x17, 0xf8, 0x73, 0xfd, 0x27
    .byte 0xf8, 0x83, 0xff, 0x37, 0xf8, 0x93, 0xfd, 0x07, 0xf8, 0xa3, 0xfb, 0x17, 0xf8, 0xb3, 0xfd, 0x27
    .byte 0xf8, 0xc3, 0xff, 0x37, 0xf8, 0xd3, 0xfd, 0x07, 0xf8, 0xe3, 0xfb, 0x17, 0xf8, 0xf3, 0xfd, 0x27
    .byte 0xfc, 0x07, 0x01, 0x20, 0xf8, 0x17, 0xfb, 0x0b, 0xf8, 0x27, 0xfb, 0x1b, 0xf8, 0x37, 0xfd, 0x2b
    .byte 0xf8, 0x47, 0xff, 0x3b, 0xf8, 0x57, 0xfb, 0x0b, 0xf8, 0x67, 0xfb, 0x1b, 0xf8, 0x77, 0xfd, 0x2b
    .byte 0xf8, 0x87, 0xff, 0x3b, 0xf8, 0x97, 0xfb, 0x0b, 0xf8, 0xa7, 0xfb, 0x1b, 0xf8, 0xb7, 0xfd, 0x2b
    .byte 0xf8, 0xc7, 0xff, 0x3b, 0xf8, 0xd7, 0xfb, 0x0b, 0xf8, 0xe7, 0xfb, 0x1b, 0xf8, 0xf7, 0xfd, 0x2b
    .byte 0xfa, 0x0b, 0x01, 0x30, 0xf8, 0x1b, 0xf9, 0x0f, 0xf8, 0x2b, 0xfb, 0x1f, 0xf8, 0x3b, 0xfd, 0x2f
    .byte 0xf8, 0x4b, 0xff, 0x3f, 0xf8, 0x5b, 0xf9, 0x0f, 0xf8, 0x6b, 0xfb, 0x1f, 0xf8, 0x7b, 0xfd, 0x2f
    .byte 0xf8, 0x8b, 0xff, 0x3f, 0xf8, 0x9b, 0xf9, 0x0f, 0xf8, 0xab, 0xfb, 0x1f, 0xf8, 0xbb, 0xfd, 0x2f
    .byte 0xf8, 0xcb, 0xff, 0x3f, 0xf8, 0xdb, 0xf9, 0x0f, 0xf8, 0xeb, 0xfb, 0x1f, 0xf8, 0xfb, 0xfd, 0x2f
.if64
.iffmt amdcl2
.kernel fill_cblocks
    .config
        .dims x
        .localsize 1024
        .useargs
        .usesetup
        .setupargs
        .arg chunk0, ulong*, global             # loaded in s[0:1]
        .arg chunk1, ulong*, global
        .arg chunk2, ulong*, global
        .arg chunk3, ulong*, global
        .arg chunk4, ulong*, global
        .arg chunk5, ulong*, global
        .arg seed, ulong*, global               # loaded in s[2:3]
        .arg out, ulong*, global                # loaded in s[4:5]
        .arg addresses, int*, global            # loaded in s[6:7]
        .arg threads_per_chunk, int             # loaded in s9
    .text
        s_mov_b32 m0, 0xffff

        v_mov_b32 v14, 512                      # magic numbers
        v_mov_b32 v39, 256
        # TODO v[25:28] are free, use those instead of v[60:63]

        v_mov_b32 v2, local_index&0xffffffff    # get local_index
        v_mov_b32 v3, local_index>>32           # local_index - higher part

        v_lshlrev_b32 v10, 2, v0
        vadd_u32 v2, vcc, v2, v10               # local_index = local_index + ID * 4
        vaddc_u32 v3, vcc, 0, v3, vcc
        flat_load_ubyte v40, v[2:3]             # load idx_0_0
        v_mov_b32 v10, 1
        vadd_u32 v2, vcc, v2, v10               # local_index = local_index + 1
        vaddc_u32 v3, vcc, 0, v3, vcc
        flat_load_ubyte v41, v[2:3]             # load idx_0_1
        vadd_u32 v2, vcc, v2, v10               # local_index = local_index + 1
        vaddc_u32 v3, vcc, 0, v3, vcc
        flat_load_ubyte v42, v[2:3]             # load idx_0_2
        vadd_u32 v2, vcc, v2, v10               # local_index = local_index + 1
        vaddc_u32 v3, vcc, 0, v3, vcc
        flat_load_ubyte v43, v[2:3]             # load idx_0_3

        v_mov_b32 v10, 125
        vadd_u32 v2, vcc, v2, v10               # local_index = local_index + 125
        vaddc_u32 v3, vcc, 0, v3, vcc
        flat_load_ubyte v44, v[2:3]             # load idx_1_0
        v_mov_b32 v10, 1
        vadd_u32 v2, vcc, v2, v10               # local_index = local_index + 1
        vaddc_u32 v3, vcc, 0, v3, vcc
        flat_load_ubyte v45, v[2:3]             # load idx_1_1
        vadd_u32 v2, vcc, v2, v10               # local_index = local_index + 1
        vaddc_u32 v3, vcc, 0, v3, vcc
        flat_load_ubyte v46, v[2:3]             # load idx_1_2
        vadd_u32 v2, vcc, v2, v10               # local_index = local_index + 1
        vaddc_u32 v3, vcc, 0, v3, vcc
        flat_load_ubyte v47, v[2:3]             # load idx_1_3

        v_mov_b32 v10, 125
        vadd_u32 v2, vcc, v2, v10               # local_index = local_index + 125
        vaddc_u32 v3, vcc, 0, v3, vcc
        flat_load_ubyte v48, v[2:3]             # load idx_2_0
        v_mov_b32 v10, 1
        vadd_u32 v2, vcc, v2, v10               # local_index = local_index + 1
        vaddc_u32 v3, vcc, 0, v3, vcc
        flat_load_ubyte v49, v[2:3]             # load idx_2_1
        vadd_u32 v2, vcc, v2, v10               # local_index = local_index + 1
        vaddc_u32 v3, vcc, 0, v3, vcc
        flat_load_ubyte v50, v[2:3]             # load idx_2_2
        vadd_u32 v2, vcc, v2, v10               # local_index = local_index + 1
        vaddc_u32 v3, vcc, 0, v3, vcc
        flat_load_ubyte v51, v[2:3]             # load idx_2_3

        v_mov_b32 v10, 125
        vadd_u32 v2, vcc, v2, v10               # local_index = local_index + 125
        vaddc_u32 v3, vcc, 0, v3, vcc
        flat_load_ubyte v52, v[2:3]             # load idx_3_0
        v_mov_b32 v10, 1
        vadd_u32 v2, vcc, v2, v10               # local_index = local_index + 1
        vaddc_u32 v3, vcc, 0, v3, vcc
        flat_load_ubyte v53, v[2:3]             # load idx_3_1
        vadd_u32 v2, vcc, v2, v10               # local_index = local_index + 1
        vaddc_u32 v3, vcc, 0, v3, vcc
        flat_load_ubyte v54, v[2:3]             # load idx_3_2
        vadd_u32 v2, vcc, v2, v10               # local_index = local_index + 1
        vaddc_u32 v3, vcc, 0, v3, vcc
        flat_load_ubyte v55, v[2:3]             # load idx_3_3

        v_mov_b32 v10, 126
        vadd_u32 v2, vcc, v2, v10               # local_index = local_index + 126 (skip one as is not needed)
        vaddc_u32 v3, vcc, 0, v3, vcc
        flat_load_ubyte v56, v[2:3]             # load idx_4_0
        v_mov_b32 v10, 1
        vadd_u32 v2, vcc, v2, v10               # local_index = local_index + 1
        vaddc_u32 v3, vcc, 0, v3, vcc
        flat_load_ubyte v57, v[2:3]             # load idx_4_1
        vadd_u32 v2, vcc, v2, v10               # local_index = local_index + 1
        vaddc_u32 v3, vcc, 0, v3, vcc
        flat_load_ubyte v58, v[2:3]             # load idx_4_2

        v_mov_b32 v10, 126
        vadd_u32 v2, vcc, v2, v10               # local_index = local_index + 126 (skip one as is not needed)
        vaddc_u32 v3, vcc, 0, v3, vcc
        flat_load_ubyte v59, v[2:3]             # load idx_5_0
        v_mov_b32 v10, 1
        vadd_u32 v2, vcc, v2, v10               # local_index = local_index + 1
        vaddc_u32 v3, vcc, 0, v3, vcc
        flat_load_ubyte v60, v[2:3]             # load idx_5_1
        vadd_u32 v2, vcc, v2, v10               # local_index = local_index + 1
        vaddc_u32 v3, vcc, 0, v3, vcc
        flat_load_ubyte v61, v[2:3]             # load idx_5_2

        s_load_dwordx2 s[2:3], s[6:7], 96       # load seed_ptr
        s_load_dwordx2 s[4:5], s[6:7], 104      # load out_ptr
        s_load_dword s9, s[6:7], 120            # load threads_per_chunk

        s_waitcnt lgkmcnt(0) & vmcnt(0)

        s_cmp_ge_u32 HASH, s9                   # chunk_index = HASH >= threads_per_chunk ? 1 : 0;
        s_cselect_b32 s12, 1, 0
        s_add_u32 s11, s9, s9
        s_cmp_ge_u32 HASH, s11                  # chunk_index = HASH >= threads_per_chunk*2 ? 2 : chunk_index;
        s_cselect_b32 s12, 2, s12
        s_add_u32 s11, s11, s9
        s_cmp_ge_u32 HASH, s11                  # chunk_index = HASH >= threads_per_chunk*3 ? 3 : chunk_index;
        s_cselect_b32 s12, 3, s12
        s_add_u32 s11, s11, s9
        s_cmp_ge_u32 HASH, s11                  # chunk_index = HASH >= threads_per_chunk*4 ? 4 : chunk_index;
        s_cselect_b32 s12, 4, s12
        s_add_u32 s11, s11, s9
        s_cmp_ge_u32 HASH, s11                  # chunk_index = HASH >= threads_per_chunk*5 ? 5 : chunk_index;
        s_cselect_b32 s12, 5, s12

        s_lshl_b32 s11, s12, 3                  # mem_ptr = s[6:7] (arg base) + 12 * 4 (kernel data) + chunk_index * 8 (64 bit addr)
        s_add_u32 s11, s11, 48
        s_load_dwordx2 s[0:1], s[6:7], s11      # load mem_ptr
        s_load_dwordx2 s[6:7], s[6:7], 112      # load addr_ptr

        s_mul_i32 s12, s9, s12                  # chunk_offset = threads_per_chunk * chunk_index;
        s_sub_u32 s12, HASH, s12                # chunk_offset = HASH - chunk_offset;
        s_mul_i32 s12, s12, 96995               # chunk_offset = chunk_offset * CBLOCKS_MEMSIZE(96995) * BLOCK_SIZE(1024)
        s_mov_b32 s13, 0
        s_lshl_b64 s[12:13], s[12:13], 10

        s_waitcnt lgkmcnt(0) & vmcnt(0)

        s_add_u32 s0, s0, s12                   # memory = memory + chunk_offset;
        s_addc_u32 s1, s1, s13

        s_lshl_b32 HASH, HASH, 11               # out_offset = HASH * 2 * BLOCK_SIZE

        s_add_u32 s2, s2, HASH                  # seed_ptr = seed_ptr + out_offset
        s_addc_u32 s3, s3, 0
        s_add_u32 s4, s4, HASH                  # out_ptr = out_ptr + out_offset
        s_addc_u32 s5, s5, 0

        v_lshlrev_b32 v40, 3, v40               # idx_0_0 -> idx_3_3 *= 8 (offsets to 64 bit numbers)
        v_lshlrev_b32 v41, 3, v41
        v_lshlrev_b32 v42, 3, v42
        v_lshlrev_b32 v43, 3, v43

        v_mov_b32 v2, s2                        # seed_addr = seed_ptr
        v_mov_b32 v3, s3

        vadd_u32 v4, vcc, v2, v40               # seed_addr += idx_0_0
        vaddc_u32 v5, vcc, 0, v3, vcc
        flat_load_dwordx2 v[6:7], v[4:5]        # load seed chunk (2 int at once)

        vadd_u32 v4, vcc, v2, v41               # seed_addr += idx_0_1
        vaddc_u32 v5, vcc, 0, v3, vcc
        flat_load_dwordx2 v[8:9], v[4:5]        # load seed chunk (2 int at once)

        vadd_u32 v4, vcc, v2, v42               # seed_addr += idx_0_2
        vaddc_u32 v5, vcc, 0, v3, vcc
        flat_load_dwordx2 v[10:11], v[4:5]      # load seed chunk (2 int at once)

        vadd_u32 v4, vcc, v2, v43               # seed_addr += idx_0_3
        vaddc_u32 v5, vcc, 0, v3, vcc
        flat_load_dwordx2 v[12:13], v[4:5]      # load seed chunk (2 int at once)

        v_lshlrev_b32 v44, 3, v44
        v_lshlrev_b32 v45, 3, v45
        v_lshlrev_b32 v46, 3, v46
        v_lshlrev_b32 v47, 3, v47
        v_lshlrev_b32 v48, 3, v48
        v_lshlrev_b32 v49, 3, v49
        v_lshlrev_b32 v50, 3, v50
        v_lshlrev_b32 v51, 3, v51
        v_lshlrev_b32 v52, 3, v52
        v_lshlrev_b32 v53, 3, v53
        v_lshlrev_b32 v54, 3, v54
        v_lshlrev_b32 v55, 3, v55
        v_lshlrev_b32 v56, 2, v56               # idx_4_0 -> idx_5_2 *= 4 (ds_bpermute_b32 address is lane id * 4 - amdgcn specs)
        v_lshlrev_b32 v57, 2, v57
        v_lshlrev_b32 v58, 2, v58
        v_lshlrev_b32 v59, 2, v59
        v_lshlrev_b32 v60, 2, v60
        v_lshlrev_b32 v61, 2, v61

        v_lshlrev_b32 v1, 4, v0                 # offset = id * 16

        s_waitcnt vmcnt(0)

        vadd_u32 v2, vcc, s0, v1                # mem_addr = mem_ptr + offset
        v_mov_b32 v3, s1
        vaddc_u32 v3, vcc, 0, v3, vcc
        flat_store_dwordx4 v[2:3], v[6:9]       # store seed chunk

        vadd_u32 v2, vcc, v2, v14               # mem_addr += 512
        vaddc_u32 v3, vcc, 0, v3, vcc
        flat_store_dwordx4 v[2:3], v[10:13]     # store seed chunk

        v_mov_b32 v2, 1024                      # seed_addr = seed_ptr + BLOCK_SIZE
        v_mov_b32 v3, s3
        vadd_u32 v2, vcc, v2, s2
        vaddc_u32 v3, vcc, 0, v3, vcc

        vadd_u32 v4, vcc, v2, v40               # seed_addr += idx_0_0
        vaddc_u32 v5, vcc, 0, v3, vcc
        flat_load_dwordx2 v[6:7], v[4:5]        # load seed chunk (2 int at once)

        vadd_u32 v4, vcc, v2, v41               # seed_addr += idx_0_1
        vaddc_u32 v5, vcc, 0, v3, vcc
        flat_load_dwordx2 v[8:9], v[4:5]        # load seed chunk (2 int at once)

        vadd_u32 v4, vcc, v2, v42               # seed_addr += idx_0_2
        vaddc_u32 v5, vcc, 0, v3, vcc
        flat_load_dwordx2 v[10:11], v[4:5]      # load seed chunk (2 int at once)

        vadd_u32 v4, vcc, v2, v43               # seed_addr += idx_0_3
        vaddc_u32 v5, vcc, 0, v3, vcc
        flat_load_dwordx2 v[12:13], v[4:5]      # load seed chunk (2 int at once)

        s_waitcnt vmcnt(0)

        v_mov_b32 v2, 1024                      # mem_addr = mem_ptr + BLOCK_SIZE + offset
		vadd_u32 v2, vcc, v2, v1
        vadd_u32 v2, vcc, s0, v2
        v_mov_b32 v3, s1
        vaddc_u32 v3, vcc, 0, v3, vcc
        flat_store_dwordx4 v[2:3], v[6:9]       # store seed chunk

        vadd_u32 v2, vcc, v2, v14               # mem_addr += 512
        vaddc_u32 v3, vcc, 0, v3, vcc
        flat_store_dwordx4 v[2:3], v[10:13]     # store seed chunk

        v_lshlrev_b32 v0, 3, v0                 # id = id * 8
        vadd_u32 v2, vcc, s6, v0                # addr_ptr = addr_ptr + id (load 32 refs at once, one ref = 2 integers, 8 bytes)
        v_mov_b32 v3, s7
        vaddc_u32 v3, vcc, 0, v3, vcc
        flat_load_dwordx2 v[37:38], v[2:3]      # load first refs batch

        s_waitcnt vmcnt(0)

        s_mov_b32 s12, 0                        # lane index for ref

        v_readlane_b32 s8, v37, s12             # store current ref to SALU
        v_readlane_b32 s9, v38, s12

        s_lshl_b32 s9, s9, 10                   # addr1 = addr1 * BLOCK_SIZE

        v_mov_b32 v62, s9                       # mem_addr = mem_ptr + addr1 + offset
		vadd_u32 v62, vcc, v62, v1
        vadd_u32 v62, vcc, s0, v62
        v_mov_b32 v63, s1
        vaddc_u32 v63, vcc, 0, v63, vcc
        flat_load_dwordx4 v[29:32], v[62:63]    # load mem chunk

        vadd_u32 v62, vcc, v62, v14             # mem_addr += 512
        vaddc_u32 v63, vcc, 0, v63, vcc
        flat_load_dwordx4 v[33:36], v[62:63]    # load mem chunk

        s_mov_b32 s10, 1                        # for(int i=1;i<=524286;i++) {

loop_cblocks:
        s_mov_b32 s11, s8                       # backup addr0
        s_add_u32 s12, s12, 1                   # increment lane index for ref

        s_cmp_eq_u32 s12, 32                    # if less than 32 skip refs batch load
        s_cbranch_scc0 skip_addr_load_cblocks

        vadd_u32 v2, vcc, v2, v39               # addr_ptr += 256 (next addr batch 32 * 2 * 4)
        vaddc_u32 v3, vcc, 0, v3, vcc
        flat_load_dwordx2 v[37:38], v[2:3]      # load refs batch
        s_mov_b32 s12, 0                        # reset lane index for ref

skip_addr_load_cblocks:
        s_waitcnt vmcnt(0)                      # load data

        v_xor_b32 v6, v6, v29                   # state = prev block ^ ref block
        v_xor_b32 v7, v7, v30
        v_xor_b32 v8, v8, v31
        v_xor_b32 v9, v9, v32
        v_xor_b32 v10, v10, v33
        v_xor_b32 v11, v11, v34
        v_xor_b32 v12, v12, v35
        v_xor_b32 v13, v13, v36

        v_mov_b32 v15, v6                       # tmp = state
        v_mov_b32 v16, v7
        v_mov_b32 v17, v8
        v_mov_b32 v18, v9
        v_mov_b32 v19, v10
        v_mov_b32 v20, v11
        v_mov_b32 v21, v12
        v_mov_b32 v22, v13

        s_cmp_eq_u32 s10, 524286                # if last ref, data has been preloaded in previous iteration and no other data is needed
        s_cbranch_scc1 skip_data_load_cblocks

        v_readlane_b32 s8, v37, s12             # get current addr0 and addr1
        v_readlane_b32 s9, v38, s12

        s_lshl_b32 s9, s9, 10                   # addr1 = addr1 * BLOCK_SIZE

        v_mov_b32 v62, s9                       # mem_addr = mem_ptr + addr1 + offset
		vadd_u32 v62, vcc, v62, v1
        vadd_u32 v62, vcc, s0, v62
        v_mov_b32 v63, s1
        vaddc_u32 v63, vcc, 0, v63, vcc
        flat_load_dwordx4 v[29:32], v[62:63]    # load mem chunk

        vadd_u32 v62, vcc, v62, v14             # mem_addr += 512
        vaddc_u32 v63, vcc, 0, v63, vcc
        flat_load_dwordx4 v[33:36], v[62:63]    # load mem chunk

skip_data_load_cblocks:
                                                # apply argon2 hash function to tmp
        v_mad_u64_u32 v[23:24], vcc, v15, v17, 0    # a = fBlaMka(a, b)
        vadd_u32 v23, vcc, v23, v23
        vaddc_u32 v24, vcc, v24, v24, vcc
        vadd_u32 v15, vcc, v15, v17
        vaddc_u32 v16, vcc, v16, v18, vcc
        vadd_u32 v15, vcc, v15, v23
        vaddc_u32 v16, vcc, v16, v24, vcc

        v_mov_b32 v23, v22                      # d = rotate(d ^ a, 32)
        v_xor_b32 v22, v21, v15
        v_xor_b32 v21, v23, v16

        v_mad_u64_u32 v[23:24], vcc, v19, v21, 0    # c = fBlaMka(c, d)
        vadd_u32 v23, vcc, v23, v23
        vaddc_u32 v24, vcc, v24, v24, vcc
        vadd_u32 v19, vcc, v19, v21
        vaddc_u32 v20, vcc, v20, v22, vcc
        vadd_u32 v19, vcc, v19, v23
        vaddc_u32 v20, vcc, v20, v24, vcc

        v_xor_b32 v23, v17, v19                 # b = rotate(b ^ c, 40)
        v_xor_b32 v24, v18, v20
        v_alignbit_b32 v18, v23, v24, 24
        v_alignbit_b32 v17, v24, v23, 24

        v_mad_u64_u32 v[23:24], vcc, v15, v17, 0    # a = fBlaMka(a, b)
        vadd_u32 v23, vcc, v23, v23
        vaddc_u32 v24, vcc, v24, v24, vcc
        vadd_u32 v15, vcc, v15, v17
        vaddc_u32 v16, vcc, v16, v18, vcc
        vadd_u32 v15, vcc, v15, v23
        vaddc_u32 v16, vcc, v16, v24, vcc

        v_xor_b32 v23, v15, v21                 # d = rotate(d ^ a, 48)
        v_xor_b32 v24, v16, v22
        v_alignbit_b32 v22, v23, v24, 16
        v_alignbit_b32 v21, v24, v23, 16

        v_mad_u64_u32 v[23:24], vcc, v19, v21, 0    # c = fBlaMka(c, d)
        vadd_u32 v23, vcc, v23, v23
        vaddc_u32 v24, vcc, v24, v24, vcc
        vadd_u32 v19, vcc, v19, v21
        vaddc_u32 v20, vcc, v20, v22, vcc
        vadd_u32 v19, vcc, v19, v23
        vaddc_u32 v20, vcc, v20, v24, vcc

        v_xor_b32 v23, v17, v19                 # b = rotate(b ^ c, 1)
        v_xor_b32 v24, v18, v20
        v_alignbit_b32 v18, v24, v23, 31
        v_alignbit_b32 v17, v23, v24, 31

        ds_bpermute_b32 v17, v56, v17           # first shuffle using cross lane data access
        ds_bpermute_b32 v18, v56, v18
        ds_bpermute_b32 v19, v57, v19
        ds_bpermute_b32 v20, v57, v20
        ds_bpermute_b32 v21, v58, v21
        ds_bpermute_b32 v22, v58, v22

        s_waitcnt lgkmcnt(0)

        v_mad_u64_u32 v[23:24], vcc, v15, v17, 0    # a = fBlaMka(a, b)
        vadd_u32 v23, vcc, v23, v23
        vaddc_u32 v24, vcc, v24, v24, vcc
        vadd_u32 v15, vcc, v15, v17
        vaddc_u32 v16, vcc, v16, v18, vcc
        vadd_u32 v15, vcc, v15, v23
        vaddc_u32 v16, vcc, v16, v24, vcc

        v_mov_b32 v23, v22                      # d = rotate(d ^ a, 32)
        v_xor_b32 v22, v21, v15
        v_xor_b32 v21, v23, v16

        v_mad_u64_u32 v[23:24], vcc, v19, v21, 0    # c = fBlaMka(c, d)
        vadd_u32 v23, vcc, v23, v23
        vaddc_u32 v24, vcc, v24, v24, vcc
        vadd_u32 v19, vcc, v19, v21
        vaddc_u32 v20, vcc, v20, v22, vcc
        vadd_u32 v19, vcc, v19, v23
        vaddc_u32 v20, vcc, v20, v24, vcc

        v_xor_b32 v23, v17, v19                 # b = rotate(b ^ c, 40)
        v_xor_b32 v24, v18, v20
        v_alignbit_b32 v18, v23, v24, 24
        v_alignbit_b32 v17, v24, v23, 24

        v_mad_u64_u32 v[23:24], vcc, v15, v17, 0    # a = fBlaMka(a, b)
        vadd_u32 v23, vcc, v23, v23
        vaddc_u32 v24, vcc, v24, v24, vcc
        vadd_u32 v15, vcc, v15, v17
        vaddc_u32 v16, vcc, v16, v18, vcc
        vadd_u32 v15, vcc, v15, v23
        vaddc_u32 v16, vcc, v16, v24, vcc

        v_xor_b32 v23, v15, v21                 # d = rotate(d ^ a, 48)
        v_xor_b32 v24, v16, v22
        v_alignbit_b32 v22, v23, v24, 16
        v_alignbit_b32 v21, v24, v23, 16

        v_mad_u64_u32 v[23:24], vcc, v19, v21, 0    # c = fBlaMka(c, d)
        vadd_u32 v23, vcc, v23, v23
        vaddc_u32 v24, vcc, v24, v24, vcc
        vadd_u32 v19, vcc, v19, v21
        vaddc_u32 v20, vcc, v20, v22, vcc
        vadd_u32 v19, vcc, v19, v23
        vaddc_u32 v20, vcc, v20, v24, vcc

        v_xor_b32 v23, v17, v19                 # b = rotate(b ^ c, 1)
        v_xor_b32 v24, v18, v20
        v_alignbit_b32 v18, v24, v23, 31
        v_alignbit_b32 v17, v23, v24, 31

        ds_write_b64 v44, v[15:16]              # second shuffle using local storage
        ds_write_b64 v45, v[17:18]
        ds_write_b64 v46, v[19:20]
        ds_write_b64 v47, v[21:22]

        ds_read_b64 v[15:16], v48
        ds_read_b64 v[17:18], v49
        ds_read_b64 v[19:20], v50
        ds_read_b64 v[21:22], v51

        s_waitcnt lgkmcnt(0)

        v_mad_u64_u32 v[23:24], vcc, v15, v17, 0    # a = fBlaMka(a, b)
        vadd_u32 v23, vcc, v23, v23
        vaddc_u32 v24, vcc, v24, v24, vcc
        vadd_u32 v15, vcc, v15, v17
        vaddc_u32 v16, vcc, v16, v18, vcc
        vadd_u32 v15, vcc, v15, v23
        vaddc_u32 v16, vcc, v16, v24, vcc

        v_mov_b32 v23, v22                      # d = rotate(d ^ a, 32)
        v_xor_b32 v22, v21, v15
        v_xor_b32 v21, v23, v16

        v_mad_u64_u32 v[23:24], vcc, v19, v21, 0    # c = fBlaMka(c, d)
        vadd_u32 v23, vcc, v23, v23
        vaddc_u32 v24, vcc, v24, v24, vcc
        vadd_u32 v19, vcc, v19, v21
        vaddc_u32 v20, vcc, v20, v22, vcc
        vadd_u32 v19, vcc, v19, v23
        vaddc_u32 v20, vcc, v20, v24, vcc

        v_xor_b32 v23, v17, v19                 # b = rotate(b ^ c, 40)
        v_xor_b32 v24, v18, v20
        v_alignbit_b32 v18, v23, v24, 24
        v_alignbit_b32 v17, v24, v23, 24

        v_mad_u64_u32 v[23:24], vcc, v15, v17, 0    # a = fBlaMka(a, b)
        vadd_u32 v23, vcc, v23, v23
        vaddc_u32 v24, vcc, v24, v24, vcc
        vadd_u32 v15, vcc, v15, v17
        vaddc_u32 v16, vcc, v16, v18, vcc
        vadd_u32 v15, vcc, v15, v23
        vaddc_u32 v16, vcc, v16, v24, vcc

        v_xor_b32 v23, v15, v21                 # d = rotate(d ^ a, 48)
        v_xor_b32 v24, v16, v22
        v_alignbit_b32 v22, v23, v24, 16
        v_alignbit_b32 v21, v24, v23, 16

        v_mad_u64_u32 v[23:24], vcc, v19, v21, 0    # c = fBlaMka(c, d)
        vadd_u32 v23, vcc, v23, v23
        vaddc_u32 v24, vcc, v24, v24, vcc
        vadd_u32 v19, vcc, v19, v21
        vaddc_u32 v20, vcc, v20, v22, vcc
        vadd_u32 v19, vcc, v19, v23
        vaddc_u32 v20, vcc, v20, v24, vcc

        v_xor_b32 v23, v17, v19                 # b = rotate(b ^ c, 1)
        v_xor_b32 v24, v18, v20
        v_alignbit_b32 v18, v24, v23, 31
        v_alignbit_b32 v17, v23, v24, 31

        ds_bpermute_b32 v17, v59, v17           # third shuffle using cross lane data access
        ds_bpermute_b32 v18, v59, v18
        ds_bpermute_b32 v19, v60, v19
        ds_bpermute_b32 v20, v60, v20
        ds_bpermute_b32 v21, v61, v21
        ds_bpermute_b32 v22, v61, v22

        s_waitcnt lgkmcnt(0)

        v_mad_u64_u32 v[23:24], vcc, v15, v17, 0    # a = fBlaMka(a, b)
        vadd_u32 v23, vcc, v23, v23
        vaddc_u32 v24, vcc, v24, v24, vcc
        vadd_u32 v15, vcc, v15, v17
        vaddc_u32 v16, vcc, v16, v18, vcc
        vadd_u32 v15, vcc, v15, v23
        vaddc_u32 v16, vcc, v16, v24, vcc

        v_mov_b32 v23, v22                      # d = rotate(d ^ a, 32)
        v_xor_b32 v22, v21, v15
        v_xor_b32 v21, v23, v16

        v_mad_u64_u32 v[23:24], vcc, v19, v21, 0    # c = fBlaMka(c, d)
        vadd_u32 v23, vcc, v23, v23
        vaddc_u32 v24, vcc, v24, v24, vcc
        vadd_u32 v19, vcc, v19, v21
        vaddc_u32 v20, vcc, v20, v22, vcc
        vadd_u32 v19, vcc, v19, v23
        vaddc_u32 v20, vcc, v20, v24, vcc

        v_xor_b32 v23, v17, v19                 # b = rotate(b ^ c, 40)
        v_xor_b32 v24, v18, v20
        v_alignbit_b32 v18, v23, v24, 24
        v_alignbit_b32 v17, v24, v23, 24

        v_mad_u64_u32 v[23:24], vcc, v15, v17, 0    # a = fBlaMka(a, b)
        vadd_u32 v23, vcc, v23, v23
        vaddc_u32 v24, vcc, v24, v24, vcc
        vadd_u32 v15, vcc, v15, v17
        vaddc_u32 v16, vcc, v16, v18, vcc
        vadd_u32 v15, vcc, v15, v23
        vaddc_u32 v16, vcc, v16, v24, vcc

        v_xor_b32 v23, v15, v21                 # d = rotate(d ^ a, 48)
        v_xor_b32 v24, v16, v22
        v_alignbit_b32 v22, v23, v24, 16
        v_alignbit_b32 v21, v24, v23, 16

        v_mad_u64_u32 v[23:24], vcc, v19, v21, 0    # c = fBlaMka(c, d)
        vadd_u32 v23, vcc, v23, v23
        vaddc_u32 v24, vcc, v24, v24, vcc
        vadd_u32 v19, vcc, v19, v21
        vaddc_u32 v20, vcc, v20, v22, vcc
        vadd_u32 v19, vcc, v19, v23
        vaddc_u32 v20, vcc, v20, v24, vcc

        v_xor_b32 v23, v17, v19                 # b = rotate(b ^ c, 1)
        v_xor_b32 v24, v18, v20
        v_alignbit_b32 v18, v24, v23, 31
        v_alignbit_b32 v17, v23, v24, 31

        ds_write_b64 v52, v[15:16]              # reorder to original shuffle using local storage
        ds_write_b64 v53, v[17:18]
        ds_write_b64 v54, v[19:20]
        ds_write_b64 v55, v[21:22]

        ds_read_b64 v[15:16], v40
        ds_read_b64 v[17:18], v41
        ds_read_b64 v[19:20], v42
        ds_read_b64 v[21:22], v43

        s_waitcnt lgkmcnt(0)

        v_xor_b32 v6, v6, v15                   # state ^= tmp
        v_xor_b32 v7, v7, v16
        v_xor_b32 v8, v8, v17
        v_xor_b32 v9, v9, v18
        v_xor_b32 v10, v10, v19
        v_xor_b32 v11, v11, v20
        v_xor_b32 v12, v12, v21
        v_xor_b32 v13, v13, v22

        s_cmp_eq_i32 s11, -1                    # if(addr0 != -1) store state
        s_cbranch_scc1 skip_blk_store_cblocks

        s_lshl_b32 s11, s11, 10                 # addr0 = addr0 * BLOCK_SIZE

        v_mov_b32 v62, s11                      # mem_addr = mem_ptr + addr0 + offset
		vadd_u32 v62, vcc, v62, v1
        vadd_u32 v62, vcc, s0, v62
        v_mov_b32 v63, s1
        vaddc_u32 v63, vcc, 0, v63, vcc
        flat_store_dwordx4 v[62:63], v[6:9]     # store mem chunk

        vadd_u32 v62, vcc, v62, v14             # mem_addr += 512
        vaddc_u32 v63, vcc, 0, v63, vcc
        flat_store_dwordx4 v[62:63], v[10:13]   # store mem chunk

skip_blk_store_cblocks:
        s_add_u32 s10, s10, 1                   # i++
        s_cmp_le_u32 s10, 524286                # } endfor
        s_cbranch_scc1 loop_cblocks

        v_mov_b32 v62, s4                       # out_addr = out_ptr
        v_mov_b32 v63, s5

        vadd_u32 v4, vcc, v62, v40              # out_addr += idx_0_0
        vaddc_u32 v5, vcc, 0, v63, vcc
        flat_store_dwordx2 v[4:5], v[6:7]       # store out chunk (2 int at once)

        vadd_u32 v4, vcc, v62, v41              # out_addr += idx_0_1
        vaddc_u32 v5, vcc, 0, v63, vcc
        flat_store_dwordx2 v[4:5], v[8:9]       # store out chunk (2 int at once)

        vadd_u32 v4, vcc, v62, v42              # out_addr += idx_0_2
        vaddc_u32 v5, vcc, 0, v63, vcc
        flat_store_dwordx2 v[4:5], v[10:11]     # store out chunk (2 int at once)

        vadd_u32 v4, vcc, v62, v43              # out_addr += idx_0_3
        vaddc_u32 v5, vcc, 0, v63, vcc
        flat_store_dwordx2 v[4:5], v[12:13]     # store out chunk (2 int at once)

        s_endpgm

.kernel fill_gblocks
    .config
        .dims x
        .localsize 4096
        .useargs
        .usesetup
        .setupargs
        .arg chunk0, ulong*, global             # loaded in s[0:1]
        .arg chunk1, ulong*, global
        .arg chunk2, ulong*, global
        .arg chunk3, ulong*, global
        .arg chunk4, ulong*, global
        .arg chunk5, ulong*, global
        .arg seed, ulong*, global               # loaded in s[2:3]
        .arg out, ulong*, global                # loaded in s[4:5]
        .arg addresses, int*, global            # loaded in s[6:7]
        .arg threads_per_chunk, int             # loaded in s9
    .text
        s_mov_b32 m0, 0xFFFF

        v_lshrrev_b32 v1, 5, v0                 # seg = id / 32
        v_and_b32 v0, v0, 31                    # id = id % 32

        s_load_dword s9, s[6:7], 120            # load threads_per_chunk
        s_load_dwordx2 s[2:3], s[6:7], 96       # load seed_ptr
        s_load_dwordx2 s[4:5], s[6:7], 104      # load out_ptr

        v_mov_b32 v54, local_index&0xffffffff   # get local_index
        v_mov_b32 v55, local_index>>32          # local_index - higher part

        v_lshlrev_b32 v56, 2, v0                # local_index = local_index + id * 4
        vadd_u32 v54, vcc, v54, v56
        vaddc_u32 v55, vcc, 0, v55, vcc

        v_mov_b32 v56, 1
        v_mov_b32 v57, 125

        flat_load_ubyte v32, v[54:55]           # load idx_0_0
        vadd_u32 v54, vcc, v54, v56             # local_index = local_index + 1
        vaddc_u32 v55, vcc, 0, v55, vcc
        flat_load_ubyte v33, v[54:55]           # load idx_0_1
        vadd_u32 v54, vcc, v54, v56             # local_index = local_index + 1
        vaddc_u32 v55, vcc, 0, v55, vcc
        flat_load_ubyte v34, v[54:55]           # load idx_0_2
        vadd_u32 v54, vcc, v54, v56             # local_index = local_index + 1
        vaddc_u32 v55, vcc, 0, v55, vcc
        flat_load_ubyte v35, v[54:55]           # load idx_0_3

        vadd_u32 v54, vcc, v54, v57             # local_index = local_index + 125
        vaddc_u32 v55, vcc, 0, v55, vcc
        flat_load_ubyte v36, v[54:55]           # load idx_1_0
        vadd_u32 v54, vcc, v54, v56             # local_index = local_index + 1
        vaddc_u32 v55, vcc, 0, v55, vcc
        flat_load_ubyte v37, v[54:55]           # load idx_1_1
        vadd_u32 v54, vcc, v54, v56             # local_index = local_index + 1
        vaddc_u32 v55, vcc, 0, v55, vcc
        flat_load_ubyte v38, v[54:55]           # load idx_1_2
        vadd_u32 v54, vcc, v54, v56             # local_index = local_index + 1
        vaddc_u32 v55, vcc, 0, v55, vcc
        flat_load_ubyte v39, v[54:55]           # load idx_1_3

        vadd_u32 v54, vcc, v54, v57             # local_index = local_index + 125
        vaddc_u32 v55, vcc, 0, v55, vcc
        flat_load_ubyte v40, v[54:55]           # load idx_2_0
        vadd_u32 v54, vcc, v54, v56             # local_index = local_index + 1
        vaddc_u32 v55, vcc, 0, v55, vcc
        flat_load_ubyte v41, v[54:55]           # load idx_2_1
        vadd_u32 v54, vcc, v54, v56             # local_index = local_index + 1
        vaddc_u32 v55, vcc, 0, v55, vcc
        flat_load_ubyte v42, v[54:55]           # load idx_2_2
        vadd_u32 v54, vcc, v54, v56             # local_index = local_index + 1
        vaddc_u32 v55, vcc, 0, v55, vcc
        flat_load_ubyte v43, v[54:55]           # load idx_2_3

        vadd_u32 v54, vcc, v54, v57             # local_index = local_index + 125
        vaddc_u32 v55, vcc, 0, v55, vcc
        flat_load_ubyte v44, v[54:55]           # load idx_3_0
        vadd_u32 v54, vcc, v54, v56             # local_index = local_index + 1
        vaddc_u32 v55, vcc, 0, v55, vcc
        flat_load_ubyte v45, v[54:55]           # load idx_3_1
        vadd_u32 v54, vcc, v54, v56             # local_index = local_index + 1
        vaddc_u32 v55, vcc, 0, v55, vcc
        flat_load_ubyte v46, v[54:55]           # load idx_3_2
        vadd_u32 v54, vcc, v54, v56             # local_index = local_index + 1
        vaddc_u32 v55, vcc, 0, v55, vcc
        flat_load_ubyte v47, v[54:55]           # load idx_3_3

        v_mov_b32 v57, 126

        vadd_u32 v54, vcc, v54, v57             # local_index = local_index + 126 (skip one as is not needed)
        vaddc_u32 v55, vcc, 0, v55, vcc
        flat_load_ubyte v48, v[54:55]           # load idx_4_0
        vadd_u32 v54, vcc, v54, v56             # local_index = local_index + 1
        vaddc_u32 v55, vcc, 0, v55, vcc
        flat_load_ubyte v49, v[54:55]           # load idx_4_1
        vadd_u32 v54, vcc, v54, v56             # local_index = local_index + 1
        vaddc_u32 v55, vcc, 0, v55, vcc
        flat_load_ubyte v50, v[54:55]           # load idx_4_2

        vadd_u32 v54, vcc, v54, v57             # local_index = local_index + 126 (skip one as is not needed)
        vaddc_u32 v55, vcc, 0, v55, vcc
        flat_load_ubyte v51, v[54:55]           # load idx_5_0
        vadd_u32 v54, vcc, v54, v56             # local_index = local_index + 1
        vaddc_u32 v55, vcc, 0, v55, vcc
        flat_load_ubyte v52, v[54:55]           # load idx_5_1
        vadd_u32 v54, vcc, v54, v56             # local_index = local_index + 1
        vaddc_u32 v55, vcc, 0, v55, vcc
        flat_load_ubyte v53, v[54:55]           # load idx_5_2

        s_waitcnt lgkmcnt(0) & vmcnt(0)

        s_cmp_ge_u32 HASH, s9                   # chunk_index = HASH >= threads_per_chunk ? 1 : 0;
        s_cselect_b32 s12, 1, 0
        s_add_u32 s11, s9, s9
        s_cmp_ge_u32 HASH, s11                  # chunk_index = HASH >= threads_per_chunk*2 ? 2 : chunk_index;
        s_cselect_b32 s12, 2, s12
        s_add_u32 s11, s11, s9
        s_cmp_ge_u32 HASH, s11                  # chunk_index = HASH >= threads_per_chunk*3 ? 3 : chunk_index;
        s_cselect_b32 s12, 3, s12
        s_add_u32 s11, s11, s9
        s_cmp_ge_u32 HASH, s11                  # chunk_index = HASH >= threads_per_chunk*4 ? 4 : chunk_index;
        s_cselect_b32 s12, 4, s12
        s_add_u32 s11, s11, s9
        s_cmp_ge_u32 HASH, s11                  # chunk_index = HASH >= threads_per_chunk*5 ? 5 : chunk_index;
        s_cselect_b32 s12, 5, s12

        s_lshl_b32 s11, s12, 3                  # mem_ptr = s[6:7] (arg base) + 12 * 4 (kernel data) + chunk_index * 8 (64 bit addr)
        s_add_u32 s11, s11, 48
        s_load_dwordx2 s[0:1], s[6:7], s11      # load mem_ptr
        s_load_dwordx2 s[6:7], s[6:7], 112      # load addr_ptr

        s_mul_i32 s12, s9, s12                  # chunk_offset = threads_per_chunk * chunk_index;
        s_sub_u32 s12, HASH, s12                # chunk_offset = HASH - chunk_offset;
        s_mov_b32 s13, 0                        # chunk_offset = chunk_offset * GBLOCKS_MEMSIZE(16384) * BLOCK_SIZE(1024) -> shift left 24 bits
        s_lshl_b64 s[12:13], s[12:13], 24

        s_lshl_b32 HASH, HASH, 13               # out_offset = HASH * 8 * BLOCK_SIZE

        s_add_u32 s2, s2, HASH                  # seed_ptr = seed_ptr + out_offset
        s_addc_u32 s3, s3, 0
        s_add_u32 s4, s4, HASH                  # out_ptr = out_ptr + out_offset
        s_addc_u32 s5, s5, 0

        v_lshlrev_b32 v54, 11, v1               # seg_offset = segment * 2 * BLOCK_SIZE

        v_lshlrev_b32 v32, 3, v32               # idx_0_0 -> idx_0_3 *= 8 (offsets to 64 bit numbers)
        v_lshlrev_b32 v33, 3, v33
        v_lshlrev_b32 v34, 3, v34
        v_lshlrev_b32 v35, 3, v35

        vadd_u32 v54, vcc, v54, s2              # seed_addr = seed_ptr + seg_offset
        v_mov_b32 v55, s3
        vaddc_u32 v55, vcc, 0, v55, vcc

        vadd_u32 v56, vcc, v54, v32             # seed_addr += idx_0_0
        vaddc_u32 v57, vcc, 0, v55, vcc
        flat_load_dwordx2 v[8:9], v[56:57] glc     # load seed chunk from idx_0_0

        vadd_u32 v56, vcc, v54, v33             # seed_addr += idx_0_1
        vaddc_u32 v57, vcc, 0, v55, vcc
        flat_load_dwordx2 v[10:11], v[56:57] glc   # load seed chunk from idx_0_1

        vadd_u32 v56, vcc, v54, v34             # seed_addr += idx_0_2
        vaddc_u32 v57, vcc, 0, v55, vcc
        flat_load_dwordx2 v[12:13], v[56:57] glc   # load seed chunk from idx_0_2

        vadd_u32 v56, vcc, v54, v35             # seed_addr += idx_0_3
        vaddc_u32 v57, vcc, 0, v55, vcc
        flat_load_dwordx2 v[14:15], v[56:57] glc   # load seed chunk from idx_0_3

        v_mov_b32 v56, 1024
        vadd_u32 v54, vcc, v54, v56             # seed_addr += BLOCK_SIZE
        vaddc_u32 v55, vcc, 0, v55, vcc

        vadd_u32 v56, vcc, v54, v32             # seed_addr += idx_0_0
        vaddc_u32 v57, vcc, 0, v55, vcc
        flat_load_dwordx2 v[16:17], v[56:57] glc   # load seed chunk from idx_0_0

        vadd_u32 v56, vcc, v54, v33             # seed_addr += idx_0_1
        vaddc_u32 v57, vcc, 0, v55, vcc
        flat_load_dwordx2 v[18:19], v[56:57] glc   # load seed chunk from idx_0_1

        vadd_u32 v56, vcc, v54, v34             # seed_addr += idx_0_2
        vaddc_u32 v57, vcc, 0, v55, vcc
        flat_load_dwordx2 v[20:21], v[56:57] glc   # load seed chunk from idx_0_2

        vadd_u32 v56, vcc, v54, v35             # seed_addr += idx_0_3
        vaddc_u32 v57, vcc, 0, v55, vcc
        flat_load_dwordx2 v[22:23], v[56:57] glc   # load seed chunk from idx_0_3

        v_lshlrev_b32 v36, 3, v36               # idx_1_0 -> idx_3_3 *= 8 (offsets to 64 bit numbers)
        v_lshlrev_b32 v37, 3, v37
        v_lshlrev_b32 v38, 3, v38
        v_lshlrev_b32 v39, 3, v39
        v_lshlrev_b32 v40, 3, v40
        v_lshlrev_b32 v41, 3, v41
        v_lshlrev_b32 v42, 3, v42
        v_lshlrev_b32 v43, 3, v43
        v_lshlrev_b32 v44, 3, v44
        v_lshlrev_b32 v45, 3, v45
        v_lshlrev_b32 v46, 3, v46
        v_lshlrev_b32 v47, 3, v47

        v_lshlrev_b32 v48, 2, v48               # idx_4_0 -> idx_5_2 *= 4 (ds_bpermute_b32 address is lane id * 4 - amdgcn specs)
        v_lshlrev_b32 v49, 2, v49
        v_lshlrev_b32 v50, 2, v50
        v_lshlrev_b32 v51, 2, v51
        v_lshlrev_b32 v52, 2, v52
        v_lshlrev_b32 v53, 2, v53

        v_and_b32 v62, v1, 1                    # wf_idx = seg % 2
        v_lshlrev_b32 v62, 7, v62               # wf_idx *= 128 (it will be 0 for seg 0 and 2, and 128 for seg 1 and 3)

        vadd_u32 v48, vcc, v48, v62             # idx_4_0 -> idx_5_2 += wf_idx
        vadd_u32 v49, vcc, v49, v62
        vadd_u32 v50, vcc, v50, v62
        vadd_u32 v51, vcc, v51, v62
        vadd_u32 v52, vcc, v52, v62
        vadd_u32 v53, vcc, v53, v62

        v_lshlrev_b32 v54, 10, v1               # lds_offset = segment * BLOCK_SIZE

        vadd_u32 v32, vcc, v32, v54             # idx_0_0 -> idx_3_3 += lds_offset
        vadd_u32 v33, vcc, v33, v54
        vadd_u32 v34, vcc, v34, v54
        vadd_u32 v35, vcc, v35, v54
        vadd_u32 v36, vcc, v36, v54
        vadd_u32 v37, vcc, v37, v54
        vadd_u32 v38, vcc, v38, v54
        vadd_u32 v39, vcc, v39, v54
        vadd_u32 v40, vcc, v40, v54
        vadd_u32 v41, vcc, v41, v54
        vadd_u32 v42, vcc, v42, v54
        vadd_u32 v43, vcc, v43, v54
        vadd_u32 v44, vcc, v44, v54
        vadd_u32 v45, vcc, v45, v54
        vadd_u32 v46, vcc, v46, v54
        vadd_u32 v47, vcc, v47, v54

        v_lshlrev_b32 v54, 22, v1               # mem_offset = segment * 4096 * BLOCK_SIZE
        v_lshlrev_b32 v61, 5, v0                # offset = id * 32

        v_mov_b32 v56, 16
        v_mov_b32 v57, 1008

        s_waitcnt vmcnt(0) & lgkmcnt(0)

        s_add_u32 s0, s0, s12                   # memory = memory + chunk_offset;
        s_addc_u32 s1, s1, s13

        vadd_u32 v54, vcc, v54, v61             # mem_addr = mem_ptr + mem_offset + offset
        vadd_u32 v54, vcc, v54, s0
        v_mov_b32 v55, s1
        vaddc_u32 v55, vcc, 0, v55, vcc
        flat_store_dwordx4 v[54:55], v[8:11]    # store seed chunk

        vadd_u32 v54, vcc, v54, v56             # mem_addr += 16
        vaddc_u32 v55, vcc, 0, v55, vcc
        flat_store_dwordx4 v[54:55], v[12:15]   # store seed chunk

        vadd_u32 v54, vcc, v54, v57             # mem_addr += 1008
        vaddc_u32 v55, vcc, 0, v55, vcc
        flat_store_dwordx4 v[54:55], v[16:19]   # store seed chunk

        vadd_u32 v54, vcc, v54, v56             # mem_addr += 16
        vaddc_u32 v55, vcc, 0, v55, vcc
        flat_store_dwordx4 v[54:55], v[20:23]   # store seed chunk

        v_mov_b32 v54, 768                      # seg_offset = 768 + seg * 64
        v_lshlrev_b32 v55, 6, v1
        vadd_u32 v63, vcc, v54, v55

        s_mov_b32 s14, 1022                     # ref_count_limit = 1022
        s_mov_b32 s13, 0                        # for(s=0;s<64;s++)

loop_gblocks_segments:

        v_mov_b32 v54, local_index&0xffffffff   # get local_index
        v_mov_b32 v55, local_index>>32          # local_index - higher part
        vadd_u32 v56, vcc, v63, s13             # seg_index = local_index + seg_offset + s
        vadd_u32 v54, vcc, v54, v56
        vaddc_u32 v55, vcc, 0, v55, vcc

        flat_load_ushort v56, v[54:55]          # load addr_start
        v_mov_b32 v58, 2
        vadd_u32 v54, vcc, v54, v58             # seg_ptr += 2
        vaddc_u32 v55, vcc, 0, v55, vcc
        flat_load_ushort v57, v[54:55]          # load prev_block

        s_waitcnt vmcnt(0) & lgkmcnt(0)

        v_lshlrev_b32 v56, 2, v56               # addr_start *= 4
        v_lshlrev_b32 v57, 10, v57              # prev_block *= BLOCK_SIZE

        vadd_u32 v57, vcc, v57, v61             # mem_addr = mem_ptr + prev_block + offset
        vadd_u32 v58, vcc, v57, s0
        v_mov_b32 v59, s1
        vaddc_u32 v59, vcc, 0, v59, vcc
        flat_load_dwordx4 v[0:3], v[58:59] glc      # load mem chunk (4 int at once)

        v_mov_b32 v57, 16
        vadd_u32 v58, vcc, v58, v57             # mem_addr += 16
        vaddc_u32 v59, vcc, 0, v59, vcc
        flat_load_dwordx4 v[4:7], v[58:59] glc      # load mem chunk (4 int at once)

 #       v_lshrrev_b32 v57, 3, v61               # id * 4 = offset / 8
 #       vadd_u32 v56, vcc, v56, v57             # addr_ptr = addr_ptr + addr_start + id * 4 (load 32 refs at once, one ref = 1 integer, 4 bytes)
        vadd_u32 v56, vcc, v56, s6
        v_mov_b32 v57, s7
        vaddc_u32 v57, vcc, 0, v57, vcc
#        flat_load_dword v58, v[56:57]           # load first refs batch
        flat_load_dword v59, v[56:57]           # load first refs batch

        s_waitcnt vmcnt(0)

#        ds_bpermute_b32 v59, v62, v58           # read ref to current lane

#        s_waitcnt lgkmcnt(0)

        v_lshrrev_b32 v28, 16, v59              # addr1 = ref >> 16
        v_lshlrev_b32 v28, 10, v28              # addr1 = addr1 * BLOCK_SIZE

        vadd_u32 v28, vcc, v28, v61             # mem_addr = mem_ptr + addr1 + offset
        vadd_u32 v28, vcc, v28, s0
        v_mov_b32 v29, s1
        vaddc_u32 v29, vcc, 0, v29, vcc
        flat_load_dwordx4 v[16:19], v[28:29] glc    # load mem chunk (4 int at once)

        v_mov_b32 v30, 16
        vadd_u32 v28, vcc, v28, v30             # mem_addr += 16
        vaddc_u32 v29, vcc, 0, v29, vcc
        flat_load_dwordx4 v[20:23], v[28:29] glc    # load mem chunk (4 int at once)

        s_cmp_lt_u32 s13, 16
        s_cbranch_scc1 skip_nextblk_load_gblock1

        v_mov_b32 v31, 0x7FFF
        v_and_b32 v31, v59, v31                 # addr0 = ref & 0x7FFF (keep only last 15 bits, bit 16 is keep block bit)
        v_lshlrev_b32 v31, 10, v31              # addr0 = addr0 * BLOCK_SIZE

        vadd_u32 v28, vcc, v31, v61             # mem_addr = mem_ptr + addr0 + offset
        vadd_u32 v28, vcc, v28, s0
        v_mov_b32 v29, s1
        vaddc_u32 v29, vcc, 0, v29, vcc
        flat_load_dwordx4 v[24:27], v[28:29] glc    # load mem chunk (4 int at once)

        vadd_u32 v28, vcc, v28, v30             # mem_addr += 16
        vaddc_u32 v29, vcc, 0, v29, vcc
        flat_load_dwordx4 v[28:31], v[28:29] glc    # load mem chunk (4 int at once)

skip_nextblk_load_gblock1:

#        s_mov_b32 s12, 0                        # lane index for ref
        s_mov_b32 s10, 1                        # for(int i=1;i<=ref_count_limit;i++) {

loop_gblocks:

        v_mov_b32 v60, v59                      # backup ref
#        s_add_u32 s12, s12, 4                   # increment lane index for ref

#        s_cmp_eq_u32 s12, 128                   # if less than 128 skip refs batch load
#        s_cbranch_scc0 skip_addr_load_gblocks

#        v_mov_b32 v54, 128
        v_mov_b32 v54, 4 # todo replace
        vadd_u32 v56, vcc, v56, v54             # addr_ptr += 128 (next addr batch 32 * 4)
        vaddc_u32 v57, vcc, 0, v57, vcc
#        flat_load_dword v58, v[56:57]           # load refs batch
        flat_load_dword v59, v[56:57] glc          # load refs batch
#        s_mov_b32 s12, 0                        # reset lane index for ref

#skip_addr_load_gblocks:

        s_waitcnt vmcnt(0)                      # load data

        v_xor_b32 v0, v0, v16                   # state = prev block ^ ref block
        v_xor_b32 v1, v1, v17
        v_xor_b32 v2, v2, v18
        v_xor_b32 v3, v3, v19
        v_xor_b32 v4, v4, v20
        v_xor_b32 v5, v5, v21
        v_xor_b32 v6, v6, v22
        v_xor_b32 v7, v7, v23

        v_mov_b32 v8, v0                        # tmp = state
        v_mov_b32 v9, v1
        v_mov_b32 v10, v2
        v_mov_b32 v11, v3
        v_mov_b32 v12, v4
        v_mov_b32 v13, v5
        v_mov_b32 v14, v6
        v_mov_b32 v15, v7

        s_cmp_lt_u32 s13, 16
        s_cbranch_scc1 skip_nextblk_xor_gblocks

        v_xor_b32 v0, v0, v24                   # state = state ^ next block
        v_xor_b32 v1, v1, v25
        v_xor_b32 v2, v2, v26
        v_xor_b32 v3, v3, v27
        v_xor_b32 v4, v4, v28
        v_xor_b32 v5, v5, v29
        v_xor_b32 v6, v6, v30
        v_xor_b32 v7, v7, v31

skip_nextblk_xor_gblocks:

        s_cmp_eq_u32 s10, s14                   # if last ref, data has been preloaded in previous iteration and no other data is needed
        s_cbranch_scc1 skip_data_load_gblocks

#        vadd_u32 v31, vcc, v62, s12             # addr_lane = wf_idx + lane_index
#        ds_bpermute_b32 v59, v31, v58           # read ref to current lane

#        s_waitcnt lgkmcnt(0)

        v_lshrrev_b32 v28, 16, v59              # addr1 = ref >> 16
        v_lshlrev_b32 v28, 10, v28              # addr1 = addr1 * BLOCK_SIZE

        vadd_u32 v28, vcc, v28, v61             # mem_addr = mem_ptr + addr1 + offset
        vadd_u32 v28, vcc, v28, s0
        v_mov_b32 v29, s1
        vaddc_u32 v29, vcc, 0, v29, vcc
        flat_load_dwordx4 v[16:19], v[28:29] glc   # load mem chunk (4 int at once)

        v_mov_b32 v30, 16
        vadd_u32 v28, vcc, v28, v30             # mem_addr += 16
        vaddc_u32 v29, vcc, 0, v29, vcc
        flat_load_dwordx4 v[20:23], v[28:29] glc   # load mem chunk (4 int at once)

        s_cmp_lt_u32 s13, 16
        s_cbranch_scc1 skip_data_load_gblocks

        v_mov_b32 v31, 0x7FFF
        v_and_b32 v31, v59, v31                 # addr0 = ref & 0x7FFF
        v_lshlrev_b32 v31, 10, v31              # addr0 = addr0 * BLOCK_SIZE

        vadd_u32 v28, vcc, v31, v61             # mem_addr = mem_ptr + addr0 + offset
        vadd_u32 v28, vcc, v28, s0
        v_mov_b32 v29, s1
        vaddc_u32 v29, vcc, 0, v29, vcc
        flat_load_dwordx4 v[24:27], v[28:29] glc   # load mem chunk (4 int at once)

        vadd_u32 v28, vcc, v28, v30             # mem_addr += 16
        vaddc_u32 v29, vcc, 0, v29, vcc
        flat_load_dwordx4 v[28:31], v[28:29] glc   # load mem chunk (4 int at once)

skip_data_load_gblocks:
                                                # apply argon2 hash function to tmp
        v_mul_lo_u32 v54, v8, v10               # a = fBlaMka(a, b)
        v_mul_hi_u32 v55, v8, v10
        v_lshlrev_b64 v[54:55], 1, v[54:55]
        vadd_u32 v8, vcc, v8, v10
        vaddc_u32 v9, vcc, v9, v11, vcc
        vadd_u32 v8, vcc, v8, v54
        vaddc_u32 v9, vcc, v9, v55, vcc

        v_mov_b32 v54, v15                      # d = rotate(d ^ a, 32)
        v_xor_b32 v15, v14, v8
        v_xor_b32 v14, v54, v9

        v_mul_lo_u32 v54, v12, v14              # c = fBlaMka(c, d)
        v_mul_hi_u32 v55, v12, v14
        v_lshlrev_b64 v[54:55], 1, v[54:55]
        vadd_u32 v12, vcc, v12, v14
        vaddc_u32 v13, vcc, v13, v15, vcc
        vadd_u32 v12, vcc, v12, v54
        vaddc_u32 v13, vcc, v13, v55, vcc

        v_xor_b32 v54, v10, v12                 # b = rotate(b ^ c, 40)
        v_xor_b32 v55, v11, v13
        v_alignbit_b32 v11, v54, v55, 24
        v_alignbit_b32 v10, v55, v54, 24

        v_mul_lo_u32 v54, v8, v10               # a = fBlaMka(a, b)
        v_mul_hi_u32 v55, v8, v10
        v_lshlrev_b64 v[54:55], 1, v[54:55]
        vadd_u32 v8, vcc, v8, v10
        vaddc_u32 v9, vcc, v9, v11, vcc
        vadd_u32 v8, vcc, v8, v54
        vaddc_u32 v9, vcc, v9, v55, vcc

        v_xor_b32 v54, v8, v14                  # d = rotate(d ^ a, 48)
        v_xor_b32 v55, v9, v15
        v_alignbit_b32 v15, v54, v55, 16
        v_alignbit_b32 v14, v55, v54, 16
#        ds_write_b64 v35, v[14:15]

        v_mul_lo_u32 v54, v12, v14              # c = fBlaMka(c, d)
        v_mul_hi_u32 v55, v12, v14
        v_lshlrev_b64 v[54:55], 1, v[54:55]
        vadd_u32 v12, vcc, v12, v14
        vaddc_u32 v13, vcc, v13, v15, vcc
        vadd_u32 v12, vcc, v12, v54
        vaddc_u32 v13, vcc, v13, v55, vcc
#        ds_write_b64 v34, v[12:13]

        v_xor_b32 v54, v10, v12                 # b = rotate(b ^ c, 1)
        v_xor_b32 v55, v11, v13
        v_alignbit_b32 v11, v55, v54, 31
        v_alignbit_b32 v10, v54, v55, 31
#        ds_write_b64 v33, v[10:11]

#        s_waitcnt lgkmcnt(0)

#        ds_read_b64 v[10:11], v37
#        ds_read_b64 v[14:15], v39
#        ds_read_b64 v[12:13], v38

        ds_bpermute_b32 v10, v48, v10           # first shuffle using cross lane data access
        ds_bpermute_b32 v11, v48, v11
        ds_bpermute_b32 v12, v49, v12
        ds_bpermute_b32 v13, v49, v13
        ds_bpermute_b32 v14, v50, v14
        ds_bpermute_b32 v15, v50, v15

        s_waitcnt lgkmcnt(0)

#        s_waitcnt lgkmcnt(0)

        v_mul_lo_u32 v54, v8, v10               # a = fBlaMka(a, b)
        v_mul_hi_u32 v55, v8, v10
        v_lshlrev_b64 v[54:55], 1, v[54:55]
        vadd_u32 v8, vcc, v8, v10
        vaddc_u32 v9, vcc, v9, v11, vcc
        vadd_u32 v8, vcc, v8, v54
        vaddc_u32 v9, vcc, v9, v55, vcc

        v_mov_b32 v54, v15                      # d = rotate(d ^ a, 32)
        v_xor_b32 v15, v14, v8
        v_xor_b32 v14, v54, v9

        v_mul_lo_u32 v54, v12, v14              # c = fBlaMka(c, d)
        v_mul_hi_u32 v55, v12, v14
        v_lshlrev_b64 v[54:55], 1, v[54:55]
        vadd_u32 v12, vcc, v12, v14
        vaddc_u32 v13, vcc, v13, v15, vcc
        vadd_u32 v12, vcc, v12, v54
        vaddc_u32 v13, vcc, v13, v55, vcc

        v_xor_b32 v54, v10, v12                 # b = rotate(b ^ c, 40)
        v_xor_b32 v55, v11, v13
        v_alignbit_b32 v11, v54, v55, 24
        v_alignbit_b32 v10, v55, v54, 24

        v_mul_lo_u32 v54, v8, v10               # a = fBlaMka(a, b)
        v_mul_hi_u32 v55, v8, v10
        v_lshlrev_b64 v[54:55], 1, v[54:55]
        vadd_u32 v8, vcc, v8, v10
        vaddc_u32 v9, vcc, v9, v11, vcc
        vadd_u32 v8, vcc, v8, v54
        vaddc_u32 v9, vcc, v9, v55, vcc
        ds_write_b64 v36, v[8:9]

        v_xor_b32 v54, v8, v14                  # d = rotate(d ^ a, 48)
        v_xor_b32 v55, v9, v15
        v_alignbit_b32 v15, v54, v55, 16
        v_alignbit_b32 v14, v55, v54, 16
        ds_write_b64 v39, v[14:15]

        v_mul_lo_u32 v54, v12, v14              # c = fBlaMka(c, d)
        v_mul_hi_u32 v55, v12, v14
        v_lshlrev_b64 v[54:55], 1, v[54:55]
        vadd_u32 v12, vcc, v12, v14
        vaddc_u32 v13, vcc, v13, v15, vcc
        vadd_u32 v12, vcc, v12, v54
        vaddc_u32 v13, vcc, v13, v55, vcc
        ds_write_b64 v38, v[12:13]

        v_xor_b32 v54, v10, v12                 # b = rotate(b ^ c, 1)
        v_xor_b32 v55, v11, v13
        v_alignbit_b32 v11, v55, v54, 31
        v_alignbit_b32 v10, v54, v55, 31
        ds_write_b64 v37, v[10:11]

#        s_waitcnt lgkmcnt(0)

        ds_read_b64 v[8:9], v40                # second shuffle using local storage
        ds_read_b64 v[10:11], v41
        ds_read_b64 v[14:15], v43
        ds_read_b64 v[12:13], v42

        s_waitcnt lgkmcnt(0)

        v_mul_lo_u32 v54, v8, v10               # a = fBlaMka(a, b)
        v_mul_hi_u32 v55, v8, v10
        v_lshlrev_b64 v[54:55], 1, v[54:55]
        vadd_u32 v8, vcc, v8, v10
        vaddc_u32 v9, vcc, v9, v11, vcc
        vadd_u32 v8, vcc, v8, v54
        vaddc_u32 v9, vcc, v9, v55, vcc

        v_mov_b32 v54, v15                      # d = rotate(d ^ a, 32)
        v_xor_b32 v15, v14, v8
        v_xor_b32 v14, v54, v9

        v_mul_lo_u32 v54, v12, v14              # c = fBlaMka(c, d)
        v_mul_hi_u32 v55, v12, v14
        v_lshlrev_b64 v[54:55], 1, v[54:55]
        vadd_u32 v12, vcc, v12, v14
        vaddc_u32 v13, vcc, v13, v15, vcc
        vadd_u32 v12, vcc, v12, v54
        vaddc_u32 v13, vcc, v13, v55, vcc

        v_xor_b32 v54, v10, v12                 # b = rotate(b ^ c, 40)
        v_xor_b32 v55, v11, v13
        v_alignbit_b32 v11, v54, v55, 24
        v_alignbit_b32 v10, v55, v54, 24

        v_mul_lo_u32 v54, v8, v10               # a = fBlaMka(a, b)
        v_mul_hi_u32 v55, v8, v10
        v_lshlrev_b64 v[54:55], 1, v[54:55]
        vadd_u32 v8, vcc, v8, v10
        vaddc_u32 v9, vcc, v9, v11, vcc
        vadd_u32 v8, vcc, v8, v54
        vaddc_u32 v9, vcc, v9, v55, vcc

        v_xor_b32 v54, v8, v14                  # d = rotate(d ^ a, 48)
        v_xor_b32 v55, v9, v15
        v_alignbit_b32 v15, v54, v55, 16
        v_alignbit_b32 v14, v55, v54, 16
#        ds_write_b64 v43, v[14:15]

        v_mul_lo_u32 v54, v12, v14              # c = fBlaMka(c, d)
        v_mul_hi_u32 v55, v12, v14
        v_lshlrev_b64 v[54:55], 1, v[54:55]
        vadd_u32 v12, vcc, v12, v14
        vaddc_u32 v13, vcc, v13, v15, vcc
        vadd_u32 v12, vcc, v12, v54
        vaddc_u32 v13, vcc, v13, v55, vcc
#        ds_write_b64 v42, v[12:13]

        v_xor_b32 v54, v10, v12                 # b = rotate(b ^ c, 1)
        v_xor_b32 v55, v11, v13
        v_alignbit_b32 v11, v55, v54, 31
        v_alignbit_b32 v10, v54, v55, 31
#        ds_write_b64 v41, v[10:11]

#        ds_read_b64 v[10:11], v45
#        ds_read_b64 v[14:15], v47
#        ds_read_b64 v[12:13], v46

        ds_bpermute_b32 v10, v51, v10           # third shuffle using cross lane data access
        ds_bpermute_b32 v11, v51, v11
        ds_bpermute_b32 v12, v52, v12
        ds_bpermute_b32 v13, v52, v13
        ds_bpermute_b32 v14, v53, v14
        ds_bpermute_b32 v15, v53, v15

        s_waitcnt lgkmcnt(0)

#        s_waitcnt lgkmcnt(0)

        v_mul_lo_u32 v54, v8, v10               # a = fBlaMka(a, b)
        v_mul_hi_u32 v55, v8, v10
        v_lshlrev_b64 v[54:55], 1, v[54:55]
        vadd_u32 v8, vcc, v8, v10
        vaddc_u32 v9, vcc, v9, v11, vcc
        vadd_u32 v8, vcc, v8, v54
        vaddc_u32 v9, vcc, v9, v55, vcc

        v_mov_b32 v54, v15                      # d = rotate(d ^ a, 32)
        v_xor_b32 v15, v14, v8
        v_xor_b32 v14, v54, v9

        v_mul_lo_u32 v54, v12, v14              # c = fBlaMka(c, d)
        v_mul_hi_u32 v55, v12, v14
        v_lshlrev_b64 v[54:55], 1, v[54:55]
        vadd_u32 v12, vcc, v12, v14
        vaddc_u32 v13, vcc, v13, v15, vcc
        vadd_u32 v12, vcc, v12, v54
        vaddc_u32 v13, vcc, v13, v55, vcc

        v_xor_b32 v54, v10, v12                 # b = rotate(b ^ c, 40)
        v_xor_b32 v55, v11, v13
        v_alignbit_b32 v11, v54, v55, 24
        v_alignbit_b32 v10, v55, v54, 24

        v_mul_lo_u32 v54, v8, v10               # a = fBlaMka(a, b)
        v_mul_hi_u32 v55, v8, v10
        v_lshlrev_b64 v[54:55], 1, v[54:55]
        vadd_u32 v8, vcc, v8, v10
        vaddc_u32 v9, vcc, v9, v11, vcc
        vadd_u32 v8, vcc, v8, v54
        vaddc_u32 v9, vcc, v9, v55, vcc
        ds_write_b64 v44, v[8:9]

        v_xor_b32 v54, v8, v14                  # d = rotate(d ^ a, 48)
        v_xor_b32 v55, v9, v15
        v_alignbit_b32 v15, v54, v55, 16
        v_alignbit_b32 v14, v55, v54, 16
        ds_write_b64 v47, v[14:15]

        v_mul_lo_u32 v54, v12, v14              # c = fBlaMka(c, d)
        v_mul_hi_u32 v55, v12, v14
        v_lshlrev_b64 v[54:55], 1, v[54:55]
        vadd_u32 v12, vcc, v12, v14
        vaddc_u32 v13, vcc, v13, v15, vcc
        vadd_u32 v12, vcc, v12, v54
        vaddc_u32 v13, vcc, v13, v55, vcc
        ds_write_b64 v46, v[12:13]

        v_xor_b32 v54, v10, v12                 # b = rotate(b ^ c, 1)
        v_xor_b32 v55, v11, v13
        v_alignbit_b32 v11, v55, v54, 31
        v_alignbit_b32 v10, v54, v55, 31
        ds_write_b64 v45, v[10:11]

#        s_waitcnt lgkmcnt(0)

        ds_read_b64 v[8:9], v32                # reorder to original shuffle using local storage
        ds_read_b64 v[10:11], v33
        ds_read_b64 v[12:13], v34
        ds_read_b64 v[14:15], v35

        s_waitcnt lgkmcnt(0)

        v_xor_b32 v0, v0, v8                    # state ^= tmp
        v_xor_b32 v1, v1, v9
        v_xor_b32 v2, v2, v10
        v_xor_b32 v3, v3, v11
        v_xor_b32 v4, v4, v12
        v_xor_b32 v5, v5, v13
        v_xor_b32 v6, v6, v14
        v_xor_b32 v7, v7, v15

        v_mov_b32 v8, 0x8000					# keep_block = ref & 0x8000
		v_and_b32 v8, v60, v8
		v_cmp_eq_u32 vcc, v8, 0
		s_and_saveexec_b64 s[16:17], vcc
		s_cbranch_execz skip_gblock_store

		v_mov_b32 v8, 0x7FFF
        v_and_b32 v8, v60, v8                   # addr0 = ref & 0x7FFF
        v_lshlrev_b32 v8, 10, v8                # addr0 = addr0 * BLOCK_SIZE

        vadd_u32 v8, vcc, v8, v61               # mem_addr = mem_ptr + addr0 + offset
        vadd_u32 v8, vcc, v8, s0
        v_mov_b32 v9, s1
        vaddc_u32 v9, vcc, 0, v9, vcc
        flat_store_dwordx4 v[8:9], v[0:3] glc      # store mem chunk (4 int at once)

        v_mov_b32 v60, 16
        vadd_u32 v8, vcc, v8, v60               # mem_addr += 16
        vaddc_u32 v9, vcc, 0, v9, vcc
        flat_store_dwordx4 v[8:9], v[4:7] glc      # store mem chunk (4 int at once)

#        s_barrier                               # wait for all lanes to finish segment

skip_gblock_store:
		s_mov_b64  exec, s[16:17]

        s_add_u32 s10, s10, 1                   # i++
        s_cmp_le_u32 s10, s14                   # } endfor
        s_cbranch_scc1 loop_gblocks

        s_barrier                               # wait for all lanes to finish segment

        s_mov_b32 s14, 1024                     # ref_count_limit = 1024
        s_add_u32 s13, s13, 4                   # s+=4
        s_cmp_lt_u32 s13, 64                    # } endfor
        s_cbranch_scc1 loop_gblocks_segments

        v_mov_b32 v6, 262112                    # out_addr_ref = addr_ptr + 65528 * 4
        vadd_u32 v4, vcc, s6, v6
        v_mov_b32 v5, s7
        vaddc_u32 v5, vcc, 0, v5, vcc
        flat_load_ushort v0, v[4:5]             # load out_ref_0
        v_mov_b32 v6, 2
        vadd_u32 v4, vcc, v4, v6
        vaddc_u32 v5, vcc, 0, v5, vcc
        flat_load_ushort v1, v[4:5]             # load out_ref_1
        v_mov_b32 v6, 4
        vadd_u32 v4, vcc, v4, v6
        vaddc_u32 v5, vcc, 0, v5, vcc
        flat_load_ushort v2, v[4:5]             # load out_ref_2
        vadd_u32 v4, vcc, v4, v6
        vaddc_u32 v5, vcc, 0, v5, vcc
        flat_load_ushort v3, v[4:5]             # load out_ref_3

        v_lshrrev_b32 v61, 5, v61               # recover id
        v_mov_b32 v62, 768                      # recover seg
        vsub_u32 v63, vcc, v63, v62
        v_lshrrev_b32 v63, 1, v63
        vadd_u32 v63, vcc, v63, v61             # wrk_id = seg * 32 + id

        v_mov_b32 v4, local_index&0xffffffff    # get local_index
        v_mov_b32 v5, local_index>>32           # local_index - higher part

        vadd_u32 v4, vcc, v4, v63               # local_index = local_index + wrk_id
        vaddc_u32 v5, vcc, 0, v5, vcc
        flat_load_ubyte v4, v[4:5]              # load store_pos

        v_lshlrev_b32 v63, 3, v63               # wrk_id *= 8

        s_waitcnt vmcnt(0) & lgkmcnt(0)         # load data

        v_lshlrev_b32 v0, 10, v0                # out_ref_x *= BLOCK_SIZE
        v_lshlrev_b32 v1, 10, v1
        v_lshlrev_b32 v2, 10, v2
        v_lshlrev_b32 v3, 10, v3
        v_lshlrev_b32 v4, 3, v4                 # store_pos *= 8

        vadd_u32 v5, vcc, v0, v63               # mem_addr = mem_ptr + out_ref_0 + wrk_id
        vadd_u32 v5, vcc, s0, v5
        v_mov_b32 v6, s1
        vaddc_u32 v6, vcc, 0, v6, vcc
        flat_load_dwordx2 v[7:8], v[5:6]        # load data

        vadd_u32 v5, vcc, v1, v63               # mem_addr = mem_ptr + out_ref_1 + wrk_id
        vadd_u32 v5, vcc, s0, v5
        v_mov_b32 v6, s1
        vaddc_u32 v6, vcc, 0, v6, vcc
        flat_load_dwordx2 v[9:10], v[5:6]       # load data

        vadd_u32 v5, vcc, v2, v63               # mem_addr = mem_ptr + out_ref_2 + wrk_id
        vadd_u32 v5, vcc, s0, v5
        v_mov_b32 v6, s1
        vaddc_u32 v6, vcc, 0, v6, vcc
        flat_load_dwordx2 v[11:12], v[5:6]      # load data

        vadd_u32 v5, vcc, v3, v63               # mem_addr = mem_ptr + out_ref_3 + wrk_id
        vadd_u32 v5, vcc, s0, v5
        v_mov_b32 v6, s1
        vaddc_u32 v6, vcc, 0, v6, vcc
        flat_load_dwordx2 v[13:14], v[5:6]      # load data

        s_waitcnt vmcnt(0)                      # load data

        v_xor_b32 v7, v7, v9                    # out_0 ^= out_1; out_0 ^= out_2; out_0 ^= out_3;
        v_xor_b32 v7, v7, v11
        v_xor_b32 v7, v7, v13
        v_xor_b32 v8, v8, v10
        v_xor_b32 v8, v8, v12
        v_xor_b32 v8, v8, v14

        vadd_u32 v5, vcc, s4, v4                # out_addr = out_ptr + store_pos
        v_mov_b32 v6, s5
        vaddc_u32 v6, vcc, 0, v6, vcc
        flat_store_dwordx2 v[5:6], v[7:8]       # store out chunk (2 int at once)

        s_endpgm
.else
        .error "Unsupported binary format"
.endif
.else
        .error "Unsupported bitness (32 bit)"
.endif
)ffDXD";


