//
// Created by Haifa Bogdan Adnan on 06/08/2018.
//

#include "../../common/common.h"

#include "opencl_kernel.h"

string opencl_kernel = "#define MEMORY_CHUNK_PER_ITEM           4\n"
                       "#define BLOCK_SIZE                      1024\n"
                       "#define BLOCK_SIZE_ULONG                128\n"
                       "\n"
                       "#define fBlaMka(x, y) ((x) + (y) + 2 * upsample(mul_hi((uint)(x), (uint)(y)), (uint)(x) * (uint)y))\n"
                       "\n"
                       "#define G1(data, vec)           \\\n"
                       "{                           \\\n"
                       "    a = data[vec[0]]; \\\n"
                       "    b = data[vec[1]]; \\\n"
                       "    c = data[vec[2]]; \\\n"
                       "    d = data[vec[3]]; \\\n"
                       "    a = fBlaMka(a, b);          \\\n"
                       "    d = rotate(d ^ a, (ulong)32);      \\\n"
                       "    c = fBlaMka(c, d);          \\\n"
                       "    b = rotate(b ^ c, (ulong)40);      \\\n"
                       "    a = fBlaMka(a, b);          \\\n"
                       "    d = rotate(d ^ a, (ulong)48);      \\\n"
                       "    c = fBlaMka(c, d);          \\\n"
                       "    b = rotate(b ^ c, (ulong)1);       \\\n"
                       "    data[vec[1]] = b; \\\n"
                       "    data[vec[2]] = c; \\\n"
                       "    data[vec[3]] = d; \\\n"
                       "}\n"
                       "\n"
                       "#define G2(data, vec)           \\\n"
                       "{                           \\\n"
                       "    b = data[vec[1]]; \\\n"
                       "    c = data[vec[2]]; \\\n"
                       "    d = data[vec[3]]; \\\n"
                       "    a = fBlaMka(a, b);          \\\n"
                       "    d = rotate(d ^ a, (ulong)32);      \\\n"
                       "    c = fBlaMka(c, d);          \\\n"
                       "    b = rotate(b ^ c, (ulong)40);      \\\n"
                       "    a = fBlaMka(a, b);          \\\n"
                       "    d = rotate(d ^ a, (ulong)48);      \\\n"
                       "    c = fBlaMka(c, d);          \\\n"
                       "    b = rotate(b ^ c, (ulong)1);       \\\n"
                       "    data[vec[0]] = a; \\\n"
                       "    data[vec[1]] = b; \\\n"
                       "    data[vec[2]] = c; \\\n"
                       "    data[vec[3]] = d; \\\n"
                       "}\n"
                       "\n"
                       "#define xor_block_4(dst1, dst2, dst3, src)                                                     \\\n"
                       "{ \\\n"
                       "ulong4 data = vload4(0, &dst1[offset]) ^ vload4(0, &src[offset]); \\\n"
                       "vstore4(data, 0, &dst2[offset]); \\\n"
                       "vstore4(data, 0, &dst3[offset]); \\\n"
                       "}\n"
                       "\n"
                       "#define xor_block_3(dst1, dst2, src)                                                     \\\n"
                       "{ \\\n"
                       "ulong4 data = vload4(0, &dst1[offset]) ^ vload4(0, &src[offset]); \\\n"
                       "vstore4(data, 0, &dst2[offset]); \\\n"
                       "}\n"
                       "\n"
                       "#define xor_block_2(dst, src)                                                     \\\n"
                       "{ \\\n"
                       "ulong4 data = vload4(0, &dst[offset]) ^ vload4(0, &src[offset]); \\\n"
                       "vstore4(data, 0, &dst[offset]); \\\n"
                       "}\n"
                       "\n"
                       "#define copy_block(dst, src)                                                    \\\n"
                       "vstore4(vload4(0, &src[offset]), 0, &dst[offset]);\n"
                       "\n"
                       "#define zero_block(dst)                                                    \\\n"
                       "vstore4(zero, 0, &dst[offset]);\n"
                       "\n"
                       "__kernel void fill_blocks(__global ulong *chunk_0,\n"
                       "        __global ulong *chunk_1,\n"
                       "        __global ulong *chunk_2,\n"
                       "        __global ulong *chunk_3,\n"
                       "        __global ulong *chunk_4,\n"
                       "        __global ulong *chunk_5,\n"
                       "        __global int *refs_1_1_524288,\n"
                       "        __global int *refs_4_4_16384,\n"
                       "        __global ulong *out,\n"
                       "        __global ulong *seed,\n"
                       "        int threads,\n"
                       "        int threads_per_chunk,\n"
                       "        int memsize,\n"
                       "        int addrsize,\n"
                       "        int xor_limit,\n"
                       "        int profile) {\n"
                       "    __local ulong state[BLOCK_SIZE_ULONG];\n"
                       "    __local ulong buffer[BLOCK_SIZE_ULONG];\n"
                       "    ulong a, b, c, d;\n"
                       "\n"
                       "    int offsets_round_1[32][4] = {\n"
                       "            { 0, 4, 8, 12 },\n"
                       "            { 1, 5, 9, 13 },\n"
                       "            { 2, 6, 10, 14 },\n"
                       "            { 3, 7, 11, 15 },\n"
                       "            { 16, 20, 24, 28 },\n"
                       "            { 17, 21, 25, 29 },\n"
                       "            { 18, 22, 26, 30 },\n"
                       "            { 19, 23, 27, 31 },\n"
                       "            { 32, 36, 40, 44 },\n"
                       "            { 33, 37, 41, 45 },\n"
                       "            { 34, 38, 42, 46 },\n"
                       "            { 35, 39, 43, 47 },\n"
                       "            { 48, 52, 56, 60 },\n"
                       "            { 49, 53, 57, 61 },\n"
                       "            { 50, 54, 58, 62 },\n"
                       "            { 51, 55, 59, 63 },\n"
                       "            { 64, 68, 72, 76 },\n"
                       "            { 65, 69, 73, 77 },\n"
                       "            { 66, 70, 74, 78 },\n"
                       "            { 67, 71, 75, 79 },\n"
                       "            { 80, 84, 88, 92 },\n"
                       "            { 81, 85, 89, 93 },\n"
                       "            { 82, 86, 90, 94 },\n"
                       "            { 83, 87, 91, 95 },\n"
                       "            { 96, 100, 104, 108 },\n"
                       "            { 97, 101, 105, 109 },\n"
                       "            { 98, 102, 106, 110 },\n"
                       "            { 99, 103, 107, 111 },\n"
                       "            { 112, 116, 120, 124 },\n"
                       "            { 113, 117, 121, 125 },\n"
                       "            { 114, 118, 122, 126 },\n"
                       "            { 115, 119, 123, 127 },\n"
                       "    };\n"
                       "\n"
                       "    int offsets_round_2[32][4] = {\n"
                       "            { 0, 5, 10, 15 },\n"
                       "            { 1, 6, 11, 12 },\n"
                       "            { 2, 7, 8, 13 },\n"
                       "            { 3, 4, 9, 14 },\n"
                       "            { 16, 21, 26, 31 },\n"
                       "            { 17, 22, 27, 28 },\n"
                       "            { 18, 23, 24, 29 },\n"
                       "            { 19, 20, 25, 30 },\n"
                       "            { 32, 37, 42, 47 },\n"
                       "            { 33, 38, 43, 44 },\n"
                       "            { 34, 39, 40, 45 },\n"
                       "            { 35, 36, 41, 46 },\n"
                       "            { 48, 53, 58, 63 },\n"
                       "            { 49, 54, 59, 60 },\n"
                       "            { 50, 55, 56, 61 },\n"
                       "            { 51, 52, 57, 62 },\n"
                       "            { 64, 69, 74, 79 },\n"
                       "            { 65, 70, 75, 76 },\n"
                       "            { 66, 71, 72, 77 },\n"
                       "            { 67, 68, 73, 78 },\n"
                       "            { 80, 85, 90, 95 },\n"
                       "            { 81, 86, 91, 92 },\n"
                       "            { 82, 87, 88, 93 },\n"
                       "            { 83, 84, 89, 94 },\n"
                       "            { 96, 101, 106, 111 },\n"
                       "            { 97, 102, 107, 108 },\n"
                       "            { 98, 103, 104, 109 },\n"
                       "            { 99, 100, 105, 110 },\n"
                       "            { 112, 117, 122, 127 },\n"
                       "            { 113, 118, 123, 124 },\n"
                       "            { 114, 119, 120, 125 },\n"
                       "            { 115, 116, 121, 126 },\n"
                       "    };\n"
                       "\n"
                       "    int offsets_round_3[32][4] = {\n"
                       "            { 0, 32, 64, 96 },\n"
                       "            { 1, 33, 65, 97 },\n"
                       "            { 16, 48, 80, 112 },\n"
                       "            { 17, 49, 81, 113 },\n"
                       "            { 2, 34, 66, 98 },\n"
                       "            { 3, 35, 67, 99 },\n"
                       "            { 18, 50, 82, 114 },\n"
                       "            { 19, 51, 83, 115 },\n"
                       "            { 4, 36, 68, 100 },\n"
                       "            { 5, 37, 69, 101 },\n"
                       "            { 20, 52, 84, 116 },\n"
                       "            { 21, 53, 85, 117 },\n"
                       "            { 6, 38, 70, 102 },\n"
                       "            { 7, 39, 71, 103 },\n"
                       "            { 22, 54, 86, 118 },\n"
                       "            { 23, 55, 87, 119 },\n"
                       "            { 8, 40, 72, 104 },\n"
                       "            { 9, 41, 73, 105 },\n"
                       "            { 24, 56, 88, 120 },\n"
                       "            { 25, 57, 89, 121 },\n"
                       "            { 10, 42, 74, 106 },\n"
                       "            { 11, 43, 75, 107 },\n"
                       "            { 26, 58, 90, 122 },\n"
                       "            { 27, 59, 91, 123 },\n"
                       "            { 12, 44, 76, 108 },\n"
                       "            { 13, 45, 77, 109 },\n"
                       "            { 28, 60, 92, 124 },\n"
                       "            { 29, 61, 93, 125 },\n"
                       "            { 14, 46, 78, 110 },\n"
                       "            { 15, 47, 79, 111 },\n"
                       "            { 30, 62, 94, 126 },\n"
                       "            { 31, 63, 95, 127 },\n"
                       "    };\n"
                       "\n"
                       "    int offsets_round_4[32][4] = {\n"
                       "            { 0, 33, 80, 113 },\n"
                       "            { 1, 48, 81, 96 },\n"
                       "            { 16, 49, 64, 97 },\n"
                       "            { 17, 32, 65, 112 },\n"
                       "            { 2, 35, 82, 115 },\n"
                       "            { 3, 50, 83, 98 },\n"
                       "            { 18, 51, 66, 99 },\n"
                       "            { 19, 34, 67, 114 },\n"
                       "            { 4, 37, 84, 117 },\n"
                       "            { 5, 52, 85, 100 },\n"
                       "            { 20, 53, 68, 101 },\n"
                       "            { 21, 36, 69, 116 },\n"
                       "            { 6, 39, 86, 119 },\n"
                       "            { 7, 54, 87, 102 },\n"
                       "            { 22, 55, 70, 103 },\n"
                       "            { 23, 38, 71, 118 },\n"
                       "            { 8, 41, 88, 121 },\n"
                       "            { 9, 56, 89, 104 },\n"
                       "            { 24, 57, 72, 105 },\n"
                       "            { 25, 40, 73, 120 },\n"
                       "            { 10, 43, 90, 123 },\n"
                       "            { 11, 58, 91, 106 },\n"
                       "            { 26, 59, 74, 107 },\n"
                       "            { 27, 42, 75, 122 },\n"
                       "            { 12, 45, 92, 125 },\n"
                       "            { 13, 60, 93, 108 },\n"
                       "            { 28, 61, 76, 109 },\n"
                       "            { 29, 44, 77, 124 },\n"
                       "            { 14, 47, 94, 127 },\n"
                       "            { 15, 62, 95, 110 },\n"
                       "            { 30, 63, 78, 111 },\n"
                       "            { 31, 46, 79, 126 },\n"
                       "    };\n"
                       "\n"
                       "    ulong4 zero = (ulong4)(0);\n"
                       "\n"
                       "    int hash = get_group_id(0);\n"
                       "    int id = get_local_id(0);\n"
                       "    int offset = id * MEMORY_CHUNK_PER_ITEM;\n"
                       "\n"
                       "    __global int *addresses = profile == 0 ? refs_1_1_524288 : refs_4_4_16384;\n"
                       "    int chunk_index = hash / threads_per_chunk;\n"
                       "    int chunk_offset = hash - chunk_index * threads_per_chunk;\n"
                       "    __global ulong *memory = chunk_index == 0 ? chunk_0 :\n"
                       "                             (chunk_index == 1 ? chunk_1 :\n"
                       "                              (chunk_index == 2 ? chunk_2 :\n"
                       "                               (chunk_index == 3 ? chunk_3 :\n"
                       "                                (chunk_index == 4 ? chunk_4 :\n"
                       "                                 chunk_5))));\n"
                       "\n"
                       "    memory = memory + chunk_offset * (memsize >> 3);\n"
                       "\n"
                       "    int mem_end = memsize >> 3;\n"
                       "    for(int i=0; i < mem_end; i += BLOCK_SIZE_ULONG) {\n"
                       "        zero_block((memory + i));\n"
                       "    }\n"
                       "\n"
                       "    int mem_seed_count = (profile == 1 ? 4 : 1);\n"
                       "    int lane_length = (profile == 1 ? 4096 : 0);\n"
                       "\n"
                       "    __global ulong *out_mem = out + hash * 2 * mem_seed_count * BLOCK_SIZE_ULONG;\n"
                       "    __global ulong *mem_seed = seed + hash * 2 * mem_seed_count * BLOCK_SIZE_ULONG;\n"
                       "\n"
                       "    for(int i = 0; i < mem_seed_count; i++) {\n"
                       "        __global ulong *src = mem_seed + i * 2 * BLOCK_SIZE_ULONG;\n"
                       "        __global ulong *dst = memory + i * lane_length * BLOCK_SIZE_ULONG;\n"
                       "        copy_block(dst, src);\n"
                       "        src += BLOCK_SIZE_ULONG;\n"
                       "        dst += BLOCK_SIZE_ULONG;\n"
                       "        copy_block(dst, src);\n"
                       "    }\n"
                       "\n"
                       "    __global ulong *next_block;\n"
                       "    __global ulong *prev_block;\n"
                       "    __global ulong *ref_block;\n"
                       "\n"
                       "    prev_block = mem_seed + BLOCK_SIZE_ULONG;\n"
                       "    copy_block(state, prev_block);\n"
                       "\n"
                       "    int final_addrsize = (profile == 1) ? (addrsize - 3) : addrsize;\n"
                       "\n"
                       "    int i=0;\n"
                       "    for(; i < xor_limit; ++i, addresses += 3) {\n"
                       "        if(addresses[1] != -1) {\n"
                       "            prev_block = (memory + addresses[1] * BLOCK_SIZE_ULONG);\n"
                       "            copy_block(state, prev_block);\n"
                       "        }\n"
                       "\n"
                       "        ref_block = memory + (addresses[2] * BLOCK_SIZE_ULONG);\n"
                       "\n"
                       "        xor_block_4(state, state, buffer, ref_block);\n"
                       "        barrier(CLK_LOCAL_MEM_FENCE);\n"
                       "\n"
                       "        G1(state, offsets_round_1[id]);\n"
                       "        barrier(CLK_LOCAL_MEM_FENCE);\n"
                       "        G2(state, offsets_round_2[id]);\n"
                       "        barrier(CLK_LOCAL_MEM_FENCE);\n"
                       "        G1(state, offsets_round_3[id]);\n"
                       "        barrier(CLK_LOCAL_MEM_FENCE);\n"
                       "        G2(state, offsets_round_4[id]);\n"
                       "        barrier(CLK_LOCAL_MEM_FENCE);\n"
                       "\n"
                       "        if(addresses[0] != -1) {\n"
                       "            next_block = memory + addresses[0] * BLOCK_SIZE_ULONG;\n"
                       "            xor_block_4(state, state, next_block, buffer);\n"
                       "        }\n"
                       "        else {\n"
                       "            xor_block_2(state, buffer);\n"
                       "        }\n"
                       "    }\n"
                       "\n"
                       "    for(; i < final_addrsize; ++i, addresses += 3) {\n"
                       "        next_block = memory + addresses[0] * BLOCK_SIZE_ULONG;\n"
                       "\n"
                       "        if(addresses[1] != -1) {\n"
                       "            prev_block = (memory + addresses[1] * BLOCK_SIZE_ULONG);\n"
                       "            copy_block(state, prev_block);\n"
                       "        }\n"
                       "\n"
                       "        ref_block = memory + (addresses[2] * BLOCK_SIZE_ULONG);\n"
                       "\n"
                       "        xor_block_4(state, state, buffer, ref_block);\n"
                       "        xor_block_2(buffer, next_block);\n"
                       "        barrier(CLK_LOCAL_MEM_FENCE);\n"
                       "\n"
                       "        G1(state, offsets_round_1[id]);\n"
                       "        barrier(CLK_LOCAL_MEM_FENCE);\n"
                       "        G2(state, offsets_round_2[id]);\n"
                       "        barrier(CLK_LOCAL_MEM_FENCE);\n"
                       "        G1(state, offsets_round_3[id]);\n"
                       "        barrier(CLK_LOCAL_MEM_FENCE);\n"
                       "        G2(state, offsets_round_4[id]);\n"
                       "        barrier(CLK_LOCAL_MEM_FENCE);\n"
                       "\n"
                       "        xor_block_4(state, state, next_block, buffer);\n"
                       "    }\n"
                       "\n"
                       "    int result_block = (profile == 1) ? addresses[1] : 0;\n"
                       "    next_block = memory + result_block * BLOCK_SIZE_ULONG;\n"
                       "    copy_block(out_mem, next_block);\n"
                       "\n"
                       "    for(;i < addrsize; ++i, addresses += 3) {\n"
                       "        next_block = memory + addresses[2] * BLOCK_SIZE_ULONG;\n"
                       "        xor_block_2(out_mem, next_block);\n"
                       "    }\n"
                       "};";
