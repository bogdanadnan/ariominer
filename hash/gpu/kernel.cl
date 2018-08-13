#define MEMORY_CHUNK_PER_ITEM           4
#define BLOCK_SIZE                      1024
#define BLOCK_SIZE_ULONG                128

#define fBlaMka(x, y) ((x) + (y) + 2 * ((x) & 0xFFFFFFFF) * ((y) & 0xFFFFFFFF))

#define G(data, vec)           \
{                           \
    ulong a = data[vec[0]]; \
    ulong b = data[vec[1]]; \
    ulong c = data[vec[2]]; \
    ulong d = data[vec[3]]; \
    a = fBlaMka(a, b);          \
    d = rotate(d ^ a, (ulong)32);      \
    c = fBlaMka(c, d);          \
    b = rotate(b ^ c, (ulong)40);      \
    a = fBlaMka(a, b);          \
    d = rotate(d ^ a, (ulong)48);      \
    c = fBlaMka(c, d);          \
    b = rotate(b ^ c, (ulong)1);       \
    data[vec[0]] = a; \
    data[vec[1]] = b; \
    data[vec[2]] = c; \
    data[vec[3]] = d; \
}

__constant ulong4 zero = (ulong4)(0);

__constant int offsets_round_1[32][4] = {
        { 0, 4, 8, 12 },
        { 1, 5, 9, 13 },
        { 2, 6, 10, 14 },
        { 3, 7, 11, 15 },
        { 16, 20, 24, 28 },
        { 17, 21, 25, 29 },
        { 18, 22, 26, 30 },
        { 19, 23, 27, 31 },
        { 32, 36, 40, 44 },
        { 33, 37, 41, 45 },
        { 34, 38, 42, 46 },
        { 35, 39, 43, 47 },
        { 48, 52, 56, 60 },
        { 49, 53, 57, 61 },
        { 50, 54, 58, 62 },
        { 51, 55, 59, 63 },
        { 64, 68, 72, 76 },
        { 65, 69, 73, 77 },
        { 66, 70, 74, 78 },
        { 67, 71, 75, 79 },
        { 80, 84, 88, 92 },
        { 81, 85, 89, 93 },
        { 82, 86, 90, 94 },
        { 83, 87, 91, 95 },
        { 96, 100, 104, 108 },
        { 97, 101, 105, 109 },
        { 98, 102, 106, 110 },
        { 99, 103, 107, 111 },
        { 112, 116, 120, 124 },
        { 113, 117, 121, 125 },
        { 114, 118, 122, 126 },
        { 115, 119, 123, 127 },
};

__constant int offsets_round_2[32][4] = {
        { 0, 5, 10, 15 },
        { 1, 6, 11, 12 },
        { 2, 7, 8, 13 },
        { 3, 4, 9, 14 },
        { 16, 21, 26, 31 },
        { 17, 22, 27, 28 },
        { 18, 23, 24, 29 },
        { 19, 20, 25, 30 },
        { 32, 37, 42, 47 },
        { 33, 38, 43, 44 },
        { 34, 39, 40, 45 },
        { 35, 36, 41, 46 },
        { 48, 53, 58, 63 },
        { 49, 54, 59, 60 },
        { 50, 55, 56, 61 },
        { 51, 52, 57, 62 },
        { 64, 69, 74, 79 },
        { 65, 70, 75, 76 },
        { 66, 71, 72, 77 },
        { 67, 68, 73, 78 },
        { 80, 85, 90, 95 },
        { 81, 86, 91, 92 },
        { 82, 87, 88, 93 },
        { 83, 84, 89, 94 },
        { 96, 101, 106, 111 },
        { 97, 102, 107, 108 },
        { 98, 103, 104, 109 },
        { 99, 100, 105, 110 },
        { 112, 117, 122, 127 },
        { 113, 118, 123, 124 },
        { 114, 119, 120, 125 },
        { 115, 116, 121, 126 },
};

__constant int offsets_round_3[32][4] = {
        { 0, 32, 64, 96 },
        { 1, 33, 65, 97 },
        { 16, 48, 80, 112 },
        { 17, 49, 81, 113 },
        { 2, 34, 66, 98 },
        { 3, 35, 67, 99 },
        { 18, 50, 82, 114 },
        { 19, 51, 83, 115 },
        { 4, 36, 68, 100 },
        { 5, 37, 69, 101 },
        { 20, 52, 84, 116 },
        { 21, 53, 85, 117 },
        { 6, 38, 70, 102 },
        { 7, 39, 71, 103 },
        { 22, 54, 86, 118 },
        { 23, 55, 87, 119 },
        { 8, 40, 72, 104 },
        { 9, 41, 73, 105 },
        { 24, 56, 88, 120 },
        { 25, 57, 89, 121 },
        { 10, 42, 74, 106 },
        { 11, 43, 75, 107 },
        { 26, 58, 90, 122 },
        { 27, 59, 91, 123 },
        { 12, 44, 76, 108 },
        { 13, 45, 77, 109 },
        { 28, 60, 92, 124 },
        { 29, 61, 93, 125 },
        { 14, 46, 78, 110 },
        { 15, 47, 79, 111 },
        { 30, 62, 94, 126 },
        { 31, 63, 95, 127 },
};

__constant int offsets_round_4[32][4] = {
        { 0, 33, 80, 113 },
        { 1, 48, 81, 96 },
        { 16, 49, 64, 97 },
        { 17, 32, 65, 112 },
        { 2, 35, 82, 115 },
        { 3, 50, 83, 98 },
        { 18, 51, 66, 99 },
        { 19, 34, 67, 114 },
        { 4, 37, 84, 117 },
        { 5, 52, 85, 100 },
        { 20, 53, 68, 101 },
        { 21, 36, 69, 116 },
        { 6, 39, 86, 119 },
        { 7, 54, 87, 102 },
        { 22, 55, 70, 103 },
        { 23, 38, 71, 118 },
        { 8, 41, 88, 121 },
        { 9, 56, 89, 104 },
        { 24, 57, 72, 105 },
        { 25, 40, 73, 120 },
        { 10, 43, 90, 123 },
        { 11, 58, 91, 106 },
        { 26, 59, 74, 107 },
        { 27, 42, 75, 122 },
        { 12, 45, 92, 125 },
        { 13, 60, 93, 108 },
        { 28, 61, 76, 109 },
        { 29, 44, 77, 124 },
        { 14, 47, 94, 127 },
        { 15, 62, 95, 110 },
        { 30, 63, 78, 111 },
        { 31, 46, 79, 126 },
};

#define xor_block_4(dst1, dst2, dst3, src)                                                     \
{ \
ulong4 data = vload4(0, &dst1[offset]) ^ vload4(0, &src[offset]); \
vstore4(data, 0, &dst2[offset]); \
vstore4(data, 0, &dst3[offset]); \
}

#define xor_block_3(dst1, dst2, src)                                                     \
{ \
ulong4 data = vload4(0, &dst1[offset]) ^ vload4(0, &src[offset]); \
vstore4(data, 0, &dst2[offset]); \
}

#define xor_block_2(dst, src)                                                     \
{ \
ulong4 data = vload4(0, &dst[offset]) ^ vload4(0, &src[offset]); \
vstore4(data, 0, &dst[offset]); \
}

#define copy_block(dst, src)                                                    \
vstore4(vload4(0, &src[offset]), 0, &dst[offset]);

#define copy_2block(dst, src)                                                    \
vstore8(vload8(0, &src[offset]), 0, &dst[offset]);

#define zero_block(dst)                                                    \
vstore4(zero, 0, &dst[offset]);

static void fill_block_noxor(__global ulong *prev_block,
        __global ulong *ref_block,
        __global ulong *next_block,
        __global ulong *global_state,
        __local ulong *state,
        __local ulong *buffer,
        int id)
{
    int offset = id * MEMORY_CHUNK_PER_ITEM;
    xor_block_4(prev_block, state, buffer, ref_block);

    barrier(CLK_LOCAL_MEM_FENCE);

    G(state, offsets_round_1[id]);
    barrier(CLK_LOCAL_MEM_FENCE);
    G(state, offsets_round_2[id]);
    barrier(CLK_LOCAL_MEM_FENCE);
    G(state, offsets_round_3[id]);
    barrier(CLK_LOCAL_MEM_FENCE);
    G(state, offsets_round_4[id]);
    barrier(CLK_LOCAL_MEM_FENCE);

    xor_block_4(state, global_state, next_block, buffer);
    barrier(CLK_GLOBAL_MEM_FENCE);
}

static void fill_block_xor(__global ulong *prev_block,
        __global ulong *ref_block,
        __global ulong *next_block,
        __global ulong *global_state,
        __local ulong *state,
        __local ulong *buffer,
        int id)
{
    int offset = id * MEMORY_CHUNK_PER_ITEM;
    xor_block_4(prev_block, state, buffer, ref_block);
    xor_block_2(buffer, next_block);

    barrier(CLK_LOCAL_MEM_FENCE);

    G(state, offsets_round_1[id]);
    barrier(CLK_LOCAL_MEM_FENCE);
    G(state, offsets_round_2[id]);
    barrier(CLK_LOCAL_MEM_FENCE);
    G(state, offsets_round_3[id]);
    barrier(CLK_LOCAL_MEM_FENCE);
    G(state, offsets_round_4[id]);
    barrier(CLK_LOCAL_MEM_FENCE);

    xor_block_4(state, global_state, next_block, buffer);
    barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel void fill_blocks(__global ulong *chunk_0,
        __global ulong *chunk_1,
        __global ulong *chunk_2,
        __global ulong *chunk_3,
        __global ulong *chunk_4,
        __global ulong *chunk_5,
        __global int *refs_1_1_524288,
        __global int *refs_4_4_16384,
        __global ulong *out,
        __global ulong *seed,
        int threads,
        int threads_per_chunk,
        int memsize,
        int addrsize,
        int xor_limit,
        int profile) {
    __local ulong state[BLOCK_SIZE_ULONG];
    __local ulong buffer[BLOCK_SIZE_ULONG];
    __global ulong *global_state;
    __global ulong *zero_blk;

    int hash = get_group_id(0);
    int id = get_local_id(0);
    int offset = id * MEMORY_CHUNK_PER_ITEM;

    __global int *addresses = profile == 0 ? refs_1_1_524288 : refs_4_4_16384;
    int chunk_index = hash / threads_per_chunk;
    int chunk_offset = hash - chunk_index * threads_per_chunk;
    __global ulong *memory = chunk_index == 0 ? chunk_0 :
                             (chunk_index == 1 ? chunk_1 :
                              (chunk_index == 2 ? chunk_2 :
                               (chunk_index == 3 ? chunk_3 :
                                (chunk_index == 4 ? chunk_4 :
                                 chunk_5))));

    memory = memory + chunk_offset * (memsize >> 3);

    int mem_end = memsize >> 3;
    for(int i=0;i<mem_end;i+=BLOCK_SIZE_ULONG){
        zero_block((memory + i));
    }

    int mem_seed_count = (profile == 1 ? 4 : 1);
    int lane_length = (profile == 1 ? 4096 : 0);

    __global ulong *out_mem = out + hash * 2 * mem_seed_count * BLOCK_SIZE_ULONG;
    __global ulong *mem_seed = seed + hash * 2 * mem_seed_count * BLOCK_SIZE_ULONG;

    for(int i = 0; i < mem_seed_count; i++) {
        __global ulong *src = mem_seed + i * 2 * BLOCK_SIZE_ULONG;
        __global ulong *dst = memory + i * lane_length * BLOCK_SIZE_ULONG;
        copy_block(dst, src);
        src += BLOCK_SIZE_ULONG;
        dst += BLOCK_SIZE_ULONG;
        copy_block(dst, src);
    }

    global_state = mem_seed + BLOCK_SIZE_ULONG;
    zero_blk = mem_seed;

    zero_block(zero_blk);
    barrier(CLK_GLOBAL_MEM_FENCE);

    __global ulong *next_block;
    __global ulong *prev_block;
    __global ulong *ref_block;
    __global ulong *xor_block;

    int final_addrsize = (profile == 1) ? (addrsize - 3) : addrsize;

    int i=0;
    for(; i < xor_limit; ++i, addresses += 3) {
        next_block = (addresses[0] == -1) ? global_state : (memory + addresses[0] * BLOCK_SIZE_ULONG);
        prev_block = (addresses[1] == -1) ? global_state : (memory + addresses[1] * BLOCK_SIZE_ULONG);
        ref_block = memory + (addresses[2] * BLOCK_SIZE_ULONG);

        fill_block_noxor(prev_block, ref_block, next_block, global_state, state, buffer, id);
    }
    for(; i < final_addrsize; ++i, addresses += 3) {
        next_block = (addresses[0] == -1) ? global_state : (memory + addresses[0] * BLOCK_SIZE_ULONG);
        prev_block = (addresses[1] == -1) ? global_state : (memory + addresses[1] * BLOCK_SIZE_ULONG);
        ref_block = memory + (addresses[2] * BLOCK_SIZE_ULONG);

        fill_block_xor(prev_block, ref_block, next_block, global_state, state, buffer, id);
    }

    int result_block = (profile == 1) ? addresses[1] : 0;
    next_block = memory + result_block * BLOCK_SIZE_ULONG;
    copy_block(out_mem, next_block);

    for(;i < addrsize; ++i, addresses += 3) {
        next_block = memory + addresses[2] * BLOCK_SIZE_ULONG;
        xor_block_2(out_mem, next_block);
    }

    barrier(CLK_GLOBAL_MEM_FENCE);
};
