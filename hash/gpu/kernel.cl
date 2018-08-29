#define MEMORY_CHUNK_PER_ITEM           4
#define ITEMS_PER_SEGMENT               32
#define BLOCK_SIZE                      1024
#define BLOCK_SIZE_ULONG                128

#define fBlaMka(x, y) ((x) + (y) + 2 * upsample(mul_hi((uint)(x), (uint)(y)), (uint)(x) * (uint)y))

#define G1(data, vec)           \
{                           \
    a = data[vec[0]]; \
    b = data[vec[1]]; \
    c = data[vec[2]]; \
    d = data[vec[3]]; \
    a = fBlaMka(a, b);          \
    d = rotate(d ^ a, (ulong)32);      \
    c = fBlaMka(c, d);          \
    b = rotate(b ^ c, (ulong)40);      \
    a = fBlaMka(a, b);          \
    d = rotate(d ^ a, (ulong)48);      \
    c = fBlaMka(c, d);          \
    b = rotate(b ^ c, (ulong)1);       \
    data[vec[1]] = b; \
    data[vec[2]] = c; \
    data[vec[3]] = d; \
}

#define G2(data, vec)           \
{                           \
    b = data[vec[1]]; \
    c = data[vec[2]]; \
    d = data[vec[3]]; \
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

#define xor_block(dst, src)                                                     \
vstore4(vload4(0, &dst[offset]) ^ vload4(0, &src[offset]), 0, &dst[offset]);

#define xor_block_small(dst, src)                                                    \
dst[local_id] ^= src[local_id];

#define copy_block(dst, src)                                                    \
vstore4(vload4(0, &src[offset]), 0, &dst[offset]);

#define copy_block_small(dst, src)                                                    \
dst[local_id] = src[local_id];


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

__kernel void fill_blocks(__global ulong *chunk_0,
        __global ulong *chunk_1,
        __global ulong *chunk_2,
        __global ulong *chunk_3,
        __global ulong *chunk_4,
        __global ulong *chunk_5,
        __global ulong *out,
        __global int *refs_1_1_524288,
        __global int *refs_4_4_16384,
        __global int *seg_1_1_524288,
        __global int *seg_4_4_16384,
        __global ulong *seed,
        int threads,
        int threads_per_chunk,
        int memsize,
        int addrsize,
        int parallelism) {
    __local ulong state[4 * BLOCK_SIZE_ULONG];
    ulong a, b, c, d;
    ulong4 buffer = 0;

    ulong chunks[6];
    chunks[0] = (ulong)chunk_0;
    chunks[1] = (ulong)chunk_1;
    chunks[2] = (ulong)chunk_2;
    chunks[3] = (ulong)chunk_3;
    chunks[4] = (ulong)chunk_4;
    chunks[5] = (ulong)chunk_5;

    int hash = get_group_id(0);
    int local_id = get_local_id(0);
    int id = local_id % ITEMS_PER_SEGMENT;
    int segment = local_id / ITEMS_PER_SEGMENT;
    int offset = id * MEMORY_CHUNK_PER_ITEM;

    __global int *addresses = parallelism == 1 ? refs_1_1_524288 : refs_4_4_16384;
    __global int *segments = parallelism == 1 ? seg_1_1_524288 : seg_4_4_16384;
    int blocks = parallelism == 1 ? 524288 : 16384;
    int segments_in_lane = parallelism == 1 ? 1 : 16;

    int chunk_index = hash / threads_per_chunk;
    int chunk_offset = hash - chunk_index * threads_per_chunk;
    __global ulong *memory = (__global ulong *)chunks[chunk_index];
    memory = memory + chunk_offset * (memsize >> 3);

    int lane_length = parallelism == 1 ? 0 : 4096;

    __global ulong *out_mem = out + hash * 2 * parallelism * BLOCK_SIZE_ULONG;
    __global ulong *mem_seed = seed + hash * 2 * parallelism * BLOCK_SIZE_ULONG;

    __global ulong *seed_src = mem_seed + segment * 2 * BLOCK_SIZE_ULONG;
    __global ulong *seed_dst = memory + segment * lane_length * BLOCK_SIZE_ULONG;
    copy_block(seed_dst, seed_src);
    seed_src += BLOCK_SIZE_ULONG;
    seed_dst += BLOCK_SIZE_ULONG;
    copy_block(seed_dst, seed_src);

    __global ulong *next_block;
    __global ulong *prev_block;
    __global ulong *ref_block;

    __local ulong *local_state = state + segment * BLOCK_SIZE_ULONG;

    for(int s=0; s<segments_in_lane; s++) {
        __global int *curr_seg = segments  + 3 * (s * parallelism + segment);
        __global int *addr = addresses + 3 * curr_seg[0];
        __global int *stop_addr = addresses + 3 * curr_seg[1];
        int with_xor = curr_seg[2];

        for(; addr < stop_addr; addr += 3) {
            if(addr[0] != -1) {
                next_block = memory + addr[0] * BLOCK_SIZE_ULONG;
            }
            if(addr[1] != -1) {
                prev_block = memory + addr[1] * BLOCK_SIZE_ULONG;
                copy_block(local_state, prev_block);
                buffer = vload4(0, &local_state[offset]);
            }
            ref_block = memory + addr[2] * BLOCK_SIZE_ULONG;

            buffer ^= vload4(0, &ref_block[offset]);
            vstore4(buffer, 0, &local_state[offset]);

            if(with_xor == 1) {
                buffer ^= vload4(0, &next_block[offset]);
            }

            G1(local_state, offsets_round_1[id]);
            barrier(CLK_LOCAL_MEM_FENCE);
            G2(local_state, offsets_round_2[id]);
            barrier(CLK_LOCAL_MEM_FENCE);
            G1(local_state, offsets_round_3[id]);
            barrier(CLK_LOCAL_MEM_FENCE);
            G2(local_state, offsets_round_4[id]);
            barrier(CLK_LOCAL_MEM_FENCE);

            buffer ^= vload4(0, &local_state[offset]);
            vstore4(buffer, 0, &local_state[offset]);

            if(addr[0] != -1) {
                vstore4(buffer, 0, &next_block[offset]);
            }
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);
    int stride = (parallelism == 1) ? 4 : 1;
    int dst_addr = (parallelism == 1) ? addrsize : (addrsize - 3);

    int result_block = (parallelism == 1) ? 0 : addresses[dst_addr * 3 + 1];

    next_block = memory + result_block * BLOCK_SIZE_ULONG;
    if(parallelism == 1) {
        copy_block(out_mem, next_block);
    }
    else {
        copy_block_small(out_mem, next_block);
    }

    for(;dst_addr < addrsize; ++dst_addr) {
        next_block = memory + addresses[dst_addr * 3 + 2] * BLOCK_SIZE_ULONG;
        if(parallelism == 1) {
            xor_block(out_mem, next_block);
        }
        else {
            xor_block_small(out_mem, next_block);
        }
    }
};
