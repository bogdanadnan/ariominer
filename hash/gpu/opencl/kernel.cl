#define ITEMS_PER_SEGMENT               32
#define BLOCK_SIZE_ULONG                128
#define CBLOCKS_MEMSIZE					96995
#define GBLOCKS_MEMSIZE					16384
#define CBLOCKS_REFSIZE					524286

#define fBlaMka(x, y) ((x) + (y) + 2 * upsample(mul_hi((uint)(x), (uint)(y)), (uint)(x) * (uint)y))

#define COMPUTE \
    a = fBlaMka(a, b);          \
    d = rotate(d ^ a, (ulong)32);      \
    c = fBlaMka(c, d);          \
    b = rotate(b ^ c, (ulong)40);      \
    a = fBlaMka(a, b);          \
    d = rotate(d ^ a, (ulong)48);      \
    c = fBlaMka(c, d);          \
    b = rotate(b ^ c, (ulong)1);

__constant char offsets_round_1[32][4] = {
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

__constant char offsets_round_2[32][4] = {
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

__constant char offsets_round_3[32][4] = {
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

__constant char offsets_round_4[32][4] = {
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

#define G1(data) \
{ \
	barrier(CLK_LOCAL_MEM_FENCE); \
	a = data[i1_0]; \
	b = data[i1_1]; \
	c = data[i1_2]; \
	d = data[i1_3]; \
	COMPUTE \
	data[i1_1] = b; \
    data[i1_2] = c; \
    data[i1_3] = d; \
    barrier(CLK_LOCAL_MEM_FENCE); \
}

#define G2(data) \
{ \
	b = data[i2_1]; \
	c = data[i2_2]; \
	d = data[i2_3]; \
	COMPUTE \
	data[i2_0] = a; \
	data[i2_1] = b; \
    data[i2_2] = c; \
    data[i2_3] = d; \
    barrier(CLK_LOCAL_MEM_FENCE); \
}

#define G3(data) \
{ \
	a = data[i3_0]; \
	b = data[i3_1]; \
	c = data[i3_2]; \
	d = data[i3_3]; \
	COMPUTE \
	data[i3_1] = b; \
    data[i3_2] = c; \
    data[i3_3] = d; \
    barrier(CLK_LOCAL_MEM_FENCE); \
}

#define G4(data) \
{ \
	b = data[i4_1]; \
	c = data[i4_2]; \
	d = data[i4_3]; \
	COMPUTE \
	data[i4_0] = a; \
	data[i4_1] = b; \
    data[i4_2] = c; \
    data[i4_3] = d; \
    barrier(CLK_LOCAL_MEM_FENCE); \
}

__kernel void fill_cblocks(__global ulong *chunk_0,
						__global ulong *chunk_1,
						__global ulong *chunk_2,
						__global ulong *chunk_3,
						__global ulong *chunk_4,
						__global ulong *chunk_5,
						__global ulong *seed,
						__global ulong *out,
						__global int *addresses,
						int threads_per_chunk) {
	__local ulong state[BLOCK_SIZE_ULONG];
	__local int addr[64];
	ulong4 tmp;

	int hash = get_group_id(0);
	int id = get_local_id(0);
	int offset = id * 4;

	ulong chunks[4];
	chunks[0] = (ulong)chunk_0;
	chunks[1] = (ulong)chunk_1;
	chunks[2] = (ulong)chunk_2;
	chunks[3] = (ulong)chunk_3;
//	chunks[4] = (ulong)chunk_4;
//	chunks[5] = (ulong)chunk_5;
	int chunk_index = hash / threads_per_chunk;
	int chunk_offset = hash - chunk_index * threads_per_chunk;
	__global ulong *memory = (__global ulong *)chunks[chunk_index] + chunk_offset * CBLOCKS_MEMSIZE * BLOCK_SIZE_ULONG;

	ulong a, b, c, d;

	int i1_0 = offsets_round_1[id][0];
	int i1_1 = offsets_round_1[id][1];
	int i1_2 = offsets_round_1[id][2];
	int i1_3 = offsets_round_1[id][3];

	int i2_0 = offsets_round_2[id][0];
	int i2_1 = offsets_round_2[id][1];
	int i2_2 = offsets_round_2[id][2];
	int i2_3 = offsets_round_2[id][3];

	int i3_0 = offsets_round_3[id][0];
	int i3_1 = offsets_round_3[id][1];
	int i3_2 = offsets_round_3[id][2];
	int i3_3 = offsets_round_3[id][3];

	int i4_0 = offsets_round_4[id][0];
	int i4_1 = offsets_round_4[id][1];
	int i4_2 = offsets_round_4[id][2];
	int i4_3 = offsets_round_4[id][3];

	__global ulong *out_mem = out + hash * 2 * BLOCK_SIZE_ULONG;
	__global ulong *seed_mem = seed + hash * 2 * BLOCK_SIZE_ULONG;

	vstore4(vload4(0, seed_mem + offset), 0, memory + offset);
	seed_mem += BLOCK_SIZE_ULONG;
	vstore4(vload4(0, seed_mem + offset), 0, memory + BLOCK_SIZE_ULONG + offset);

	__global int *stop_addr = addresses + CBLOCKS_REFSIZE * 2;

	tmp = vload4(0, seed_mem + offset);

	for(; addresses < stop_addr; addresses += 64) {
		addr[id] = addresses[id];
		addr[id + 32] = addresses[id + 32];

		uint i_limit = (stop_addr - addresses) >> 1;
		if(i_limit > 32) i_limit = 32;

		for(int i=0;i<i_limit;i++) {
			int addr0 = addr[i];

			tmp ^= vload4(0, memory + addr[i + 32] * BLOCK_SIZE_ULONG + offset);
			vstore4(tmp, 0, state + offset);

			G1(state);
			G2(state);
			G3(state);
			G4(state);

			tmp ^= vload4(0, state + offset);

			if (addr0 != -1)
				vstore4(tmp, 0, memory + addr0 * BLOCK_SIZE_ULONG + offset);
		}
	}
	vstore4(tmp, 0, out_mem + offset);
};

__kernel void fill_gblocks(__global ulong *chunk_0,
						__global ulong *chunk_1,
						__global ulong *chunk_2,
						__global ulong *chunk_3,
						__global ulong *chunk_4,
						__global ulong *chunk_5,
						__global ulong *seed,
						__global ulong *out,
						__global int *addresses,
						__global int *segments,
						int threads_per_chunk) {
	__local ulong scratchpad[4 * BLOCK_SIZE_ULONG];
	ulong4 tmp;
	ulong a, b, c, d;

	int hash = get_group_id(0);
	int local_id = get_local_id(0);

	int id = local_id % ITEMS_PER_SEGMENT;
	int segment = local_id / ITEMS_PER_SEGMENT;
	int offset = id * 4;

	ulong chunks[6];
	chunks[0] = (ulong)chunk_0;
	chunks[1] = (ulong)chunk_1;
	chunks[2] = (ulong)chunk_2;
	chunks[3] = (ulong)chunk_3;
	chunks[4] = (ulong)chunk_4;
	chunks[5] = (ulong)chunk_5;
	int chunk_index = hash / threads_per_chunk;
	int chunk_offset = hash - chunk_index * threads_per_chunk;
	__global ulong *memory = (__global ulong *)chunks[chunk_index] + chunk_offset * GBLOCKS_MEMSIZE * BLOCK_SIZE_ULONG;

	int i1_0 = offsets_round_1[id][0];
	int i1_1 = offsets_round_1[id][1];
	int i1_2 = offsets_round_1[id][2];
	int i1_3 = offsets_round_1[id][3];

	int i2_0 = offsets_round_2[id][0];
	int i2_1 = offsets_round_2[id][1];
	int i2_2 = offsets_round_2[id][2];
	int i2_3 = offsets_round_2[id][3];

	int i3_0 = offsets_round_3[id][0];
	int i3_1 = offsets_round_3[id][1];
	int i3_2 = offsets_round_3[id][2];
	int i3_3 = offsets_round_3[id][3];

	int i4_0 = offsets_round_4[id][0];
	int i4_1 = offsets_round_4[id][1];
	int i4_2 = offsets_round_4[id][2];
	int i4_3 = offsets_round_4[id][3];

	__global ulong *out_mem = out + hash * 8 * BLOCK_SIZE_ULONG;
	__global ulong *seed_mem = seed + hash * 8 * BLOCK_SIZE_ULONG + segment * 2 * BLOCK_SIZE_ULONG;

	__global ulong *seed_dst = memory + segment * 4096 * BLOCK_SIZE_ULONG;

	vstore4(vload4(0, seed_mem + offset), 0, seed_dst + offset);

	seed_mem += BLOCK_SIZE_ULONG;
	seed_dst += BLOCK_SIZE_ULONG;

	vstore4(vload4(0, seed_mem + offset), 0, seed_dst + offset);

	__global ulong *next_block;
	__global ulong *prev_block;
	__global ulong *ref_block;

	__local ulong *state = scratchpad + segment * BLOCK_SIZE_ULONG;

	segments += segment;
	int inc = 1022;

	for(int s=0; s<16; s++) {
		__global ushort *curr_seg = (__global ushort *)(segments + s * 4);

		ushort addr_start_idx = curr_seg[0];
		ushort prev_blk_idx = curr_seg[1];

		__global short *start_addr = (__global short *)(addresses + addr_start_idx);
		__global short *stop_addr = (__global short *)(addresses + addr_start_idx + inc);
		inc = 1024;

		prev_block = memory + prev_blk_idx * BLOCK_SIZE_ULONG;

		tmp = vload4(0, prev_block + offset);
		ulong4 ref = 0, next = 0;
		ulong4 nextref = 0;
		ref = vload4(0, memory + start_addr[1] * BLOCK_SIZE_ULONG + offset);

		for(; start_addr < stop_addr; start_addr+=2) {
			short addr0 = start_addr[0];
			next_block = memory + addr0 * BLOCK_SIZE_ULONG;

			if(s >= 4)
				next = vload4(0, next_block + offset);

			if(start_addr + 2 < stop_addr)
				nextref = vload4(0, memory + start_addr[3] * BLOCK_SIZE_ULONG + offset);

			tmp ^= ref;
			vstore4(tmp, 0, state + offset);

			G1(state);
			G2(state);
			G3(state);
			G4(state);

			if(s >= 4)
				tmp ^= next;

			tmp ^= vload4(0, state + offset);
			vstore4(tmp, 0, next_block + offset);
			barrier(CLK_GLOBAL_MEM_FENCE);

			ref = nextref;
		}
	}

	__global short *out_addr = (__global short *)(addresses + 65528);

	ulong out_data = (memory + out_addr[0] * BLOCK_SIZE_ULONG)[local_id];
	out_data ^= (memory + out_addr[1] * BLOCK_SIZE_ULONG)[local_id];
	out_data ^= (memory + out_addr[3] * BLOCK_SIZE_ULONG)[local_id];
	out_data ^= (memory + out_addr[5] * BLOCK_SIZE_ULONG)[local_id];

	out_mem[local_id] = out_data;
};
