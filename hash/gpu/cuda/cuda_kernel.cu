#include <driver_types.h>

#include "../../../common/common.h"
#include "../../../app/arguments.h"

#include "../../hasher.h"
#include "../../argon2/argon2.h"

#include "cuda_hasher.h"

#define ITEMS_PER_SEGMENT               32
#define BLOCK_SIZE_UINT4                64
#define BLOCK_SIZE_UINT                256
#define KERNEL_WORKGROUP_SIZE   		32

#define COMPUTE	\
	asm ("{"	\
		".reg .u32 s1, s2, s3, s4;\n\t"	\
		"mul.lo.u32 s3, %0, %2;\n\t"	\
		"mul.hi.u32 s4, %0, %2;\n\t"	\
		"add.cc.u32 s3, s3, s3;\n\t"	\
		"addc.u32 s4, s4, s4;\n\t"	\
		"add.cc.u32 s1, %0, %2;\n\t"	\
		"addc.u32 s2, %1, %3;\n\t"	\
		"add.cc.u32 %0, s1, s3;\n\t"	\
		"addc.u32 %1, s2, s4;\n\t"	\
		"xor.b32 s1, %0, %6;\n\t"	\
		"xor.b32 %6, %1, %7;\n\t"	\
		"mov.b32 %7, s1;\n\t"	\
		"mul.lo.u32 s3, %4, %6;\n\t"	\
		"mul.hi.u32 s4, %4, %6;\n\t"	\
		"add.cc.u32 s3, s3, s3;\n\t"	\
		"addc.u32 s4, s4, s4;\n\t"	\
		"add.cc.u32 s1, %4, %6;\n\t"	\
		"addc.u32 s2, %5, %7;\n\t"	\
		"add.cc.u32 %4, s1, s3;\n\t"	\
		"addc.u32 %5, s2, s4;\n\t"	\
		"xor.b32 s3, %2, %4;\n\t"	\
		"xor.b32 s4, %3, %5;\n\t"	\
		"shf.r.wrap.b32 %3, s4, s3, 24;\n\t"	\
		"shf.r.wrap.b32 %2, s3, s4, 24;\n\t"	\
		"mul.lo.u32 s3, %0, %2;\n\t"	\
		"mul.hi.u32 s4, %0, %2;\n\t"	\
		"add.cc.u32 s3, s3, s3;\n\t"	\
		"addc.u32 s4, s4, s4;\n\t"	\
		"add.cc.u32 s1, %0, %2;\n\t"	\
		"addc.u32 s2, %1, %3;\n\t"	\
		"add.cc.u32 %0, s1, s3;\n\t"	\
		"addc.u32 %1, s2, s4;\n\t"	\
		"xor.b32 s3, %0, %6;\n\t"	\
		"xor.b32 s4, %1, %7;\n\t"	\
		"shf.r.wrap.b32 %7, s4, s3, 16;\n\t"	\
		"shf.r.wrap.b32 %6, s3, s4, 16;\n\t"	\
		"mul.lo.u32 s3, %4, %6;\n\t"	\
		"mul.hi.u32 s4, %4, %6;\n\t"	\
		"add.cc.u32 s3, s3, s3;\n\t"	\
		"addc.u32 s4, s4, s4;\n\t"	\
		"add.cc.u32 s1, %4, %6;\n\t"	\
		"addc.u32 s2, %5, %7;\n\t"	\
		"add.cc.u32 %4, s1, s3;\n\t"	\
		"addc.u32 %5, s2, s4;\n\t"	\
		"xor.b32 s3, %2, %4;\n\t"	\
		"xor.b32 s4, %3, %5;\n\t"	\
		"shf.r.wrap.b32 %3, s3, s4, 31;\n\t"	\
		"shf.r.wrap.b32 %2, s4, s3, 31;\n\t"	\
	"}" : "+r"(tmp_a.x), "+r"(tmp_a.y), "+r"(tmp_a.z), "+r"(tmp_a.w), "+r"(tmp_b.x), "+r"(tmp_b.y), "+r"(tmp_b.z), "+r"(tmp_b.w));

#define G1(data)           \
{                           \
	COMPUTE \
	tmp_a.z = __shfl_sync(0xffffffff, tmp_a.z, i_shfl1_1); \
	tmp_a.w = __shfl_sync(0xffffffff, tmp_a.w, i_shfl1_1); \
	tmp_b.x = __shfl_sync(0xffffffff, tmp_b.x, i_shfl1_2); \
	tmp_b.y = __shfl_sync(0xffffffff, tmp_b.y, i_shfl1_2); \
	tmp_b.z = __shfl_sync(0xffffffff, tmp_b.z, i_shfl1_3); \
	tmp_b.w = __shfl_sync(0xffffffff, tmp_b.w, i_shfl1_3); \
}

#define G2(data)           \
{ \
	COMPUTE \
    data[i2_0_0] = tmp_a.x; \
    data[i2_0_1] = tmp_a.y; \
    data[i2_1_0] = tmp_a.z; \
    data[i2_1_1] = tmp_a.w; \
    data[i2_2_0] = tmp_b.x; \
    data[i2_2_1] = tmp_b.y; \
    data[i2_3_0] = tmp_b.z; \
    data[i2_3_1] = tmp_b.w; \
    __syncwarp(); \
}

#define G3(data)           \
{                           \
    tmp_a.x = data[i3_0_0]; \
    tmp_a.y = data[i3_0_1]; \
    tmp_a.z = data[i3_1_0]; \
    tmp_a.w = data[i3_1_1]; \
    tmp_b.x = data[i3_2_0]; \
    tmp_b.y = data[i3_2_1]; \
    tmp_b.z = data[i3_3_0]; \
    tmp_b.w = data[i3_3_1]; \
	COMPUTE \
	tmp_a.z = __shfl_sync(0xffffffff, tmp_a.z, i_shfl2_1); \
	tmp_a.w = __shfl_sync(0xffffffff, tmp_a.w, i_shfl2_1); \
	tmp_b.x = __shfl_sync(0xffffffff, tmp_b.x, i_shfl2_2); \
	tmp_b.y = __shfl_sync(0xffffffff, tmp_b.y, i_shfl2_2); \
	tmp_b.z = __shfl_sync(0xffffffff, tmp_b.z, i_shfl2_3); \
	tmp_b.w = __shfl_sync(0xffffffff, tmp_b.w, i_shfl2_3); \
}

#define G4(data)           \
{                           \
	COMPUTE \
    data[i4_0_0] = tmp_a.x; \
    data[i4_0_1] = tmp_a.y; \
    data[i4_1_0] = tmp_a.z; \
    data[i4_1_1] = tmp_a.w; \
    data[i4_2_0] = tmp_b.x; \
    data[i4_2_1] = tmp_b.y; \
    data[i4_3_0] = tmp_b.z; \
    data[i4_3_1] = tmp_b.w; \
    __syncwarp(); \
    tmp_a.x = data[i1_0_0]; \
    tmp_a.y = data[i1_0_1]; \
    tmp_a.z = data[i1_1_0]; \
    tmp_a.w = data[i1_1_1]; \
    tmp_b.x = data[i1_2_0]; \
    tmp_b.y = data[i1_2_1]; \
    tmp_b.z = data[i1_3_0]; \
    tmp_b.w = data[i1_3_1]; \
}

__constant__ int offsets[768] = {
		0, 4, 8, 12,
		1, 5, 9, 13,
		2, 6, 10, 14,
		3, 7, 11, 15,
		16, 20, 24, 28,
		17, 21, 25, 29,
		18, 22, 26, 30,
		19, 23, 27, 31,
		32, 36, 40, 44,
		33, 37, 41, 45,
		34, 38, 42, 46,
		35, 39, 43, 47,
		48, 52, 56, 60,
		49, 53, 57, 61,
		50, 54, 58, 62,
		51, 55, 59, 63,
		64, 68, 72, 76,
		65, 69, 73, 77,
		66, 70, 74, 78,
		67, 71, 75, 79,
		80, 84, 88, 92,
		81, 85, 89, 93,
		82, 86, 90, 94,
		83, 87, 91, 95,
		96, 100, 104, 108,
		97, 101, 105, 109,
		98, 102, 106, 110,
		99, 103, 107, 111,
		112, 116, 120, 124,
		113, 117, 121, 125,
		114, 118, 122, 126,
		115, 119, 123, 127,
		0, 5, 10, 15,
		1, 6, 11, 12,
		2, 7, 8, 13,
		3, 4, 9, 14,
		16, 21, 26, 31,
		17, 22, 27, 28,
		18, 23, 24, 29,
		19, 20, 25, 30,
		32, 37, 42, 47,
		33, 38, 43, 44,
		34, 39, 40, 45,
		35, 36, 41, 46,
		48, 53, 58, 63,
		49, 54, 59, 60,
		50, 55, 56, 61,
		51, 52, 57, 62,
		64, 69, 74, 79,
		65, 70, 75, 76,
		66, 71, 72, 77,
		67, 68, 73, 78,
		80, 85, 90, 95,
		81, 86, 91, 92,
		82, 87, 88, 93,
		83, 84, 89, 94,
		96, 101, 106, 111,
		97, 102, 107, 108,
		98, 103, 104, 109,
		99, 100, 105, 110,
		112, 117, 122, 127,
		113, 118, 123, 124,
		114, 119, 120, 125,
		115, 116, 121, 126,
		0, 32, 64, 96,
		1, 33, 65, 97,
		2, 34, 66, 98,
		3, 35, 67, 99,
		4, 36, 68, 100,
		5, 37, 69, 101,
		6, 38, 70, 102,
		7, 39, 71, 103,
		8, 40, 72, 104,
		9, 41, 73, 105,
		10, 42, 74, 106,
		11, 43, 75, 107,
		12, 44, 76, 108,
		13, 45, 77, 109,
		14, 46, 78, 110,
		15, 47, 79, 111,
		16, 48, 80, 112,
		17, 49, 81, 113,
		18, 50, 82, 114,
		19, 51, 83, 115,
		20, 52, 84, 116,
		21, 53, 85, 117,
		22, 54, 86, 118,
		23, 55, 87, 119,
		24, 56, 88, 120,
		25, 57, 89, 121,
		26, 58, 90, 122,
		27, 59, 91, 123,
		28, 60, 92, 124,
		29, 61, 93, 125,
		30, 62, 94, 126,
		31, 63, 95, 127,
		0, 33, 80, 113,
		1, 48, 81, 96,
		2, 35, 82, 115,
		3, 50, 83, 98,
		4, 37, 84, 117,
		5, 52, 85, 100,
		6, 39, 86, 119,
		7, 54, 87, 102,
		8, 41, 88, 121,
		9, 56, 89, 104,
		10, 43, 90, 123,
		11, 58, 91, 106,
		12, 45, 92, 125,
		13, 60, 93, 108,
		14, 47, 94, 127,
		15, 62, 95, 110,
		16, 49, 64, 97,
		17, 32, 65, 112,
		18, 51, 66, 99,
		19, 34, 67, 114,
		20, 53, 68, 101,
		21, 36, 69, 116,
		22, 55, 70, 103,
		23, 38, 71, 118,
		24, 57, 72, 105,
		25, 40, 73, 120,
		26, 59, 74, 107,
		27, 42, 75, 122,
		28, 61, 76, 109,
		29, 44, 77, 124,
		30, 63, 78, 111,
		31, 46, 79, 126,
        0, 1, 2, 3,
        1, 2, 3, 0,
        2, 3, 0, 1,
        3, 0, 1, 2,
        4, 5, 6, 7,
        5, 6, 7, 4,
        6, 7, 4, 5,
        7, 4, 5, 6,
        8, 9, 10, 11,
        9, 10, 11, 8,
        10, 11, 8, 9,
        11, 8, 9, 10,
        12, 13, 14, 15,
        13, 14, 15, 12,
        14, 15, 12, 13,
        15, 12, 13, 14,
        16, 17, 18, 19,
        17, 18, 19, 16,
        18, 19, 16, 17,
        19, 16, 17, 18,
        20, 21, 22, 23,
        21, 22, 23, 20,
        22, 23, 20, 21,
        23, 20, 21, 22,
        24, 25, 26, 27,
        25, 26, 27, 24,
        26, 27, 24, 25,
        27, 24, 25, 26,
        28, 29, 30, 31,
        29, 30, 31, 28,
        30, 31, 28, 29,
        31, 28, 29, 30,
        0, 1, 16, 17,
        1, 16, 17, 0,
        2, 3, 18, 19,
        3, 18, 19, 2,
        4, 5, 20, 21,
        5, 20, 21, 4,
        6, 7, 22, 23,
        7, 22, 23, 6,
        8, 9, 24, 25,
        9, 24, 25, 8,
        10, 11, 26, 27,
        11, 26, 27, 10,
        12, 13, 28, 29,
        13, 28, 29, 12,
        14, 15, 30, 31,
        15, 30, 31, 14,
        16, 17, 0, 1,
        17, 0, 1, 16,
        18, 19, 2, 3,
        19, 2, 3, 18,
        20, 21, 4, 5,
        21, 4, 5, 20,
        22, 23, 6, 7,
        23, 6, 7, 22,
        24, 25, 8, 9,
        25, 8, 9, 24,
        26, 27, 10, 11,
        27, 10, 11, 26,
        28, 29, 12, 13,
        29, 12, 13, 28,
        30, 31, 14, 15,
        31, 14, 15, 30
};

inline __host__ __device__ void operator^=( uint4& a, uint4 s) {
   a.x ^= s.x; a.y ^= s.y; a.z ^= s.z; a.w ^= s.w;
}

__global__ void fill_blocks_cpu(uint32_t *scratchpad0,
                                uint32_t *scratchpad1,
                                uint32_t *scratchpad2,
                                uint32_t *scratchpad3,
                                uint32_t *scratchpad4,
                                uint32_t *scratchpad5,
                                uint32_t *seed,
                                uint32_t *out,
                                int32_t *addresses,
                                int memsize,
                                int threads_per_chunk,
								int thread_idx) {
	__shared__ uint32_t state[BLOCK_SIZE_UINT];
	__shared__ int32_t addr[64];

	uint4 tmp_a, tmp_b, tmp_c, tmp_d, tmp_p, tmp_q;

	int hash = blockIdx.x;
    int mem_hash = hash + thread_idx;
	int id = threadIdx.x;

	int offset = id << 2;

	int i1_0_0 = 2 * offsets[offset];
	int i1_0_1 = i1_0_0 + 1;
	int i1_1_0 = 2 * offsets[offset + 1];
    int i1_1_1 = i1_1_0 + 1;
	int i1_2_0 = 2 * offsets[offset + 2];
    int i1_2_1 = i1_2_0 + 1;
	int i1_3_0 = 2 * offsets[offset + 3];
    int i1_3_1 = i1_3_0 + 1;

	int i2_0_0 = 2 * offsets[offset + 128];
    int i2_0_1 = i2_0_0 + 1;
	int i2_1_0 = 2 * offsets[offset + 129];
    int i2_1_1 = i2_1_0 + 1;
	int i2_2_0 = 2 * offsets[offset + 130];
    int i2_2_1 = i2_2_0 + 1;
	int i2_3_0 = 2 * offsets[offset + 131];
    int i2_3_1 = i2_3_0 + 1;

	int i3_0_0 = 2 * offsets[offset + 256];
    int i3_0_1 = i3_0_0 + 1;
	int i3_1_0 = 2 * offsets[offset + 257];
    int i3_1_1 = i3_1_0 + 1;
	int i3_2_0 = 2 * offsets[offset + 258];
    int i3_2_1 = i3_2_0 + 1;
	int i3_3_0 = 2 * offsets[offset + 259];
    int i3_3_1 = i3_3_0 + 1;

	int i4_0_0 = 2 * offsets[offset + 384];
    int i4_0_1 = i4_0_0 + 1;
	int i4_1_0 = 2 * offsets[offset + 385];
    int i4_1_1 = i4_1_0 + 1;
	int i4_2_0 = 2 * offsets[offset + 386];
    int i4_2_1 = i4_2_0 + 1;
	int i4_3_0 = 2 * offsets[offset + 387];
    int i4_3_1 = i4_3_0 + 1;

	int i_shfl1_1 = offsets[offset + 513];
	int i_shfl1_2 = offsets[offset + 514];
	int i_shfl1_3 = offsets[offset + 515];
	int i_shfl2_1 = offsets[offset + 641];
	int i_shfl2_2 = offsets[offset + 642];
	int i_shfl2_3 = offsets[offset + 643];

    int scratchpad_location = mem_hash / threads_per_chunk;
    uint4 *memory = reinterpret_cast<uint4*>(scratchpad0);
    if(scratchpad_location == 1) memory = reinterpret_cast<uint4*>(scratchpad1);
    if(scratchpad_location == 2) memory = reinterpret_cast<uint4*>(scratchpad2);
    if(scratchpad_location == 3) memory = reinterpret_cast<uint4*>(scratchpad3);
    if(scratchpad_location == 4) memory = reinterpret_cast<uint4*>(scratchpad4);
    if(scratchpad_location == 5) memory = reinterpret_cast<uint4*>(scratchpad5);
    int hash_offset = mem_hash - scratchpad_location * threads_per_chunk;
    memory = memory + hash_offset * (memsize >> 4);

	uint32_t *seed_src = seed + hash * 2 * BLOCK_SIZE_UINT;

	uint4 *seed_dst = memory;
	seed_dst[id] = make_uint4(seed_src[i1_0_0], seed_src[i1_0_1], seed_src[i1_1_0], seed_src[i1_1_1]);
	seed_dst[id + 32] = make_uint4(seed_src[i1_2_0], seed_src[i1_2_1], seed_src[i1_3_0], seed_src[i1_3_1]);
	seed_src += BLOCK_SIZE_UINT;
	seed_dst += BLOCK_SIZE_UINT4;
    tmp_a = make_uint4(seed_src[i1_0_0], seed_src[i1_0_1], seed_src[i1_1_0], seed_src[i1_1_1]);
    tmp_b = make_uint4(seed_src[i1_2_0], seed_src[i1_2_1], seed_src[i1_3_0], seed_src[i1_3_1]);
	seed_dst[id] = tmp_a;
	seed_dst[id + 32] = tmp_b;

	uint4 *next_block;
	uint4 *ref_block;

	int32_t *stop_addr = addresses + 524286 * 2;

	for(; addresses < stop_addr; addresses += 64) {
		addr[id] = addresses[id];
		addr[id + 32] = addresses[id + 32];

		uint64_t i_limit = (stop_addr - addresses) >> 1;
		if(i_limit > 32) i_limit = 32;

		int32_t addr_n = 0;
		int32_t addr0 = addr[0];
		int32_t addr1 = addr[32];
		ref_block = memory + addr1 * BLOCK_SIZE_UINT4;
        tmp_p = ref_block[id];
        tmp_q = ref_block[id + 32];

		for(int i=0;i<i_limit;i++) {
			addr_n = addr0;
			tmp_a ^= tmp_p; tmp_b ^= tmp_q;

			if (i < i_limit - 1) {
				addr0 = addr[i + 1];
				addr1 = addr[i + 33];

				ref_block = memory + addr1 * BLOCK_SIZE_UINT4;
                tmp_p = ref_block[id];
                tmp_q = ref_block[id + 32];
			}

			tmp_c = tmp_a; tmp_d = tmp_b;

			G1(state);
			G2(state);
			G3(state);
			G4(state);

			tmp_a ^= tmp_c; tmp_b ^= tmp_d;

			if (addr_n != -1) {
				next_block = memory + addr_n * BLOCK_SIZE_UINT4;
                next_block[id] = tmp_a;
                next_block[id + 32] = tmp_b;
			}
		}
	}

	uint32_t *out_mem = out + hash * 2 * BLOCK_SIZE_UINT;
    out_mem[i1_0_0] = tmp_a.x;
    out_mem[i1_0_1] = tmp_a.y;
    out_mem[i1_1_0] = tmp_a.z;
    out_mem[i1_1_1] = tmp_a.w;
    out_mem[i1_2_0] = tmp_b.x;
    out_mem[i1_2_1] = tmp_b.y;
    out_mem[i1_3_0] = tmp_b.z;
    out_mem[i1_3_1] = tmp_b.w;
};

__global__ void fill_blocks_gpu(uint32_t *scratchpad0,
                                uint32_t *scratchpad1,
                                uint32_t *scratchpad2,
                                uint32_t *scratchpad3,
                                uint32_t *scratchpad4,
                                uint32_t *scratchpad5,
                                uint32_t *seed,
                                uint32_t *out,
                                uint32_t *addresses,
                                uint32_t *segments,
                                int memsize,
                                int threads_per_chunk,
								int thread_idx) {
	__shared__ uint32_t state[4 * BLOCK_SIZE_UINT];
	__shared__ uint32_t addr[4 * 32];

	uint4 tmp_a, tmp_b, tmp_c, tmp_d, tmp_p, tmp_q, tmp_l, tmp_m;

	int hash = blockIdx.x;
	int mem_hash = hash + thread_idx;
	int local_id = threadIdx.x;

	int id = local_id % ITEMS_PER_SEGMENT;
	int segment = local_id / ITEMS_PER_SEGMENT;

	int offset = id << 2;

	int i1_0_0 = 2 * offsets[offset];
	int i1_0_1 = i1_0_0 + 1;
	int i1_1_0 = 2 * offsets[offset + 1];
	int i1_1_1 = i1_1_0 + 1;
	int i1_2_0 = 2 * offsets[offset + 2];
	int i1_2_1 = i1_2_0 + 1;
	int i1_3_0 = 2 * offsets[offset + 3];
	int i1_3_1 = i1_3_0 + 1;

	int i2_0_0 = 2 * offsets[offset + 128];
	int i2_0_1 = i2_0_0 + 1;
	int i2_1_0 = 2 * offsets[offset + 129];
	int i2_1_1 = i2_1_0 + 1;
	int i2_2_0 = 2 * offsets[offset + 130];
	int i2_2_1 = i2_2_0 + 1;
	int i2_3_0 = 2 * offsets[offset + 131];
	int i2_3_1 = i2_3_0 + 1;

	int i3_0_0 = 2 * offsets[offset + 256];
	int i3_0_1 = i3_0_0 + 1;
	int i3_1_0 = 2 * offsets[offset + 257];
	int i3_1_1 = i3_1_0 + 1;
	int i3_2_0 = 2 * offsets[offset + 258];
	int i3_2_1 = i3_2_0 + 1;
	int i3_3_0 = 2 * offsets[offset + 259];
	int i3_3_1 = i3_3_0 + 1;

	int i4_0_0 = 2 * offsets[offset + 384];
	int i4_0_1 = i4_0_0 + 1;
	int i4_1_0 = 2 * offsets[offset + 385];
	int i4_1_1 = i4_1_0 + 1;
	int i4_2_0 = 2 * offsets[offset + 386];
	int i4_2_1 = i4_2_0 + 1;
	int i4_3_0 = 2 * offsets[offset + 387];
	int i4_3_1 = i4_3_0 + 1;

	int i_shfl1_1 = offsets[offset + 513];
	int i_shfl1_2 = offsets[offset + 514];
	int i_shfl1_3 = offsets[offset + 515];
	int i_shfl2_1 = offsets[offset + 641];
	int i_shfl2_2 = offsets[offset + 642];
	int i_shfl2_3 = offsets[offset + 643];

    int scratchpad_location = mem_hash / threads_per_chunk;
    uint4 *memory = reinterpret_cast<uint4*>(scratchpad0);
    if(scratchpad_location == 1) memory = reinterpret_cast<uint4*>(scratchpad1);
    if(scratchpad_location == 2) memory = reinterpret_cast<uint4*>(scratchpad2);
    if(scratchpad_location == 3) memory = reinterpret_cast<uint4*>(scratchpad3);
    if(scratchpad_location == 4) memory = reinterpret_cast<uint4*>(scratchpad4);
    if(scratchpad_location == 5) memory = reinterpret_cast<uint4*>(scratchpad5);
    int hash_offset = mem_hash - scratchpad_location * threads_per_chunk;
    memory = memory + hash_offset * (memsize >> 4);

	uint32_t *mem_seed = seed + hash * 8 * BLOCK_SIZE_UINT;

	uint32_t *seed_src = mem_seed + segment * 2 * BLOCK_SIZE_UINT;
	uint4 *seed_dst = memory + segment * 4096 * BLOCK_SIZE_UINT4;

	seed_dst[id] = make_uint4(seed_src[i1_0_0], seed_src[i1_0_1], seed_src[i1_1_0], seed_src[i1_1_1]);
	seed_dst[id + 32] = make_uint4(seed_src[i1_2_0], seed_src[i1_2_1], seed_src[i1_3_0], seed_src[i1_3_1]);
	seed_src += BLOCK_SIZE_UINT;
	seed_dst += BLOCK_SIZE_UINT4;
	seed_dst[id] = make_uint4(seed_src[i1_0_0], seed_src[i1_0_1], seed_src[i1_1_0], seed_src[i1_1_1]);
	seed_dst[id + 32] = make_uint4(seed_src[i1_2_0], seed_src[i1_2_1], seed_src[i1_3_0], seed_src[i1_3_1]);

	uint4 *next_block;
	uint4 *prev_block;
	uint4 *ref_block;

	uint32_t *local_state = state + segment * BLOCK_SIZE_UINT;
	uint32_t *local_addr = addr + segment * 32;

	segments += segment;
	uint16_t addr_start_idx = 0;
	uint16_t prev_blk_idx;
	int inc = 1022;

	//without xor
	for(int s=0; s<4; s++) {
		uint32_t curr_seg = segments[s * 4];

		asm("mov.b32 {%0, %1}, %2;"
		: "=h"(addr_start_idx), "=h"(prev_blk_idx) : "r"(curr_seg));

		uint32_t *addr = addresses + addr_start_idx;
		uint32_t *stop_addr = addresses + addr_start_idx + inc;
		inc = 1024;

		prev_block = memory + prev_blk_idx * BLOCK_SIZE_UINT4;
		__syncthreads();

		tmp_a = prev_block[id];
		tmp_b = prev_block[id + 32];

		for(; addr < stop_addr; addr += 32) {
			local_addr[id] = addr[id];

			uint64_t i_limit = stop_addr - addr;
			if(i_limit > 32) i_limit = 32;

			int16_t addr0, addr1;
			asm("{mov.b32 {%0, %1}, %2;}": "=h"(addr0), "=h"(addr1) : "r"(local_addr[0]));
			ref_block = memory + addr1 * BLOCK_SIZE_UINT4;
			tmp_p = ref_block[id];
			tmp_q = ref_block[id + 32];

			for(int i=0;i<i_limit;i++) {
				next_block = memory + addr0 * BLOCK_SIZE_UINT4;

                tmp_a ^= tmp_p; tmp_b ^= tmp_q;

				if (i < (i_limit - 1)) {
					asm("{mov.b32 {%0, %1}, %2;}": "=h"(addr0), "=h"(addr1) : "r"(local_addr[i + 1]));
					ref_block = memory + addr1 * BLOCK_SIZE_UINT4;
                    tmp_p = ref_block[id];
                    tmp_q = ref_block[id + 32];
				}

				tmp_c = tmp_a; tmp_d = tmp_b;

				G1(local_state);
				G2(local_state);
				G3(local_state);
				G4(local_state);

				tmp_a ^= tmp_c; tmp_b ^= tmp_d;

				next_block[id] = tmp_a;
				next_block[id + 32] = tmp_b;
			}
		}
	}

	// with xor
	for(int s=4; s<16; s++) {
		uint32_t curr_seg = segments[s * 4];

		asm("mov.b32 {%0, %1}, %2;"
		: "=h"(addr_start_idx), "=h"(prev_blk_idx) : "r"(curr_seg));

		uint32_t *addr = addresses + addr_start_idx;
		uint32_t *stop_addr = addresses + addr_start_idx + 1024;

		prev_block = memory + prev_blk_idx * BLOCK_SIZE_UINT4;
		__syncthreads();

		tmp_a = prev_block[id];
		tmp_b = prev_block[id + 32];

		for(; addr < stop_addr; addr += 32) {
			local_addr[id] = addr[id];

			uint16_t addr0, addr1, dont_store;
			asm("{mov.b32 {%0, %1}, %2;}": "=h"(addr0), "=h"(addr1) : "r"(local_addr[0]));
			ref_block = memory + addr1 * BLOCK_SIZE_UINT4;

            tmp_p = ref_block[id];
			tmp_q = ref_block[id + 32];

			for (int i = 0; i < 32; i++) {
				if(s >= 12) {
					dont_store = addr0 >> 15;
					addr0 = addr0 & 32767;
				}
				else {
					dont_store = 0;
				}
				next_block = memory + addr0 * BLOCK_SIZE_UINT4;

				tmp_l = next_block[id];
				tmp_m = next_block[id + 32];

                tmp_a ^= tmp_p; tmp_b ^= tmp_q;

				if (i < 31) {
					asm("{mov.b32 {%0, %1}, %2;}": "=h"(addr0), "=h"(addr1) : "r"(local_addr[i + 1]));
					ref_block = memory + addr1 * BLOCK_SIZE_UINT4;
                    tmp_p = ref_block[id];
                    tmp_q = ref_block[id + 32];
				}

				tmp_c = tmp_a; tmp_d = tmp_b;

				G1(local_state);
				G2(local_state);
				G3(local_state);
				G4(local_state);

				tmp_c ^= tmp_l; tmp_d ^= tmp_m;
				tmp_a ^= tmp_c; tmp_b ^= tmp_d;

                if(!dont_store) {
                    next_block[id] = tmp_a;
                    next_block[id + 32] = tmp_b;
				}
			}
		}
	}

	__syncthreads();

	int dst_addr = 65528;

	offset = ((segment / 2) << 7) + (id << 2) + ((segment % 2) << 1);
	uint32_t *block = reinterpret_cast<uint32_t*>(memory) + ((int16_t*)(&addresses[dst_addr]))[0] * BLOCK_SIZE_UINT;

    uint32_t data0 = block[offset];
    uint32_t data1 = block[offset + 1];

	for(;dst_addr < 65531; ++dst_addr) {
		block = reinterpret_cast<uint32_t*>(memory) + ((int16_t*)(&addresses[dst_addr]))[1] * BLOCK_SIZE_UINT;
        data0 ^= block[offset];
        data1 ^= block[offset + 1];
	}

	int idx0 = i1_0_0; int idx1 = i1_0_1;
	if (segment == 1) { idx0 = i1_1_0; idx1 = i1_1_1; }
	if (segment == 2) { idx0 = i1_2_0; idx1 = i1_2_1; }
	if (segment == 3) { idx0 = i1_3_0; idx1 = i1_3_1; }

	uint32_t *out_mem = out + hash * 8 * BLOCK_SIZE_UINT;
    out_mem[idx0] = data0;
    out_mem[idx1] = data1;
};

void cuda_allocate(cuda_device_info *device, double chunks, size_t chunk_size) {
	device->error = cudaSetDevice(device->cuda_index);
	if(device->error != cudaSuccess) {
		device->error_message = "Error setting current device for memory allocation.";
		return;
	}

	size_t allocated_mem_for_current_chunk = 0;

	if (chunks > 0) {
		allocated_mem_for_current_chunk = chunks > 1 ? chunk_size : (size_t)ceil(chunk_size * chunks);
		chunks -= 1;
	}
	else {
		allocated_mem_for_current_chunk = 1;
	}
	device->error = cudaMalloc(&device->arguments.memory_chunk_0, allocated_mem_for_current_chunk);
	if (device->error != cudaSuccess) {
		device->error_message = "Error allocating memory.";
		return;
	}
	if (chunks > 0) {
		allocated_mem_for_current_chunk = chunks > 1 ? chunk_size : (size_t)ceil(chunk_size * chunks);
		chunks -= 1;
	}
	else {
		allocated_mem_for_current_chunk = 1;
	}
	device->error = cudaMalloc(&device->arguments.memory_chunk_1, allocated_mem_for_current_chunk);
	if (device->error != cudaSuccess) {
		device->error_message = "Error allocating memory.";
		return;
	}
	if (chunks > 0) {
		allocated_mem_for_current_chunk = chunks > 1 ? chunk_size : (size_t)ceil(chunk_size * chunks);
		chunks -= 1;
	}
	else {
		allocated_mem_for_current_chunk = 1;
	}
	device->error = cudaMalloc(&device->arguments.memory_chunk_2, allocated_mem_for_current_chunk);
	if (device->error != cudaSuccess) {
		device->error_message = "Error allocating memory.";
		return;
	}
	if (chunks > 0) {
		allocated_mem_for_current_chunk = chunks > 1 ? chunk_size : (size_t)ceil(chunk_size * chunks);
		chunks -= 1;
	}
	else {
		allocated_mem_for_current_chunk = 1;
	}
	device->error = cudaMalloc(&device->arguments.memory_chunk_3, allocated_mem_for_current_chunk);
	if (device->error != cudaSuccess) {
		device->error_message = "Error allocating memory.";
		return;
	}
	if (chunks > 0) {
		allocated_mem_for_current_chunk = chunks > 1 ? chunk_size : (size_t)ceil(chunk_size * chunks);
		chunks -= 1;
	}
	else {
		allocated_mem_for_current_chunk = 1;
	}
	device->error = cudaMalloc(&device->arguments.memory_chunk_4, allocated_mem_for_current_chunk);
	if (device->error != cudaSuccess) {
		device->error_message = "Error allocating memory.";
		return;
	}
	if (chunks > 0) {
		allocated_mem_for_current_chunk = chunks > 1 ? chunk_size : (size_t)ceil(chunk_size * chunks);
		chunks -= 1;
	}
	else {
		allocated_mem_for_current_chunk = 1;
	}
	device->error = cudaMalloc(&device->arguments.memory_chunk_5, allocated_mem_for_current_chunk);
	if (device->error != cudaSuccess) {
		device->error_message = "Error allocating memory.";
		return;
	}

	//optimise address sizes
	int32_t *addresses_1_1_524288 = (int32_t *)malloc((argon2profile_1_1_524288.block_refs_size + 2) * 2 * sizeof(int32_t)); //add 2 to ref_size to be exact multiple of 32

	for(int i=0;i<argon2profile_1_1_524288.block_refs_size;i++) {
		int ref_chunk_idx = (i / 32) * 64;
		int ref_idx = i % 32;

		addresses_1_1_524288[ref_chunk_idx + ref_idx] = argon2profile_1_1_524288.block_refs[i*4];
		addresses_1_1_524288[ref_chunk_idx + ref_idx + 32] = argon2profile_1_1_524288.block_refs[i*4 + 2];
	}
	device->error = cudaMalloc(&device->arguments.address_profile_1_1_524288, (argon2profile_1_1_524288.block_refs_size + 2) * 2 * sizeof(int32_t));
	if(device->error != cudaSuccess) {
		device->error_message = "Error allocating memory.";
		return;
	}
	device->error = cudaMemcpy(device->arguments.address_profile_1_1_524288, addresses_1_1_524288, (argon2profile_1_1_524288.block_refs_size + 2) * 2 * sizeof(int32_t), cudaMemcpyHostToDevice);
	if(device->error != cudaSuccess) {
		device->error_message = "Error copying memory.";
		return;
	}
	free(addresses_1_1_524288);

	//optimise address sizes
	uint16_t *addresses_4_4_16384 = (uint16_t *)malloc(argon2profile_4_4_16384.block_refs_size * 2 * sizeof(uint16_t));
	for(int i=0;i<argon2profile_4_4_16384.block_refs_size;i++) {
		addresses_4_4_16384[i*2] = argon2profile_4_4_16384.block_refs[i*4 + (i >= 65528 ? 1 : 0)];
		addresses_4_4_16384[i*2 + 1] = argon2profile_4_4_16384.block_refs[i*4 + 2];
		if(argon2profile_4_4_16384.block_refs[i*4 + 3] == 0) {
            addresses_4_4_16384[i*2] |= 32768;
		}
	}
	device->error = cudaMalloc(&device->arguments.address_profile_4_4_16384, argon2profile_4_4_16384.block_refs_size * 2 * sizeof(uint16_t));
	if(device->error != cudaSuccess) {
		device->error_message = "Error allocating memory.";
		return;
	}
	device->error = cudaMemcpy(device->arguments.address_profile_4_4_16384, addresses_4_4_16384, argon2profile_4_4_16384.block_refs_size * 2 * sizeof(uint16_t), cudaMemcpyHostToDevice);
	if(device->error != cudaSuccess) {
		device->error_message = "Error copying memory.";
		return;
	}
	free(addresses_4_4_16384);

	//reorganize segments data
	uint16_t *segments_4_4_16384 = (uint16_t *)malloc(64 * 2 * sizeof(uint16_t));
	for(int i=0;i<64;i++) {
		int seg_start = argon2profile_4_4_16384.segments[i*3];
		segments_4_4_16384[i*2] = seg_start;
		segments_4_4_16384[i*2 + 1] = argon2profile_4_4_16384.block_refs[seg_start*4 + 1];
	}
	device->error = cudaMalloc(&device->arguments.segments_profile_4_4_16384, 64 * 2 * sizeof(uint16_t));
	if(device->error != cudaSuccess) {
		device->error_message = "Error allocating memory.";
		return;
	}
	device->error = cudaMemcpy(device->arguments.segments_profile_4_4_16384, segments_4_4_16384, 64 * 2 * sizeof(uint16_t), cudaMemcpyHostToDevice);
	if(device->error != cudaSuccess) {
		device->error_message = "Error copying memory.";
		return;
	}
	free(segments_4_4_16384);

    size_t max_threads = max(device->profile_info.threads_profile_4_4_16384, device->profile_info.threads_profile_1_1_524288);
    size_t accessory_memory_size = max_threads * 8 * ARGON2_BLOCK_SIZE;
    device->error = cudaMalloc(&device->arguments.seed_memory[0], accessory_memory_size);
    if (device->error != cudaSuccess) {
        device->error_message = "Error allocating memory.";
        return;
    }
    device->error = cudaMalloc(&device->arguments.out_memory[0], accessory_memory_size);
    if (device->error != cudaSuccess) {
        device->error_message = "Error allocating memory.";
        return;
    }
    device->error = cudaMallocHost(&device->arguments.host_seed_memory[0], accessory_memory_size);
    if (device->error != cudaSuccess) {
        device->error_message = "Error allocating pinned memory.";
        return;
    }
    device->error = cudaMalloc(&device->arguments.seed_memory[1], accessory_memory_size);
    if (device->error != cudaSuccess) {
        device->error_message = "Error allocating memory.";
        return;
    }
    device->error = cudaMalloc(&device->arguments.out_memory[1], accessory_memory_size);
    if (device->error != cudaSuccess) {
        device->error_message = "Error allocating memory.";
        return;
    }
    device->error = cudaMallocHost(&device->arguments.host_seed_memory[1], accessory_memory_size);
    if (device->error != cudaSuccess) {
        device->error_message = "Error allocating pinned memory.";
        return;
    }
}

void cuda_free(cuda_device_info *device) {
	cudaSetDevice(device->cuda_index);

	if(device->arguments.address_profile_1_1_524288 != NULL) {
		cudaFree(device->arguments.address_profile_1_1_524288);
		device->arguments.address_profile_1_1_524288 = NULL;
	}

	if(device->arguments.address_profile_4_4_16384 != NULL) {
		cudaFree(device->arguments.address_profile_4_4_16384);
		device->arguments.address_profile_4_4_16384 = NULL;
	}

	if(device->arguments.segments_profile_4_4_16384 != NULL) {
		cudaFree(device->arguments.segments_profile_4_4_16384);
		device->arguments.segments_profile_4_4_16384 = NULL;
	}

    if(device->arguments.memory_chunk_0 != NULL) {
        cudaFree(device->arguments.memory_chunk_0);
        device->arguments.memory_chunk_0 = NULL;
    }

    if(device->arguments.memory_chunk_1 != NULL) {
        cudaFree(device->arguments.memory_chunk_1);
        device->arguments.memory_chunk_1 = NULL;
    }

    if(device->arguments.memory_chunk_2 != NULL) {
        cudaFree(device->arguments.memory_chunk_2);
        device->arguments.memory_chunk_2 = NULL;
    }

    if(device->arguments.memory_chunk_3 != NULL) {
        cudaFree(device->arguments.memory_chunk_3);
        device->arguments.memory_chunk_3 = NULL;
    }

    if(device->arguments.memory_chunk_4 != NULL) {
        cudaFree(device->arguments.memory_chunk_4);
        device->arguments.memory_chunk_4 = NULL;
    }

    if(device->arguments.memory_chunk_5 != NULL) {
        cudaFree(device->arguments.memory_chunk_5);
        device->arguments.memory_chunk_5 = NULL;
    }

	if(device->arguments.seed_memory != NULL) {
		for(int i=0;i<2;i++) {
			if(device->arguments.seed_memory[i] != NULL)
				cudaFree(device->arguments.seed_memory[i]);
			device->arguments.seed_memory[i] = NULL;
		}
	}

	if(device->arguments.out_memory != NULL) {
		for(int i=0;i<2;i++) {
			if(device->arguments.out_memory[i] != NULL)
				cudaFree(device->arguments.out_memory[i]);
			device->arguments.out_memory[i] = NULL;
		}
	}

	if(device->arguments.host_seed_memory != NULL) {
		for(int i=0;i<2;i++) {
			if(device->arguments.host_seed_memory[i] != NULL)
				cudaFreeHost(device->arguments.host_seed_memory[i]);
			device->arguments.host_seed_memory[i] = NULL;
		}
	}

	cudaDeviceReset();
}

void *cuda_kernel_filler(void *memory, int threads, argon2profile *profile, void *user_data) {
	cuda_gpumgmt_thread_data *gpumgmt_thread = (cuda_gpumgmt_thread_data *)user_data;
	cuda_device_info *device = gpumgmt_thread->device;
	cudaStream_t stream = (cudaStream_t)gpumgmt_thread->device_data;

	int mem_seed_count = profile->thr_cost;
	size_t work_items;

	uint32_t memsize;
	uint32_t parallelism;

	if(strcmp(profile->profile_name, "1_1_524288") == 0) {
		memsize = (uint32_t)argon2profile_1_1_524288.memsize;
		parallelism = argon2profile_1_1_524288.thr_cost;
	}
	else {
		memsize = (uint32_t)argon2profile_4_4_16384.memsize;
		parallelism = argon2profile_4_4_16384.thr_cost;
	}
	work_items = KERNEL_WORKGROUP_SIZE * parallelism;

	gpumgmt_thread->lock();

	device->error = cudaMemcpyAsync(device->arguments.seed_memory[gpumgmt_thread->thread_id], memory, threads * 2 * mem_seed_count * ARGON2_BLOCK_SIZE, cudaMemcpyHostToDevice, stream);
	if (device->error != cudaSuccess) {
		device->error_message = "Error writing to gpu memory.";
		gpumgmt_thread->unlock();
		return NULL;
	}

	if(parallelism == 1) {
		fill_blocks_cpu<<<threads, work_items, 0, stream>>>((uint32_t*)device->arguments.memory_chunk_0,
                (uint32_t*)device->arguments.memory_chunk_1,
                (uint32_t*)device->arguments.memory_chunk_2,
                (uint32_t*)device->arguments.memory_chunk_3,
                (uint32_t*)device->arguments.memory_chunk_4,
                (uint32_t*)device->arguments.memory_chunk_5,
				device->arguments.seed_memory[gpumgmt_thread->thread_id],
				device->arguments.out_memory[gpumgmt_thread->thread_id],
				device->arguments.address_profile_1_1_524288,
				memsize, device->profile_info.threads_per_chunk_profile_1_1_524288, gpumgmt_thread->threads_profile_1_1_524288_idx);
	}
	else {
		fill_blocks_gpu<<<threads, work_items, 0, stream>>> ((uint32_t*)device->arguments.memory_chunk_0,
                (uint32_t*)device->arguments.memory_chunk_1,
                (uint32_t*)device->arguments.memory_chunk_2,
                (uint32_t*)device->arguments.memory_chunk_3,
                (uint32_t*)device->arguments.memory_chunk_4,
                (uint32_t*)device->arguments.memory_chunk_5,
				device->arguments.seed_memory[gpumgmt_thread->thread_id],
				device->arguments.out_memory[gpumgmt_thread->thread_id],
				device->arguments.address_profile_4_4_16384,
				device->arguments.segments_profile_4_4_16384,
				memsize, device->profile_info.threads_per_chunk_profile_4_4_16384, gpumgmt_thread->threads_profile_4_4_16384_idx);
	}

	device->error = cudaMemcpyAsync(memory, device->arguments.out_memory[gpumgmt_thread->thread_id], threads * 2 * mem_seed_count * ARGON2_BLOCK_SIZE, cudaMemcpyDeviceToHost, stream);
	if (device->error != cudaSuccess) {
		device->error_message = "Error reading gpu memory.";
		gpumgmt_thread->unlock();
		return NULL;
	}

	while(cudaStreamQuery(stream) != cudaSuccess) {
		this_thread::sleep_for(chrono::milliseconds(10));
		continue;
	}

	gpumgmt_thread->unlock();

	return memory;
}
