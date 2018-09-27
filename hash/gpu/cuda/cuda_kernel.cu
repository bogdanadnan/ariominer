#include <driver_types.h>
#include "../../../common/common.h"
#include "../../../app/arguments.h"

#include "../../hasher.h"
#include "../../argon2/argon2.h"

#include "cuda_hasher.h"

#define MEMORY_CHUNK_PER_ITEM           4
#define ITEMS_PER_SEGMENT               32
#define BLOCK_SIZE_ULONG                128
#define KERNEL_WORKGROUP_SIZE   		32

__device__ uint64_t upsample(uint32_t hi, uint32_t lo)
{
	return ((uint64_t)hi << 32) | (uint64_t)lo;
}

__device__ uint64_t rotate(uint64_t x, uint32_t n)
{
	return (x >> (64-n)) | (x << n);
}

__device__ uint64_t rotr_32(uint64_t x)
{
	asm("{"
		".reg .u32 x1, x2;"
		"mov.b64 {x1, x2}, %1;"
		"mov.b64 %0, {x2, x1};"
		"}" : "=l"(x) : "l"(x));
	return x;
}

__device__ uint64_t rotr_16(uint64_t x)
{
	asm("{"
		".reg .u16 x1, x2, x3, x4;"
		"mov.b64 {x1, x2, x3, x4}, %1;"
		"mov.b64 %0, {x2, x3, x4, x1};"
		"}" : "=l"(x) : "l"(x));
	return x;
}

__device__ uint64_t rotr_24(uint64_t x)
{
	asm("{"
		".reg .u32 x1, x2, x3, x4;"
		"mov.b64 {x1, x2}, %1;"
		"prmt.b32 x3, x1, x2, 0x6543;"
		"prmt.b32 x4, x1, x2, 0x2107;"
		"mov.b64 %0, {x3, x4};"
		"}" : "=l"(x) : "l"(x));
	return x;
}

__device__ uint64_t rotr_63(uint64_t x)
{
	asm("{"
		".reg .u64 t1, t2;"
		"shl.b64 t1, %1, 1;"
		"shr.b64 t2, %1, 63;"
		"add.u64 %0, t1, t2;"
		"}" : "=l"(x) : "l"(x));
	return x;
}

__device__ uint64_t fBlaMka(uint64_t x, uint64_t y) {
	asm ("{.reg .u64 d1, d2;\n\t"
		 ".reg .u32 s1, s2, s3, s4;\n\t"
		 "cvt.u32.u64    s1, %1;\n\t"
		 "cvt.u32.u64    s2, %2;\n\t"
		 "mul.lo.u32 s3, s1, s2;\n\t"
		 "mul.hi.u32 s4, s1, s2;\n\t"
		 "mov.b64 d1, {s3, s4};\n\t"
		 "shl.b64 d2, d1, 1;\n\t"
		 "add.u64 d1, d2, %1;\n\t"
		 "add.u64 %0, d1, %2;\n\t"
		 "}" : "=l"(x) : "l"(x), "l"(y));
	return x;
}
//#define fBlaMka(x, y) ((x) + (y) + 2 * upsample(__umulhi((uint32_t)(x), (uint32_t)(y)), (uint32_t)(x) * (uint32_t)y))

//    a = fBlaMka(a, b);          \
//    d = rotate(d ^ a, 32);      \
//    c = fBlaMka(c, d);          \
//    b = rotate(b ^ c, 40);      \
//    a = fBlaMka(a, b);          \
//    d = rotate(d ^ a, 48);      \
//    c = fBlaMka(c, d);          \
//    b = rotate(b ^ c, 1);       \

#define COMPUTE_PTX            \
	     ".reg .u64 d1, d2, a, b, c, d;\n\t"     \
		 ".reg .u32 s1, s2, s3, s4;\n\t"     \
		 "add.u64 d1, %0, %1;\n\t"     \
		 "cvt.u32.u64 s1, %0;\n\t"     \
		 "cvt.u32.u64 s2, %1;\n\t"     \
		 "mul.lo.u32 s3, s1, s2;\n\t"     \
		 "mul.hi.u32 s4, s1, s2;\n\t"     \
		 "mov.b64 a, {s3, s4};\n\t"     \
		 "shl.b64 d2, a, 1;\n\t"     \
		 "add.u64 a, d1, d2;\n\t"     \
		 "xor.b64 d1, %3, a;\n\t"     \
		 "mov.b64 {s1, s2}, d1;\n\t"     \
		 "mov.b64 d, {s2, s1};\n\t"     \
		 "add.u64 d1, %2, d;\n\t"     \
		 "cvt.u32.u64 s1, %2;\n\t"     \
		 "mul.lo.u32 s3, s1, s2;\n\t"     \
		 "mul.hi.u32 s4, s1, s2;\n\t"     \
		 "mov.b64 c, {s3, s4};\n\t"     \
		 "shl.b64 d2, c, 1;\n\t"     \
		 "add.u64 c, d1, d2;\n\t"     \
		 "xor.b64 d1, %1, c;\n\t"     \
		 "mov.b64 {s3, s4}, d1;\n\t"     \
		 "prmt.b32 s2, s3, s4, 0x6543;\n\t"     \
		 "prmt.b32 s1, s3, s4, 0x2107;\n\t"     \
		 "mov.b64 b, {s2, s1};\n\t"     \
		 "add.u64 d1, a, b;\n\t"     \
		 "cvt.u32.u64 s1, a;\n\t"     \
		 "mul.lo.u32 s3, s1, s2;\n\t"     \
		 "mul.hi.u32 s4, s1, s2;\n\t"     \
		 "mov.b64 a, {s3, s4};\n\t"     \
		 "shl.b64 d2, a, 1;\n\t"     \
		 "add.u64 %0, d1, d2;\n\t"     \
		 "xor.b64 d1, d, %0;\n\t"     \
		 "mov.b64 {s3, s4}, d1;\n\t"     \
		 "prmt.b32 s2, s3, s4, 0x5432;\n\t"     \
		 "prmt.b32 s1, s3, s4, 0x1076;\n\t"     \
		 "mov.b64 %3, {s2, s1};\n\t"     \
		 "add.u64 d1, c, %3;\n\t"     \
		 "cvt.u32.u64 s1, c;\n\t"     \
		 "mul.lo.u32 s3, s1, s2;\n\t"     \
		 "mul.hi.u32 s4, s1, s2;\n\t"     \
		 "mov.b64 c, {s3, s4};\n\t"     \
		 "shl.b64 d2, c, 1;\n\t"     \
		 "add.u64 %2, d1, d2;\n\t"     \
		 "xor.b64 d1, b, %2;\n\t"     \
		 "shl.b64 a, d1, 1;\n\t"     \
		 "shr.b64 b, d1, 63;\n\t"     \
		 "add.u64 %1, a, b;\n\t"

#define G_VEC(data, vec)           \
{                           \
    b = data[vec[1]]; \
    c = data[vec[2]]; \
    d = data[vec[3]]; \
	asm ("{"  \
			COMPUTE_PTX \
		 "}" : "+l"(a), "+l"(b), "+l"(c), "+l"(d)); \
    data[vec[0]] = a; \
    data[vec[1]] = b; \
    data[vec[2]] = c; \
    data[vec[3]] = d; \
}

#define G1(data)           \
{                           \
    a = data[i1_0]; \
    b = data[i1_1]; \
    c = data[i1_2]; \
    d = data[i1_3]; \
	asm ("{"  \
			COMPUTE_PTX \
		 "}" : "+l"(a), "+l"(b), "+l"(c), "+l"(d)); \
	data[i1_1] = b; \
    data[i1_2] = c; \
    data[i1_3] = d; \
}


#define G2(data)           \
{ \
    b = data[i2_1]; \
    c = data[i2_2]; \
    d = data[i2_3]; \
	asm ("{"  \
			COMPUTE_PTX \
		 "}" : "+l"(a), "+l"(b), "+l"(c), "+l"(d)); \
    data[i2_0] = a; \
    data[i2_1] = b; \
    data[i2_2] = c; \
    data[i2_3] = d; \
}

#define G3(data)           \
{                           \
    a = data[i3_0]; \
    b = data[i3_1]; \
    c = data[i3_2]; \
    d = data[i3_3]; \
	asm ("{"  \
			COMPUTE_PTX \
		 "}" : "+l"(a), "+l"(b), "+l"(c), "+l"(d)); \
	data[i3_1] = b; \
    data[i3_2] = c; \
    data[i3_3] = d; \
}

#define G4(data)           \
{                           \
    b = data[i4_1]; \
    c = data[i4_2]; \
    d = data[i4_3]; \
	asm ("{"  \
			COMPUTE_PTX \
		 "}" : "+l"(a), "+l"(b), "+l"(c), "+l"(d)); \
    data[i4_0] = a; \
    data[i4_1] = b; \
    data[i4_2] = c; \
    data[i4_3] = d; \
}

#define copy_block(dst, src) for(int i=0;i<4;i++) (dst)[i] = (src)[i]
#define xor_block(dst, src) for(int i=0;i<4;i++) (dst)[i] ^= (src)[i]

#define copy_block_small(dst, src) dst[local_id] = src[local_id]
#define xor_block_small(dst, src) dst[local_id] ^= src[local_id]

__constant__ int offsets_round_1[32][4] = {
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

__constant__ int offsets_round_2[32][4] = {
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

__constant__ int offsets_round_3[32][4] = {
		{ 0, 32, 64, 96 },
		{ 1, 33, 65, 97 },
		{ 2, 34, 66, 98 },
		{ 3, 35, 67, 99 },
		{ 4, 36, 68, 100 },
		{ 5, 37, 69, 101 },
		{ 6, 38, 70, 102 },
		{ 7, 39, 71, 103 },
		{ 8, 40, 72, 104 },
		{ 9, 41, 73, 105 },
		{ 10, 42, 74, 106 },
		{ 11, 43, 75, 107 },
		{ 12, 44, 76, 108 },
		{ 13, 45, 77, 109 },
		{ 14, 46, 78, 110 },
		{ 15, 47, 79, 111 },
		{ 16, 48, 80, 112 },
		{ 17, 49, 81, 113 },
		{ 18, 50, 82, 114 },
		{ 19, 51, 83, 115 },
		{ 20, 52, 84, 116 },
		{ 21, 53, 85, 117 },
		{ 22, 54, 86, 118 },
		{ 23, 55, 87, 119 },
		{ 24, 56, 88, 120 },
		{ 25, 57, 89, 121 },
		{ 26, 58, 90, 122 },
		{ 27, 59, 91, 123 },
		{ 28, 60, 92, 124 },
		{ 29, 61, 93, 125 },
		{ 30, 62, 94, 126 },
		{ 31, 63, 95, 127 },
};

__constant__ int offsets_round_4[32][4] = {
		{ 0, 33, 80, 113 },
		{ 1, 48, 81, 96 },
		{ 2, 35, 82, 115 },
		{ 3, 50, 83, 98 },
		{ 4, 37, 84, 117 },
		{ 5, 52, 85, 100 },
		{ 6, 39, 86, 119 },
		{ 7, 54, 87, 102 },
		{ 8, 41, 88, 121 },
		{ 9, 56, 89, 104 },
		{ 10, 43, 90, 123 },
		{ 11, 58, 91, 106 },
		{ 12, 45, 92, 125 },
		{ 13, 60, 93, 108 },
		{ 14, 47, 94, 127 },
		{ 15, 62, 95, 110 },
		{ 16, 49, 64, 97 },
		{ 17, 32, 65, 112 },
		{ 18, 51, 66, 99 },
		{ 19, 34, 67, 114 },
		{ 20, 53, 68, 101 },
		{ 21, 36, 69, 116 },
		{ 22, 55, 70, 103 },
		{ 23, 38, 71, 118 },
		{ 24, 57, 72, 105 },
		{ 25, 40, 73, 120 },
		{ 26, 59, 74, 107 },
		{ 27, 42, 75, 122 },
		{ 28, 61, 76, 109 },
		{ 29, 44, 77, 124 },
		{ 30, 63, 78, 111 },
		{ 31, 46, 79, 126 },
};

__global__ void fill_blocks(uint64_t *scratchpad,
							uint64_t *out,
							int *refs_1_1_524288,
							int *refs_4_4_16384,
							int *seg_1_1_524288,
							int *seg_4_4_16384,
							uint64_t *seed,
							int memsize,
							int addrsize,
							int parallelism) {
	__shared__ uint64_t state[4 * BLOCK_SIZE_ULONG];
	uint64_t a, b, c, d;
	uint64_t buffer[4];

	buffer[0] = buffer[1] = buffer[2] = buffer[3] = 0;

	int hash = blockIdx.x;
	int local_id = threadIdx.x;

	int id = local_id % ITEMS_PER_SEGMENT;
	int segment = local_id / ITEMS_PER_SEGMENT;
	int offset = id * MEMORY_CHUNK_PER_ITEM;

	int *addresses = parallelism == 1 ? refs_1_1_524288 : refs_4_4_16384;
	int *segments = parallelism == 1 ? seg_1_1_524288 : seg_4_4_16384;
	int segments_in_lane = parallelism == 1 ? 1 : 16;

	uint64_t *memory = scratchpad + hash * (memsize >> 3);

	int lane_length = parallelism == 1 ? 0 : 4096;

	uint64_t *out_mem = out + hash * 2 * parallelism * BLOCK_SIZE_ULONG;
	uint64_t *mem_seed = seed + hash * 2 * parallelism * BLOCK_SIZE_ULONG;

	uint64_t *seed_src = mem_seed + segment * 2 * BLOCK_SIZE_ULONG;
	uint64_t *seed_dst = memory + segment * lane_length * BLOCK_SIZE_ULONG;
	copy_block(&seed_dst[offset], &seed_src[offset]);
	seed_src += BLOCK_SIZE_ULONG;
	seed_dst += BLOCK_SIZE_ULONG;
	copy_block(&seed_dst[offset], &seed_src[offset]);

	uint64_t *next_block;
	uint64_t *prev_block;
	uint64_t *ref_block;

	uint64_t *local_state = state + segment * BLOCK_SIZE_ULONG;

	int off = (id >> 2 << 4);
	int i0_0 = off;
	int i0_1 = off + 4;
	int i0_2 = off + 8;
	int i0_3 = off + 12;

	int id0_x3 = id & 0x3;
	int id1_x3 = (id+1) & 0x3;
	int id2_x3 = (id+2) & 0x3;
	int id3_x3 = (id+3) & 0x3;

	int i1_0 = i0_0 + id0_x3;
	int i1_1 = i0_1 + id0_x3;
	int i1_2 = i0_2 + id0_x3;
	int i1_3 = i0_3 + id0_x3;

	int i2_0 = i0_0 + id0_x3;
	int i2_1 = i0_1 + id1_x3;
	int i2_2 = i0_2 + id2_x3;
	int i2_3 = i0_3 + id3_x3;

	int i3_0 = id;
	int i3_1 = id + 32;
	int i3_2 = id + 64;
	int i3_3 = id + 96;

	int i4_0 = id;
	int i4_1 = id + ((~(id % 2)) & 1) * 33 + (id % 2) * ((id / 16) * 15 + ((~(id / 16)) & 1) * 47);
	int i4_2 = id + (id / 16) * 48 + ((~(id / 16)) & 1) * 80;
	int i4_3 = id + ((~(id % 2)) & 1) * ((id / 16) * 81 + ((~(id / 16)) & 1) * 113) + (id % 2) * 95;

/*	int i1_0 = offsets_round_1[id][0];
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
	int i4_3 = offsets_round_4[id][3]; */

	for(int s=0; s<segments_in_lane; s++) {
		int *curr_seg = segments  + 3 * (s * parallelism + segment);
		int *addr = addresses + 3 * curr_seg[0];
		int *stop_addr = addresses + 3 * curr_seg[1];
		int with_xor = curr_seg[2];

		for(; addr < stop_addr; addr += 3) {
			__syncthreads();

			if(addr[0] != -1) {
				next_block = memory + addr[0] * BLOCK_SIZE_ULONG;
			}
			if(addr[1] != -1) {
				prev_block = memory + addr[1] * BLOCK_SIZE_ULONG;
				copy_block(&local_state[offset], &prev_block[offset]);
				copy_block(buffer, &local_state[offset]);
			}
			ref_block = memory + addr[2] * BLOCK_SIZE_ULONG;

			xor_block(buffer, &ref_block[offset]);
			copy_block(&local_state[offset], buffer);

			if(with_xor == 1) {
				xor_block(buffer, &next_block[offset]);
			}

			G1(local_state);
			__syncthreads();
			G2(local_state);
			__syncthreads();
			G3(local_state);
			__syncthreads();
			G4(local_state);
			__syncthreads();

			xor_block(buffer, &local_state[offset]);
			copy_block(&local_state[offset], buffer);

			if(addr[0] != -1) {
				copy_block(&next_block[offset], buffer);
			}
		}
	}

	__syncthreads();
	int dst_addr = (parallelism == 1) ? addrsize : (addrsize - 3);

	int result_block = (parallelism == 1) ? 0 : addresses[dst_addr * 3 + 1];

	next_block = memory + result_block * BLOCK_SIZE_ULONG;
	if(parallelism == 1) {
		copy_block(&out_mem[offset], &next_block[offset]);
	}
	else {
		copy_block_small(out_mem, next_block);
	}

	for(;dst_addr < addrsize; ++dst_addr) {
		next_block = memory + addresses[dst_addr * 3 + 2] * BLOCK_SIZE_ULONG;
		if(parallelism == 1) {
			xor_block(&out_mem[offset], &next_block[offset]);
		}
		else {
			xor_block_small(out_mem, next_block);
		}
	}
};

void cuda_allocate(cuda_device_info *device) {
	int max_threads = max(device->threads_profile_1_1_524288, device->threads_profile_4_4_16384);

	device->error = cudaSetDevice(device->device_index);
	if(device->error != cudaSuccess) {
		device->error_message = "Error setting current device for memory allocation.";
		return;
	}

	device->error = cudaMalloc(&device->arguments.memory, device->arguments.memory_size);
	if(device->error != cudaSuccess) {
		device->error_message = "Error allocating memory.";
		return;
	}

	device->error = cudaMalloc(&device->arguments.address_profile_1_1_524288, argon2profile_1_1_524288.block_refs_size * 3 * sizeof(int32_t));
	if(device->error != cudaSuccess) {
		device->error_message = "Error allocating memory.";
		return;
	}
	device->error = cudaMemcpy(device->arguments.address_profile_1_1_524288, argon2profile_1_1_524288.block_refs, argon2profile_1_1_524288.block_refs_size * 3 * sizeof(int32_t), cudaMemcpyHostToDevice);
	if(device->error != cudaSuccess) {
		device->error_message = "Error copying memory.";
		return;
	}

	device->error = cudaMalloc(&device->arguments.address_profile_4_4_16384, argon2profile_4_4_16384.block_refs_size * 3 * sizeof(int32_t));
	if(device->error != cudaSuccess) {
		device->error_message = "Error allocating memory.";
		return;
	}
	device->error = cudaMemcpy(device->arguments.address_profile_4_4_16384, argon2profile_4_4_16384.block_refs, argon2profile_4_4_16384.block_refs_size * 3 * sizeof(int32_t), cudaMemcpyHostToDevice);
	if(device->error != cudaSuccess) {
		device->error_message = "Error copying memory.";
		return;
	}

	device->error = cudaMalloc(&device->arguments.segments_profile_1_1_524288, 3 * sizeof(int32_t));
	if(device->error != cudaSuccess) {
		device->error_message = "Error allocating memory.";
		return;
	}
	device->error = cudaMemcpy(device->arguments.segments_profile_1_1_524288, argon2profile_1_1_524288.segments, 3 * sizeof(int32_t), cudaMemcpyHostToDevice);
	if(device->error != cudaSuccess) {
		device->error_message = "Error copying memory.";
		return;
	}

	device->error = cudaMalloc(&device->arguments.segments_profile_4_4_16384, 64 * 3 * sizeof(int32_t));
	if(device->error != cudaSuccess) {
		device->error_message = "Error allocating memory.";
		return;
	}
	device->error = cudaMemcpy(device->arguments.segments_profile_4_4_16384, argon2profile_4_4_16384.segments, 64 * 3 * sizeof(int32_t), cudaMemcpyHostToDevice);
	if(device->error != cudaSuccess) {
		device->error_message = "Error copying memory.";
		return;
	}

	device->error = cudaMalloc(&device->arguments.seed_memory[0], max_threads * 8 * ARGON2_BLOCK_SIZE);
	if(device->error != cudaSuccess) {
		device->error_message = "Error allocating memory.";
		return;
	}
	device->error = cudaMalloc(&device->arguments.seed_memory[1], max_threads * 8 * ARGON2_BLOCK_SIZE);
	if(device->error != cudaSuccess) {
		device->error_message = "Error allocating memory.";
		return;
	}
	device->error = cudaMalloc(&device->arguments.out_memory[0], max_threads * 8 * ARGON2_BLOCK_SIZE);
	if(device->error != cudaSuccess) {
		device->error_message = "Error allocating memory.";
		return;
	}
	device->error = cudaMalloc(&device->arguments.out_memory[1], max_threads * 8 * ARGON2_BLOCK_SIZE);
	if(device->error != cudaSuccess) {
		device->error_message = "Error allocating memory.";
		return;
	}

	device->error = cudaMallocHost(&device->arguments.host_seed_memory[0], max_threads * 8 * ARGON2_BLOCK_SIZE);
	if(device->error != cudaSuccess) {
		device->error_message = "Error allocating pinned memory.";
		return;
	}
	device->error = cudaMallocHost(&device->arguments.host_seed_memory[1], max_threads * 8 * ARGON2_BLOCK_SIZE);
	if(device->error != cudaSuccess) {
		device->error_message = "Error allocating pinned memory.";
		return;
	}
}

void cuda_free(cuda_device_info *device) {
	cudaSetDevice(device->device_index);

	if(device->arguments.memory != NULL)
		cudaFree(device->arguments.memory);
	if(device->arguments.address_profile_1_1_524288 != NULL)
		cudaFree(device->arguments.address_profile_1_1_524288);
	if(device->arguments.address_profile_4_4_16384 != NULL)
		cudaFree(device->arguments.address_profile_4_4_16384);
	if(device->arguments.segments_profile_1_1_524288 != NULL)
		cudaFree(device->arguments.segments_profile_1_1_524288);
	if(device->arguments.segments_profile_4_4_16384 != NULL)
		cudaFree(device->arguments.segments_profile_4_4_16384);
	if(device->arguments.seed_memory[0] != NULL)
		cudaFree(device->arguments.seed_memory[0]);
	if(device->arguments.seed_memory[1] != NULL)
		cudaFree(device->arguments.seed_memory[1]);
	if(device->arguments.out_memory[0] != NULL)
		cudaFree(device->arguments.out_memory[0]);
	if(device->arguments.out_memory[1] != NULL)
		cudaFree(device->arguments.out_memory[1]);
}

void *cuda_kernel_filler(void *memory, int threads, argon2profile *profile, void *user_data) {
	//    uint64_t start_log = microseconds();
	//    printf("Waiting for lock: %lld\n", microseconds() - start_log);
	//    start_log = microseconds();
	cuda_gpumgmt_thread_data *gpumgmt_thread = (cuda_gpumgmt_thread_data *)user_data;
	cuda_device_info *device = gpumgmt_thread->device;

	int mem_seed_count = profile->thr_cost;
	size_t work_items;

	uint32_t memsize;
	uint32_t addrsize;
	uint32_t parallelism;

	if(strcmp(profile->profile_name, "1_1_524288") == 0) {
		memsize = (uint32_t)argon2profile_1_1_524288.memsize;
		addrsize = (uint32_t)argon2profile_1_1_524288.block_refs_size;
		parallelism = argon2profile_1_1_524288.thr_cost;
	}
	else {
		memsize = (uint32_t)argon2profile_4_4_16384.memsize;
		addrsize = (uint32_t)argon2profile_4_4_16384.block_refs_size;
		parallelism = argon2profile_4_4_16384.thr_cost;
	}
	work_items = KERNEL_WORKGROUP_SIZE * parallelism;

	device->device_lock.lock();

	device->error = cudaMemcpy(device->arguments.seed_memory[gpumgmt_thread->thread_id], memory, threads * 2 * mem_seed_count * ARGON2_BLOCK_SIZE, cudaMemcpyHostToDevice);
	if (device->error != cudaSuccess) {
		device->error_message = "Error writing to gpu memory.";
		device->device_lock.unlock();
		return NULL;
	}

	fill_blocks<<<threads, work_items>>>((uint64_t*)device->arguments.memory,
			device->arguments.out_memory[gpumgmt_thread->thread_id],
			device->arguments.address_profile_1_1_524288,
			device->arguments.address_profile_4_4_16384,
			device->arguments.segments_profile_1_1_524288,
			device->arguments.segments_profile_4_4_16384,
			device->arguments.seed_memory[gpumgmt_thread->thread_id],
			memsize,
			addrsize,
			parallelism);

	device->error = cudaMemcpy(memory, device->arguments.out_memory[gpumgmt_thread->thread_id], threads * 2 * mem_seed_count * ARGON2_BLOCK_SIZE, cudaMemcpyDeviceToHost);
	if (device->error != cudaSuccess) {
		device->error_message = "Error reading gpu memory.";
		device->device_lock.unlock();
		return NULL;
	}

	device->device_lock.unlock();

	return memory;
}