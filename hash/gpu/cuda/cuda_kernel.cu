#include <driver_types.h>
#include "../../../common/common.h"
#include "../../../app/arguments.h"

#include "../../hasher.h"
#include "../../argon2/argon2.h"

#include "cuda_hasher.h"

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

#define fBlaMka(x, y) ((x) + (y) + 2 * upsample(__umulhi((uint32_t)(x), (uint32_t)(y)), (uint32_t)(x) * (uint32_t)y))

#if defined(USE_NON_PTX_CUDA)

	#define COMPUTE            \
		a = fBlaMka(a, b);          \
		d = rotate(d ^ a, 32);      \
		c = fBlaMka(c, d);          \
		b = rotate(b ^ c, 40);      \
		a = fBlaMka(a, b);          \
		d = rotate(d ^ a, 48);      \
		c = fBlaMka(c, d);          \
		b = rotate(b ^ c, 1);
#else

    #define COMPUTE            \
	asm ("{"  \
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
		 "add.u64 %1, a, b;\n\t" \
	"}" : "+l"(a), "+l"(b), "+l"(c), "+l"(d));

#endif

#define G1(data)           \
{                           \
	COMPUTE \
	data[i1_1] = b; \
    data[i1_2] = c; \
    data[i1_3] = d; \
    __syncthreads(); \
}

#define G2(data)           \
{ \
    b = data[i2_1]; \
    c = data[i2_2]; \
    d = data[i2_3]; \
	COMPUTE \
    data[i2_0] = a; \
    data[i2_1] = b; \
    data[i2_2] = c; \
    data[i2_3] = d; \
    __syncthreads(); \
}

#define G3(data)           \
{                           \
    a = data[i3_0]; \
    b = data[i3_1]; \
    c = data[i3_2]; \
    d = data[i3_3]; \
	COMPUTE \
	data[i3_1] = b; \
    data[i3_2] = c; \
    data[i3_3] = d; \
    __syncthreads(); \
}

#define G4(data)           \
{                           \
    b = data[i4_1]; \
    c = data[i4_2]; \
    d = data[i4_3]; \
	COMPUTE \
    data[i4_0] = a; \
    data[i4_1] = b; \
    data[i4_2] = c; \
    data[i4_3] = d; \
    __syncthreads(); \
    a = data[i1_0]; \
    b = data[i1_1]; \
    c = data[i1_2]; \
    d = data[i1_3]; \
}

#define copy_block(D, S) \
	dst = (uint32_t*)(D); \
	src = (uint32_t*)(S); \
	for(int k=0;k<8;k++)  \
		dst[id + 32 * k] = src[id + 32 * k];

int offsets[512] = {
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
		31, 46, 79, 126
};

__global__ void fill_blocks_cpu(uint64_t *scratchpad,
							uint64_t *seed,
							uint64_t *out,
							uint64_t *addresses,
							int *offsets_,
							int memsize) {
	__shared__ uint64_t state[BLOCK_SIZE_ULONG];
	uint32_t *src, *dst;

	uint64_t a, b, c, d, x, y, z, w;

	int hash = blockIdx.x;
	int id = threadIdx.x;

	int offset = id << 2;

	int i1_0 = offsets_[offset];
	int i1_1 = offsets_[offset + 1];
	int i1_2 = offsets_[offset + 2];
	int i1_3 = offsets_[offset + 3];

	int i2_0 = offsets_[offset + 128];
	int i2_1 = offsets_[offset + 129];
	int i2_2 = offsets_[offset + 130];
	int i2_3 = offsets_[offset + 131];

	int i3_0 = offsets_[offset + 256];
	int i3_1 = offsets_[offset + 257];
	int i3_2 = offsets_[offset + 258];
	int i3_3 = offsets_[offset + 259];

	int i4_0 = offsets_[offset + 384];
	int i4_1 = offsets_[offset + 385];
	int i4_2 = offsets_[offset + 386];
	int i4_3 = offsets_[offset + 387];

	uint64_t *memory = scratchpad + hash * (memsize >> 3);

	uint64_t *out_mem = out + hash * 2 * BLOCK_SIZE_ULONG;
	uint64_t *mem_seed = seed + hash * 2 * BLOCK_SIZE_ULONG;

	uint64_t *seed_dst = memory;
	copy_block(seed_dst, mem_seed);
	mem_seed += BLOCK_SIZE_ULONG;
	seed_dst += BLOCK_SIZE_ULONG;
	copy_block(seed_dst, mem_seed);

	uint64_t *next_block;
	uint64_t *ref_block;

	uint64_t *stop_addr = addresses + 524286;

	a = seed_dst[i1_0];
	b = seed_dst[i1_1];
	c = seed_dst[i1_2];
	d = seed_dst[i1_3];

	for(; addresses < stop_addr; addresses += 32) {
		int64_t store_addr = addresses[id];
		uint64_t i_limit = stop_addr - addresses;
		if(i_limit > 32) i_limit = 32;

		for(int i=0;i<i_limit;i++) {
			uint64_t addr = __shfl_sync(0xFFFFFFFF, store_addr, i, 32);
			int32_t addr0, addr1;
			asm("mov.b64 {%0, %1}, %2;": "=r"(addr0), "=r"(addr1) : "l"(addr));

			if (addr0 != -1) {
				next_block = memory + addr0 * BLOCK_SIZE_ULONG;
			}
			ref_block = memory + addr1 * BLOCK_SIZE_ULONG;

			a ^= ref_block[i1_0];
			b ^= ref_block[i1_1];
			c ^= ref_block[i1_2];
			d ^= ref_block[i1_3];

			x = a; y = b; z = c; w = d;

			G1(state);
			G2(state);
			G3(state);
			G4(state);

			a ^= x; b ^= y; c ^= z; d ^= w;

			if (addr0 != -1) {
				next_block[i1_0] = a;
				next_block[i1_1] = b;
				next_block[i1_2] = c;
				next_block[i1_3] = d;
			}
		}
	}

	out_mem[i1_0] = a;
	out_mem[i1_1] = b;
	out_mem[i1_2] = c;
	out_mem[i1_3] = d;
};

__global__ void fill_blocks_gpu(uint64_t *scratchpad,
							uint64_t *seed,
							uint64_t *out,
							uint32_t *addresses,
							uint32_t *segments,
							int *offsets_,
							int memsize) {
	__shared__ uint64_t state[4 * BLOCK_SIZE_ULONG];

	uint64_t a, b, c, d, x, y, z, w;
	uint32_t *src, *dst;

	int hash = blockIdx.x;
	int local_id = threadIdx.x;

	int id = local_id % ITEMS_PER_SEGMENT;
	int segment = local_id / ITEMS_PER_SEGMENT;

	int offset = id << 2;

	int i1_0 = offsets_[offset];
	int i1_1 = offsets_[offset + 1];
	int i1_2 = offsets_[offset + 2];
	int i1_3 = offsets_[offset + 3];

	int i2_0 = offsets_[offset + 128];
	int i2_1 = offsets_[offset + 129];
	int i2_2 = offsets_[offset + 130];
	int i2_3 = offsets_[offset + 131];

	int i3_0 = offsets_[offset + 256];
	int i3_1 = offsets_[offset + 257];
	int i3_2 = offsets_[offset + 258];
	int i3_3 = offsets_[offset + 259];

	int i4_0 = offsets_[offset + 384];
	int i4_1 = offsets_[offset + 385];
	int i4_2 = offsets_[offset + 386];
	int i4_3 = offsets_[offset + 387];

	uint64_t *memory = scratchpad + hash * (memsize >> 3);

	uint64_t *out_mem = out + hash * 8 * BLOCK_SIZE_ULONG;
	uint64_t *mem_seed = seed + hash * 8 * BLOCK_SIZE_ULONG;

	uint64_t *seed_src = mem_seed + segment * 2 * BLOCK_SIZE_ULONG;
	uint64_t *seed_dst = memory + segment * 4096 * BLOCK_SIZE_ULONG;
	copy_block(seed_dst, seed_src);
	seed_src += BLOCK_SIZE_ULONG;
	seed_dst += BLOCK_SIZE_ULONG;
	copy_block(seed_dst, seed_src);

	uint64_t *next_block;
	uint64_t *prev_block;
	uint64_t *ref_block;

	uint64_t *local_state = state + segment * BLOCK_SIZE_ULONG;

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

		prev_block = memory + prev_blk_idx * BLOCK_SIZE_ULONG;
		__syncthreads();

		a = prev_block[i1_0];
		b = prev_block[i1_1];
		c = prev_block[i1_2];
		d = prev_block[i1_3];

		for(; addr < stop_addr; addr += 32) {
			uint32_t store_addr = addr[id];

			uint64_t i_limit = stop_addr - addr;
			if(i_limit > 32) i_limit = 32;

			for(int i=0;i<i_limit;i++) {
				uint32_t local_addr = __shfl_sync(0xFFFFFFFF, store_addr, i, 32);
				int16_t addr0, addr1;
				asm("{mov.b32 {%0, %1}, %2;}": "=h"(addr0), "=h"(addr1) : "r"(local_addr));

				next_block = memory + addr0 * BLOCK_SIZE_ULONG;
				ref_block = memory + addr1 * BLOCK_SIZE_ULONG;

				copy_block(local_state, ref_block);

				a ^= local_state[i1_0];
				b ^= local_state[i1_1];
				c ^= local_state[i1_2];
				d ^= local_state[i1_3];

				x = a; y = b; z = c; w = d;

				G1(local_state);
				G2(local_state);
				G3(local_state);
				G4(local_state);

				a ^= x; b ^= y; c ^= z; d ^= w;

				next_block[i1_0] = a;
				next_block[i1_1] = b;
				next_block[i1_2] = c;
				next_block[i1_3] = d;
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

		prev_block = memory + prev_blk_idx * BLOCK_SIZE_ULONG;
		__syncthreads();

		a = prev_block[i1_0];
		b = prev_block[i1_1];
		c = prev_block[i1_2];
		d = prev_block[i1_3];

		for(; addr < stop_addr; addr += 32) {
			uint32_t store_addr = addr[id];

			for (int i = 0; i < 32; i++) {
				uint32_t local_addr = __shfl_sync(0xFFFFFFFF, store_addr, i, 32);
				int16_t addr0, addr1;
				asm("{mov.b32 {%0, %1}, %2;}": "=h"(addr0), "=h"(addr1) : "r"(local_addr));

				next_block = memory + addr0 * BLOCK_SIZE_ULONG;
				ref_block = memory + addr1 * BLOCK_SIZE_ULONG;

				copy_block(local_state, ref_block);

				a ^= local_state[i1_0];
				b ^= local_state[i1_1];
				c ^= local_state[i1_2];
				d ^= local_state[i1_3];

				x = a; y = b; z = c; w = d;

				x ^= next_block[i1_0];
				y ^= next_block[i1_1];
				z ^= next_block[i1_2];
				w ^= next_block[i1_3];

				G1(local_state);
				G2(local_state);
				G3(local_state);
				G4(local_state);

				a ^= x; b ^= y; c ^= z; d ^= w;

				next_block[i1_0] = a;
				next_block[i1_1] = b;
				next_block[i1_2] = c;
				next_block[i1_3] = d;
			}
		}
	}

	__syncthreads();

	int dst_addr = 65528;

	next_block = memory + ((int16_t*)(&addresses[dst_addr]))[0] * BLOCK_SIZE_ULONG;
	out_mem[local_id] = next_block[local_id];

	for(;dst_addr < 65531; ++dst_addr) {
		next_block = memory + ((int16_t*)(&addresses[dst_addr]))[1] * BLOCK_SIZE_ULONG;
		out_mem[local_id] ^= next_block[local_id];
	}
};

void cuda_allocate(cuda_device_info *device) {
	int max_threads = max(device->threads_profile_1_1_524288, device->threads_profile_4_4_16384);

	device->error = cudaSetDevice(device->device_index);
	if(device->error != cudaSuccess) {
		device->error_message = "Error setting current device for memory allocation.";
		return;
	}

	//optimise address sizes
	uint32_t *addresses_1_1_524288 = (uint32_t *)malloc(argon2profile_1_1_524288.block_refs_size * 2 * sizeof(int32_t));
	for(int i=0;i<argon2profile_1_1_524288.block_refs_size;i++) {
		addresses_1_1_524288[i*2] = argon2profile_1_1_524288.block_refs[i*3];
		addresses_1_1_524288[i*2+1] = argon2profile_1_1_524288.block_refs[i*3 + 2];
	}
	device->error = cudaMalloc(&device->arguments.address_profile_1_1_524288, (argon2profile_1_1_524288.block_refs_size + 2) * 2 * sizeof(int32_t));
	if(device->error != cudaSuccess) {
		device->error_message = "Error allocating memory.";
		return;
	}
	device->error = cudaMemcpy(device->arguments.address_profile_1_1_524288, addresses_1_1_524288, argon2profile_1_1_524288.block_refs_size * 2 * sizeof(int32_t), cudaMemcpyHostToDevice);
	if(device->error != cudaSuccess) {
		device->error_message = "Error copying memory.";
		return;
	}
	free(addresses_1_1_524288);

	//optimise address sizes
	int16_t *addresses_4_4_16384 = (int16_t *)malloc(argon2profile_4_4_16384.block_refs_size * 2 * sizeof(int16_t));
	for(int i=0;i<argon2profile_4_4_16384.block_refs_size;i++) {
		addresses_4_4_16384[i*2] = argon2profile_4_4_16384.block_refs[i*3 + (i == 65528 ? 1 : 0)];
		addresses_4_4_16384[i*2 + 1] = argon2profile_4_4_16384.block_refs[i*3 + 2];
	}
	device->error = cudaMalloc(&device->arguments.address_profile_4_4_16384, argon2profile_4_4_16384.block_refs_size * 2 * sizeof(int16_t));
	if(device->error != cudaSuccess) {
		device->error_message = "Error allocating memory.";
		return;
	}
	device->error = cudaMemcpy(device->arguments.address_profile_4_4_16384, addresses_4_4_16384, argon2profile_4_4_16384.block_refs_size * 2 * sizeof(int16_t), cudaMemcpyHostToDevice);
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
		segments_4_4_16384[i*2 + 1] = argon2profile_4_4_16384.block_refs[seg_start*3 + 1];
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

	device->error = cudaMalloc(&device->arguments.offsets, 512 * sizeof(int));
	if(device->error != cudaSuccess) {
		device->error_message = "Error allocating memory.";
		return;
	}
	device->error = cudaMemcpy(device->arguments.offsets, offsets, 512 * sizeof(int), cudaMemcpyHostToDevice);
	if(device->error != cudaSuccess) {
		device->error_message = "Error copying memory.";
		return;
	}

	for(int i=0;i<device->device_threads;i++) {
		device->error = cudaMalloc(&device->arguments.memory[i], device->threads_per_stream[i].memory_size);
		if(device->error != cudaSuccess) {
			device->error_message = "Error allocating memory.";
			return;
		}

		size_t accessory_memory_size = max_threads * 8 * ARGON2_BLOCK_SIZE;
		device->error = cudaMalloc(&device->arguments.seed_memory[i], accessory_memory_size);
		if (device->error != cudaSuccess) {
			device->error_message = "Error allocating memory.";
			return;
		}
		device->error = cudaMalloc(&device->arguments.out_memory[i], accessory_memory_size);
		if (device->error != cudaSuccess) {
			device->error_message = "Error allocating memory.";
			return;
		}
		device->error = cudaMallocHost(&device->arguments.host_seed_memory[i], accessory_memory_size);
		if (device->error != cudaSuccess) {
			device->error_message = "Error allocating pinned memory.";
			return;
		}
	}
}

void cuda_free(cuda_device_info *device) {
	cudaSetDevice(device->device_index);

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

	if(device->arguments.offsets != NULL) {
		cudaFree(device->arguments.offsets);
		device->arguments.offsets = NULL;
	}

	if(device->arguments.memory != NULL) {
		for(int i=0;i<device->device_threads;i++) {
			if(device->arguments.memory[i] != NULL)
				cudaFree(device->arguments.memory[i]);
			device->arguments.memory[i] = NULL;
		}
	}

	if(device->arguments.seed_memory != NULL) {
		for(int i=0;i<device->device_threads;i++) {
			if(device->arguments.seed_memory[i] != NULL)
				cudaFree(device->arguments.seed_memory[i]);
			device->arguments.seed_memory[i] = NULL;
		}
	}

	if(device->arguments.out_memory != NULL) {
		for(int i=0;i<device->device_threads;i++) {
			if(device->arguments.out_memory[i] != NULL)
				cudaFree(device->arguments.out_memory[i]);
			device->arguments.out_memory[i] = NULL;
		}
	}

	if(device->arguments.host_seed_memory != NULL) {
		for(int i=0;i<device->device_threads;i++) {
			if(device->arguments.host_seed_memory[i] != NULL)
				cudaFreeHost(device->arguments.host_seed_memory[i]);
			device->arguments.host_seed_memory[i] = NULL;
		}
	}

	cudaDeviceReset();
}

void *cuda_kernel_filler(void *memory, int threads, argon2profile *profile, void *user_data) {
	//    uint64_t start_log = microseconds();
	//    printf("Waiting for lock: %lld\n", microseconds() - start_log);
	//    start_log = microseconds();
	cuda_gpumgmt_thread_data *gpumgmt_thread = (cuda_gpumgmt_thread_data *)user_data;
	cuda_device_info *device = gpumgmt_thread->device;
	cudaStream_t *stream = (cudaStream_t *)gpumgmt_thread->cuda_info;

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

	device->error = cudaMemcpyAsync(device->arguments.seed_memory[gpumgmt_thread->thread_id], memory, threads * 2 * mem_seed_count * ARGON2_BLOCK_SIZE, cudaMemcpyHostToDevice, *stream);
	if (device->error != cudaSuccess) {
		device->error_message = "Error writing to gpu memory.";
		return NULL;
	}

	if(parallelism == 1) {
		fill_blocks_cpu<<<threads, work_items, 0, *stream>>>((uint64_t*)device->arguments.memory[gpumgmt_thread->thread_id],
				device->arguments.seed_memory[gpumgmt_thread->thread_id],
				device->arguments.out_memory[gpumgmt_thread->thread_id],
				device->arguments.address_profile_1_1_524288,
				device->arguments.offsets,
				memsize);
	}
	else {
		fill_blocks_gpu<< < threads, work_items, 0, *stream >> > ((uint64_t *) device->arguments.memory[gpumgmt_thread->thread_id],
				device->arguments.seed_memory[gpumgmt_thread->thread_id],
				device->arguments.out_memory[gpumgmt_thread->thread_id],
				device->arguments.address_profile_4_4_16384,
				device->arguments.segments_profile_4_4_16384,
				device->arguments.offsets,
				memsize);
	}

	device->error = cudaMemcpyAsync(memory, device->arguments.out_memory[gpumgmt_thread->thread_id], threads * 2 * mem_seed_count * ARGON2_BLOCK_SIZE, cudaMemcpyDeviceToHost, *stream);
	if (device->error != cudaSuccess) {
		device->error_message = "Error reading gpu memory.";
		return NULL;
	}

	while(cudaStreamQuery(*stream) != cudaSuccess) {
		this_thread::sleep_for(chrono::milliseconds(10));
		continue;
	}

	return memory;
}