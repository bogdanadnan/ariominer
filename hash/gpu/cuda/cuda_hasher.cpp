//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#include "../../../common/common.h"
#include "../../../app/arguments.h"

#include "../../hasher.h"
#include "../../argon2/argon2.h"

#include "cuda_hasher.h"

#include <cuda_runtime.h>

cuda_hasher::cuda_hasher()
{
	_type = "GPU";
	_subtype = "CUDA";
	_priority = 2;
	_intensity = 0;
	__running = false;
	_description = "";
}


cuda_hasher::~cuda_hasher()
{
}

bool cuda_hasher::initialize() {
	int devCount = 0;
	cudaGetDeviceCount(&devCount);

	// Iterate through devices
	for (int i = 0; i < devCount; ++i)
	{
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, i);
		_description += devProp.name;
		_description += "\n";
	}	
	return devCount > 0;
}

extern void host_fill_block();
bool cuda_hasher::configure(arguments &args) {
	host_fill_block();
	return true;
}

void cuda_hasher::cleanup() {
}

REGISTER_HASHER(cuda_hasher);