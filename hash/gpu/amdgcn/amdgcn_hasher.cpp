//
// Created by Haifa Bogdan Adnan on 07/10/2018.
//

#if defined(WITH_AMDGCN)

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CLRX/amdasm/Assembler.h>
#include <CLRX/clhelper/CLHelper.h>

#include "../../../common/common.h"
#include "../../../app/arguments.h"

#include "../../hasher.h"
#include "../../argon2/argon2.h"

#include "amdgcn_hasher.h"
#include "amdgcn_kernel.h"
#include "../../../common/dllexport.h"

#define KERNEL_WORKGROUP_SIZE   32

amdgcn_hasher::amdgcn_hasher() {
	_type = "GPU";
	_subtype = "AMDGCN";
	_short_subtype = "GCN";
	_priority = 0;
	_intensity = 0;
	__running = false;
	_description = "";
}

amdgcn_hasher::~amdgcn_hasher() {
//	this->cleanup();
}

bool amdgcn_hasher::initialize() {
	cl_int error = CL_SUCCESS;
	string error_message;

	__devices = __query_amdgcn_devices(error, error_message);

	if(error != CL_SUCCESS) {
		_description = "No compatible GPU detected: " + error_message;
		return false;
	}

	if (__devices.empty()) {
		_description = "No compatible GPU detected.";
		return false;
	}

	return true;
}

#ifndef CL_DEVICE_TOPOLOGY_AMD
#define CL_DEVICE_TOPOLOGY_AMD                      0x4037
#endif

typedef union
{
    struct { cl_uint type; cl_uint data[5]; } raw;
    struct { cl_uint type; cl_char unused[17]; cl_char bus; cl_char device; cl_char function; } pcie;
} device_topology_amd;

bool amdgcn_hasher::configure(arguments &args) {
	int index = args.get_cards_count();
	double intensity_cpu = 0;
	double intensity_gpu = 0;

	for(vector<double>::iterator it = args.gpu_intensity_cblocks().begin(); it != args.gpu_intensity_cblocks().end(); it++) {
		intensity_cpu += *it;
	}
	intensity_cpu /= args.gpu_intensity_cblocks().size();

	for(vector<double>::iterator it = args.gpu_intensity_gblocks().begin(); it != args.gpu_intensity_gblocks().end(); it++) {
		intensity_gpu += *it;
	}
	intensity_gpu /= args.gpu_intensity_gblocks().size();

	vector<string> filter = _get_gpu_filters(args);

	int total_threads_profile_4_4_16384 = 0;
	int total_threads_profile_1_1_524288 = 0;

	if (intensity_cpu == 0 && intensity_gpu == 0) {
		_intensity = 0;
		_description = "Status: DISABLED - by user.";
		return false;
	}

	bool cards_selected = false;

	for(vector<amdgcn_device_info *>::iterator d = __devices.begin(); d != __devices.end(); d++, index++) {
		stringstream ss;
		ss << "["<< (index + 1) << "] " << (*d)->device_string;
		string device_description = ss.str();
		(*d)->device_index = index;

		if(filter.size() > 0) {
			bool found = false;
			for(vector<string>::iterator fit = filter.begin(); fit != filter.end(); fit++) {
				if(device_description.find(*fit) != string::npos) {
					found = true;
					break;
				}
			}
			if(!found) {
				(*d)->profile_info.threads_profile_4_4_16384 = 0;
				(*d)->profile_info.threads_profile_1_1_524288 = 0;
				ss << " - DISABLED" << endl;
				_description += ss.str();
				continue;
			}
			else {
				cards_selected = true;
			}
		}
		else {
			cards_selected = true;
		}

		ss << endl;

		double device_intensity_cpu = 0;
		if(args.gpu_intensity_cblocks().size() == 1 || (*d)->device_index >= args.gpu_intensity_cblocks().size())
			device_intensity_cpu = args.gpu_intensity_cblocks()[0];
		else
			device_intensity_cpu = args.gpu_intensity_cblocks()[(*d)->device_index];

		double device_intensity_gpu = 0;
		if(args.gpu_intensity_gblocks().size() == 1 || (*d)->device_index >= args.gpu_intensity_gblocks().size())
			device_intensity_gpu = args.gpu_intensity_gblocks()[0];
		else
			device_intensity_gpu = args.gpu_intensity_gblocks()[(*d)->device_index];

		_description += ss.str();

		if(!(__setup_device_info((*d), device_intensity_cpu, device_intensity_gpu))) {
			_description += (*d)->error_message;
			_description += "\n";
			continue;
		};

		device_info device;

		device_topology_amd amdtopo;
		if(clGetDeviceInfo((*d)->device, CL_DEVICE_TOPOLOGY_AMD, sizeof(amdtopo), &amdtopo, NULL) == CL_SUCCESS) {
			char bus_id[50];
			sprintf(bus_id, "%02x:%02x.%x", amdtopo.pcie.bus, amdtopo.pcie.device, amdtopo.pcie.function);
			device.bus_id = bus_id;
		}

		device.name = (*d)->device_string;
		device.cblocks_intensity = device_intensity_cpu;
		device.gblocks_intensity = device_intensity_gpu;
		_store_device_info((*d)->device_index, device);

		total_threads_profile_4_4_16384 += (*d)->profile_info.threads_profile_4_4_16384;
		total_threads_profile_1_1_524288 += (*d)->profile_info.threads_profile_1_1_524288;
	}

	args.set_cards_count(index);

	if(!cards_selected) {
		_intensity = 0;
		_description += "Status: DISABLED - no card enabled because of filtering.";
		return false;
	}

	if (total_threads_profile_4_4_16384 == 0 && total_threads_profile_1_1_524288 == 0) {
		_intensity = 0;
		_description += "Status: DISABLED - not enough resources.";
		return false;
	}

	_intensity = (intensity_cpu + intensity_gpu) / 2.0;

	__running = true;
	_update_running_status(__running);
	for(vector<amdgcn_device_info *>::iterator d = __devices.begin(); d != __devices.end(); d++) {
		if((*d)->profile_info.threads_profile_1_1_524288 != 0 || (*d)->profile_info.threads_profile_4_4_16384 != 0) {
			__runners.push_back(new thread([&](amdgcn_device_info *device) {
				this->__run(device, 0);
			}, (*d)));
			__runners.push_back(new thread([&](amdgcn_device_info *device) {
				this->__run(device, 1);
			}, (*d)));
		}
	}

	_description += "Status: ENABLED - with " + to_string(total_threads_profile_1_1_524288) + " threads for CPU blocks and " + to_string(total_threads_profile_4_4_16384) + " threads for GPU blocks.";

	return true;
}

void amdgcn_hasher::cleanup() {
	__running = false;
	for(vector<thread*>::iterator it = __runners.begin();it != __runners.end();++it) {
		(*it)->join();
		delete *it;
	}
	__runners.clear();

	vector<cl_platform_id> platforms;

	for(vector<amdgcn_device_info *>::iterator it=__devices.begin(); it != __devices.end(); it++) {
		if ((*it)->profile_info.threads_profile_1_1_524288 != 0 || (*it)->profile_info.threads_profile_4_4_16384 != 0) {
			clReleaseMemObject((*it)->arguments.memory_chunk_0);
			clReleaseMemObject((*it)->arguments.memory_chunk_1);
			clReleaseMemObject((*it)->arguments.memory_chunk_2);
			clReleaseMemObject((*it)->arguments.memory_chunk_3);
			clReleaseMemObject((*it)->arguments.memory_chunk_4);
			clReleaseMemObject((*it)->arguments.memory_chunk_5);
			clReleaseMemObject((*it)->arguments.address_profile_1_1_524288);
			clReleaseMemObject((*it)->arguments.address_profile_4_4_16384);
			clReleaseMemObject((*it)->arguments.seed_memory[0]);
			clReleaseMemObject((*it)->arguments.seed_memory[1]);
			clReleaseMemObject((*it)->arguments.out_memory[0]);
			clReleaseMemObject((*it)->arguments.out_memory[1]);

			clReleaseKernel((*it)->kernel_cblocks);
			clReleaseKernel((*it)->kernel_gblocks);
			clReleaseProgram((*it)->program);
			clReleaseCommandQueue((*it)->queue);
			clReleaseContext((*it)->context);
		}
		clReleaseDevice((*it)->device);
		delete (*it);
	}
	__devices.clear();
}

amdgcn_device_info *amdgcn_hasher::__get_device_info(cl_platform_id platform, cl_device_id device) {
	amdgcn_device_info *device_info = new amdgcn_device_info(CL_SUCCESS, "");

	device_info->platform = platform;
	device_info->device = device;

	char *buffer;
	size_t sz;

	// device name
	string device_name;
	sz = 0;
	clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &sz);
	buffer = (char *)malloc(sz + 1);
	device_info->error = clGetDeviceInfo(device, CL_DEVICE_NAME, sz, buffer, &sz);
	if(device_info->error != CL_SUCCESS) {
		free(buffer);
		device_info->error_message = "Error querying device name.";
		return device_info;
	}
	else {
		buffer[sz] = 0;
		device_name = buffer;
		free(buffer);
	}

	string board_name;

#ifndef CL_DEVICE_BOARD_NAME_AMD
#define CL_DEVICE_BOARD_NAME_AMD                    0x4038
#endif

    	sz = 0;
	clGetDeviceInfo(device, CL_DEVICE_BOARD_NAME_AMD, 0, NULL, &sz);
	buffer = (char *)malloc(sz + 1);
	device_info->error = clGetDeviceInfo(device, CL_DEVICE_BOARD_NAME_AMD, sz, buffer, &sz);
	if(device_info->error != CL_SUCCESS) {
		free(buffer);
		device_info->error_message = "Error querying card name.";
		return device_info;
	}
	else {
		buffer[sz] = 0;
		board_name = buffer;
		free(buffer);
	}

	device_info->device_string = (!board_name.empty()) ? (board_name + " (" + device_name + ")") : device_name;

	device_info->error = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(device_info->max_mem_size), &(device_info->max_mem_size), NULL);
	if(device_info->error != CL_SUCCESS) {
		device_info->error_message = "Error querying device global memory size.";
		return device_info;
	}

	device_info->error = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(device_info->max_allocable_mem_size), &(device_info->max_allocable_mem_size), NULL);
	if(device_info->error != CL_SUCCESS) {
		device_info->error_message = "Error querying device max memory allocation.";
		return device_info;
	}

	double mem_in_gb = device_info->max_mem_size / 1073741824.0;
	stringstream ss;
	ss << setprecision(2) << mem_in_gb;
	device_info->device_string += (" (" + ss.str() + "GB)");

	return device_info;
}

bool amdgcn_hasher::__setup_device_info(amdgcn_device_info *device, double intensity_cpu, double intensity_gpu) {
	cl_int error;

	CLRX::CLAsmSetup asmSetup;
	try {
		asmSetup = CLRX::assemblerSetupForCLDevice(device->device, 0);
	}
	catch(const CLRX::Exception &ex) {
		device->error = CL_INVALID_DEVICE_TYPE;
		device->error_message = string("Error retrieving assembler setup: ") + ex.what();
		return false;
	}

	cl_context_properties properties[]={
			CL_CONTEXT_PLATFORM, (cl_context_properties)device->platform,
			0};

	device->context=clCreateContext(properties, 1, &(device->device), NULL, NULL, &error);
	if(error != CL_SUCCESS)  {
		device->error = error;
		device->error_message = "Error getting device context.";
		return false;
	}

	device->queue = clCreateCommandQueue(device->context, device->device, CL_QUEUE_PROFILING_ENABLE, &error);
	if(error != CL_SUCCESS)  {
		device->error = error;
		device->error_message = "Error getting device command queue.";
		return false;
	}

	CLRX::CString build_log;
	try
	{
		device->program = CLRX::createProgramForCLDevice(device->context, device->device,
										 asmSetup, amdgcn_kernel.c_str(), 0, &build_log);
	}
	catch(const CLRX::Exception &ex)
	{
		device->error = CL_INVALID_BINARY;
		device->error_message = "Error building AMDGCN program for device: " + string(build_log.c_str());
		return false;
	}

	device->kernel_cblocks = clCreateKernel(device->program, "fill_cblocks", &error);
	if(error != CL_SUCCESS)  {
		device->error = error;
		device->error_message = "Error creating AMDGCN kernel for device.";
		return false;
	}

	device->kernel_gblocks = clCreateKernel(device->program, "fill_gblocks", &error);
	if(error != CL_SUCCESS)  {
		device->error = error;
		device->error_message = "Error creating AMDGCN kernel for device.";
		return false;
	}

	device->profile_info.threads_per_chunk_profile_1_1_524288 = (uint32_t)(device->max_allocable_mem_size / argon2profile_1_1_524288.memsize);
	size_t chunk_size_profile_1_1_524288 = device->profile_info.threads_per_chunk_profile_1_1_524288 * argon2profile_1_1_524288.memsize;

	device->profile_info.threads_per_chunk_profile_4_4_16384 = (uint32_t)(device->max_allocable_mem_size / argon2profile_4_4_16384.memsize);
	size_t chunk_size_profile_4_4_16384 = device->profile_info.threads_per_chunk_profile_4_4_16384 * argon2profile_4_4_16384.memsize;

	if(chunk_size_profile_1_1_524288 == 0 && chunk_size_profile_4_4_16384 == 0) {
		device->error = -1;
		device->error_message = "Not enough memory on GPU.";
		return false;
	}

	size_t chunk_size = max(chunk_size_profile_1_1_524288, chunk_size_profile_4_4_16384);
	uint64_t usable_memory = device->max_mem_size;
	double chunks = (double)usable_memory / (double)chunk_size;

	uint32_t max_threads_1_1_524288 = (uint32_t)(device->profile_info.threads_per_chunk_profile_1_1_524288 * chunks);
	uint32_t max_threads_4_4_16384 = (uint32_t)(device->profile_info.threads_per_chunk_profile_4_4_16384 * chunks);

	if(max_threads_1_1_524288 == 0 && max_threads_4_4_16384 == 0) {
		device->error = -1;
		device->error_message = "Not enough memory on GPU.";
		return false;
	}

	device->profile_info.threads_profile_1_1_524288 = (uint32_t)(max_threads_1_1_524288 * intensity_cpu / 100.0);
	if(max_threads_1_1_524288 > 0 && device->profile_info.threads_profile_1_1_524288 == 0 && intensity_cpu > 0)
		device->profile_info.threads_profile_1_1_524288 = 1;
	device->profile_info.threads_profile_4_4_16384 = (uint32_t)(max_threads_4_4_16384 * intensity_gpu / 100.0);
	if(max_threads_4_4_16384 > 0 && device->profile_info.threads_profile_4_4_16384 == 0 && intensity_gpu > 0)
		device->profile_info.threads_profile_4_4_16384 = 1;

	size_t max_threads = max(device->profile_info.threads_profile_4_4_16384, device->profile_info.threads_profile_1_1_524288);

	double chunks_1_1_524288 = (double)device->profile_info.threads_profile_1_1_524288 / (double)device->profile_info.threads_per_chunk_profile_1_1_524288;
	double chunks_4_4_16384 = (double)device->profile_info.threads_profile_4_4_16384 / (double)device->profile_info.threads_per_chunk_profile_4_4_16384;

	double counter = max(chunks_1_1_524288, chunks_4_4_16384);
	size_t allocated_mem_for_current_chunk = 0;

	if(counter > 0) {
		if (counter > 1) {
			allocated_mem_for_current_chunk = chunk_size;
		} else {
			allocated_mem_for_current_chunk = (size_t)ceil(chunk_size * counter);
		}
		counter -= 1;
	}
	else {
		allocated_mem_for_current_chunk = 1;
	}
	device->arguments.memory_chunk_0 = clCreateBuffer(device->context, CL_MEM_READ_WRITE, allocated_mem_for_current_chunk, NULL, &error);
	if(error != CL_SUCCESS) {
		device->error = error;
		device->error_message = "Error creating memory buffer.";
		return false;
	}

	if(counter > 0) {
		if (counter > 1) {
			allocated_mem_for_current_chunk = chunk_size;
		} else {
			allocated_mem_for_current_chunk = (size_t)ceil(chunk_size * counter);
		}
		counter -= 1;
	}
	else {
		allocated_mem_for_current_chunk = 1;
	}
	device->arguments.memory_chunk_1 = clCreateBuffer(device->context, CL_MEM_READ_WRITE, allocated_mem_for_current_chunk, NULL, &error);
	if(error != CL_SUCCESS) {
		device->error = error;
		device->error_message = "Error creating memory buffer.";
		return false;
	}

	if(counter > 0) {
		if (counter > 1) {
			allocated_mem_for_current_chunk = chunk_size;
		} else {
			allocated_mem_for_current_chunk = (size_t)ceil(chunk_size * counter);
		}
		counter -= 1;
	}
	else {
		allocated_mem_for_current_chunk = 1;
	}
	device->arguments.memory_chunk_2 = clCreateBuffer(device->context, CL_MEM_READ_WRITE, allocated_mem_for_current_chunk, NULL, &error);
	if(error != CL_SUCCESS) {
		device->error = error;
		device->error_message = "Error creating memory buffer.";
		return false;
	}

	if(counter > 0) {
		if (counter > 1) {
			allocated_mem_for_current_chunk = chunk_size;
		} else {
			allocated_mem_for_current_chunk = (size_t)ceil(chunk_size * counter);
		}
		counter -= 1;
	}
	else {
		allocated_mem_for_current_chunk = 1;
	}
	device->arguments.memory_chunk_3 = clCreateBuffer(device->context, CL_MEM_READ_WRITE, allocated_mem_for_current_chunk, NULL, &error);
	if(error != CL_SUCCESS) {
		device->error = error;
		device->error_message = "Error creating memory buffer.";
		return false;
	}

	if(counter > 0) {
		if (counter > 1) {
			allocated_mem_for_current_chunk = chunk_size;
		} else {
			allocated_mem_for_current_chunk = (size_t)ceil(chunk_size * counter);
		}
		counter -= 1;
	}
	else {
		allocated_mem_for_current_chunk = 1;
	}
	device->arguments.memory_chunk_4 = clCreateBuffer(device->context, CL_MEM_READ_WRITE, allocated_mem_for_current_chunk, NULL, &error);
	if(error != CL_SUCCESS) {
		device->error = error;
		device->error_message = "Error creating memory buffer.";
		return false;
	}

	if(counter > 0) {
		if (counter > 1) {
			allocated_mem_for_current_chunk = chunk_size;
		} else {
			allocated_mem_for_current_chunk = (size_t)ceil(chunk_size * counter);
		}
		counter -= 1;
	}
	else {
		allocated_mem_for_current_chunk = 1;
	}
	device->arguments.memory_chunk_5 = clCreateBuffer(device->context, CL_MEM_READ_WRITE, allocated_mem_for_current_chunk, NULL, &error);
	if(error != CL_SUCCESS) {
		device->error = error;
		device->error_message = "Error creating memory buffer.";
		return false;
	}

	device->arguments.address_profile_1_1_524288 = clCreateBuffer(device->context, CL_MEM_READ_ONLY, (argon2profile_1_1_524288.block_refs_size + 2) * 2 * sizeof(int32_t), NULL, &error);
	if(error != CL_SUCCESS) {
		device->error = error;
		device->error_message = "Error creating memory buffer.";
		return false;
	}

	device->arguments.address_profile_4_4_16384 = clCreateBuffer(device->context, CL_MEM_READ_ONLY, argon2profile_4_4_16384.block_refs_size * 2 * sizeof(int16_t), NULL, &error);
	if(error != CL_SUCCESS) {
		device->error = error;
		device->error_message = "Error creating memory buffer.";
		return false;
	}

	device->arguments.seed_memory[0] = clCreateBuffer(device->context, CL_MEM_READ_ONLY, max_threads * 8 * ARGON2_BLOCK_SIZE, NULL, &error);
	if(error != CL_SUCCESS) {
		device->error = error;
		device->error_message = "Error creating memory buffer.";
		return false;
	}

	device->arguments.seed_memory[1] = clCreateBuffer(device->context, CL_MEM_READ_ONLY, max_threads * 8 * ARGON2_BLOCK_SIZE, NULL, &error);
	if(error != CL_SUCCESS) {
		device->error = error;
		device->error_message = "Error creating memory buffer.";
		return false;
	}

	device->arguments.out_memory[0] = clCreateBuffer(device->context, CL_MEM_WRITE_ONLY, max_threads * 8 * ARGON2_BLOCK_SIZE, NULL, &error);
	if(error != CL_SUCCESS) {
		device->error = error;
		device->error_message = "Error creating memory buffer.";
		return false;
	}

	device->arguments.out_memory[1] = clCreateBuffer(device->context, CL_MEM_WRITE_ONLY, max_threads * 8 * ARGON2_BLOCK_SIZE, NULL, &error);
	if(error != CL_SUCCESS) {
		device->error = error;
		device->error_message = "Error creating memory buffer.";
		return false;
	}

	//optimise address sizes
	int32_t *addresses_1_1_524288 = (int32_t *)malloc((argon2profile_1_1_524288.block_refs_size + 2) * 2 * sizeof(int32_t)); //add 2 to ref_size to be exact$

	for(int i=0;i<argon2profile_1_1_524288.block_refs_size;i++) {
		addresses_1_1_524288[i * 2] = argon2profile_1_1_524288.block_refs[i*4];
		addresses_1_1_524288[i * 2 + 1] = argon2profile_1_1_524288.block_refs[i*4 + 2];
	}
	error=clEnqueueWriteBuffer(device->queue, device->arguments.address_profile_1_1_524288, CL_TRUE, 0, (argon2profile_1_1_524288.block_refs_size + 2) * 2 * sizeof(int32_t), addresses_1_1_524288, 0, NULL, NULL);
	if(error != CL_SUCCESS) {
		device->error = error;
		device->error_message = "Error writing to gpu memory.";
		return false;
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
	error=clEnqueueWriteBuffer(device->queue, device->arguments.address_profile_4_4_16384, CL_TRUE, 0, argon2profile_4_4_16384.block_refs_size * 2 * sizeof(uint16_t), addresses_4_4_16384, 0, NULL, NULL);
	if(error != CL_SUCCESS) {
		device->error = error;
		device->error_message = "Error writing to gpu memory.";
		return false;
	}
	free(addresses_4_4_16384);

/*	for(int seg = 0; seg < 4; seg++) {
		int offset = seg;
		printf("\t.byte ");
		for(int s=0;s<16;s++) {
			int idx = offset + s * 4;
			int seg_start = argon2profile_4_4_16384.segments[idx * 3];
			int prev_blk = argon2profile_4_4_16384.block_refs[seg_start * 4 + 1];
			printf("0x%02hhx, 0x%02hhx, 0x%02hhx, 0x%02hhx, ", ((uint8_t *) &seg_start)[0],
				   ((uint8_t *) &seg_start)[1], ((uint8_t *) &prev_blk)[0],
				   ((uint8_t *) &prev_blk)[1]);
			if((s + 1) % 4 == 0)
				printf("\n\t.byte ");
		}
		printf("\n");
	}*/

	clSetKernelArg(device->kernel_cblocks, 0, sizeof(device->arguments.memory_chunk_0), &device->arguments.memory_chunk_0);
	clSetKernelArg(device->kernel_cblocks, 1, sizeof(device->arguments.memory_chunk_1), &device->arguments.memory_chunk_1);
	clSetKernelArg(device->kernel_cblocks, 2, sizeof(device->arguments.memory_chunk_2), &device->arguments.memory_chunk_2);
	clSetKernelArg(device->kernel_cblocks, 3, sizeof(device->arguments.memory_chunk_3), &device->arguments.memory_chunk_3);
	clSetKernelArg(device->kernel_cblocks, 4, sizeof(device->arguments.memory_chunk_4), &device->arguments.memory_chunk_4);
	clSetKernelArg(device->kernel_cblocks, 5, sizeof(device->arguments.memory_chunk_5), &device->arguments.memory_chunk_5);
	clSetKernelArg(device->kernel_cblocks, 8, sizeof(device->arguments.address_profile_1_1_524288), &device->arguments.address_profile_1_1_524288);
	clSetKernelArg(device->kernel_cblocks, 9, sizeof(int32_t), &device->profile_info.threads_per_chunk_profile_1_1_524288);

	clSetKernelArg(device->kernel_gblocks, 0, sizeof(device->arguments.memory_chunk_0), &device->arguments.memory_chunk_0);
	clSetKernelArg(device->kernel_gblocks, 1, sizeof(device->arguments.memory_chunk_1), &device->arguments.memory_chunk_1);
	clSetKernelArg(device->kernel_gblocks, 2, sizeof(device->arguments.memory_chunk_2), &device->arguments.memory_chunk_2);
	clSetKernelArg(device->kernel_gblocks, 3, sizeof(device->arguments.memory_chunk_3), &device->arguments.memory_chunk_3);
	clSetKernelArg(device->kernel_gblocks, 4, sizeof(device->arguments.memory_chunk_4), &device->arguments.memory_chunk_4);
	clSetKernelArg(device->kernel_gblocks, 5, sizeof(device->arguments.memory_chunk_5), &device->arguments.memory_chunk_5);
	clSetKernelArg(device->kernel_gblocks, 8, sizeof(device->arguments.address_profile_4_4_16384), &device->arguments.address_profile_4_4_16384);
	clSetKernelArg(device->kernel_gblocks, 9, sizeof(int32_t), &device->profile_info.threads_per_chunk_profile_4_4_16384);

	return true;
}

vector<amdgcn_device_info*> amdgcn_hasher::__query_amdgcn_devices(cl_int &error, string &error_message) {
	cl_int err;

	cl_uint platform_count = 0;
	cl_uint device_count = 0;

	vector<amdgcn_device_info*> result;

	vector<cl_platform_id> platforms;

	try {
		platforms = CLRX::chooseCLPlatformsForCLRX();
	}
	catch(const CLRX::Exception &err) {
		error = CL_INVALID_DEVICE;
		error_message = err.what();
		return result;
	}

	uint32_t counter = 0;

	for(int i=0; i < platforms.size(); i++) {
		device_count = 0;
		clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &device_count);
		if(device_count == 0) {
			continue;
		}

		cl_device_id * devices = (cl_device_id*)malloc(device_count * sizeof(cl_device_id));
		err=clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, device_count, devices, &device_count);

		if(err != CL_SUCCESS)  {
			free(devices);
			error = err;
			error_message = "Error querying for AMDGCN devices.";
			continue;
		}

		for(uint32_t j=0; j < device_count; j++) {
			amdgcn_device_info *info = __get_device_info(platforms[i], devices[j]);
			if(info->error != CL_SUCCESS) {
				error = info->error;
				error_message = info->error_message;
			}
			else {
				info->device_index = counter;
				result.push_back(info);
				counter++;
			}
		}

		free(devices);
	}

	return result;
}

struct amdgcn_gpumgmt_thread_data {
	int thread_id;
	amdgcn_device_info *device;
};

void print_block(void *data) {
    for(int i=0;i<256;i++) {
    	printf("%u, ", ((uint32_t *)data)[i]);
    }
    printf("\n");
}

void print_cksum(void *data) {
    uint64_t x = 0;

    for(int i=0;i<128;i++) {
        x += ((uint64_t *)data)[i];
    }
    printf("%llu\n", x);
}

void *amdgcn_kernel_filler(void *memory, int threads, argon2profile *profile, void *user_data) {
	amdgcn_gpumgmt_thread_data *gpumgmt_thread = (amdgcn_gpumgmt_thread_data *)user_data;
	amdgcn_device_info *device = gpumgmt_thread->device;

	cl_int error;

	int mem_seed_count = profile->thr_cost;
	size_t total_work_items = threads * KERNEL_WORKGROUP_SIZE * profile->thr_cost;
	size_t local_work_items = KERNEL_WORKGROUP_SIZE * profile->thr_cost;

	device->device_lock.lock();

	error = clEnqueueWriteBuffer(device->queue, device->arguments.seed_memory[gpumgmt_thread->thread_id], CL_FALSE, 0, threads * 2 * mem_seed_count * ARGON2_BLOCK_SIZE, memory, 0, NULL, NULL);
	if (error != CL_SUCCESS) {
		device->error = error;
		device->error_message = "Error writing to gpu memory.";
		device->device_lock.unlock();
		return NULL;
	}

	if(strcmp(profile->profile_name, "1_1_524288") == 0) {
		clSetKernelArg(device->kernel_cblocks, 6, sizeof(device->arguments.seed_memory[gpumgmt_thread->thread_id]), &device->arguments.seed_memory[gpumgmt_thread->thread_id]);
		clSetKernelArg(device->kernel_cblocks, 7, sizeof(device->arguments.out_memory[gpumgmt_thread->thread_id]), &device->arguments.out_memory[gpumgmt_thread->thread_id]);
		error=clEnqueueNDRangeKernel(device->queue, device->kernel_cblocks, 1, NULL, &total_work_items, &local_work_items, 0, NULL, NULL);
		if(error != CL_SUCCESS) {
			device->error = error;
			device->error_message = "Error running the kernel.";
			device->device_lock.unlock();
			return NULL;
		}
	}
	else {
		clSetKernelArg(device->kernel_gblocks, 6, sizeof(device->arguments.seed_memory[gpumgmt_thread->thread_id]), &device->arguments.seed_memory[gpumgmt_thread->thread_id]);
		clSetKernelArg(device->kernel_gblocks, 7, sizeof(device->arguments.out_memory[gpumgmt_thread->thread_id]), &device->arguments.out_memory[gpumgmt_thread->thread_id]);
		error=clEnqueueNDRangeKernel(device->queue, device->kernel_gblocks, 1, NULL, &total_work_items, &local_work_items, 0, NULL, NULL);
		if(error != CL_SUCCESS) {
			device->error = error;
			device->error_message = "Error running the kernel.";
			device->device_lock.unlock();
			return NULL;
		}
	}

	error = clEnqueueReadBuffer(device->queue, device->arguments.out_memory[gpumgmt_thread->thread_id], CL_FALSE, 0, threads * 2 * mem_seed_count * ARGON2_BLOCK_SIZE, memory, 0, NULL, NULL);
	if (error != CL_SUCCESS) {
		device->error = error;
		device->error_message = "Error reading gpu memory.";
		device->device_lock.unlock();
		return NULL;
	}

	error=clFinish(device->queue);
	if(error != CL_SUCCESS) {
		device->error = error;
		device->error_message = "Error flushing GPU queue.";
		device->device_lock.unlock();
		return NULL;
	}

	device->device_lock.unlock();
//	for(int i=0;i<threads;i++) {
//		print_cksum(((uint8_t *)memory + i * 8 * 1024));
//		print_block(((uint8_t *)memory + i * 8 * 1024));
//	}
	return memory;
}

void amdgcn_hasher::__run(amdgcn_device_info *device, int thread_id) {
	void *memory = malloc(8 * ARGON2_BLOCK_SIZE * max(device->profile_info.threads_profile_1_1_524288, device->profile_info.threads_profile_4_4_16384));

	amdgcn_gpumgmt_thread_data thread_data;
	thread_data.device = device;
	thread_data.thread_id = thread_id;

	argon2 hash_factory(amdgcn_kernel_filler, memory, &thread_data);
	hash_factory.set_lane_length(2);

	while(__running) {
		if(_should_pause()) {
			this_thread::sleep_for(chrono::milliseconds(100));
			continue;
		}

		hash_data input = _get_input();
		argon2profile *profile = _get_argon2profile();

		if(!input.base.empty()) {
			if(strcmp(profile->profile_name, "1_1_524288") == 0) {
				if(device->profile_info.threads_profile_1_1_524288 == 0) {
					this_thread::sleep_for(chrono::milliseconds(100));
					continue;
				}
				hash_factory.set_seed_memory_offset(2 * ARGON2_BLOCK_SIZE);
				hash_factory.set_threads(device->profile_info.threads_profile_1_1_524288);
			}
			else {
				if(device->profile_info.threads_profile_4_4_16384 == 0) {
					this_thread::sleep_for(chrono::milliseconds(100));
					continue;
				}
				hash_factory.set_seed_memory_offset(8 * ARGON2_BLOCK_SIZE);
				hash_factory.set_threads(device->profile_info.threads_profile_4_4_16384);
			}

			vector<string> hashes = hash_factory.generate_hashes(*profile, input.base, input.salt);

			if (device->error != CL_SUCCESS) {
				LOG("Error running kernel: (" + to_string(device->error) + ")" + device->error_message);
				__running = false;
				exit(0);
			}
			vector<hash_data> stored_hashes;
			for(vector<string>::iterator it = hashes.begin(); it != hashes.end(); ++it) {
				input.hash = *it;
				stored_hashes.push_back(input);
			}
			_store_hash(stored_hashes, device->device_index);
		}
	}

	free(memory);
	_update_running_status(__running);
}

REGISTER_HASHER(amdgcn_hasher);

#endif //WITH_AMDGCN
