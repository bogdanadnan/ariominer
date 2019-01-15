//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#include "../../../common/common.h"
#include "../../../app/arguments.h"

#include "../../hasher.h"
#include "../../argon2/argon2.h"

#include "opencl_hasher.h"
#include "opencl_kernel.h"
#include "../../../common/dllexport.h"

#if defined(WITH_OPENCL)

#define KERNEL_WORKGROUP_SIZE   32

opencl_device_info *opencl_hasher::__get_device_info(cl_platform_id platform, cl_device_id device) {
    opencl_device_info *device_info = new opencl_device_info(CL_SUCCESS, "");

    device_info->platform = platform;
    device_info->device = device;

    char *buffer;
    size_t sz;

    // device name
    string device_vendor;
    sz = 0;
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, 0, NULL, &sz);
    buffer = (char *)malloc(sz + 1);
    device_info->error = clGetDeviceInfo(device, CL_DEVICE_VENDOR, sz, buffer, &sz);
    if(device_info->error != CL_SUCCESS) {
        free(buffer);
        device_info->error_message = "Error querying device vendor.";
        return device_info;
    }
    else {
        buffer[sz] = 0;
        device_vendor = buffer;
        free(buffer);
    }

    string device_name;
	cl_device_info query_type = CL_DEVICE_NAME;

#ifdef	CL_DEVICE_BOARD_NAME_AMD
    if(device_vendor.find("Advanced Micro Devices") != string::npos)
		query_type = CL_DEVICE_BOARD_NAME_AMD;
#endif

	sz = 0;
	clGetDeviceInfo(device, query_type, 0, NULL, &sz);
	buffer = (char *) malloc(sz + 1);
	device_info->error = clGetDeviceInfo(device, query_type, sz, buffer, &sz);
	if (device_info->error != CL_SUCCESS) {
		free(buffer);
		device_info->error_message = "Error querying device name.";
		return device_info;
	} else {
		buffer[sz] = 0;
		device_name = buffer;
		free(buffer);
	}

    string device_version;
    sz = 0;
    clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, NULL, &sz);
    buffer = (char *)malloc(sz + 1);
    device_info->error = clGetDeviceInfo(device, CL_DEVICE_VERSION, sz, buffer, &sz);
    if(device_info->error != CL_SUCCESS) {
        free(buffer);
        device_info->error_message = "Error querying device version.";
        return device_info;
    }
    else {
        buffer[sz] = 0;
        device_version = buffer;
        free(buffer);
    }

    device_info->device_string = device_vendor + " - " + device_name + " : " + device_version;

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

    return device_info;
}

bool opencl_hasher::__setup_device_info(opencl_device_info *device, double intensity_cpu, double intensity_gpu) {
    cl_int error;

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

    const char *srcptr[] = { opencl_kernel.c_str() };
    size_t srcsize = opencl_kernel.size();

    device->program = clCreateProgramWithSource(device->context, 1, srcptr, &srcsize, &error);
    if(error != CL_SUCCESS)  {
        device->error = error;
        device->error_message = "Error creating opencl program for device.";
        return false;
    }

    error=clBuildProgram(device->program, 1, &device->device, "", NULL, NULL);
    if(error != CL_SUCCESS)  {
        size_t log_size;
        clGetProgramBuildInfo(device->program, device->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *) malloc(log_size + 1);
        clGetProgramBuildInfo(device->program, device->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        log[log_size] = 0;
        string build_log = log;
        free(log);

        device->error = error;
        device->error_message = "Error building opencl program for device: " + build_log;
        return false;
    }

    device->kernel_cblocks = clCreateKernel(device->program, "fill_cblocks", &error);
    if(error != CL_SUCCESS)  {
        device->error = error;
        device->error_message = "Error creating opencl kernel for device.";
        return false;
    }

    device->kernel_gblocks = clCreateKernel(device->program, "fill_gblocks", &error);
    if(error != CL_SUCCESS)  {
        device->error = error;
        device->error_message = "Error creating opencl kernel for device.";
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

    device->arguments.segments_profile_4_4_16384 = clCreateBuffer(device->context, CL_MEM_READ_ONLY, 64 * 2 * sizeof(uint16_t), NULL, &error);
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
		int ref_chunk_idx = (i / 32) * 64;
		int ref_idx = i % 32;

		addresses_1_1_524288[ref_chunk_idx + ref_idx] = argon2profile_1_1_524288.block_refs[i*3];
		addresses_1_1_524288[ref_chunk_idx + ref_idx + 32] = argon2profile_1_1_524288.block_refs[i*3 + 2];
	}
    error=clEnqueueWriteBuffer(device->queue, device->arguments.address_profile_1_1_524288, CL_TRUE, 0, (argon2profile_1_1_524288.block_refs_size + 2) * 2 * sizeof(int32_t), addresses_1_1_524288, 0, NULL, NULL);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error writing to gpu memory.";
        return false;
    }
    free(addresses_1_1_524288);

	//optimise address sizes
	int16_t *addresses_4_4_16384 = (int16_t *)malloc(argon2profile_4_4_16384.block_refs_size * 2 * sizeof(int16_t));
	for(int i=0;i<argon2profile_4_4_16384.block_refs_size;i++) {
		addresses_4_4_16384[i*2] = argon2profile_4_4_16384.block_refs[i*3 + (i == 65528 ? 1 : 0)];
		addresses_4_4_16384[i*2 + 1] = argon2profile_4_4_16384.block_refs[i*3 + 2];
	}
    error=clEnqueueWriteBuffer(device->queue, device->arguments.address_profile_4_4_16384, CL_TRUE, 0, argon2profile_4_4_16384.block_refs_size * 2 * sizeof(int16_t), addresses_4_4_16384, 0, NULL, NULL);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error writing to gpu memory.";
        return false;
    }
    free(addresses_4_4_16384);

	//reorganize segments data
	uint16_t *segments_4_4_16384 = (uint16_t *)malloc(64 * 2 * sizeof(uint16_t));
	for(int i=0;i<64;i++) {
		int seg_start = argon2profile_4_4_16384.segments[i*3];
		segments_4_4_16384[i*2] = seg_start;
		segments_4_4_16384[i*2 + 1] = argon2profile_4_4_16384.block_refs[seg_start*3 + 1];
	}
    error=clEnqueueWriteBuffer(device->queue, device->arguments.segments_profile_4_4_16384, CL_TRUE, 0, 64 * 2 * sizeof(uint16_t), segments_4_4_16384, 0, NULL, NULL);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error writing to gpu memory.";
        return false;
    }
	free(segments_4_4_16384);

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
	clSetKernelArg(device->kernel_gblocks, 9, sizeof(device->arguments.segments_profile_4_4_16384), &device->arguments.segments_profile_4_4_16384);
	clSetKernelArg(device->kernel_gblocks, 10, sizeof(int32_t), &device->profile_info.threads_per_chunk_profile_4_4_16384);

    return true;
}

vector<opencl_device_info*> opencl_hasher::__query_opencl_devices(cl_int &error, string &error_message) {
    cl_int err;

    cl_uint platform_count = 0;
    cl_uint device_count = 0;

    vector<opencl_device_info*> result;

    clGetPlatformIDs(0, NULL, &platform_count);
    if(platform_count == 0) {
        return result;
    }

    cl_platform_id *platforms = (cl_platform_id*)malloc(platform_count * sizeof(cl_platform_id));

    err=clGetPlatformIDs(platform_count, platforms, &platform_count);
    if(err != CL_SUCCESS)  {
        free(platforms);
        error = err;
        error_message = "Error querying for opencl platforms.";
        return result;
    }

    int counter = 0;

    for(uint32_t i=0; i < platform_count; i++) {
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
            error_message = "Error querying for opencl devices.";
            continue;
        }

        for(uint32_t j=0; j < device_count; j++) {
            opencl_device_info *info = __get_device_info(platforms[i], devices[j]);
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

    free(platforms);

    return result;
}

opencl_hasher::opencl_hasher() {
    _type = "GPU";
	_subtype = "OPENCL";
	_priority = 1;
    _intensity = 0;
    __running = false;
    _description = "";
}

opencl_hasher::~opencl_hasher() {
//    this->cleanup();
}

bool opencl_hasher::configure(arguments &args) {
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

    vector<string> filter = args.gpu_filter();

    int total_threads_profile_4_4_16384 = 0;
    int total_threads_profile_1_1_524288 = 0;

    if (intensity_cpu == 0 && intensity_gpu == 0) {
        _intensity = 0;
        _description = "Status: DISABLED - by user.";
        return false;
    }

    for(vector<opencl_device_info *>::iterator d = __devices.begin(); d != __devices.end(); d++, index++) {
        stringstream ss;
        ss << "["<< (index + 1) << "] " << (*d)->device_string << endl;
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
                continue;
            }
        }

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
        total_threads_profile_4_4_16384 += (*d)->profile_info.threads_profile_4_4_16384;
        total_threads_profile_1_1_524288 += (*d)->profile_info.threads_profile_1_1_524288;
    }

    args.set_cards_count(index);

    if (total_threads_profile_4_4_16384 == 0 && total_threads_profile_1_1_524288 == 0) {
        _intensity = 0;
        _description += "Status: DISABLED - not enough resources.";
        return false;
    }

    _intensity = (intensity_cpu + intensity_gpu) / 2.0;

    __running = true;
    _update_running_status(__running);
    for(vector<opencl_device_info *>::iterator d = __devices.begin(); d != __devices.end(); d++) {
        if((*d)->profile_info.threads_profile_1_1_524288 != 0 || (*d)->profile_info.threads_profile_4_4_16384 != 0) {
            __runners.push_back(new thread([&](opencl_device_info *device) {
                this->__run(device, 0);
            }, (*d)));
            __runners.push_back(new thread([&](opencl_device_info *device) {
                this->__run(device, 1);
            }, (*d)));
        }
	}

    _description += "Status: ENABLED - with " + to_string(total_threads_profile_1_1_524288) + " threads for CPU blocks and " + to_string(total_threads_profile_4_4_16384) + " threads for GPU blocks.";

    return true;
}

struct opencl_gpumgmt_thread_data {
    int thread_id;
    opencl_device_info *device;
};

void *opencl_kernel_filler(void *memory, int threads, argon2profile *profile, void *user_data) {
	//    uint64_t start_log = microseconds();
	//    printf("Waiting for lock: %lld\n", microseconds() - start_log);
	//    start_log = microseconds();
	opencl_gpumgmt_thread_data *gpumgmt_thread = (opencl_gpumgmt_thread_data *)user_data;
    opencl_device_info *device = gpumgmt_thread->device;

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

	return memory;
}

void opencl_hasher::__run(opencl_device_info *device, int thread_id) {
	void *memory = malloc(8 * ARGON2_BLOCK_SIZE * max(device->profile_info.threads_profile_1_1_524288, device->profile_info.threads_profile_4_4_16384));
	
	opencl_gpumgmt_thread_data thread_data;
    thread_data.device = device;
    thread_data.thread_id = thread_id;

    argon2 hash_factory(opencl_kernel_filler, memory, &thread_data);
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
			_store_hash(stored_hashes);
		}
    }
	free(memory);
    _update_running_status(__running);
}

void opencl_hasher::cleanup() {
    __running = false;
    for(vector<thread*>::iterator it = __runners.begin();it != __runners.end();++it) {
        (*it)->join();
        delete *it;
    }
    __runners.clear();

    vector<cl_platform_id> platforms;

    for(vector<opencl_device_info *>::iterator it=__devices.begin(); it != __devices.end(); it++) {
		if ((*it)->profile_info.threads_profile_1_1_524288 != 0 || (*it)->profile_info.threads_profile_4_4_16384 != 0) {
			clReleaseMemObject((*it)->arguments.memory_chunk_0);
			clReleaseMemObject((*it)->arguments.memory_chunk_1);
			clReleaseMemObject((*it)->arguments.memory_chunk_2);
			clReleaseMemObject((*it)->arguments.memory_chunk_3);
			clReleaseMemObject((*it)->arguments.memory_chunk_4);
			clReleaseMemObject((*it)->arguments.memory_chunk_5);
			clReleaseMemObject((*it)->arguments.address_profile_1_1_524288);
			clReleaseMemObject((*it)->arguments.address_profile_4_4_16384);
			clReleaseMemObject((*it)->arguments.segments_profile_4_4_16384);
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

bool opencl_hasher::initialize() {
    cl_int error = CL_SUCCESS;
    string error_message;

    __devices = __query_opencl_devices(error, error_message);

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

REGISTER_HASHER(opencl_hasher);

#endif // WITH_OPENCL