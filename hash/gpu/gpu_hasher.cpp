//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#include "../../common/common.h"
#include "../../app/arguments.h"

#include "../hasher.h"
#include "../argon2/argon2.h"

#include "gpu_hasher.h"
#include "opencl_kernel.h"

#define KERNEL_WORKGROUP_SIZE   32

gpu_device_info gpu_hasher::__get_device_info(cl_platform_id platform, cl_device_id device) {
    gpu_device_info device_info(CL_SUCCESS, "");

    device_info.platform = platform;
    device_info.device = device;

    cl_int error;
    char buffer[100];
    size_t sz;

    // device name
    string device_vendor;
    error = clGetDeviceInfo(device, CL_DEVICE_VENDOR, 100, buffer, &sz);
    buffer[sz] = 0;
    device_vendor = buffer;
    if(error != CL_SUCCESS) {
        return gpu_device_info(error, "Error querying device vendor.");
    }

    string device_name;
    error = clGetDeviceInfo(device, CL_DEVICE_NAME, 100, buffer, &sz);
    buffer[sz] = 0;
    device_name = buffer;
    if(error != CL_SUCCESS) {
        return gpu_device_info(error, "Error querying device name.");
    }

    string device_version;
    error = clGetDeviceInfo(device, CL_DEVICE_VERSION, 100, buffer, &sz);
    buffer[sz] = 0;
    device_version = buffer;
    if(error != CL_SUCCESS) {
        return gpu_device_info(error, "Error querying device version.");
    }

    device_info.device_string = device_vendor + " - " + device_name + " : " + device_version;

    error = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(device_info.max_mem_size), &(device_info.max_mem_size), NULL);
    if(error != CL_SUCCESS) {
        return gpu_device_info(error, "Error querying device global memory size.");
    }
    error = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(device_info.max_allocable_mem_size), &(device_info.max_allocable_mem_size), NULL);
    if(error != CL_SUCCESS) {
        return gpu_device_info(error, "Error querying device max memory allocation.");
    }

    cl_context_properties properties[]={
            CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
            0};

    device_info.context=clCreateContext(properties, 1, &device, NULL, NULL, &error);
    if(error != CL_SUCCESS)  {
        return gpu_device_info(error, "Error getting device context.");
    }

    device_info.queue = clCreateCommandQueue(device_info.context, device, 0, &error);
    if(error != CL_SUCCESS)  {
        return gpu_device_info(error, "Error getting device command queue.");
    }

    return device_info;
}

bool gpu_hasher::__setup_device_info(gpu_device_info &device, int intensity) {
    cl_int error;

    const char *srcptr[] = { opencl_kernel.c_str() };
    size_t srcsize = opencl_kernel.size();

    device.program = clCreateProgramWithSource(device.context, 1, srcptr, &srcsize, &error);
    if(error != CL_SUCCESS)  {
        device.error = error;
        device.error_message = "Error creating opencl program for device.";
        return false;
    }

    error=clBuildProgram(device.program, 1, &device.device, "", NULL, NULL);
    if(error != CL_SUCCESS)  {
        size_t log_size;
        clGetProgramBuildInfo(device.program, device.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *) malloc(log_size);
        clGetProgramBuildInfo(device.program, device.device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        string build_log = log;
        free(log);

        device.error = error;
        device.error_message = "Error building opencl program for device: " + build_log;
        return false;
    }

    device.kernel = clCreateKernel(device.program, "fill_blocks", &error);
    if(error != CL_SUCCESS)  {
        device.error = error;
        device.error_message = "Error creating opencl kernel for device.";
        return false;
    }

    device.profile_info.threads_per_chunk_profile_1_1_524288 = device.max_allocable_mem_size / argon2profile_1_1_524288.memsize;
    size_t chunk_size_profile_1_1_524288 = device.profile_info.threads_per_chunk_profile_1_1_524288 * argon2profile_1_1_524288.memsize;

    device.profile_info.threads_per_chunk_profile_4_4_16384 = device.max_allocable_mem_size / argon2profile_4_4_16384.memsize;
    size_t chunk_size_profile_4_4_16384 = device.profile_info.threads_per_chunk_profile_4_4_16384 * argon2profile_4_4_16384.memsize;

    size_t chunk_size = max(chunk_size_profile_1_1_524288, chunk_size_profile_4_4_16384);
    uint64_t usable_memory = (uint64_t)((device.max_mem_size * 0.9 * intensity) / 100.0); // leave 10% of memory for other tasks
    double chunks = (double)usable_memory / (double)chunk_size;

    device.profile_info.threads_profile_1_1_524288 = device.profile_info.threads_per_chunk_profile_1_1_524288 * chunks;
    device.profile_info.threads_profile_4_4_16384 = device.profile_info.threads_per_chunk_profile_4_4_16384 * chunks;

    size_t max_threads = max(device.profile_info.threads_profile_1_1_524288, device.profile_info.threads_profile_4_4_16384);

    double counter = chunks;
    double allocated_mem_for_current_chunk = 0;

    if(counter > 0) {
        if (counter > 1) {
            allocated_mem_for_current_chunk = chunk_size;
        } else {
            allocated_mem_for_current_chunk = chunk_size * counter;
        }
        counter -= 1;
    }
    else {
        allocated_mem_for_current_chunk = 1;
    }
    device.arguments.memory_chunk_0 = clCreateBuffer(device.context, CL_MEM_READ_WRITE, allocated_mem_for_current_chunk, NULL, &error);
    if(error != CL_SUCCESS) {
        device.error = error;
        device.error_message = "Error creating memory buffer.";
        return false;
    }

    if(counter > 0) {
        if (counter > 1) {
            allocated_mem_for_current_chunk = chunk_size;
        } else {
            allocated_mem_for_current_chunk = chunk_size * counter;
        }
        counter -= 1;
    }
    else {
        allocated_mem_for_current_chunk = 1;
    }
    device.arguments.memory_chunk_1 = clCreateBuffer(device.context, CL_MEM_READ_WRITE, allocated_mem_for_current_chunk, NULL, &error);
    if(error != CL_SUCCESS) {
        device.error = error;
        device.error_message = "Error creating memory buffer.";
        return false;
    }

    if(counter > 0) {
        if (counter > 1) {
            allocated_mem_for_current_chunk = chunk_size;
        } else {
            allocated_mem_for_current_chunk = chunk_size * counter;
        }
        counter -= 1;
    }
    else {
        allocated_mem_for_current_chunk = 1;
    }
    device.arguments.memory_chunk_2 = clCreateBuffer(device.context, CL_MEM_READ_WRITE, allocated_mem_for_current_chunk, NULL, &error);
    if(error != CL_SUCCESS) {
        device.error = error;
        device.error_message = "Error creating memory buffer.";
        return false;
    }

    if(counter > 0) {
        if (counter > 1) {
            allocated_mem_for_current_chunk = chunk_size;
        } else {
            allocated_mem_for_current_chunk = chunk_size * counter;
        }
        counter -= 1;
    }
    else {
        allocated_mem_for_current_chunk = 1;
    }
    device.arguments.memory_chunk_3 = clCreateBuffer(device.context, CL_MEM_READ_WRITE, allocated_mem_for_current_chunk, NULL, &error);
    if(error != CL_SUCCESS) {
        device.error = error;
        device.error_message = "Error creating memory buffer.";
        return false;
    }

    if(counter > 0) {
        if (counter > 1) {
            allocated_mem_for_current_chunk = chunk_size;
        } else {
            allocated_mem_for_current_chunk = chunk_size * counter;
        }
        counter -= 1;
    }
    else {
        allocated_mem_for_current_chunk = 1;
    }
    device.arguments.memory_chunk_4 = clCreateBuffer(device.context, CL_MEM_READ_WRITE, allocated_mem_for_current_chunk, NULL, &error);
    if(error != CL_SUCCESS) {
        device.error = error;
        device.error_message = "Error creating memory buffer.";
        return false;
    }

    if(counter > 0) {
        if (counter > 1) {
            allocated_mem_for_current_chunk = chunk_size;
        } else {
            allocated_mem_for_current_chunk = chunk_size * counter;
        }
        counter -= 1;
    }
    else {
        allocated_mem_for_current_chunk = 1;
    }
    device.arguments.memory_chunk_5 = clCreateBuffer(device.context, CL_MEM_READ_WRITE, allocated_mem_for_current_chunk, NULL, &error);
    if(error != CL_SUCCESS) {
        device.error = error;
        device.error_message = "Error creating memory buffer.";
        return false;
    }

    device.arguments.address_profile_1_1_524288 = clCreateBuffer(device.context, CL_MEM_READ_ONLY, argon2profile_1_1_524288.block_refs_size * 3 * sizeof(uint32_t), NULL, &error);
    if(error != CL_SUCCESS) {
        device.error = error;
        device.error_message = "Error creating memory buffer.";
        return false;
    }

    device.arguments.address_profile_4_4_16384 = clCreateBuffer(device.context, CL_MEM_READ_ONLY, argon2profile_4_4_16384.block_refs_size * 3 * sizeof(uint32_t), NULL, &error);
    if(error != CL_SUCCESS) {
        device.error = error;
        device.error_message = "Error creating memory buffer.";
        return false;
    }

    device.arguments.seed_memory = clCreateBuffer(device.context, CL_MEM_READ_WRITE, max_threads * 8 * ARGON2_BLOCK_SIZE, NULL, &error);
    if(error != CL_SUCCESS) {
        device.error = error;
        device.error_message = "Error creating memory buffer.";
        return false;
    }

    device.arguments.out_memory = clCreateBuffer(device.context, CL_MEM_WRITE_ONLY, max_threads * 8 * ARGON2_BLOCK_SIZE, NULL, &error);
    if(error != CL_SUCCESS) {
        device.error = error;
        device.error_message = "Error creating memory buffer.";
        return false;
    }

    error=clEnqueueWriteBuffer(device.queue, device.arguments.address_profile_1_1_524288, CL_TRUE, 0, argon2profile_1_1_524288.block_refs_size * 3 * sizeof(uint32_t), argon2profile_1_1_524288.block_refs, 0, NULL, NULL);
    if(error != CL_SUCCESS) {
        device.error = error;
        device.error_message = "Error writing to gpu memory.";
        return false;
    }
    error=clEnqueueWriteBuffer(device.queue, device.arguments.address_profile_4_4_16384, CL_TRUE, 0, argon2profile_4_4_16384.block_refs_size * 3 * sizeof(uint32_t), argon2profile_4_4_16384.block_refs, 0, NULL, NULL);
    if(error != CL_SUCCESS) {
        device.error = error;
        device.error_message = "Error writing to gpu memory.";
        return false;
    }
    error=clFinish(device.queue);
    if(error != CL_SUCCESS) {
        device.error = error;
        device.error_message = "Error writing to gpu memory.";
        return false;
    }

    clSetKernelArg(device.kernel, 0, sizeof(device.arguments.memory_chunk_0), &device.arguments.memory_chunk_0);
    clSetKernelArg(device.kernel, 1, sizeof(device.arguments.memory_chunk_1), &device.arguments.memory_chunk_1);
    clSetKernelArg(device.kernel, 2, sizeof(device.arguments.memory_chunk_2), &device.arguments.memory_chunk_2);
    clSetKernelArg(device.kernel, 3, sizeof(device.arguments.memory_chunk_3), &device.arguments.memory_chunk_3);
    clSetKernelArg(device.kernel, 4, sizeof(device.arguments.memory_chunk_4), &device.arguments.memory_chunk_4);
    clSetKernelArg(device.kernel, 5, sizeof(device.arguments.memory_chunk_5), &device.arguments.memory_chunk_5);
    clSetKernelArg(device.kernel, 6, sizeof(device.arguments.address_profile_1_1_524288), &device.arguments.address_profile_1_1_524288);
    clSetKernelArg(device.kernel, 7, sizeof(device.arguments.address_profile_4_4_16384), &device.arguments.address_profile_4_4_16384);
    clSetKernelArg(device.kernel, 8, sizeof(device.arguments.out_memory), &device.arguments.out_memory);

    return true;
}

vector<gpu_device_info> gpu_hasher::__query_opencl_devices(cl_int &error, string &error_message) {
    cl_int err;

    cl_uint platform_count = 0;
    cl_uint device_count = 0;

    vector<gpu_device_info> result;

    clGetPlatformIDs(0, NULL, &platform_count);
    if(platform_count == 0) {
        return vector<gpu_device_info>();
    }

    cl_platform_id *platforms = (cl_platform_id*)malloc(platform_count * sizeof(cl_platform_id));

    err=clGetPlatformIDs(platform_count, platforms, &platform_count);
    if(err != CL_SUCCESS)  {
        free(platforms);
        error = err;
        error_message = "Error querying for opencl platforms.";
        return vector<gpu_device_info>();
    }

    for(int i=0;i<platform_count;i++) {
        device_count = 0;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &device_count);
        if(device_count == 0) {
            continue;
        }

        cl_device_id * devices = (cl_device_id*)malloc(device_count);
        err=clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, device_count, devices, &device_count);

        if(err != CL_SUCCESS)  {
            free(devices);
            error = err;
            error_message = "Error querying for opencl devices.";
            continue;
        }

        for(int j=0; j<device_count;j++) {
            gpu_device_info info = __get_device_info(platforms[i], devices[j]);
            if(info.error != CL_SUCCESS) {
                error = info.error;
                error_message = info.error_message;
            }
            else {
                result.push_back(info);
            }
        }

        free(devices);
    }

    free(platforms);

    return result;
}

void kernel_filler(void *memory, int threads, argon2profile *profile, void *user_data) {
    gpu_device_info *device = (gpu_device_info*) user_data;

    cl_int error;

    int mem_seed_count = profile->thr_cost;

    error=clEnqueueWriteBuffer(device->queue, device->arguments.seed_memory, CL_TRUE, 0, threads * 2 * mem_seed_count * ARGON2_BLOCK_SIZE, memory, 0, NULL, NULL);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error writing to gpu memory.";
        return;
    }

    error=clFinish(device->queue);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error writing to gpu memory.";
        return;
    }

    clSetKernelArg(device->kernel, 9, sizeof(device->arguments.seed_memory), &device->arguments.seed_memory);

    size_t total_work_items = threads * 32;
    size_t local_work_items = 32;

    uint32_t threads_per_chunk;
    uint32_t memsize;
    uint32_t addrsize;
    uint32_t xor_limit;
    uint32_t profile_id;

    if(strcmp(profile->profile_name, "1_1_524288") == 0) {
        threads_per_chunk = device->profile_info.threads_per_chunk_profile_1_1_524288;
        memsize = argon2profile_1_1_524288.memsize;
        addrsize = argon2profile_1_1_524288.block_refs_size;
        xor_limit = argon2profile_1_1_524288.xor_limit;
        profile_id = 0;
    }
    else {
        threads_per_chunk = device->profile_info.threads_per_chunk_profile_4_4_16384;
        memsize = argon2profile_4_4_16384.memsize;
        addrsize = argon2profile_4_4_16384.block_refs_size;
        xor_limit = argon2profile_4_4_16384.xor_limit;
        profile_id = 1;
    }

    clSetKernelArg(device->kernel, 10, sizeof(int), &threads);
    clSetKernelArg(device->kernel, 11, sizeof(uint32_t), &threads_per_chunk);
    clSetKernelArg(device->kernel, 12, sizeof(uint32_t), &memsize);
    clSetKernelArg(device->kernel, 13, sizeof(uint32_t), &addrsize);
    clSetKernelArg(device->kernel, 14, sizeof(uint32_t), &xor_limit);
    clSetKernelArg(device->kernel, 15, sizeof(uint32_t), &profile_id);

    error=clEnqueueNDRangeKernel(device->queue, device->kernel, 1, NULL, &total_work_items, &local_work_items, 0, NULL, NULL);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error running the kernel.";
        return;
    }

    error=clFinish(device->queue);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error running the kernel.";
        return;
    }

    error=clEnqueueReadBuffer(device->queue, device->arguments.out_memory, CL_TRUE, 0, threads * 2 * mem_seed_count * ARGON2_BLOCK_SIZE, memory, 0, NULL, NULL);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error reading gpu memory.";
        return;
    }

    error=clFinish(device->queue);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error reading gpu memory.";
        return;
    }
}

gpu_hasher::gpu_hasher() {
    _type = "GPU";
    _intensity = 0;
    __running = false;
    _description = "";
}

gpu_hasher::~gpu_hasher() {
    __running = false;
    for(vector<thread*>::iterator it = __runners.begin();it != __runners.end();++it) {
        (*it)->join();
        delete *it;
    }
}

bool gpu_hasher::configure(arguments &args) {
    int index = 1;
    int intensity = args.gpu_intensity();
    int total_threads = 0;

    if (intensity == 0) {
        _intensity = 0;
        _description += "Status: DISABLED - by user.";
        return false;
    }

    stringstream ss;

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

    for(vector<gpu_device_info>::iterator d = __devices.begin(); d != __devices.end(); d++) {
        ss << "["<< index << "] " << d->device_string << endl;

        _description += ss.str();

        if(!(__setup_device_info(*d, intensity))) {
            _description += d->error_message;
            _description += "\n";
            continue;
        };
        total_threads += d->profile_info.threads_profile_4_4_16384;
    }

    if (total_threads == 0) {
        _intensity = 0;
        _description += "Status: DISABLED - not enough resources.";
        return false;
    }

    _intensity = intensity;

    __running = true;
    for(auto d = __devices.begin(); d != __devices.end(); d++) {
        __runners.push_back(new thread(&gpu_hasher::__run, ref(*this), &(*d)));
    }

    _description += "Status: ENABLED - with " + to_string(total_threads) + " threads.";

    return false;
}

void gpu_hasher::__run(gpu_device_info *device) {
    void *memory = malloc(8 * ARGON2_BLOCK_SIZE * max(device->profile_info.threads_profile_1_1_524288, device->profile_info.threads_profile_4_4_16384));

    argon2 hash_factory(kernel_filler, memory, device);
    hash_factory.set_lane_length(2);

    while(__running) {
        if(should_pause()) {
            this_thread::sleep_for(chrono::milliseconds(100));
            continue;
        }

        hash_data input = get_input();
        argon2profile *profile = get_argon2profile();

        if(!input.base.empty()) {
            if(strcmp(profile->profile_name, "1_1_524288") == 0) {
                hash_factory.set_seed_memory_offset(2 * ARGON2_BLOCK_SIZE);
                hash_factory.set_threads(device->profile_info.threads_profile_1_1_524288);
            }
            else {
                hash_factory.set_seed_memory_offset(8 * ARGON2_BLOCK_SIZE);
                hash_factory.set_threads(device->profile_info.threads_profile_4_4_16384);
            }

            vector<string> hashes = hash_factory.generate_hashes(*profile, input.base, input.salt);
            for(vector<string>::iterator it = hashes.begin(); it != hashes.end(); ++it) {
                input.hash = *it;
                _store_hash(input);
            }
        }
    }

    free(memory);
}

REGISTER_HASHER(gpu_hasher);