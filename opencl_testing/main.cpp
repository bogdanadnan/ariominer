//
// Created by Haifa Bogdan Adnan on 09/08/2018.
//

#include <stdio.h>
#include <string.h>

#include "../common/common.h"
#include "../hash/argon2/argon2.h"

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif // !__APPLE__

struct kernel_arguments {
    cl_mem memory_chunk_0;
    cl_mem memory_chunk_1;
    cl_mem memory_chunk_2;
    cl_mem memory_chunk_3;
    cl_mem memory_chunk_4;
    cl_mem memory_chunk_5;
    cl_mem address_profile_1_1_524288;
    cl_mem address_profile_4_4_16384;
    cl_mem seed_memory;
    cl_mem out_memory;
};

struct argon2profile_info {
    uint32_t threads_profile_1_1_524288;
    uint32_t threads_per_chunk_profile_1_1_524288;
    uint32_t threads_profile_4_4_16384;
    uint32_t threads_per_chunk_profile_4_4_16384;
};

struct gpu_device_info {
    gpu_device_info(cl_int err, const string &err_msg) {
        error = err;
        error_message = err_msg;
    }

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    cl_program program;
    cl_kernel kernel;

    kernel_arguments arguments;
    argon2profile_info profile_info;

    string device_string;
    uint64_t max_mem_size;
    uint64_t max_allocable_mem_size;

    cl_int error;
    string error_message;
};

gpu_device_info __get_device_info(cl_platform_id platform, cl_device_id device) {
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

    char src[65000];
    FILE *fil=fopen("../hash/gpu/kernel.cl","r");
    size_t srcsize=fread(src, sizeof src, 1, fil);
    fclose(fil);

    const char *srcptr[] = { src };

    device_info.program = clCreateProgramWithSource(device_info.context, 1, srcptr, &srcsize, &error);
    if(error != CL_SUCCESS)  {
        return gpu_device_info(error, "Error creating opencl program for device.");
    }

    error=clBuildProgram(device_info.program, 1, &device_info.device, "", NULL, NULL);
    if(error != CL_SUCCESS)  {
        size_t log_size;
        clGetProgramBuildInfo(device_info.program, device_info.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *) malloc(log_size);
        clGetProgramBuildInfo(device_info.program, device_info.device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        string build_log = log;
        free(log);

        return gpu_device_info(error, "Error building opencl program for device: " + build_log);
    }

    device_info.kernel = clCreateKernel(device_info.program, "fill_blocks", &error);
    if(error != CL_SUCCESS)  {
        return gpu_device_info(error, "Error creating opencl kernel for device.");
    }

    return device_info;
}

bool __setup_device_info(gpu_device_info &device, int intensity) {
    cl_int error;

    device.profile_info.threads_per_chunk_profile_1_1_524288 = device.max_allocable_mem_size / argon2profile_1_1_524288.memsize;
    size_t chunk_size_profile_1_1_524288 = device.profile_info.threads_per_chunk_profile_1_1_524288 * argon2profile_1_1_524288.memsize;

    device.profile_info.threads_per_chunk_profile_4_4_16384 = device.max_allocable_mem_size / argon2profile_4_4_16384.memsize;
    size_t chunk_size_profile_4_4_16384 = device.profile_info.threads_per_chunk_profile_4_4_16384 * argon2profile_4_4_16384.memsize;

    size_t chunk_size = max(chunk_size_profile_1_1_524288, chunk_size_profile_4_4_16384);
    uint64_t usable_memory = (uint64_t)((device.max_mem_size * 0.95 * intensity) / 100.0); // leave 5% of memory for other tasks
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

vector<gpu_device_info> __query_opencl_devices(cl_int &error, string &error_message) {
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

uint64_t checksum(block *data) {
    uint64_t sum = 0;
    for(int i=0;i<ARGON2_QWORDS_IN_BLOCK;i++) {
        sum += data->v[i];
    }
    return sum;
}

void fill_memory_blocks(void *memory, int threads, argon2profile *profile, void *user_data) {
    for(unsigned int i=0;i<8192;i+=1024) {
        cout<<i / 1024 << " " << checksum((block*)((uint8_t*)memory + i)) << endl;
    }

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
    uint32_t profile_id;
    uint32_t xor_limit;

    if(strcmp(profile->profile_name, "1_1_524288") == 0) {
        threads_per_chunk = device->profile_info.threads_per_chunk_profile_1_1_524288;
        memsize = argon2profile_1_1_524288.memsize;
        addrsize = argon2profile_1_1_524288.block_refs_size;
        profile_id = 0;
        xor_limit = argon2profile_1_1_524288.xor_limit;
    }
    else {
        threads_per_chunk = device->profile_info.threads_per_chunk_profile_4_4_16384;
        memsize = argon2profile_4_4_16384.memsize;
        addrsize = argon2profile_4_4_16384.block_refs_size;
        profile_id = 1;
        xor_limit = argon2profile_4_4_16384.xor_limit;
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

    void *mem = malloc(2 * argon2profile_4_4_16384.memsize);
    error=clEnqueueReadBuffer(device->queue, device->arguments.memory_chunk_0, CL_TRUE, 0, argon2profile_4_4_16384.memsize, mem, 0, NULL, NULL);
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

    printf("%lld \n", checksum((block *)(memory)));
    printf("%lld \n", checksum((block *)((uint8_t*)memory + 1024)));
    for(unsigned int i=0;i<argon2profile_4_4_16384.memsize;i+=1024) {
        printf("%d %lld \n", i / 1024, checksum((block*)((uint8_t*)mem + i)));
    }
}

int main() {
    cl_int error;
    string error_message;


    vector<gpu_device_info> devices = __query_opencl_devices(error, error_message);

    if(!__setup_device_info(devices[0], 90)) {
        cout << devices[0].error_message << endl;
        return 0;
    }

    uint8_t *mem = (uint8_t*)malloc(max(devices[0].profile_info.threads_profile_1_1_524288, devices[0].profile_info.threads_profile_4_4_16384) * 8 * ARGON2_BLOCK_SIZE);
    argon2 hasher(fill_memory_blocks, mem, &devices[0]);

    hasher.set_seed_memory_offset(8 * ARGON2_BLOCK_SIZE);
    hasher.set_lane_length(2);
    hasher.set_threads(devices[0].profile_info.threads_profile_4_4_16384);

    vector<string> hashes = hasher.generate_hashes(argon2profile_4_4_16384, "PZ8Tyr4Nx8MHsRAGMpZmZ6TWY63dXWSCy7AEg3h9oYjeR74yj73q3gPxbxq9R3nxSSUV4KKgu1sQZu9Qj9v2q2HhT5H3LTHwW7HzAA28SjWFdzkNoovBMncD-sauULo1zM4tt9DhGEnO8qPe5nlzItJwwIKiIcAUDg-4KhqbBhShBf36zYeen943tS6KhgFmQixtUoVbf2egtBmD6j3NQtcueEBite2zjzdpK2ShaA28icRfJM9yPUQ6azN-56262626",
                                                   "NSHFFAg.iATJ0sfM");

    for(int i=0;i<hashes.size();i++) {
        cout<< hashes[i] << endl;
    }
}
