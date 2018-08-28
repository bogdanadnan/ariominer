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
    char *buffer;
    size_t sz;

    // device name
    string device_vendor;
    sz = 0;
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, 0, NULL, &sz);
    buffer = (char *)malloc(sz + 1);
    error = clGetDeviceInfo(device, CL_DEVICE_VENDOR, sz, buffer, &sz);
    if(error != CL_SUCCESS) {
        free(buffer);
        return gpu_device_info(error, "Error querying device vendor.");
    }
    else {
        buffer[sz] = 0;
        device_vendor = buffer;
        free(buffer);
    }

    string device_name;
    sz = 0;
    clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &sz);
    buffer = (char *)malloc(sz + 1);
    error = clGetDeviceInfo(device, CL_DEVICE_NAME, sz, buffer, &sz);
    if(error != CL_SUCCESS) {
        free(buffer);
        return gpu_device_info(error, "Error querying device name.");
    }
    else {
        buffer[sz] = 0;
        device_name = buffer;
        free(buffer);
    }

    string device_version;
    sz = 0;
    clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, NULL, &sz);
    buffer = (char *)malloc(sz + 1);
    error = clGetDeviceInfo(device, CL_DEVICE_VERSION, sz, buffer, &sz);
    if(error != CL_SUCCESS) {
        free(buffer);
        return gpu_device_info(error, "Error querying device version.");
    }
    else {
        buffer[sz] = 0;
        device_version = buffer;
        free(buffer);
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

    return device_info;
}

bool gpu_hasher::__setup_device_info(gpu_device_info &device, double intensity_cpu, double intensity_gpu) {
    cl_int error;

    cl_context_properties properties[]={
            CL_CONTEXT_PLATFORM, (cl_context_properties)device.platform,
            0};

    device.context=clCreateContext(properties, 1, &(device.device), NULL, NULL, &error);
    if(error != CL_SUCCESS)  {
        device.error = error;
        device.error_message = "Error getting device context.";
        return false;
    }

    device.queue = clCreateCommandQueue(device.context, device.device, 0, &error);
    if(error != CL_SUCCESS)  {
        device.error = error;
        device.error_message = "Error getting device command queue.";
        return false;
    }

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
        char *log = (char *) malloc(log_size + 1);
        clGetProgramBuildInfo(device.program, device.device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        log[log_size] = 0;
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

    device.profile_info.threads_per_chunk_profile_1_1_524288 = (uint32_t)(device.max_allocable_mem_size / argon2profile_1_1_524288.memsize);
    size_t chunk_size_profile_1_1_524288 = device.profile_info.threads_per_chunk_profile_1_1_524288 * argon2profile_1_1_524288.memsize;

    device.profile_info.threads_per_chunk_profile_4_4_16384 = (uint32_t)(device.max_allocable_mem_size / argon2profile_4_4_16384.memsize);
    size_t chunk_size_profile_4_4_16384 = device.profile_info.threads_per_chunk_profile_4_4_16384 * argon2profile_4_4_16384.memsize;

    if(chunk_size_profile_1_1_524288 == 0 || chunk_size_profile_4_4_16384 == 0) {
        device.error = -1;
        device.error_message = "Not enough memory on GPU.";
        return false;
    }

    size_t chunk_size = max(chunk_size_profile_1_1_524288, chunk_size_profile_4_4_16384);
    uint64_t usable_memory = (uint64_t)(device.max_mem_size * 0.9); // leave 10% of memory for other tasks
    double chunks = (double)usable_memory / (double)chunk_size;

    uint32_t max_threads_1_1_524288 = (uint32_t)(device.profile_info.threads_per_chunk_profile_1_1_524288 * chunks);
    uint32_t max_threads_4_4_16384 = (uint32_t)(device.profile_info.threads_per_chunk_profile_4_4_16384 * chunks);

    if(max_threads_1_1_524288 == 0 && max_threads_4_4_16384 == 0) {
        device.error = -1;
        device.error_message = "Not enough memory on GPU.";
        return false;
    }

    device.profile_info.threads_profile_1_1_524288 = (uint32_t)(max_threads_1_1_524288 * intensity_cpu / 100.0);
    if(device.profile_info.threads_profile_1_1_524288 == 0 && intensity_cpu > 0)
        device.profile_info.threads_profile_1_1_524288 = 1;
    device.profile_info.threads_profile_4_4_16384 = (uint32_t)(max_threads_4_4_16384 * intensity_gpu / 100.0);
    if(device.profile_info.threads_profile_4_4_16384 == 0 && intensity_gpu > 0)
        device.profile_info.threads_profile_4_4_16384 = 1;

    size_t max_threads = max(device.profile_info.threads_profile_4_4_16384, device.profile_info.threads_profile_1_1_524288);

    double chunks_1_1_524288 = (double)device.profile_info.threads_profile_1_1_524288 / (double)device.profile_info.threads_per_chunk_profile_1_1_524288;
    double chunks_4_4_16384 = (double)device.profile_info.threads_profile_4_4_16384 / (double)device.profile_info.threads_per_chunk_profile_4_4_16384;

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
			allocated_mem_for_current_chunk = (size_t)ceil(chunk_size * counter);
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
			allocated_mem_for_current_chunk = (size_t)ceil(chunk_size * counter);
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
			allocated_mem_for_current_chunk = (size_t)ceil(chunk_size * counter);
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
			allocated_mem_for_current_chunk = (size_t)ceil(chunk_size * counter);
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
			allocated_mem_for_current_chunk = (size_t)ceil(chunk_size * counter);
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

    device.arguments.address_profile_1_1_524288 = clCreateBuffer(device.context, CL_MEM_READ_ONLY, argon2profile_1_1_524288.block_refs_size * 3 * sizeof(int32_t), NULL, &error);
    if(error != CL_SUCCESS) {
        device.error = error;
        device.error_message = "Error creating memory buffer.";
        return false;
    }

    device.arguments.address_profile_4_4_16384 = clCreateBuffer(device.context, CL_MEM_READ_ONLY, argon2profile_4_4_16384.block_refs_size * 3 * sizeof(int32_t), NULL, &error);
    if(error != CL_SUCCESS) {
        device.error = error;
        device.error_message = "Error creating memory buffer.";
        return false;
    }

    device.arguments.segments_profile_1_1_524288 = clCreateBuffer(device.context, CL_MEM_READ_ONLY, 3 * sizeof(int32_t), NULL, &error);
    if(error != CL_SUCCESS) {
        device.error = error;
        device.error_message = "Error creating memory buffer.";
        return false;
    }

    device.arguments.segments_profile_4_4_16384 = clCreateBuffer(device.context, CL_MEM_READ_ONLY, 64 * 3 * sizeof(int32_t), NULL, &error);
    if(error != CL_SUCCESS) {
        device.error = error;
        device.error_message = "Error creating memory buffer.";
        return false;
    }

    device.arguments.seed_memory[0] = clCreateBuffer(device.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, max_threads * 8 * ARGON2_BLOCK_SIZE, NULL, &error);
    if(error != CL_SUCCESS) {
        device.error = error;
        device.error_message = "Error creating memory buffer.";
        return false;
    }

    device.arguments.seed_memory[1] = clCreateBuffer(device.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, max_threads * 8 * ARGON2_BLOCK_SIZE, NULL, &error);
    if(error != CL_SUCCESS) {
        device.error = error;
        device.error_message = "Error creating memory buffer.";
        return false;
    }

    device.arguments.out_memory[0] = clCreateBuffer(device.context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, max_threads * 8 * ARGON2_BLOCK_SIZE, NULL, &error);
    if(error != CL_SUCCESS) {
        device.error = error;
        device.error_message = "Error creating memory buffer.";
        return false;
    }

    device.arguments.out_memory[1] = clCreateBuffer(device.context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, max_threads * 8 * ARGON2_BLOCK_SIZE, NULL, &error);
    if(error != CL_SUCCESS) {
        device.error = error;
        device.error_message = "Error creating memory buffer.";
        return false;
    }

    error=clEnqueueWriteBuffer(device.queue, device.arguments.address_profile_1_1_524288, CL_TRUE, 0, argon2profile_1_1_524288.block_refs_size * 3 * sizeof(int32_t), argon2profile_1_1_524288.block_refs, 0, NULL, NULL);
    if(error != CL_SUCCESS) {
        device.error = error;
        device.error_message = "Error writing to gpu memory.";
        return false;
    }

    error=clEnqueueWriteBuffer(device.queue, device.arguments.address_profile_4_4_16384, CL_TRUE, 0, argon2profile_4_4_16384.block_refs_size * 3 * sizeof(int32_t), argon2profile_4_4_16384.block_refs, 0, NULL, NULL);
    if(error != CL_SUCCESS) {
        device.error = error;
        device.error_message = "Error writing to gpu memory.";
        return false;
    }

    error=clEnqueueWriteBuffer(device.queue, device.arguments.segments_profile_1_1_524288, CL_TRUE, 0, 3 * sizeof(int32_t), argon2profile_1_1_524288.segments, 0, NULL, NULL);
    if(error != CL_SUCCESS) {
        device.error = error;
        device.error_message = "Error writing to gpu memory.";
        return false;
    }

    error=clEnqueueWriteBuffer(device.queue, device.arguments.segments_profile_4_4_16384, CL_TRUE, 0, 64 * 3 * sizeof(int32_t), argon2profile_4_4_16384.segments, 0, NULL, NULL);
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
    clSetKernelArg(device.kernel, 7, sizeof(device.arguments.address_profile_1_1_524288), &device.arguments.address_profile_1_1_524288);
    clSetKernelArg(device.kernel, 8, sizeof(device.arguments.address_profile_4_4_16384), &device.arguments.address_profile_4_4_16384);
    clSetKernelArg(device.kernel, 9, sizeof(device.arguments.segments_profile_1_1_524288), &device.arguments.segments_profile_1_1_524288);
    clSetKernelArg(device.kernel, 10, sizeof(device.arguments.segments_profile_4_4_16384), &device.arguments.segments_profile_4_4_16384);

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
    double intensity_cpu = args.gpu_intensity_cblocks();
    double intensity_gpu = args.gpu_intensity_gblocks();
    vector<string> filter = args.gpu_filter();

    int total_threads_profile_4_4_16384 = 0;
    int total_threads_profile_1_1_524288 = 0;

    if (intensity_cpu == 0 && intensity_gpu == 0) {
        _intensity = 0;
        _description += "Status: DISABLED - by user.";
        return false;
    }

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

    for(vector<gpu_device_info>::iterator d = __devices.begin(); d != __devices.end(); d++, index++) {
        stringstream ss;
        ss << "["<< index << "] " << d->device_string << endl;
        string device_description = ss.str();

        if(filter.size() > 0) {
            bool found = false;
            for(vector<string>::iterator fit = filter.begin(); fit != filter.end(); fit++) {
                if(device_description.find(*fit) != string::npos) {
                    found = true;
                    break;
                }
            }
            if(!found) {
                d->profile_info.threads_profile_4_4_16384 = 0;
                d->profile_info.threads_profile_1_1_524288 = 0;
                continue;
            }
        }

        _description += ss.str();

        if(!(__setup_device_info(*d, intensity_cpu, intensity_gpu))) {
            _description += d->error_message;
            _description += "\n";
            continue;
        };
        total_threads_profile_4_4_16384 += d->profile_info.threads_profile_4_4_16384;
        total_threads_profile_1_1_524288 += d->profile_info.threads_profile_1_1_524288;
    }

    if (total_threads_profile_4_4_16384 == 0 && total_threads_profile_1_1_524288 == 0) {
        _intensity = 0;
        _description += "Status: DISABLED - not enough resources.";
        return false;
    }

    _intensity = (intensity_cpu + intensity_gpu) / 2.0;

    __running = true;
    for(vector<gpu_device_info>::iterator d = __devices.begin(); d != __devices.end(); d++) {
        if(d->profile_info.threads_profile_1_1_524288 != 0 || d->profile_info.threads_profile_4_4_16384 != 0) {
            __runners.push_back(new thread([&](gpu_device_info *device) {
                this->__run(device, 0);
            }, &(*d)));
            __runners.push_back(new thread([&](gpu_device_info *device) {
                this->__run(device, 1);
            }, &(*d)));
        }
	}

    _description += "Status: ENABLED - with " + to_string(total_threads_profile_1_1_524288) + " threads for CPU blocks and " + to_string(total_threads_profile_4_4_16384) + " threads for GPU blocks.";

    return true;
}

void *map_input_memory(gpu_device_info *device, int id, size_t sz) {
    cl_int error;
    void *mem = clEnqueueMapBuffer(device->queue, device->arguments.seed_memory[id], CL_TRUE, CL_MAP_WRITE, 0, sz, 0, NULL, NULL, &error);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error mapping input GPU memory.";
        return NULL;
    }
    return mem;
}

void unmap_input_memory(gpu_device_info *device, int id, void *mem) {
    cl_int error = clEnqueueUnmapMemObject(device->queue, device->arguments.seed_memory[id], mem, 0, NULL, NULL);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error unmapping input GPU memory.";
    }
}

void *map_output_memory(gpu_device_info *device, int id, size_t sz) {
    cl_int error;
    void *mem = clEnqueueMapBuffer(device->queue, device->arguments.out_memory[id], CL_TRUE, CL_MAP_READ, 0, sz, 0, NULL, NULL, &error);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error mapping output GPU memory.";
        return NULL;
    }
    return mem;
}

void unmap_output_memory(gpu_device_info *device, int id, void *mem) {
     cl_int error = clEnqueueUnmapMemObject(device->queue, device->arguments.out_memory[id], mem, 0, NULL, NULL);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error unmapping output GPU memory.";
    }
}

struct gpumgmt_thread_data {
    int thread_id;
    gpu_device_info *device;
};

void *kernel_filler(void *memory, int threads, argon2profile *profile, void *user_data) {
    gpumgmt_thread_data *gpumgmt_thread = (gpumgmt_thread_data *)user_data;
    gpu_device_info *device = gpumgmt_thread->device;

    cl_int error;

    int mem_seed_count = profile->thr_cost;

    unmap_input_memory(device, gpumgmt_thread->thread_id, memory);

    size_t total_work_items;
    size_t local_work_items;

    uint32_t threads_per_chunk;
    uint32_t memsize;
    uint32_t addrsize;
    uint32_t parallelism;

    if(strcmp(profile->profile_name, "1_1_524288") == 0) {
        threads_per_chunk = device->profile_info.threads_per_chunk_profile_1_1_524288;
        memsize = (uint32_t)argon2profile_1_1_524288.memsize;
        addrsize = (uint32_t)argon2profile_1_1_524288.block_refs_size;
        parallelism = argon2profile_1_1_524288.thr_cost;
        total_work_items = threads * KERNEL_WORKGROUP_SIZE * parallelism;
        local_work_items = KERNEL_WORKGROUP_SIZE * parallelism;
    }
    else {
        threads_per_chunk = device->profile_info.threads_per_chunk_profile_4_4_16384;
        memsize = (uint32_t)argon2profile_4_4_16384.memsize;
        addrsize = (uint32_t)argon2profile_4_4_16384.block_refs_size;
        parallelism = argon2profile_4_4_16384.thr_cost;
        total_work_items = threads * KERNEL_WORKGROUP_SIZE * parallelism;
        local_work_items = KERNEL_WORKGROUP_SIZE * parallelism;
    }

//    uint64_t start_log = microseconds();

    device->device_lock->lock();

//    printf("Waiting for lock: %lld\n", microseconds() - start_log);
//    start_log = microseconds();

    clSetKernelArg(device->kernel, 6, sizeof(device->arguments.out_memory[gpumgmt_thread->thread_id]), &device->arguments.out_memory[gpumgmt_thread->thread_id]);
    clSetKernelArg(device->kernel, 12, sizeof(int), &threads);
    clSetKernelArg(device->kernel, 13, sizeof(uint32_t), &threads_per_chunk);
    clSetKernelArg(device->kernel, 14, sizeof(uint32_t), &memsize);
    clSetKernelArg(device->kernel, 15, sizeof(uint32_t), &addrsize);
    clSetKernelArg(device->kernel, 16, sizeof(uint32_t), &parallelism);
    clSetKernelArg(device->kernel, 11, sizeof(device->arguments.seed_memory[gpumgmt_thread->thread_id]), &device->arguments.seed_memory[gpumgmt_thread->thread_id]);

    error=clEnqueueNDRangeKernel(device->queue, device->kernel, 1, NULL, &total_work_items, &local_work_items, 0, NULL, NULL);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error running the kernel.";
        device->device_lock->unlock();
        return NULL;
    }

    error=clFinish(device->queue);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error running the kernel.";
        device->device_lock->unlock();
        return NULL;
    }

//    printf("Execute kernel: %lld\n", microseconds() - start_log);

    device->device_lock->unlock();
    return map_output_memory(device, gpumgmt_thread->thread_id, threads * 2 * mem_seed_count * ARGON2_BLOCK_SIZE);
}

void gpu_hasher::__run(gpu_device_info *device, int thread_id) {
    gpumgmt_thread_data thread_data;
    thread_data.device = device;
    thread_data.thread_id = thread_id;

    argon2 hash_factory(kernel_filler, NULL, &thread_data);
    hash_factory.set_lane_length(2);

    while(__running) {
        if(should_pause()) {
            this_thread::sleep_for(chrono::milliseconds(100));
            continue;
        }

//        uint64_t start_log = microseconds();

        hash_data input = get_input();
        argon2profile *profile = get_argon2profile();

        if(!input.base.empty()) {
            if(strcmp(profile->profile_name, "1_1_524288") == 0) {
                if(device->profile_info.threads_profile_1_1_524288 == 0) {
                    this_thread::sleep_for(chrono::milliseconds(100));
                    continue;
                }
                void *mem = map_input_memory(device, thread_id, device->profile_info.threads_profile_1_1_524288 * 2 * ARGON2_BLOCK_SIZE);
                if(mem == NULL) {
                    LOG("Error running kernel: (" + to_string(device->error) + ")" + device->error_message);
                    this_thread::sleep_for(chrono::milliseconds(100));
                    continue;
                }
                hash_factory.set_seed_memory((uint8_t *)mem);
                hash_factory.set_seed_memory_offset(2 * ARGON2_BLOCK_SIZE);
                hash_factory.set_threads(device->profile_info.threads_profile_1_1_524288);
            }
            else {
                if(device->profile_info.threads_profile_4_4_16384 == 0) {
                    this_thread::sleep_for(chrono::milliseconds(100));
                    continue;
                }
                void *mem = map_input_memory(device, thread_id, device->profile_info.threads_profile_4_4_16384 * 8 * ARGON2_BLOCK_SIZE);
                if(mem == NULL) {
                    LOG("Error running kernel: (" + to_string(device->error) + ")" + device->error_message);
                    this_thread::sleep_for(chrono::milliseconds(100));
                    continue;
                }
                hash_factory.set_seed_memory((uint8_t *)mem);
                hash_factory.set_seed_memory_offset(8 * ARGON2_BLOCK_SIZE);
                hash_factory.set_threads(device->profile_info.threads_profile_4_4_16384);
            }

            vector<string> hashes = hash_factory.generate_hashes(*profile, input.base, input.salt);
            if(hashes.size() > 0) {
                unmap_output_memory(device, thread_id, (void *) hash_factory.get_output_memory());
            }

			if (device->error != CL_SUCCESS) {
				LOG("Error running kernel: (" + to_string(device->error) + ")" + device->error_message);
			}
            for(vector<string>::iterator it = hashes.begin(); it != hashes.end(); ++it) {
                input.hash = *it;
                _store_hash(input);
            }
        }
//        printf("Total time: %lld\n", microseconds() - start_log);
    }
}

REGISTER_HASHER(gpu_hasher);