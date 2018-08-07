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

struct kernel_filler_params {
    cl::Buffer addresses;
    cl::Buffer gpu_memory;
    cl::Buffer seed_memory;
    cl::Buffer gpu_output;
    cl::Device device;
    cl::Kernel kernel;
    cl::CommandQueue queue;
    cl::Context context;
    uint8_t *output;
    int threads;
};

void kernel_filler(void *memory, int threads, void *user_data) {
    kernel_filler_params *params = (kernel_filler_params *)user_data;

    auto result = params->queue.enqueueWriteBuffer(params->seed_memory, CL_TRUE, 0, 2 * ARGON2_BLOCK_SIZE * params->threads, memory);
    result = params->queue.finish();

    result = params->kernel.setArg(2, params->seed_memory);
    result = params->queue.enqueueNDRangeKernel(params->kernel, cl::NDRange(1), cl::NDRange(KERNEL_WORKGROUP_SIZE * params->threads), cl::NDRange(KERNEL_WORKGROUP_SIZE));
    result = params->queue.finish();

    result = params->queue.enqueueReadBuffer(params->gpu_output, CL_TRUE, 0, ARGON2_BLOCK_SIZE * params->threads, params->output);
    result = params->queue.finish();

    for(int i=0;i<params->threads;i++) {
        memcpy(((uint8_t*)memory) + i * 2 * ARGON2_BLOCK_SIZE, params->output + i * ARGON2_BLOCK_SIZE, ARGON2_BLOCK_SIZE);
    }
}

gpu_hasher::gpu_hasher() {
    _type = "GPU";
    _intensity = 0;
    __running = false;
    _description = __detect_features_and_make_description();

}

gpu_hasher::~gpu_hasher() {
    __running = false;
    for(vector<thread*>::iterator it = __runners.begin();it != __runners.end();++it) {
        (*it)->join();
        delete *it;
    }
}

bool gpu_hasher::configure(int intensity) {
    int total_threads = 0;

    for(auto d = __devices.begin(); d != __devices.end(); d++) {
        d->threads_count = min(d->available_processing_thr, d->available_memory_thr);

        if (d->threads_count == 0) // not enough resources
            continue;

        d->threads_count = d->threads_count * intensity / 100;
        if (d->threads_count == 0)
            d->threads_count = 1;

        total_threads += d->threads_count;
    }

    if (total_threads == 0) {
        _intensity = 0;
        _description += "Status: DISABLED - not enough resources.";
        return false;
    }

    if (intensity == 0) {
        _intensity = 0;
        _description += "Status: DISABLED - by user.";
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

string gpu_hasher::__detect_features_and_make_description() {
    stringstream ss;
    vector<cl::Platform> platform;

    cl::Platform::get(&platform);

    if (platform.empty()) {
        return string("No OpenCL platform detected.");
    }

    for(auto p = platform.begin(); p != platform.end(); p++) {
        std::vector<cl::Device> pldev;

        p->getDevices(CL_DEVICE_TYPE_GPU, &pldev);

        for(auto d = pldev.begin(); d != pldev.end(); d++) {
            if (!d->getInfo<CL_DEVICE_AVAILABLE>())
                continue;

            opencl_device new_device;
            new_device.platform = *p;
            new_device.device = *d;

            __devices.push_back(new_device);
        }
    }

    if (__devices.empty()) {
        return string("No compatible GPU detected.");
    }

    int index = 1;
    for(auto d = __devices.begin(); d != __devices.end(); d++, index++) {
        auto device_vendor = d->device.getInfo<CL_DEVICE_VENDOR>();
        string device_version = d->device.getInfo<CL_DEVICE_VERSION>();
        string device_name = d->device.getInfo<CL_DEVICE_NAME>();

        ss << "["<< index << "] " << device_vendor << " - " << device_name << " : " << device_version << endl;

        d->available_processing_thr = d->device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() / KERNEL_WORKGROUP_SIZE;
        d->available_memory_thr = d->device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / ARGON2_MEMORY_SIZE;

        ss << "Parallelism: " << d->available_processing_thr << " concurent threads supported." << endl;
        ss << "Memory: there is enough memory for " << d->available_memory_thr << " concurent threads." << endl;
    }

    return ss.str();
}

void gpu_hasher::__run(opencl_device *device) {
    void *memory = malloc(2 * ARGON2_BLOCK_SIZE * device->threads_count + 256);
    size_t allocated_size = 2 * ARGON2_BLOCK_SIZE * device->threads_count + 256;
    uint8_t* aligned = (uint8_t*)align(32, 2 * ARGON2_BLOCK_SIZE, memory, allocated_size);

    kernel_filler_params params;

    params.device = device->device;
    params.context = cl::Context(device->device);
    params.queue = cl::CommandQueue(params.context, device->device);

    cl::Program program(params.context, cl::Program::Sources(
            1, make_pair(opencl_kernel.c_str(), opencl_kernel.length())
    ));

    vector<cl::Device> dvs;
    dvs.push_back(device->device);
    string thr_def = "-DTHREADS=" + to_string(device->threads_count);
    if (program.build(dvs, thr_def.c_str()) != CL_SUCCESS) {
        stringstream ss;
        ss << "OpenCL compilation error" << endl
           << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device->device)
           << endl;
        LOG(ss.str());
        return;
    }

    params.kernel = cl::Kernel(program, "fill_blocks");

    params.addresses = cl::Buffer(params.context, CL_MEM_READ_ONLY, BLOCKS_ADDRESSES_SIZE * sizeof(int32_t));
    params.queue.enqueueWriteBuffer(params.addresses, CL_TRUE, 0, BLOCKS_ADDRESSES_SIZE * sizeof(int32_t),
                                    blocks_addresses);

    argon2 *hash_factory = NULL;

    while(true) {
        params.gpu_memory = cl::Buffer(params.context, CL_MEM_READ_WRITE, ARGON2_MEMORY_SIZE * device->threads_count);
        params.seed_memory = cl::Buffer(params.context, CL_MEM_READ_ONLY,
                                        (2 * ARGON2_BLOCK_SIZE * device->threads_count));
        params.gpu_output = cl::Buffer(params.context, CL_MEM_WRITE_ONLY, ARGON2_BLOCK_SIZE * device->threads_count);

        params.threads = device->threads_count;
        params.output = (uint8_t *) malloc(ARGON2_BLOCK_SIZE * device->threads_count);

        hash_factory = new argon2(kernel_filler, device->threads_count, aligned, ARGON2_BLOCK_SIZE * 2, &params);

        params.kernel.setArg(0, params.addresses);
        params.kernel.setArg(1, params.gpu_memory);
        params.kernel.setArg(3, params.gpu_output);

        string test_base = "PZ8Tyr4Nx8MHsRAGMpZmZ6TWY63dXWSCy7AEg3h9oYjeR74yj73q3gPxbxq9R3nxSSUV4KKgu1sQZu9Qj9v2q2HhT5H3LTHwW7HzAA28SjWFdzkNoovBMncD-IwByKLm3TUstklR5Y9Kk0iiWriHl28eB650C6LKg4w0-kdtd8pVPieNigRVj3ELd8zsR3FtrT9HCZbBtjZXkCt6BP7MUmTRgc3GPVeQEmiJW6shAH5wvwzZhLRZAkEKjwsg-14146465";
        string test_salt = "GTmEqKhy1LYzs9Z.";
        vector<string> test_hashes = hash_factory->generate_hashes(test_base, test_salt);

        bool test_pass = true;
        for (auto it = test_hashes.begin(); it != test_hashes.end(); it++) {
            if (*it !=
                "$argon2i$v=19$m=524288,t=1,p=1$R1RtRXFLaHkxTFl6czlaLg$LVsqAUbYUxsWJu1BfgV6/qsi6JEQJNFM0Qr5h0yPJ+8") {
                test_pass = false;
            }
        }
        if (!test_pass) {
            device->threads_count --;
            delete params.output;
            delete hash_factory;

            if(device->threads_count == 0) {
                LOG("GPU hashing test failed, disabling hashing for it.");

                if(__devices.size() == 1) {
                    _intensity = 0;
                }

                free(memory);
                return;
            }
            else {
                LOG("GPU hashing test failed with " + to_string(device->threads_count + 1) + " threads, trying with " + to_string(device->threads_count) + " threads.");
            }
        } else
            break;
    }

    while(__running) {
        string base = get_base();
        if (!base.empty()) {
            vector<string> hashes = hash_factory->generate_hashes(base);
            for(auto it = hashes.begin(); it != hashes.end(); ++it) {
                _store_hash(*it);
            }
        }
    }

    delete params.output;
    delete hash_factory;

    free(memory);
}

REGISTER_HASHER(gpu_hasher);