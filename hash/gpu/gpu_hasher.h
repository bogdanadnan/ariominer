//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#ifndef PROJECT_GPU_HASHER_H
#define PROJECT_GPU_HASHER_H

#include "OpenCL_Cpp/cl.hpp"

struct opencl_device {
    cl::Platform platform;
    cl::Device device;

    int available_processing_thr;
    int available_memory_thr;
    int threads_count;
};

class gpu_hasher : public hasher {
public:
    gpu_hasher();
    ~gpu_hasher();

    virtual bool configure(int intensity);

private:
    string __detect_features_and_make_description();

    void __run(opencl_device *device);

    vector<opencl_device> __devices;

    bool __running;
    vector<thread*> __runners;
};

#endif //PROJECT_GPU_HASHER_H
