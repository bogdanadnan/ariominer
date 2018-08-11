//
// Created by Haifa Bogdan Adnan on 09/08/2018.
//

#include <stdio.h>
#include <string.h>

#include "../common/common.h"

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif // !__APPLE__

const char rot13_cl[] = "				\
__kernel void rot13					\
    (   __global    const   char*    in			\
    ,   __global            char*    out		\
    )							\
{							\
    const uint index = get_global_id(0);		\
							\
    char c=in[index];					\
    if (c<'A' || c>'z' || (c>'Z' && c<'a')) {		\
        out[index] = in[index];				\
    } else {						\
        if (c>'m' || (c>'M' && c<'a')) {		\
	    out[index] = in[index]-13;			\
	} else {					\
	    out[index] = in[index]+13;			\
	}						\
    }							\
}							\
";

void rot13 (char *buf)
{
    int index=0;
    char c=buf[index];
    while (c!=0) {
        if (c<'A' || c>'z' || (c>'Z' && c<'a')) {
            buf[index] = buf[index];
        } else {
            if (c>'m' || (c>'M' && c<'a')) {
                buf[index] = buf[index]-13;
            } else {
                buf[index] = buf[index]+13;
            }
        }
        c=buf[++index];
    }
}

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

    string device_string;
    uint max_workgroup_size;
    uint max_mem_size;
    uint max_allocable_mem_size;

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
    error = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(device_info.max_workgroup_size), &(device_info.max_workgroup_size), NULL);
    if(error != CL_SUCCESS) {
        return gpu_device_info(error, "Error querying device max workgroup size.");
    }

    cl_context_properties properties[]={
            CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
            0};

    device_info.context=clCreateContext(properties, 1, &device, NULL, NULL, &error);
    if(error != CL_SUCCESS)  {
        return gpu_device_info(error, "Error getting device context.");
    }

    cl_command_queue cq = clCreateCommandQueue(device_info.context, device, 0, &error);
    if(error != CL_SUCCESS)  {
        return gpu_device_info(error, "Error getting device command queue.");
    }

    const char *srcptr[] = { rot13_cl };
    size_t srcsize=strlen(rot13_cl);

    device_info.program = clCreateProgramWithSource(device_info.context, 1, srcptr, &srcsize, &error);
    if(error != CL_SUCCESS)  {
        return gpu_device_info(error, "Error creating opencl program for device.");
    }

    error=clBuildProgram(device_info.program, 1, &device_info.device, "", NULL, NULL);
    if(error != CL_SUCCESS)  {
        return gpu_device_info(error, "Error building opencl program for device.");
    }

    device_info.kernel = clCreateKernel(device_info.program, "rot13", &error);
    error=clBuildProgram(device_info.program, 1, &device_info.device, "", NULL, NULL);
    if(error != CL_SUCCESS)  {
        return gpu_device_info(error, "Error creating opencl kernel for device.");
    }

    return device_info;
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

int main() {
    char buf[]="Hello, World test  askdj askdj  alksjdlkjasd lkjasdlkjasd lkjas dlkajsdalskdj alskdjas dlkajsd laksjd alskdjlkajsdlkaj alskdjalskdj alskdj!";
    size_t srcsize, worksize=strlen(buf);
    worksize = 140;


    cl_int error;
    string error_message;
    vector<gpu_device_info> devices = __query_opencl_devices(error, error_message);

    rot13(buf);	// scramble using the CPU
    puts(buf);	// Just to demonstrate the plaintext is destroyed

    //char src[8192];
    //FILE *fil=fopen("rot13.cl","r");
    //srcsize=fread(src, sizeof src, 1, fil);
    //fclose(fil);

    // Allocate memory for the kernel to work with
    cl_mem mem1, mem2;
    mem1=clCreateBuffer(devices[0].context, CL_MEM_READ_ONLY, worksize, NULL, &error);
    mem2=clCreateBuffer(devices[0].context, CL_MEM_WRITE_ONLY, worksize, NULL, &error);

    clSetKernelArg(devices[0].kernel, 0, sizeof(mem1), &mem1);
    clSetKernelArg(devices[0].kernel, 1, sizeof(mem2), &mem2);

    // Target buffer just so we show we got the data from OpenCL
    char buf2[sizeof buf];
    buf2[0]='?';
    buf2[worksize]=0;

    // Send input data to OpenCL (async, don't alter the buffer!)
    error=clEnqueueWriteBuffer(devices[0].queue, mem1, CL_FALSE, 0, worksize, buf, 0, NULL, NULL);
    // Perform the operation
    error=clEnqueueNDRangeKernel(devices[0].queue, devices[0].kernel, 1, NULL, &worksize, &worksize, 0, NULL, NULL);
    // Read the result back into buf2
    error=clEnqueueReadBuffer(devices[0].queue, mem2, CL_FALSE, 0, worksize, buf2, 0, NULL, NULL);
    // Await completion of all the above
    error=clFinish(devices[0].queue);

    // Finally, output out happy message.
    puts(buf2);
}
