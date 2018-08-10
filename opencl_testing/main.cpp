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
    string device_string;
    uint max_workgroup_size;
    uint max_mem_size;
    uint max_allocable_mem_size;
};

gpu_device_info __get_device_info(cl_device_id device) {
    gpu_device_info device_info;
    cl_int error;
    char buffer[100];
    size_t sz;

    // device name
    string device_vendor;
    error = clGetDeviceInfo(device, CL_DEVICE_VENDOR, 100, buffer, &sz);
    buffer[sz] = 0;
    device_vendor = buffer;

    string device_name;
    error = clGetDeviceInfo(device, CL_DEVICE_NAME, 100, buffer, &sz);
    buffer[sz] = 0;
    device_name = buffer;

    string device_version;
    error = clGetDeviceInfo(device, CL_DEVICE_VERSION, 100, buffer, &sz);
    buffer[sz] = 0;
    device_version = buffer;

    device_info.device_string = device_vendor + " - " + device_name + " : " + device_version;

    error = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(device_info.max_mem_size), &(device_info.max_mem_size), NULL);
    error = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(device_info.max_allocable_mem_size), &(device_info.max_allocable_mem_size), NULL);
    error = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(device_info.max_workgroup_size), &(device_info.max_workgroup_size), NULL);
    return device_info;
}

int main() {
    char buf[]="Hello, World test  askdj askdj  alksjdlkjasd lkjasdlkjasd lkjas dlkajsdalskdj alskdjas dlkajsd laksjd alskdjlkajsdlkaj alskdjalskdj alskdj!";
    size_t srcsize, worksize=strlen(buf);
    worksize = 140;


    cl_int error;
    cl_platform_id platform;
    cl_device_id device;
    cl_uint platforms, devices;

    // Fetch the Platform and Device IDs; we only want one.
    error=clGetPlatformIDs(1, &platform, &platforms);
    error=clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &devices);
    gpu_device_info info = __get_device_info(device);




    cl_context_properties properties[]={
            CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
            0};
    // Note that nVidia's OpenCL requires the platform property
    cl_context context=clCreateContext(properties, 1, &device, NULL, NULL, &error)
    ;
    cl_command_queue cq = clCreateCommandQueue(context, device, 0, &error);

    rot13(buf);	// scramble using the CPU
    puts(buf);	// Just to demonstrate the plaintext is destroyed

    //char src[8192];
    //FILE *fil=fopen("rot13.cl","r");
    //srcsize=fread(src, sizeof src, 1, fil);
    //fclose(fil);

    const char *src=rot13_cl;
    srcsize=strlen(rot13_cl);

    const char *srcptr[]={src};
    // Submit the source code of the rot13 kernel to OpenCL
    cl_program prog=clCreateProgramWithSource(context,
                                              1, srcptr, &srcsize, &error);
    // and compile it (after this we could extract the compiled version)
    error=clBuildProgram(prog, 0, NULL, "", NULL, NULL);

    // Allocate memory for the kernel to work with
    cl_mem mem1, mem2;
    mem1=clCreateBuffer(context, CL_MEM_READ_ONLY, worksize, NULL, &error);
    mem2=clCreateBuffer(context, CL_MEM_WRITE_ONLY, worksize, NULL, &error);

    // get a handle and map parameters for the kernel
    cl_kernel k_rot13=clCreateKernel(prog, "rot13", &error);
    clSetKernelArg(k_rot13, 0, sizeof(mem1), &mem1);
    clSetKernelArg(k_rot13, 1, sizeof(mem2), &mem2);

    size_t val = 0;
    clGetKernelWorkGroupInfo(k_rot13, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(val), &val, NULL);

    // Target buffer just so we show we got the data from OpenCL
    char buf2[sizeof buf];
    buf2[0]='?';
    buf2[worksize]=0;

    // Send input data to OpenCL (async, don't alter the buffer!)
    error=clEnqueueWriteBuffer(cq, mem1, CL_FALSE, 0, worksize, buf, 0, NULL, NULL);
    // Perform the operation
    error=clEnqueueNDRangeKernel(cq, k_rot13, 1, NULL, &worksize, &worksize, 0, NULL, NULL);
    // Read the result back into buf2
    error=clEnqueueReadBuffer(cq, mem2, CL_FALSE, 0, worksize, buf2, 0, NULL, NULL);
    // Await completion of all the above
    error=clFinish(cq);

    // Finally, output out happy message.
    puts(buf2);
}