//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#include "../../../common/common.h"
#include "../../../app/arguments.h"

#include "../../hasher.h"
#include "../../argon2/argon2.h"

#if defined(WITH_CUDA)

#include <cuda_runtime.h>
#include <driver_types.h>

#include "cuda_hasher.h"

cuda_hasher::cuda_hasher() {
	_type = "GPU";
	_subtype = "CUDA";
	_priority = 2;
	_intensity = 0;
	__running = false;
	_description = "";
}


cuda_hasher::~cuda_hasher() {
	this->cleanup();
}

bool cuda_hasher::initialize() {
	cudaError_t error = cudaSuccess;
	string error_message;

	__devices = __query_cuda_devices(error, error_message);

	if(error != cudaSuccess) {
		_description = "No compatible GPU detected: " + error_message;
		return false;
	}

	if (__devices.empty()) {
		_description = "No compatible GPU detected.";
		return false;
	}

	return true;
}

bool cuda_hasher::configure(arguments &args) {
	int index = 1;
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

	for(vector<cuda_device_info *>::iterator d = __devices.begin(); d != __devices.end(); d++, index++) {
		stringstream ss;
		ss << "["<< index << "] " << (*d)->device_string << endl;
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
				(*d)->threads_profile_4_4_16384 = 0;
				(*d)->threads_profile_1_1_524288 = 0;
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

		int host_threads = 0;
		if(args.gpu_threads().size() == 1 || (*d)->device_index >= args.gpu_threads().size())
			host_threads = args.gpu_threads()[0];
		else
			host_threads = args.gpu_threads()[(*d)->device_index];

		_description += ss.str();

		if(!(__setup_device_info((*d), device_intensity_cpu, device_intensity_gpu, host_threads))) {
			_description += (*d)->error_message;
			_description += "\n";
			continue;
		};
		total_threads_profile_4_4_16384 += (*d)->threads_profile_4_4_16384;
		total_threads_profile_1_1_524288 += (*d)->threads_profile_1_1_524288;
	}

	if (total_threads_profile_4_4_16384 == 0 && total_threads_profile_1_1_524288 == 0) {
		_intensity = 0;
		_description += "Status: DISABLED - not enough resources.";
		return false;
	}

	_intensity = (intensity_cpu + intensity_gpu) / 2.0;

	__running = true;
	_update_running_status(__running);
	for(vector<cuda_device_info *>::iterator d = __devices.begin(); d != __devices.end(); d++) {
		if((*d)->threads_profile_1_1_524288 != 0 || (*d)->threads_profile_4_4_16384 != 0) {
			for(int i=0;i<(*d)->device_threads;i++) {
				__runners.push_back(new thread([&](cuda_device_info *device, int index) {
					this->__run(device, index);
				}, (*d), i));
			}
		}
	}

	_description += "Status: ENABLED - with " + to_string(total_threads_profile_1_1_524288) + " threads for CPU blocks and " + to_string(total_threads_profile_4_4_16384) + " threads for GPU blocks.";

	return true;
}

void cuda_hasher::cleanup() {
	__running = false;
	for(vector<thread*>::iterator it = __runners.begin();it != __runners.end();++it) {
		(*it)->join();
		delete *it;
	}
	__runners.clear();

	for(vector<cuda_device_info *>::iterator d = __devices.begin(); d != __devices.end(); d++) {
		cuda_free(*d);
	}
}

cuda_device_info *cuda_hasher::__get_device_info(int device_index) {
	cuda_device_info *device_info = new cuda_device_info();
	device_info->error = cudaSuccess;
	device_info->device_index = device_index;

	device_info->error = cudaSetDevice(device_index);
	if(device_info->error != cudaSuccess) {
		device_info->error_message = "Error setting current device.";
		return device_info;
	}

    cudaDeviceProp devProp;
	device_info->error = cudaGetDeviceProperties(&devProp, device_index);
	if(device_info->error != cudaSuccess) {
		device_info->error_message = "Error setting current device.";
		return device_info;
	}

    device_info->device_string = devProp.name;

    size_t freemem, totalmem;
    device_info->error = cudaMemGetInfo(&freemem, &totalmem);
	if(device_info->error != cudaSuccess) {
		device_info->error_message = "Error setting current device.";
		return device_info;
	}

    device_info->max_mem_size = totalmem;
    device_info->free_mem_size = freemem;

    return device_info;
}

bool cuda_hasher::__setup_device_info(cuda_device_info *device, double intensity_cpu, double intensity_gpu, int threads) {
	double cblocks_mem_size = device->max_mem_size * intensity_cpu / 100.0;
	double gblocks_mem_size = device->max_mem_size * intensity_gpu / 100.0;

	device->threads_profile_1_1_524288 = floor(cblocks_mem_size / argon2profile_1_1_524288.memsize);
	device->threads_profile_4_4_16384 = floor(gblocks_mem_size / argon2profile_4_4_16384.memsize);

	cblocks_mem_size = device->threads_profile_1_1_524288 * argon2profile_1_1_524288.memsize;
	gblocks_mem_size = device->threads_profile_4_4_16384 * argon2profile_4_4_16384.memsize;

	device->device_threads = threads;
	if(device->device_threads < 1)
		device->device_threads = 1;
	device->arguments.set_threads(device->device_threads);
	device->threads_per_stream = new cuda_threads_per_stream[device->device_threads];

	int threads_per_stream_1_1_524288 = device->threads_profile_1_1_524288 / device->device_threads;
	int threads_per_stream_4_4_16384 = device->threads_profile_4_4_16384 / device->device_threads;
	int threads_left_1_1_524288 = device->threads_profile_1_1_524288;
	int threads_left_4_4_16384 = device->threads_profile_4_4_16384;

	for(int i=0;i<device->device_threads;i++) {
		device->threads_per_stream[i].threads_profile_1_1_524288 = (i <= device->device_threads - 1) ? threads_per_stream_1_1_524288 : threads_left_1_1_524288;
		device->threads_per_stream[i].threads_profile_4_4_16384 = (i <= device->device_threads - 1) ? threads_per_stream_4_4_16384 : threads_left_4_4_16384;

		threads_left_1_1_524288 -= threads_per_stream_1_1_524288;
		threads_left_4_4_16384 -= threads_per_stream_4_4_16384;

		device->threads_per_stream[i].memory_size = max(device->threads_per_stream[i].threads_profile_1_1_524288 * argon2profile_1_1_524288.memsize,
														device->threads_per_stream[i].threads_profile_4_4_16384 * argon2profile_4_4_16384.memsize);
	}

	cuda_allocate(device);

	if(device->error != cudaSuccess)
		return false;

    return true;
}

vector<cuda_device_info *> cuda_hasher::__query_cuda_devices(cudaError_t &error, string &error_message) {
	vector<cuda_device_info *> devices;
	int devCount = 0;
	error = cudaGetDeviceCount(&devCount);

	if(error != cudaSuccess) {
		error_message = "Error querying CUDA device count.";
		return devices;
	}

	if(devCount == 0)
		return devices;

	for (int i = 0; i < devCount; ++i)
	{
		cuda_device_info *dev = __get_device_info(i);
		if(dev == NULL)
			continue;
		if(dev->error != cudaSuccess) {
			error = dev->error;
			error_message = dev->error_message;
			continue;
		}
		devices.push_back(dev);
	}
	return devices;
}

void cuda_hasher::__run(cuda_device_info *device, int thread_id) {
	cudaSetDevice(device->device_index);

	cudaStream_t current_stream;
	cudaStreamCreate(&current_stream);

	cuda_gpumgmt_thread_data thread_data;
	thread_data.device = device;
	thread_data.thread_id = thread_id;
	thread_data.cuda_info = &current_stream;

	void *memory = device->arguments.host_seed_memory[thread_id];
	argon2 hash_factory(cuda_kernel_filler, memory, &thread_data);
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
				if(device->threads_per_stream[thread_id].threads_profile_1_1_524288 == 0) {
					this_thread::sleep_for(chrono::milliseconds(100));
					continue;
				}
				hash_factory.set_seed_memory_offset(2 * ARGON2_BLOCK_SIZE);
				hash_factory.set_threads(device->threads_per_stream[thread_id].threads_profile_1_1_524288);
			}
			else {
				if(device->threads_per_stream[thread_id].threads_profile_4_4_16384 == 0) {
					this_thread::sleep_for(chrono::milliseconds(100));
					continue;
				}
				hash_factory.set_seed_memory_offset(8 * ARGON2_BLOCK_SIZE);
				hash_factory.set_threads(device->threads_per_stream[thread_id].threads_profile_4_4_16384);
			}

			vector<string> hashes = hash_factory.generate_hashes(*profile, input.base, input.salt);

			if (device->error != cudaSuccess) {
				LOG("Error running kernel: (" + to_string(device->error) + ")" + device->error_message);
				__running = false;
				continue;
			}
			vector<hash_data> stored_hashes;
			for(vector<string>::iterator it = hashes.begin(); it != hashes.end(); ++it) {
				input.hash = *it;
				stored_hashes.push_back(input);
			}
			_store_hash(stored_hashes);
		}
	}

	_update_running_status(__running);
}

REGISTER_HASHER(cuda_hasher);

#endif //WITH_CUDA