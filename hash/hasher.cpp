//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#include "../common/common.h"
#include "../crypt/base64.h"
#include "../crypt/random_generator.h"
#include "../app/arguments.h"

#include "../common/dllexport.h"
#include "argon2/argon2.h"
#include "hasher.h"

hasher::hasher() {
    _intensity = 0;
    _type = "";
	_subtype = "";
    _description = "";
	_priority = 0;

    __public_key = "";
    __blk = "";
    __difficulty = "";
    __pause = false;
    __is_running = false;
    __argon2profile = argon2profile_default;

    __begin_round_time = __hashrate_time = microseconds();
    __hashrate = 0;

    __total_hash_count_cblocks = 0;
    __total_hash_count_gblocks = 0;

    __hash_count = 0;

    if(__registered_hashers == NULL) {
        __registered_hashers = new vector<hasher*>();
    }
    __registered_hashers->push_back(this);
}

hasher::~hasher() {

};

string hasher::get_type() {
	return _type;
}

string hasher::get_subtype(bool short_subtype) {
    if(short_subtype && !(_short_subtype.empty())) {
        string short_version = _short_subtype;
        short_version.erase(3);
        return short_version;
    }
    else
    	return _subtype;
}

int hasher::get_priority() {
	return _priority;
}

string hasher::get_info() {
    return _description;
}

void hasher::set_input(const string &public_key, const string &blk, const string &difficulty, const string &argon2profile_string, const string &recommendation) {
    bool profile_change = false;
    __input_mutex.lock();
    __public_key = public_key;
    __blk = blk;
    __difficulty = difficulty;
    if(argon2profile_string == "4_4_16384") {
        if(strcmp(__argon2profile->profile_name, "1_1_524288") == 0) {
            __argon2profile = &argon2profile_4_4_16384;
            profile_change = true;
        }
    }
    else {
        if(strcmp(__argon2profile->profile_name, "4_4_16384") == 0) {
            __argon2profile = &argon2profile_1_1_524288;
            profile_change = true;
        }
    }
    __pause = (recommendation == "pause");
    __input_mutex.unlock();

    if(profile_change) {
        uint64_t timestamp = microseconds();
        __hashes_mutex.lock();
        __hash_timings.push_back(hash_timing{timestamp - __begin_round_time, __hash_count, (argon2profile_string == "4_4_16384" ? 0 : 1)});
        __hash_count = 0;
        __hashes_mutex.unlock();

        if (__hash_timings.size() > 20) //we average over 20 blocks
            __hash_timings.pop_front();
        __begin_round_time = timestamp;
    }
}

hash_data hasher::_get_input() {
    string tmp_public_key = "";
    string tmp_blk = "";
    string tmp_difficulty = "";
    string profile_name = "";
    __input_mutex.lock();
    tmp_public_key = __public_key;
    tmp_blk = __blk;
    tmp_difficulty = __difficulty;
    profile_name = __argon2profile->profile_name;
    __input_mutex.unlock();

    hash_data new_hash;
    new_hash.nonce = __make_nonce();
    new_hash.base = tmp_public_key + "-" + new_hash.nonce + "-" + tmp_blk + "-" + tmp_difficulty;
    new_hash.salt = "";
    new_hash.block = tmp_blk;
    new_hash.profile_name = profile_name;
//    new_hash.base = "PZ8Tyr4Nx8MHsRAGMpZmZ6TWY63dXWSCy7AEg3h9oYjeR74yj73q3gPxbxq9R3nxSSUV4KKgu1sQZu9Qj9v2q2HhT5H3LTHwW7HzAA28SjWFdzkNoovBMncD-sauULo1zM4tt9DhGEnO8qPe5nlzItJwwIKiIcAUDg-4KhqbBhShBf36zYeen943tS6KhgFmQixtUoVbf2egtBmD6j3NQtcueEBite2zjzdpK2ShaA28icRfJM9yPUQ6azN-56262626";
//    new_hash.salt = "NSHFFAg.iATJ0sfM";
    return new_hash;
}

double hasher::get_current_hash_rate() {
    double hashrate = 0;
    __hashes_mutex.lock();
    __update_hashrate();
    hashrate = __hashrate;
    __hashes_mutex.unlock();
    return hashrate;
}

double hasher::get_avg_hash_rate_cblocks() {
    size_t total_hashes = 0;
    uint64_t total_time = 0;
    for(list<hash_timing>::iterator it = __hash_timings.begin(); it != __hash_timings.end();it++) {
        if(it->profile == 0) {
            total_time += it->time_info;
            total_hashes += it->hash_count;
        }
    }
    if(strcmp(_get_argon2profile()->profile_name, "1_1_524288") == 0) {
        total_time += (microseconds() - __begin_round_time);
        __hashes_mutex.lock();
        total_hashes += __hash_count;
        __hashes_mutex.unlock();
    }
    if(total_time == 0)
        return 0;
    else
        return total_hashes / (total_time / 1000000.0);
}

double hasher::get_avg_hash_rate_gblocks() {
    size_t total_hashes = 0;
    uint64_t total_time = 0;
    for(list<hash_timing>::iterator it = __hash_timings.begin(); it != __hash_timings.end();it++) {
        if(it->profile == 1) {
            total_time += it->time_info;
            total_hashes += it->hash_count;
        }
    }
    if(strcmp(_get_argon2profile()->profile_name, "4_4_16384") == 0) {
        total_time += (microseconds() - __begin_round_time);
        __hashes_mutex.lock();
        total_hashes += __hash_count;
        __hashes_mutex.unlock();
    }

    if(total_time == 0)
        return 0;
    else
        return total_hashes / (total_time / 1000000.0);
}

uint32_t hasher::get_hash_count_cblocks() {
    return __total_hash_count_cblocks;
}

uint32_t hasher::get_hash_count_gblocks() {
    return __total_hash_count_gblocks;
}

vector<hash_data> hasher::get_hashes() {
    vector<hash_data> tmp;
    __hashes_mutex.lock();
    tmp.insert(tmp.end(), __hashes.begin(), __hashes.end());
    __hashes.clear();
    __hashes_mutex.unlock();
    return tmp;
}

void hasher::_store_hash(const hash_data &hash, int device_id) {
	__hashes_mutex.lock();
	__hashes.push_back(hash);
	__hash_count++;
    __device_infos[device_id].hashcount++;
	if (hash.profile_name == "1_1_524288") {
		__total_hash_count_cblocks++;
	}
	else {
		__total_hash_count_gblocks++;
	}

	__update_hashrate();

	__hashes_mutex.unlock();
}

void hasher::_store_hash(const vector<hash_data> &hashes, int device_id) {
	if (hashes.size() == 0) return;

	__hashes_mutex.lock();
	__hashes.insert(__hashes.end(), hashes.begin(), hashes.end());
	__hash_count+=hashes.size();
	__device_infos[device_id].hashcount += hashes.size();

	if (hashes[0].profile_name == "1_1_524288") {
		__total_hash_count_cblocks+=hashes.size();
	}
	else {
		__total_hash_count_gblocks+=hashes.size();
	}

	__update_hashrate();

//	for(int i=0;i<hashes.size();i++)
//	    LOG(hashes[i].hash);
	__hashes_mutex.unlock();
}

void hasher::__update_hashrate() {
    uint64_t timestamp = microseconds();

    if (timestamp - __hashrate_time > 5000000) { //we calculate hashrate every 5 seconds
        string profile;
        __input_mutex.lock();
        profile = __argon2profile->profile_name;
        __input_mutex.unlock();

        size_t hashcount = 0;
        for(map<int, device_info>::iterator iter = __device_infos.begin(); iter != __device_infos.end(); ++iter) {
            hashcount += iter->second.hashcount;
            if(profile == "1_1_524288")
                iter->second.cblock_hashrate = iter->second.hashcount / ((timestamp - __hashrate_time) / 1000000.0);
            else
                iter->second.gblock_hashrate = iter->second.hashcount / ((timestamp - __hashrate_time) / 1000000.0);
            iter->second.hashcount = 0;
        }
        __hashrate = hashcount / ((timestamp - __hashrate_time) / 1000000.0);
        __hashrate_time = timestamp;
    }
}

vector<hasher *> hasher::get_hashers() {
    return *__registered_hashers;
}

vector<hasher *> hasher::get_active_hashers() {
    vector<hasher *> filtered;
    for(vector<hasher*>::iterator it = __registered_hashers->begin();it != __registered_hashers->end();++it) {
        if((*it)->_intensity != 0)
            filtered.push_back(*it);
    }
    return filtered;
}

argon2profile *hasher::_get_argon2profile() {
    argon2profile * profile = NULL;
    __input_mutex.lock();
    profile = __argon2profile;
    __input_mutex.unlock();

    return profile;
}

bool hasher::_should_pause() {
    bool pause = false;
    __input_mutex.lock();
    pause = __pause;
    __input_mutex.unlock();

    return pause;
}

string hasher::__make_nonce() {
    char input[32];
    char output[50];

    random_generator::instance().get_random_data(input, 32);

    base64::encode(input, 32, output);
    return regex_replace (string(output), regex("[^a-zA-Z0-9]"), "");
}

vector<hasher*> *hasher::__registered_hashers = NULL;

typedef void *(*hasher_loader)();

void hasher::load_hashers() {
	string module_path = arguments::get_app_folder() + "/modules/";
	vector<string> files = get_files(module_path);
	for(vector<string>::iterator iter = files.begin();iter != files.end();iter++) {
		if(iter->find(".hsh") != string::npos) {
			void *__dll_handle = dlopen((module_path + *iter).c_str(), RTLD_LAZY);
			if(__dll_handle != NULL) {
				hasher_loader hasher_loader_ptr = (hasher_loader) dlsym(__dll_handle, "hasher_loader");
				(*hasher_loader_ptr)();
			}
		}
	}
}

bool hasher::is_running() {
    return __is_running;
}

void hasher::_update_running_status(bool running) {
    __is_running = running;
}

vector<string> hasher::_get_gpu_filters(arguments &args) {
    vector<string> local_filters = args.gpu_filter();
    vector<hasher*> gpu_hashers = get_hashers_of_type("GPU");
    for(vector<string>::iterator it = local_filters.end(); it-- != local_filters.begin();) {
        string filter = *it;
        string filter_type = "";
        for(vector<hasher*>::iterator hit = gpu_hashers.begin(); hit != gpu_hashers.end(); hit++) {
            if(filter.find((*hit)->_subtype + ":") == 0) {
                filter_type = (*hit)->_subtype;
                break;
            }
        }
        if(filter_type != "" && filter_type != this->_subtype) {
            local_filters.erase(it);
        }
        else if(filter_type != "") { //cleanup subtype prefix
            it->erase(0, this->_subtype.size() + 1);
        }
    }
    return local_filters;
}

vector<hasher *> hasher::get_hashers_of_type(const string &type) {
    vector<hasher *> filtered;
    for(vector<hasher*>::iterator it = __registered_hashers->begin();it != __registered_hashers->end();++it) {
        if((*it)->_type == type)
            filtered.push_back(*it);
    }
    return filtered;
}

map<int, device_info> &hasher::get_device_infos() {
//    map<int, device_info> device_infos_copy;
//    __hashes_mutex.lock();
//    device_infos_copy.insert(__device_infos.begin(), __device_infos.end());
//    __hashes_mutex.unlock();
    return __device_infos;
}

void hasher::_store_device_info(int device_id, device_info device) {
    __hashes_mutex.lock();
    __device_infos[device_id] = device;
    __hashes_mutex.unlock();
}

