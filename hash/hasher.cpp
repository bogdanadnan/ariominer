//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#include "../common/common.h"

#include "argon2/argon2.h"

#include "hasher.h"
#include "gpu/gpu_hasher.h"
#include "cpu/cpu_hasher.h"

hasher::hasher() {
    _intensity = 0;
    _type = "";
    _description = "";

    __nonce = "";
    __base = "";

    __hash_rate = 0;
    __avg_hash_rate = 0;
    __hash_count = 0;

    __begin_time = __hashrate_time = microseconds();

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

string hasher::get_info() {
    return _description;
}

void hasher::set_input(const string &nonce, const string &base) {
    __input_mutex.lock();
    __nonce = nonce;
    __base = base;
    __input_mutex.unlock();
}

string hasher::get_base() {
    string tmp = "";
    __input_mutex.lock();
    tmp = __base;
    __input_mutex.unlock();
    return tmp;
}

int hasher::get_intensity() {
    return _intensity;
}

double hasher::get_current_hash_rate() {
    return __hash_rate;
}

double hasher::get_avg_hash_rate() {
    return __avg_hash_rate;
}

uint hasher::get_hash_count() {
    return __hash_count;
}

vector<hash_data> hasher::get_hashes() {
    vector<hash_data> tmp;
    __hashes_mutex.lock();
    tmp.insert(tmp.end(), __hashes.begin(), __hashes.end());
    __hashes.clear();
    __hashes_mutex.unlock();
    return tmp;
}

void hasher::_store_hash(const string &hash) {
    hash_data h;
    __input_mutex.lock();
    h.nonce = __nonce;
    h.base = __base;
    __input_mutex.unlock();

    h.hash = hash;

    __hashes_mutex.lock();
    __hashes.push_back(h);
    __hashes_mutex.unlock();

    __hash_count++;
    if(__hash_count % 50 == 0) { //we count hashrate for 50 hashes to get an accurate measurement
        uint64_t timestamp = microseconds();
        __hash_rate = 50 / ((timestamp - __hashrate_time) / 1000000.0);
        __hashrate_time = timestamp;
        __avg_hash_rate = (__hash_count) / ((timestamp - __begin_time) / 1000000.0);
    }
}

vector<hasher *> hasher::get_hashers() {
    return *__registered_hashers;
}

vector<hasher *> hasher::get_active_hashers() {
    vector<hasher *> filtered;
    for(vector<hasher*>::iterator it = __registered_hashers->begin();it != __registered_hashers->end();++it) {
        if((*it)->get_intensity() != 0)
            filtered.push_back(*it);
    }
    return filtered;
}

vector<hasher*> *hasher::__registered_hashers = NULL;

