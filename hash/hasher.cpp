//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#include "../common/common.h"
#include "../crypt/base64.h"
#include "../crypt/random_generator.h"

#include "argon2/argon2.h"

#include "hasher.h"
#include "gpu/gpu_hasher.h"
#include "cpu/cpu_hasher.h"

hasher::hasher() {
    _intensity = 0;
    _type = "";
    _description = "";

    __public_key = "";
    __blk = "";
    __difficulty = "";
    __pause = false;
    __argon2profile = argon2profile_default;

    __begin_round_time = __hashrate_time = microseconds();
    __hashrate_hashcount = 0;
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

hash_data hasher::get_input() {
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

double hasher::get_intensity() {
    return _intensity;
}

double hasher::get_current_hash_rate() {
    uint64_t timestamp = microseconds();

    if(timestamp - __hashrate_time > 5000000) { //we calculate hashrate every 5 seconds
        __hashes_mutex.lock();
        __hashrate = __hashrate_hashcount / ((timestamp - __hashrate_time) / 1000000.0);
        __hashrate_hashcount = 0;
        __hashes_mutex.unlock();
        __hashrate_time = timestamp;
    }

    return __hashrate;
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
    if(strcmp(get_argon2profile()->profile_name, "1_1_524288") == 0) {
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
    if(strcmp(get_argon2profile()->profile_name, "4_4_16384") == 0) {
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

void hasher::_store_hash(const hash_data &hash) {
//    LOG(hash.hash);
    __hashes_mutex.lock();
    __hashes.push_back(hash);
    __hash_count++;
    __hashrate_hashcount++;
    if(hash.profile_name == "1_1_524288") {
        __total_hash_count_cblocks++;
    }
    else {
        __total_hash_count_gblocks++;
    }
    __hashes_mutex.unlock();
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

argon2profile *hasher::get_argon2profile() {
    argon2profile * profile = NULL;
    __input_mutex.lock();
    profile = __argon2profile;
    __input_mutex.unlock();

    return profile;
//    return &argon2profile_4_4_16384;
}

bool hasher::should_pause() {
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

