//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#include "../http/mongoose/mongoose.h"

#include "../common/common.h"
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

    __hash_rate = 0;
    __avg_hash_rate = 0;
    __hash_count = 0;
    __total_hash_count = 0;

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

void hasher::set_input(const string &public_key, const string &blk, const string &difficulty, const string &argon2profile_string, const string &recommendation) {
    __input_mutex.lock();
    __public_key = public_key;
    __blk = blk;
    __difficulty = difficulty;
    if(argon2profile_string == "4_4_16384") {
        __argon2profile = &argon2profile_4_4_16384;
    }
    else {
        __argon2profile = &argon2profile_1_1_524288;
    }
    __pause = (recommendation == "pause");
    __input_mutex.unlock();
}

hash_data hasher::get_input() {
    string tmp_public_key = "";
    string tmp_blk = "";
    string tmp_difficulty = "";
    __input_mutex.lock();
    tmp_public_key = __public_key;
    tmp_blk = __blk;
    tmp_difficulty = __difficulty;
    __input_mutex.unlock();

    hash_data new_hash;
    new_hash.nonce = __make_nonce();
    new_hash.base = tmp_public_key + "-" + new_hash.nonce + "-" + tmp_blk + "-" + tmp_difficulty;
    new_hash.salt = "";
//    new_hash.base = "PZ8Tyr4Nx8MHsRAGMpZmZ6TWY63dXWSCy7AEg3h9oYjeR74yj73q3gPxbxq9R3nxSSUV4KKgu1sQZu9Qj9v2q2HhT5H3LTHwW7HzAA28SjWFdzkNoovBMncD-sauULo1zM4tt9DhGEnO8qPe5nlzItJwwIKiIcAUDg-4KhqbBhShBf36zYeen943tS6KhgFmQixtUoVbf2egtBmD6j3NQtcueEBite2zjzdpK2ShaA28icRfJM9yPUQ6azN-56262626";
//    new_hash.salt = "NSHFFAg.iATJ0sfM";
    return new_hash;
}

int hasher::get_intensity() {
    return _intensity;
}

double hasher::get_current_hash_rate() {
    __hashes_mutex.lock();
    uint64_t timestamp = microseconds();
    if(timestamp - __hashrate_time > 5000000) { //we calculate hashrate every 5 seconds
        __hash_rate = __hash_count / ((timestamp - __hashrate_time) / 1000000.0);
        __avg_hash_rate = (__total_hash_count) / ((timestamp - __begin_time) / 1000000.0);
        __hashrate_time = timestamp;
        __hash_count = 0;
    }
    __hashes_mutex.unlock();

    return __hash_rate;
}

double hasher::get_avg_hash_rate() {
    return __avg_hash_rate;
}

uint32_t hasher::get_hash_count() {
    return __total_hash_count;
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
    __total_hash_count++;
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
    return __argon2profile;
//    return &argon2profile_1_1_524288;
}

bool hasher::should_pause() {
    return __pause;
}

string hasher::__make_nonce() {
    unsigned char input[32];
    char output[50];

    for(int i=0;i<32;i++) {
        double rnd_scaler = rand()/(1.0 + RAND_MAX);
        input[i] = (unsigned char)(rnd_scaler * 256);
    }

    mg_base64_encode(input, 32, output);
    return regex_replace (string(output), regex("[^a-zA-Z0-9]"), "");
}

vector<hasher*> *hasher::__registered_hashers = NULL;

