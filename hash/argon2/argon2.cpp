//
// Created by Haifa Bogdan Adnan on 05/08/2018.
//

#include "../../http/mongoose/mongoose.h"
#include "../../common/common.h"
#include "blake2/blake2.h"

#include "argon2.h"
#include "defs.h"

argon2::argon2(argon2_blocks_filler_ptr filler, void *seed_memory, void *user_data) {
    __filler = filler;
    __threads = 1;
    __seed_memory = (uint8_t*)seed_memory;
    __seed_memory_offset = argon2profile_default->memsize;
    __lane_length = -1;
    __user_data = user_data;
}

vector<string> argon2::generate_hashes(const argon2profile &profile, const string &base, const string &salt_) {
    vector<string> result;
    vector<string> salts;

    uint8_t blockhash[ARGON2_PREHASH_SEED_LENGTH];
    uint8_t raw_hash[ARGON2_RAW_LENGTH];

    for(int i=0;i<__threads;i++) {
        string salt;

        if(salt_.empty()) {
            salt = __make_salt();
        }
        else {
            salt = salt_;
        }
        salts.push_back(salt);

        __initial_hash(profile, blockhash, base, salt);

        memset(blockhash + ARGON2_PREHASH_DIGEST_LENGTH, 0,
               ARGON2_PREHASH_SEED_LENGTH -
               ARGON2_PREHASH_DIGEST_LENGTH);

        __fill_first_blocks(profile, blockhash, i);
    }

    (*__filler) (__seed_memory, __threads, (argon2profile*)&profile, __user_data);

    for(int i=0;i<__threads;i++) {
        blake2b_long((void *) raw_hash, ARGON2_RAW_LENGTH,
                     (void *) (__seed_memory + i * __seed_memory_offset), ARGON2_BLOCK_SIZE);

        string hash = __encode_string(profile, salts[i], raw_hash);
        result.push_back(hash);
    }
    return result;
}

string argon2::__make_salt() {
    unsigned char input[13];
    char output[25];

    for(int i=0;i<13;i++) {
        double rnd_scaler = rand()/(1.0 + RAND_MAX);
        input[i] = (unsigned char)(rnd_scaler * 256);
    }

    mg_base64_encode(input, 13, output);

    for (int i = 0; i < 16; i++) {
        if (output[i] == '+') {
            output[i] = '.';
        }
    }

    output[16] = 0;
    return string(output);
}

void argon2::__initial_hash(const argon2profile &profile, uint8_t *blockhash, const string &base, const string &salt) {
    blake2b_state BlakeHash;
    uint32_t value;

    blake2b_init(&BlakeHash, ARGON2_PREHASH_DIGEST_LENGTH);

    value = profile.thr_cost;
    blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));

    value = ARGON2_RAW_LENGTH;
    blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));

    value = profile.mem_cost;
    blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));

    value = profile.tm_cost;
    blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));

    value = ARGON2_VERSION;
    blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));

    value = ARGON2_TYPE_VALUE;
    blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));

    value = base.length();
    blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));

    blake2b_update(&BlakeHash, (const uint8_t *)base.c_str(),
                   base.length());

    value = salt.length();
    blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));

    blake2b_update(&BlakeHash, (const uint8_t *)salt.c_str(),
                   salt.length());

    value = 0;
    blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));
    blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));

    blake2b_final(&BlakeHash, blockhash, ARGON2_PREHASH_DIGEST_LENGTH);
}

void argon2::__fill_first_blocks(const argon2profile &profile, uint8_t *blockhash, int thread) {
    block *blocks = (block *)(__seed_memory + thread * __seed_memory_offset);

    size_t lane_length;
    if(__lane_length == -1) {
        lane_length = profile.mem_cost / profile.thr_cost;
    }
    else {
        lane_length = __lane_length;
    }

    for (uint32_t l = 0; l < profile.thr_cost; ++l) {
        *((uint32_t*)(blockhash + ARGON2_PREHASH_DIGEST_LENGTH)) = 0;
        *((uint32_t*)(blockhash + ARGON2_PREHASH_DIGEST_LENGTH + 4)) = l;

        blake2b_long((void *)(blocks + l * lane_length), ARGON2_BLOCK_SIZE, blockhash,
                     ARGON2_PREHASH_SEED_LENGTH);

        *((uint32_t*)(blockhash + ARGON2_PREHASH_DIGEST_LENGTH)) = 1;

        blake2b_long((void *)(blocks + l * lane_length + 1), ARGON2_BLOCK_SIZE, blockhash,
                     ARGON2_PREHASH_SEED_LENGTH);
    }
}

string argon2::__encode_string(const argon2profile &profile, const string &salt, uint8_t *hash) {
    char salt_b64[50];
    char hash_b64[50];

    mg_base64_encode((unsigned char *)salt.c_str(), salt.length(), salt_b64);
    mg_base64_encode(hash, ARGON2_RAW_LENGTH, hash_b64);

    salt_b64[22] = 0;
    hash_b64[43] = 0;

    stringstream ss;
    ss << "$argon2i$v=19$m=" << profile.mem_cost << ",t=" << profile.tm_cost << ",p=" << profile.thr_cost << "$" << salt_b64 << "$" << hash_b64;
    return ss.str();
}

void argon2::set_seed_memory(uint8_t *memory) {
    __seed_memory = memory;
}

void argon2::set_seed_memory_offset(size_t offset) {
    __seed_memory_offset = offset;
}

void argon2::set_threads(int threads) {
    __threads = threads;
}

void argon2::set_lane_length(int length) {
    if(length > 0)
        __lane_length = length;
}

