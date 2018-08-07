//
// Created by Haifa Bogdan Adnan on 05/08/2018.
//

#ifndef ARIOMINER_ARGON2_H
#define ARIOMINER_ARGON2_H

#include "defs.h"

typedef void (*argon2_blocks_filler_ptr)(void *, int, void *);

class argon2 {
public:
    argon2(argon2_blocks_filler_ptr filler, int threads, void *seed_memory, size_t seed_memory_offset, void *user_data);

    vector<string> generate_hashes(const string &base, const string &salt_ = "");

private:
    string __make_salt();
    void __initial_hash(uint8_t *blockhash, const string &base, const string &salt);
    void __fill_first_blocks(uint8_t *blockhash, int thread);
    string __encode_string(const string &salt, uint8_t *hash);

    argon2_blocks_filler_ptr __filler;
    int __threads;
    uint8_t *__seed_memory;
    size_t __seed_memory_offset;
    void *__user_data;

};


#endif //ARIOMINER_ARGON2_H
