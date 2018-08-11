//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#ifndef ARIOMINER_HASHER_H
#define ARIOMINER_HASHER_H

#include "argon2/defs.h"
#include "../app/arguments.h"

struct hash_data {
    hash_data() {
        realloc_flag = NULL;
    };
    string nonce;
    string base;
    string hash;
    bool *realloc_flag;
};

#define REGISTER_HASHER(x)          x __##x

class hasher {
public:
    hasher();
    virtual ~hasher();

    virtual bool configure(arguments &args) = 0;

    string get_type();
    string get_info();
    void set_input(const string &public_key, const string &blk, const string &difficulty, const string &argon2profile_string, const string &recommendation);

    hash_data get_input();
    argon2profile *get_argon2profile();
    bool should_pause();
    int get_intensity();

    double get_current_hash_rate();
    double get_avg_hash_rate();
    uint get_hash_count();

    vector<hash_data> get_hashes();

    static vector<hasher*> get_hashers();
    static vector<hasher*> get_active_hashers();

protected:
    int _intensity;
    string _type;
    string _description;

    void _store_hash(const hash_data &hash);

private:
    string __make_nonce();

    static vector<hasher*> *__registered_hashers;

    double __hash_rate;
    double __avg_hash_rate;
    uint __hash_count;

    mutex __input_mutex;
    string __public_key;
    string __blk;
    string __difficulty;
    bool __pause;
    argon2profile *__argon2profile;

    mutex __hashes_mutex;
    vector<hash_data> __hashes;

    uint64_t __begin_time;
    uint64_t __hashrate_time;

};

#endif //ARIOMINER_HASHER_H
