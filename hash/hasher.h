//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#ifndef ARIOMINER_HASHER_H
#define ARIOMINER_HASHER_H

#include "argon2/defs.h"

struct hash_data {
    hash_data() {
        realloc_flag = NULL;
    };
    string nonce;
    string salt;
    string base;
    string block;
    string hash;
    string profile_name;
    bool *realloc_flag;
};

struct hash_timing {
    uint64_t time_info;
    size_t hash_count;
    int profile; //0 CPU 1 GPU
};

struct device_info {
	device_info() {
		hashcount = 0;
		cblock_hashrate = 0;
		gblock_hashrate = 0;
		cblocks_intensity = 0;
		gblocks_intensity = 0;
	}

	string name;
	string bus_id;
	double cblocks_intensity;
	double gblocks_intensity;
	double cblock_hashrate;
	double gblock_hashrate;
	size_t hashcount;
};

#define REGISTER_HASHER(x)        extern "C"  { DLLEXPORT void hasher_loader() { x *instance = new x(); } }

class DLLEXPORT hasher {
public:
    hasher();
    virtual ~hasher();

    virtual bool initialize() = 0;
    virtual bool configure(arguments &args) = 0;
    virtual void cleanup() = 0;

    string get_type();
	string get_subtype();
	int get_priority();
    string get_info();
    void set_input(const string &public_key, const string &blk, const string &difficulty, const string &argon2profile_string, const string &recommendation);

    double get_current_hash_rate();
    double get_avg_hash_rate_cblocks();
    double get_avg_hash_rate_gblocks();

    uint32_t get_hash_count_cblocks();
    uint32_t get_hash_count_gblocks();

    vector<hash_data> get_hashes();
    map<int, device_info> &get_device_infos();
    bool is_running();

    static vector<hasher*> get_hashers_of_type(const string &type);
    static vector<hasher*> get_hashers();
    static vector<hasher*> get_active_hashers();
    static void load_hashers();

protected:
    double _intensity;
    string _type;
	string _subtype;
	int _priority;
    string _description;

	void _store_hash(const hash_data &hash, int device_id);
	void _store_hash(const vector<hash_data> &hashes, int device_id);

	void _store_device_info(int device_id, device_info device);

    hash_data _get_input();
    argon2profile *_get_argon2profile();
    bool _should_pause();
    void _update_running_status(bool running);
	vector<string> _get_gpu_filters(arguments &args);

private:
    string __make_nonce();
	void __update_hashrate();

    static vector<hasher*> *__registered_hashers;

    mutex __input_mutex;
    string __public_key;
    string __blk;
    string __difficulty;
    bool __pause;
    bool __is_running;
    argon2profile *__argon2profile;

    mutex __hashes_mutex;
    vector<hash_data> __hashes;

    uint64_t __hashrate_time;
    map<int, device_info> __device_infos;
    double __hashrate;

    size_t __total_hash_count_cblocks;
    size_t __total_hash_count_gblocks;

    size_t __hash_count;
    uint64_t __begin_round_time;
    list<hash_timing> __hash_timings;
};

#endif //ARIOMINER_HASHER_H
