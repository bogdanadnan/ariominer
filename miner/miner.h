//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#ifndef PROJECT_MINER_H
#define PROJECT_MINER_H

#include "../http/client.h"
#include "../app/runner.h"

class miner : public runner {
public:
    miner(arguments &args);
    ~miner();

    virtual void run();
    virtual void stop();

    string get_status();

	static string calc_duration(const string &base, const string &hash);
	static uint64_t calc_compare(const string &duration, const string &difficulty);

private:
    bool __update_pool_data();
    bool __display_report();
    void __disconnect_from_pool();

    string __argon2profile;
    string __recommendation;
    string __nonce;
    string __blk;
    string __difficulty;
    uint32_t __limit;
    string __public_key;
    uint32_t __height;
    uint32_t __found;
	uint32_t __confirmed_cblocks;
	uint32_t __confirmed_gblocks;
	uint32_t __rejected_cblocks;
	uint32_t __rejected_gblocks;
    int __chs_threshold_hit;
    int __ghs_threshold_hit;
    int __blocks_count;
	uint64_t __display_hits;

    time_t __begin_time;

    bool __running;

    arguments &__args;
    ariopool_client __client;
};
#endif //PROJECT_MINER_H
