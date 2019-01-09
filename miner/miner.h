//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#ifndef PROJECT_MINER_H
#define PROJECT_MINER_H

#define GOLD_RESULT         240

#include "../http/client.h"
#include "../app/runner.h"

class miner : public runner {
public:
    miner(arguments &args);
    ~miner();

    virtual void run();
    virtual void stop();

private:
    string __calc_duration(const string &base, const string &hash);
    uint64_t __calc_compare(const string &duration);
    bool __update_pool_data();
    bool __display_report();

    string __argon2profile;
    string __recommendation;
    string __nonce;
    string __blk;
    string __difficulty;
    uint32_t __limit;
    string __public_key;
    uint32_t __height;
    uint32_t __found;
    uint32_t __confirmed;
    uint32_t __rejected;
    int __chs_threshold_hit;
    int __ghs_threshold_hit;

    time_t __begin_time;

    bool __running;

    arguments &__args;
    ariopool_client __client;
};
#endif //PROJECT_MINER_H
