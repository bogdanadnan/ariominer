//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#ifndef PROJECT_MINER_H
#define PROJECT_MINER_H

#define GOLD_RESULT         240

#include "../http/client.h"

class miner {
public:
    miner(arguments &args);
    ~miner();

    void run();

private:
    string __make_nonce();
    string __calc_duration(const string &base, const string &hash);
    uint64_t __calc_compare(const string &duration);
    bool __update_pool_data();
    void __display_report();

    string __nonce;
    string __blk;
    string __difficulty;
    uint __limit;
    string __public_key;
    uint __height;
    uint __found;
    uint __confirmed;
    uint __rejected;

    uint64_t __total_time;

    arguments &__args;
    ariopool_client __client;
};
#endif //PROJECT_MINER_H
