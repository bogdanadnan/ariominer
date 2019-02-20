//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#ifndef PROJECT_PROXY_H
#define PROJECT_PROXY_H

#include "../app/runner.h"
#include "../app/arguments.h"
#include "../http/client.h"

struct miner_client {
    miner_client() {
        cblocks_hashrate = 0;
        gblocks_hashrate = 0;
        timestamp = 0;
    }
    string worker_name;
    double cblocks_hashrate;
    double gblocks_hashrate;
    time_t timestamp;
};

class proxy : public runner {
public:
    proxy(arguments &args);
    ~proxy();

    virtual void run();
    virtual void stop();

    string process_info_request(const string &ip, const string &miner_id, const string &miner_name, double cblocks_hashrate, double gblocks_hashrate);
    string process_submit_request(const string &ip, const string &miner_id, const string &miner_name, const string &argon, const string &nonce, const string &public_key);

private:
    bool __update_pool_data();

    mutex __pool_block_settings_lock;
    ariopool_update_result __pool_block_settings;

    mutex __miner_clients_lock;
    map<string, miner_client> __miner_clients;

    arguments &__args;
    bool __running;

    ariopool_client __client;
};


#endif //PROJECT_PROXY_H
