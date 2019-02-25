//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#ifndef PROJECT_PROXY_H
#define PROJECT_PROXY_H

#include "../app/runner.h"
#include "../app/arguments.h"
#include "../http/client.h"
#include "../http/node_api.h"

struct miner_hashrate {
    double cblocks_hashrate;
    double gblocks_hashrate;
    time_t timestamp;
};

struct miner_client {
    miner_client() {
        cblocks_hashrate = 0;
        gblocks_hashrate = 0;
        timestamp = 0;
        created = time(NULL);
    }
    string worker_name;
    double cblocks_hashrate;
    double gblocks_hashrate;
    time_t timestamp;
    time_t created;
    string details;

    list<miner_hashrate> hashrate_history;
};

struct global_status {
    global_status() {
        cblocks_hashrate = 0;
        gblocks_hashrate = 0;
        uptime = 0;
        cblocks_shares = 0;
        gblocks_shares = 0;
        cblocks_rejects = 0;
        gblocks_rejects = 0;
        workers_count = 0;
        current_block = 0;
        cblocks_dl = 0;
        gblocks_dl = 0;
        blocks = 0;
        best_dl = 0;
    }

    double cblocks_hashrate;
    double gblocks_hashrate;
    time_t uptime;
    int cblocks_shares;
    int gblocks_shares;
    int cblocks_rejects;
    int gblocks_rejects;
    int workers_count;
    int current_block;
    int cblocks_dl;
    int gblocks_dl;
    int blocks;
    int best_dl;
};

struct miner_list_item {
    miner_list_item() {};
    miner_list_item(miner_client &mc, time_t timestamp) {
        worker_name = mc.worker_name;
        cblocks_hashrate = mc.cblocks_hashrate;
        gblocks_hashrate = mc.gblocks_hashrate;
        uptime = timestamp - mc.created;
    };

    string worker_name;
    double cblocks_hashrate;
    double gblocks_hashrate;
    time_t uptime;
};

struct miner_status {
    miner_status() {
        uptime = 0;
        cblocks_hashrate = 0;
        gblocks_hashrate = 0;
        cblocks_shares = 0;
        gblocks_shares = 0;
        cblocks_rejects = 0;
        gblocks_rejects = 0;
        devices_count = 0;
        blocks = 0;
    };

    time_t uptime;
    double cblocks_hashrate;
    double gblocks_hashrate;
    int cblocks_shares;
    int gblocks_shares;
    int cblocks_rejects;
    int gblocks_rejects;
    int devices_count;
    int blocks;
};

struct device_details {
    device_details() {
        cblocks_hashrate = 0;
        gblocks_hashrate = 0;
    }

    string hasher_name;
    string device_name;
    double cblocks_hashrate;
    double gblocks_hashrate;
};

class proxy : public runner {
public:
    proxy(arguments &args);
    ~proxy();

    virtual void run();
    virtual void stop();

    string process_info_request(const string &ip, const string &miner_id, const string &miner_name, double cblocks_hashrate, double gblocks_hashrate, const string &details);
    string process_submit_request(const string &ip, const string &miner_id, const string &miner_name, const string &argon, const string &nonce, const string &public_key);

    map<string, string> get_workers();

    string get_status();

    global_status get_global_status();
    account_balance get_account_balance();
    void get_global_hashrate_history(list<miner_hashrate> &history);
    void get_workers_list(vector<miner_list_item> &workers);

    miner_status get_worker_status(const string &worker_id);
    void get_worker_devices(const string &worker_id, vector<device_details> &devices);
    void get_worker_hashrate_history(const string &worker_id, list<miner_hashrate> &history);

private:
    bool __update_pool_data();
    void __update_global_history();

    mutex __pool_block_settings_lock;
    ariopool_update_result __pool_block_settings;
    int __cblocks_dl;
    int __gblocks_dl;

    mutex __miner_clients_lock;
    map<string, miner_client> __miner_clients;

    arguments &__args;
    bool __running;
    time_t __start;

    uint32_t __found;
    uint32_t __confirmed_cblocks;
    uint32_t __confirmed_gblocks;
    uint32_t __rejected_cblocks;
    uint32_t __rejected_gblocks;
    uint32_t __best_dl;

    mutex __global_hashrate_history_lock;
    list<miner_hashrate> __global_hashrate_history;

    ariopool_client __client;
};


#endif //PROJECT_PROXY_H
