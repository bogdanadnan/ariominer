//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#include "../common/common.h"

#include "../app/arguments.h"
#include "../http/client.h"

#include "proxy.h"
#include "proxy_server.h"

proxy::proxy(arguments &args) : __args(args), __client(args, NULL) {
    __running = false;
}

proxy::~proxy() { }

void proxy::run() {
    __running = true;
    uint64_t last_update = 0;

    if(__args.proxy_port() > 0) {
        vector<string> options;
        options.push_back("listening_ports");
        options.push_back(to_string(__args.proxy_port()));
        LOG("Starting proxy server on port " + to_string(__args.proxy_port()) + ".");
        proxy_server server(options, *this, __args);
        LOG("Waiting for clients...");
        while(__running) {
            if (microseconds() - last_update > __args.update_interval()) {
                __update_pool_data();
                last_update = microseconds();
            }

            this_thread::sleep_for(chrono::milliseconds(100));
        }
    }
    else {
        LOG("Proxy port not set, exiting.");
        __running = false;
    }
}

void proxy::stop() {
    __running = false;
}

bool proxy::__update_pool_data() {
    double hash_rate_cblocks = 0;
    double hash_rate_gblocks = 0;

    time_t timestamp = time(NULL);

    __miner_clients_lock.lock();
    for(map<string, miner_client>::iterator iter=__miner_clients.begin(); iter != __miner_clients.end(); iter++) {
        if(timestamp - iter->second.timestamp < 1200) {
            hash_rate_cblocks += iter->second.cblocks_hashrate;
            hash_rate_gblocks += iter->second.gblocks_hashrate;
        }
    }
    __miner_clients_lock.unlock();

    ariopool_update_result new_settings = __client.update(hash_rate_cblocks, hash_rate_gblocks);

    bool changed = false;
    if(new_settings.success) {
        __pool_block_settings_lock.lock();
        changed = __pool_block_settings.update(new_settings);
        __pool_block_settings_lock.unlock();
    }

    if(changed && __args.is_verbose()) {
        stringstream ss;
        ss << "-----------------------------------------------------------------------------------------------------------------------------------------" << endl;
        ss << "--> Pool data updated   Block: " << new_settings.block << endl;
        ss << "--> " << ((new_settings.argon2profile == "1_1_524288") ? "CPU round" : (new_settings.recommendation == "pause" ? "Masternode round" : "GPU round"));
        ss << "  Height: " << new_settings.height << "  Limit: " << new_settings.limit << "  Difficulty: " << new_settings.difficulty << "  Proxy: " << __args.name() << endl;
        ss << "-----------------------------------------------------------------------------------------------------------------------------------------";

        LOG(ss.str());

        return true;
    }

    return false;
}

string proxy::process_info_request(const string &ip, const string &miner_id, const string &miner_name, double cblocks_hashrate, double gblocks_hashrate) {
    string miner_key = miner_id + "_" + miner_name;

    __miner_clients_lock.lock();
    if(__miner_clients.find(miner_key) == __miner_clients.end()) {
        __miner_clients.insert(make_pair(miner_key, miner_client()));
        LOG("New client from " + ip + " id: " + miner_id + " worker: " + miner_name);
    }
    miner_client &client = __miner_clients[miner_key];
    client.worker_name = miner_name;
    if(cblocks_hashrate > 0)
       client.cblocks_hashrate = cblocks_hashrate;
    if(gblocks_hashrate > 0)
        client.gblocks_hashrate = gblocks_hashrate;
    client.timestamp = time(NULL);
    __miner_clients_lock.unlock();

    __pool_block_settings_lock.lock();
    string response = __pool_block_settings.response();
    __pool_block_settings_lock.unlock();

    return response;
}

string proxy::process_submit_request(const string &ip, const string &miner_id, const string &miner_name, const string &argon, const string &nonce, const string &public_key) {
    string hash;

    __pool_block_settings_lock.lock();
    int height = __pool_block_settings.height;
    __pool_block_settings_lock.unlock();

    if(height%2)
        hash = "$argon2i$v=19$m=524288,t=1,p=1" + argon;
    else
        hash = "$argon2i$v=19$m=16384,t=4,p=4" + argon;

    ariopool_submit_result result = __client.submit(hash, nonce, public_key);

    return result.pool_response;
}
