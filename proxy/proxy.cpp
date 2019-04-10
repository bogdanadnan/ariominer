//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#include "../common/common.h"

#include "../app/arguments.h"
#include "../http/client.h"
#include "../http/simplejson/json.h"

#include "../miner/miner.h"

#include "proxy.h"
#include "proxy_server.h"

proxy::proxy(arguments &args) : __args(args), __client(args, [&]() { return this->get_status(); }) {
    __running = false;
    __cblocks_dl = 0;
    __gblocks_dl = 0;
    __found = 0;
    __confirmed_cblocks = 0;
    __confirmed_gblocks = 0;
    __rejected_cblocks = 0;
    __rejected_gblocks = 0;
    __best_dl = 0;
    __start = time(NULL);
}

proxy::~proxy() { }

void proxy::run() {
    __running = true;
    uint64_t last_update = 0;
    uint64_t last_history = microseconds() - __args.hash_report_interval() + 60000000; // first entry at 1 min after startup

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

            if(microseconds() - last_history > __args.hash_report_interval()) {
                __update_global_history();
                last_history = microseconds();
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
    vector<string> to_erase;
    for(map<string, miner_client>::iterator iter=__miner_clients.begin(); iter != __miner_clients.end(); iter++) {
        if(timestamp - iter->second.timestamp < 660) { // take into account only miners that responded in the last 11 min
            hash_rate_cblocks += iter->second.cblocks_hashrate;
            hash_rate_gblocks += iter->second.gblocks_hashrate;
        }
        else { // delete older miners
            to_erase.push_back(iter->first);
        }
    }
    for(vector<string>::iterator iter = to_erase.begin();iter != to_erase.end();iter++) {
        LOG("--> Client " + __miner_clients[*iter].worker_name + " disconnected.");
        __miner_clients.erase(*iter);
    }
    __miner_clients_lock.unlock();

    ariopool_update_result new_settings = __client.update(hash_rate_cblocks, hash_rate_gblocks);

    bool changed = false;
    if(new_settings.success) {
        __pool_block_settings_lock.lock();

        // inject Proxy and Details pool extensions, necessary for proper reporting
        if(new_settings.extensions.find("Details") == string::npos)
            new_settings.extensions = new_settings.extensions.empty() ? "Details" : ("Details, " + new_settings.extensions);
        if(new_settings.extensions.find("Proxy") == string::npos)
            new_settings.extensions = new_settings.extensions.empty() ? "Proxy" : ("Proxy, " + new_settings.extensions);
        new_settings.version = "Ariominer Proxy v." ArioMiner_VERSION_MAJOR "." ArioMiner_VERSION_MINOR "." ArioMiner_VERSION_REVISION;

        changed = __pool_block_settings.update(new_settings);
        if(__pool_block_settings.argon2profile == "1_1_524288")
            __cblocks_dl = __pool_block_settings.limit;
        else
            __gblocks_dl = __pool_block_settings.limit;
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

string proxy::process_info_request(const string &ip, const string &miner_id, const string &miner_name, double cblocks_hashrate, double gblocks_hashrate, const string &details) {
    string miner_key = miner_id + "_" + miner_name;

    __miner_clients_lock.lock();
    if(__miner_clients.find(miner_key) == __miner_clients.end()) {
        __miner_clients.insert(make_pair(miner_key, miner_client()));
        LOG("--> New client from " + ip + " id: " + miner_id + " worker: " + miner_name + ".");
    }
    miner_client &client = __miner_clients[miner_key];
    client.worker_name = miner_name;
    if(cblocks_hashrate > 0)
       client.cblocks_hashrate = cblocks_hashrate;
    if(gblocks_hashrate > 0)
        client.gblocks_hashrate = gblocks_hashrate;
    if(!details.empty())
        client.details = details;
    client.timestamp = time(NULL);

    if(cblocks_hashrate > 0 || gblocks_hashrate > 0) { // add hashrate_history record
        miner_hashrate mhr;
        mhr.cblocks_hashrate = cblocks_hashrate;
        mhr.gblocks_hashrate = gblocks_hashrate;
        mhr.timestamp = client.timestamp;
        client.hashrate_history.push_back(mhr);

        list<miner_hashrate>::iterator history_iterator = client.hashrate_history.begin();
        while(history_iterator != client.hashrate_history.end() && (client.timestamp - history_iterator->timestamp) > 86400) {
            client.hashrate_history.erase(history_iterator);
            history_iterator = client.hashrate_history.begin();
        }
    }
    __miner_clients_lock.unlock();

    if(cblocks_hashrate > 0 || gblocks_hashrate > 0) {
        LOG("--> Update from " + miner_name + ": (C) " + to_string(cblocks_hashrate) + " h/s  (G) " + to_string(gblocks_hashrate) + " h/s.");
    }
    __pool_block_settings_lock.lock();
    string response = __pool_block_settings.response();
    __pool_block_settings_lock.unlock();

    return response;
}

string proxy::process_submit_request(const string &ip, const string &miner_id, const string &miner_name, const string &argon, const string &nonce, const string &public_key) {
    string hash;

    __pool_block_settings_lock.lock();
    int height = __pool_block_settings.height;
    string argon2profile = __pool_block_settings.argon2profile;
    string difficulty = __pool_block_settings.difficulty;
    string block = __pool_block_settings.block;
    __pool_block_settings_lock.unlock();

    if(argon2profile == "1_1_524288")
        hash = "$argon2i$v=19$m=524288,t=1,p=1" + argon;
    else
        hash = "$argon2i$v=19$m=16384,t=4,p=4" + argon;

    ariopool_submit_result response = __client.submit(hash, nonce, public_key);

    string base = public_key + "-" + nonce + "-" + block + "-" + difficulty;
    string duration = miner::calc_duration(base, hash);
    uint64_t result = miner::calc_compare(duration, difficulty);

    bool is_block = (result <= GOLD_RESULT);
    if (__args.is_verbose())
        LOG("--> Submitting nonce: " + nonce + " / " + hash.substr(30));
    if (response.success) {
        if (is_block) {
            if (__args.is_verbose()) LOG("--> Block found.");
            __found++;
        } else {
            if (__args.is_verbose()) LOG("--> Nonce confirmed.");
            if(argon2profile == "1_1_524288")
                __confirmed_cblocks++;
            else
                __confirmed_gblocks++;
        }
        if(__best_dl == 0 || result < __best_dl)
            __best_dl = result;
    } else {
        if (__args.is_verbose()) {
            LOG("--> The nonce did not confirm.");
            LOG("--> Pool response: ");
            LOG(response.pool_response);
        }
        if(argon2profile == "1_1_524288")
            __rejected_cblocks++;
        else
            __rejected_gblocks++;
    }

    return response.pool_response;
}

string proxy::get_status() {
    vector<string> allDetails;
    time_t timestamp = time(NULL);

    __miner_clients_lock.lock();
    for(map<string, miner_client>::iterator iter=__miner_clients.begin(); iter != __miner_clients.end(); iter++) {
        if(timestamp - iter->second.timestamp < 660) { // take into account only miners that responded in the last 11 min
            if(!iter->second.details.empty()) {
                allDetails.push_back(iter->second.details);
            }
        }
    }
    __miner_clients_lock.unlock();

    json::JSON combinedDetails = json::JSON::Make(json::JSON::Class::Array);
    for(vector<string>::iterator iter=allDetails.begin(); iter != allDetails.end(); iter++) {
        json::JSON d = json::JSON::Load(*iter);
        if(d.JSONType() == json::JSON::Class::Array && d.length()) {
            for(int i=0;i<d.length();i++) {
                combinedDetails.append(d[i]);
            }
        }
    }

    string response = combinedDetails.dump();

    return response;
}

map<string, string> proxy::get_workers() {
    map<string, string> response;
    time_t timestamp = time(NULL);

    __miner_clients_lock.lock();
    for(map<string, miner_client>::iterator iter=__miner_clients.begin(); iter != __miner_clients.end(); iter++) {
        if(timestamp - iter->second.timestamp < 660) {
            response[iter->first] = iter->second.worker_name;
        }
    }
    __miner_clients_lock.unlock();

    return response;
}

global_status proxy::get_global_status() {
    global_status status;
    time_t timestamp = time(NULL);

    __miner_clients_lock.lock();
    for(map<string, miner_client>::iterator iter=__miner_clients.begin(); iter != __miner_clients.end(); iter++) {
        if(timestamp - iter->second.timestamp < 660) {
            status.cblocks_hashrate += iter->second.cblocks_hashrate;
            status.gblocks_hashrate += iter->second.gblocks_hashrate;
            status.workers_count++;
        }
    }
    __miner_clients_lock.unlock();

    __pool_block_settings_lock.lock();
    status.current_block = __pool_block_settings.height;
    status.cblocks_dl = __cblocks_dl;
    status.gblocks_dl = __gblocks_dl;
    status.cblocks_shares = __confirmed_cblocks;
    status.gblocks_shares = __confirmed_gblocks;
    status.cblocks_rejects = __rejected_cblocks;
    status.gblocks_rejects = __rejected_gblocks;
    status.blocks = __found;
    status.best_dl = __best_dl;
    __pool_block_settings_lock.unlock();

    status.uptime = time(NULL) - __start;
    return status;
}

account_balance proxy::get_account_balance() {
    node_api api(__args.wallet());
    return api.get_account_balance();
}

void proxy::__update_global_history() {
    miner_hashrate mhr;
    mhr.cblocks_hashrate = 0;
    mhr.gblocks_hashrate = 0;
    mhr.timestamp = time(NULL);

    __miner_clients_lock.lock();
    for(map<string, miner_client>::iterator iter=__miner_clients.begin(); iter != __miner_clients.end(); iter++) {
        if(mhr.timestamp - iter->second.timestamp < 660) {
            mhr.cblocks_hashrate += iter->second.cblocks_hashrate;
            mhr.gblocks_hashrate += iter->second.gblocks_hashrate;
        }
    }
    __miner_clients_lock.unlock();

    __global_hashrate_history_lock.lock();
    __global_hashrate_history.push_back(mhr);
    list<miner_hashrate>::iterator history_iterator = __global_hashrate_history.begin();
    while(history_iterator != __global_hashrate_history.end() && (mhr.timestamp - history_iterator->timestamp) > 86400) {
        __global_hashrate_history.erase(history_iterator);
        history_iterator = __global_hashrate_history.begin();
    }
    __global_hashrate_history_lock.unlock();
}

void proxy::get_global_hashrate_history(list<miner_hashrate> &history) {
    __global_hashrate_history_lock.lock();
    copy(__global_hashrate_history.begin(), __global_hashrate_history.end(), back_inserter(history));
    __global_hashrate_history_lock.unlock();
}

void proxy::get_workers_list(vector<miner_list_item> &workers) {
    time_t timestamp = time(NULL);
    __miner_clients_lock.lock();
    for(map<string, miner_client>::iterator iter=__miner_clients.begin(); iter != __miner_clients.end(); iter++) {
        if(timestamp - iter->second.timestamp < 660) {
            miner_list_item mli(iter->second, timestamp);
            workers.push_back(mli);
        }
    }
    __miner_clients_lock.unlock();
}

miner_status proxy::get_worker_status(const string &worker_id) {
    miner_client client; bool found = false;

    __miner_clients_lock.lock();
    if(__miner_clients.find(worker_id) != __miner_clients.end()) {
        client = __miner_clients[worker_id];
        found = true;
    }
    __miner_clients_lock.unlock();

    if(!found) {
        return miner_status();
    }

    miner_status status;
    status.uptime = client.timestamp - client.created;
    status.cblocks_hashrate = client.cblocks_hashrate;
    status.gblocks_hashrate = client.gblocks_hashrate;

    if(!client.details.empty()) {
        json::JSON details = json::JSON::Load(client.details);
        if(details.JSONType() == json::JSON::Class::Array && details.length() > 0) {
            json::JSON worker = details[0];
            if(worker.JSONType() == json::JSON::Class::Object) {
                if(worker.hasKey("cblocks_shares"))
                    status.cblocks_shares = worker["cblocks_shares"].ToInt();
                if(worker.hasKey("gblocks_shares"))
                    status.gblocks_shares = worker["gblocks_shares"].ToInt();
                if(worker.hasKey("cblocks_rejects"))
                    status.cblocks_rejects = worker["cblocks_rejects"].ToInt();
                if(worker.hasKey("gblocks_rejects"))
                    status.gblocks_rejects = worker["gblocks_rejects"].ToInt();
                if(worker.hasKey("blocks_earned"))
                    status.blocks = worker["blocks_earned"].ToInt();

                if(worker.hasKey("hashers")) {
                    json::JSON hashers = worker["hashers"];
                    if(hashers.JSONType() == json::JSON::Class::Array && hashers.length() > 0) {
                        for(int i=0; i < hashers.length(); i++) {
                            json::JSON hasher = hashers[i];
                            if(hasher.JSONType() == json::JSON::Class::Object &&
                               hasher.hasKey("devices") &&
                               hasher["devices"].JSONType() == json::JSON::Class::Array) {
                                status.devices_count += hasher["devices"].length();
                            }
                        }
                    }
                }
            }
        }
    }

    return status;
}

void proxy::get_worker_devices(const string &worker_id, vector<device_details> &devices_) {
    miner_client client; bool found = false;

    __miner_clients_lock.lock();
    if(__miner_clients.find(worker_id) != __miner_clients.end()) {
        client = __miner_clients[worker_id];
        found = true;
    }
    __miner_clients_lock.unlock();

    if(!found || client.details.empty()) {
        return;
    }

    json::JSON details = json::JSON::Load(client.details);
    if(details.JSONType() == json::JSON::Class::Array && details.length() > 0) {
        json::JSON worker = details[0];
        if(worker.JSONType() == json::JSON::Class::Object) {
            if(worker.hasKey("hashers")) {
                json::JSON hashers = worker["hashers"];
                if(hashers.JSONType() == json::JSON::Class::Array && hashers.length() > 0) {
                    for(int i=0; i < hashers.length(); i++) {
                        json::JSON hasher = hashers[i];
                        if(hasher.JSONType() == json::JSON::Class::Object &&
                           hasher.hasKey("devices") &&
                           hasher["devices"].JSONType() == json::JSON::Class::Array) {
                            json::JSON devices = hasher["devices"];
                            for(int j=0; j < devices.length(); j++) {
                                json::JSON device = devices[j];
                                if(device.JSONType() == json::JSON::Class::Object) {
                                    device_details dev_det;
                                    if(hasher.hasKey("subtype"))
                                        dev_det.hasher_name = hasher["subtype"].ToString();
                                    if(device.hasKey("name"))
                                        dev_det.device_name = device["name"].ToString();
                                    if(device.hasKey("cblocks_hashrate"))
                                        dev_det.cblocks_hashrate = device["cblocks_hashrate"].ToFloat();
                                    if(device.hasKey("gblocks_hashrate"))
                                        dev_det.gblocks_hashrate = device["gblocks_hashrate"].ToFloat();

                                    devices_.push_back(dev_det);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void proxy::get_worker_hashrate_history(const string &worker_id, list<miner_hashrate> &history) {
    __miner_clients_lock.lock();
    if(__miner_clients.find(worker_id) != __miner_clients.end()) {
        miner_client &client = __miner_clients[worker_id];
        copy(client.hashrate_history.begin(), client.hashrate_history.end(), back_inserter(history));
    }
    __miner_clients_lock.unlock();
}

string proxy::process_disconnect_request(const string &ip, const string &miner_id, const string &miner_name) {
    string miner_key = miner_id + "_" + miner_name;
    bool erased = false;

    __miner_clients_lock.lock();
    if(__miner_clients.find(miner_key) != __miner_clients.end()) {
        LOG("--> Client " + __miner_clients[miner_key].worker_name + " disconnected.");
        __miner_clients.erase(miner_key);
        erased = true;
    }
    __miner_clients_lock.unlock();

    if(erased)
        return "{ \"status\": \"ok\", \"data\": \"disconnected\", \"coin\": \"arionum\" }";
    else
        return "{ \"status\": \"error\", \"data\": \"invalid client\", \"coin\": \"arionum\" }";
}
