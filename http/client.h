//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#ifndef PROJECT_CLIENT_H
#define PROJECT_CLIENT_H

#include "http.h"
#include "pool_settings_provider.h"

struct ariopool_result {
    bool success;
};

struct ariopool_update_result : public ariopool_result {
    string block;
    string difficulty;
    uint32_t limit;
    string public_key;
    uint32_t height;
    string argon2profile;
    string recommendation;
    string version;
    string extensions;

    bool update(ariopool_update_result &src) {
        if(block != src.block ||
                difficulty != src.difficulty ||
                limit != src.limit ||
                public_key != src.public_key ||
                height != src.height ||
                argon2profile != src.argon2profile ||
                recommendation != src.recommendation ||
                version != src.version ||
                extensions != src.extensions) {
            block = src.block;
            difficulty = src.difficulty;
            limit = src.limit;
            public_key = src.public_key;
            height = src.height;
            argon2profile = src.argon2profile;
            recommendation = src.recommendation;
            version = src.version;
            extensions = src.extensions;

            return true;
        }

        return false;
    }

    string response() {
        stringstream ss;
        string argon_mem = "524288";
        string argon_threads = "1";
        string argon_time = "1";
        if(argon2profile == "4_4_16384") {
            argon_mem = "16384";
            argon_threads = "4";
            argon_time = "4";
        }

        ss << "{ \"status\": \"ok\", \"data\": { \"recommendation\": \"" << recommendation << "\", \"argon_mem\": " << argon_mem
           << ", \"argon_threads\": " << argon_threads << ", \"argon_time\": " << argon_time <<", \"difficulty\": \"" << difficulty
           << "\", \"block\": \"" << block << "\", \"height\": " << height << ", \"public_key\": \"" << public_key
           << "\", \"limit\": " << limit << " }, \"coin\": \"arionum\", \"version\": \"" << version << "\", \"extensions\": \"" << extensions << "\" }";

        return ss.str();
    }
};

struct ariopool_submit_result : public ariopool_result {
    string pool_response;
};

typedef function<string ()> get_status_ptr;

class ariopool_client : public http {
public:
    ariopool_client(arguments &args, get_status_ptr get_status);

    ariopool_update_result update(double hash_rate_cblocks, double hash_rate_gblocks);
    ariopool_submit_result submit(const string &hash, const string &nonce, const string &public_key);
    void disconnect();

private:
    bool __validate_response(const string &response);
    pool_settings &__get_pool_settings();

    pool_settings_provider __pool_settings_provider;
    bool __is_devfee_time;
    string __miner_version;
    string __worker_id;
    string __worker_name;
    string __force_argon2profile;
    int64_t __hash_report_interval;

    bool __force_hashrate_report;
    bool __show_pool_requests;

    uint64_t __timestamp;
    uint64_t __last_hash_report;
    get_status_ptr __get_status;
};

#endif //PROJECT_CLIENT_H
