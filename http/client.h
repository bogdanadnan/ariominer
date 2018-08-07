//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#ifndef PROJECT_CLIENT_H
#define PROJECT_CLIENT_H

#include "http.h"

struct ariopool_result {
    bool success;
};

struct ariopool_update_result : public ariopool_result {
    string block;
    string difficulty;
    uint limit;
    string public_key;
    uint height;
};

struct ariopool_submit_result : public ariopool_result {

};

class ariopool_client : public http {
public:
    ariopool_client(const string &pool_address, const string &worker_id, const string &wallet_address);

    ariopool_update_result update(double hash_rate);
    ariopool_submit_result submit(const string &hash, const string &nonce, const string &public_key);

private:
    bool __validate_response(const string &response);

    string __pool_address;
    string __worker_id;
    string __wallet_address;
    string __encoded_wallet_address;
};

#endif //PROJECT_CLIENT_H
