//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#include "../common/common.h"

#include "client.h"

#include "simplejson/json.h"

ariopool_client::ariopool_client(const string &pool_address, const string &worker_id, const string &wallet_address) {
    __pool_address = pool_address;
    __worker_id = worker_id;
    __wallet_address = wallet_address;
    __encoded_wallet_address = __encode(wallet_address);
}

ariopool_update_result ariopool_client::update(double hash_rate) {
    ariopool_update_result result;
    result.success = false;

    string url = __pool_address + "/mine.php?q=info&worker=" + __worker_id + "&address=" + __wallet_address + "&hashrate=" + to_string(hash_rate);

    string response = __http_get(url);

    if(!__validate_response(response)) {
        LOG("Error connecting to " + __pool_address + ".");
        return result;
    }

    json::JSON info = json::JSON::Load(response);

    result.success = (info["status"].ToString() == "ok");

    if (result.success) {
        json::JSON data = info["data"];
        result.block = data["block"].ToString();
        result.difficulty = data["difficulty"].ToString();
        result.limit = (uint)data["limit"].ToInt();
        result.public_key = data["public_key"].ToString();
        result.height = (uint)data["height"].ToInt();
    }

    return result;
}

ariopool_submit_result ariopool_client::submit(const string &hash, const string &nonce, const string &public_key) {
    ariopool_submit_result result;
    string argon_data = hash.substr(30);

    string payload = "argon=" + __encode(argon_data) +
            "&nonce=" + __encode(nonce) +
            "&private_key=" + __wallet_address +
            "&public_key=" + __encode(public_key) +
            "&address=" + __wallet_address;
    string url = __pool_address + "/mine.php?q=submitNonce";

    string response = __http_post(url, payload);

    if(!__validate_response(response)) {
        LOG("Error connecting to " + __pool_address + ".");
        return result;
    }

    json::JSON info = json::JSON::Load(response);

    result.success = (info["status"].ToString() == "ok");

    return result;
}

bool ariopool_client::__validate_response(const string &response) {
    return !response.empty() && response.find("status") != string::npos;
}
