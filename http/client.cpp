//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#include "../common/common.h"
#include "../app/arguments.h"
#include "client.h"

#include "simplejson/json.h"

#define DEV_WALLET_ADDRESS      "3ykJiMsURozMLgazT97A5iidWiPLRvY5CQW9ziFJcJAZNJ9AjZimSUQe8nfwQTJqukch2JXEF48sLdoFqzKB9FVL"
//#define DEVELOPER_OWN_BUILD

ariopool_client::ariopool_client(arguments &args, get_status_ptr get_status) {
    __pool_address = args.pool();
    __worker_id = args.uid();
    __worker_name = args.name();
    __client_wallet_address = __used_wallet_address = args.wallet();
    __force_argon2profile = args.argon2_profile();
    __hash_report_interval = args.hash_report_interval();
    __timestamp = __last_hash_report = microseconds();
    __force_hashrate_report = false;
    __show_pool_requests = args.show_pool_requests();
    __get_status = get_status;
}

ariopool_update_result ariopool_client::update(double hash_rate_cblocks, double hash_rate_gblocks) {
    ariopool_update_result result;
    result.success = false;

    string wallet = __get_wallet_address();

#ifndef DEVELOPER_OWN_BUILD
    if(wallet == DEV_WALLET_ADDRESS) {
        hash_rate_cblocks = hash_rate_cblocks / 100;
        hash_rate_gblocks = hash_rate_gblocks / 100;
    }
#endif

    uint64_t current_timestamp = microseconds();
    string hash_report_query = "";

    if(__force_hashrate_report || (current_timestamp - __last_hash_report) > __hash_report_interval) {
        hash_report_query = "&hashrate=" + to_string(hash_rate_cblocks) + "&hrgpu=" + to_string(hash_rate_gblocks);

        __last_hash_report = current_timestamp;
        __force_hashrate_report = false;
    }
    string url = __pool_address + "/mine.php?q=info&id=" + __worker_id + "&worker=" + __worker_name + "&address=" + __get_wallet_address() + hash_report_query;

    if(__show_pool_requests && url.find("hashrate") != string::npos) // log only hashrate requests
        LOG("--> Pool request: " + url);

    string response;
    if(__pool_extensions.find("Details") != string::npos && url.find("hashrate") != string::npos) {
	string payload = "";
        if(__get_status != NULL)
            payload = __get_status();
	if(!payload.empty())
	        response = _http_post(url, payload, "application/json");
	else
	        response = _http_get(url);
    }
    else {
        response = _http_get(url);
    }

    if(__show_pool_requests && url.find("hashrate") != string::npos) // log only hashrate responses
        LOG("--> Pool response: " + response);

    if(!__validate_response(response)) {
        LOG("Error connecting to " + __pool_address + ".");
        return result;
    }

    json::JSON info = json::JSON::Load(response);

    result.success = (info["status"].ToString() == "ok");

    if(info.hasKey("version")) {
        result.version = __pool_version = info["version"].ToString();
    }
    if(info.hasKey("extensions")) {
        result.extensions = __pool_extensions = info["extensions"].ToString();
    }

    if (result.success) {
        json::JSON data = info["data"];
        result.block = data["block"].ToString();
        result.difficulty = data["difficulty"].ToString();
        result.limit = (uint32_t)data["limit"].ToInt();
        result.public_key = data["public_key"].ToString();
        result.height = (uint32_t)data["height"].ToInt();
        if(__force_argon2profile == "") {
            result.argon2profile = to_string(data["argon_threads"].ToInt()) + "_" + to_string(data["argon_time"].ToInt()) + "_" + to_string(data["argon_mem"].ToInt());
        }
        else {
            result.argon2profile = __force_argon2profile;
        }
        result.recommendation = data["recommendation"].ToString();
    }

    return result;
}

ariopool_submit_result ariopool_client::submit(const string &hash, const string &nonce, const string &public_key) {
    ariopool_submit_result result;
    result.success = false;

    string argon_data = "";
    if(hash.find("$argon2i$v=19$m=16384,t=4,p=4") == 0)
        argon_data = hash.substr(29);
    else
        argon_data = hash.substr(30);

    string __wallet = __get_wallet_address();
    string payload = "argon=" + _encode(argon_data) +
            "&nonce=" + _encode(nonce) +
            "&private_key=" + _encode(__wallet) +
            "&public_key=" + _encode(public_key) +
            "&address=" + _encode(__wallet);

    string url = __pool_address + "/mine.php?q=submitNonce";

    if(__show_pool_requests)
        LOG("--> Pool request: " + url + "/" +payload);

    string response = "";

    for(int i=0;i<2;i++) { //try resubmitting if first submit fails
        response = _http_post(url, payload, "x-www-form-urlencoded");
        result.pool_response = response;
        if(response != "") {
            break;
        }
    }

    if(__show_pool_requests)
        LOG("--> Pool response: " + response);

    if(!__validate_response(response)) {
        LOG("Error connecting to " + __pool_address + ".");
        return result;
    }


    json::JSON info = json::JSON::Load(response);

    result.success = (info["status"].ToString() == "ok");

    return result;
}

bool ariopool_client::__validate_response(const string &response) {
    return !response.empty() && response.find("status") != string::npos && response.find(":null") == string::npos;
}

string ariopool_client::__get_wallet_address() {
    uint64_t minutes = (microseconds() - __timestamp) / 60000000;
    if(minutes != 0 && (minutes % 100 == 0)) {
        if(__used_wallet_address != DEV_WALLET_ADDRESS) {
            LOG("--> Switching to dev wallet for 1 minute.");
            __used_wallet_address = DEV_WALLET_ADDRESS;
            __force_hashrate_report = true;
        }
    }
    else {
        if(__used_wallet_address != __client_wallet_address) {
            LOG("--> Switching back to client wallet.");
            __used_wallet_address = __client_wallet_address;
        }

        if(minutes % 100 == 1) { // force hashrate report one minute after dev fee period
            if(!__first_minute_hashrate) {
                __force_hashrate_report = true;
                __first_minute_hashrate = true;
            }
        }
        else {
            __first_minute_hashrate = false;
        }

        if(minutes % 100 == 99) { // force hashrate report before dev fee period
            if(!__last_minute_hashrate) {
                __force_hashrate_report = true;
                __last_minute_hashrate = true;
            }
        }
        else {
            __last_minute_hashrate = false;
        }
    }

    return __used_wallet_address;
}
