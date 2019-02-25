//
// Created by Haifa Bogdan Adnan on 20/02/2019.
//

#include "../common/common.h"

#include "proxy_server.h"
#include "../http/simplejson/json.h"
#include "../http/node_api.h"

proxy_server::proxy_server(vector<string> &options, proxy &prx, arguments &args) : CivetServer(options), __proxy(prx), __args(args),
                                                          __proxy_mine_handler(*this),
                                                          __proxy_api_handler(*this),
                                                          __proxy_base_handler(*this)
{
    addHandler("/mine.php", __proxy_mine_handler);
    addHandler("/api", __proxy_api_handler);
    addHandler("/", __proxy_base_handler);
}

proxy_server::~proxy_server() {

}

proxy &proxy_server::get_proxy() {
    return __proxy;
}

proxy_mine_handler::proxy_mine_handler(proxy_server &server) : __server(server) {

}

bool proxy_mine_handler::handleGet(CivetServer *server, struct mg_connection *conn) {
    mg_printf(conn,
              "HTTP/1.1 200 OK\r\nContent-Type: "
              "application/json\r\nConnection: close\r\n\r\n");

    const mg_request_info *req_info = mg_get_request_info(conn);
    string queryString = "";
    if(req_info != NULL)
        queryString = req_info->query_string;

    string query;
    CivetServer::getParam(queryString.c_str(), queryString.size(), "q", query);

    if(query == "info") {
        return __handleMining(queryString, server, conn, "");
    }

    return false;
}

bool proxy_mine_handler::handlePost(CivetServer *server, struct mg_connection *conn) {
    mg_printf(conn,
              "HTTP/1.1 200 OK\r\nContent-Type: "
              "application/json\r\nConnection: close\r\n\r\n");

    const mg_request_info *req_info = mg_get_request_info(conn);
    string queryString = "";
    if(req_info != NULL)
        queryString = req_info->query_string;

    string query;
    CivetServer::getParam(queryString.c_str(), queryString.size(), "q", query);

    string payload = "";
    while(true) {
        char buffer[1024]; buffer[0] = 0;
        int dlen = mg_read(conn, buffer, sizeof(buffer) - 1);
        if(dlen <= 0)
            break;
        buffer[dlen] = 0;
        payload += buffer;
    }

    if(query == "info") {
        return __handleMining(queryString, server, conn, payload);
    }
    else if(query == "submitNonce") {
        return __handleSubmit(server, conn, payload);
    }

    return false;
}

bool proxy_mine_handler::__handleMining(const string &query, CivetServer *server, struct mg_connection *conn, const string &payload) {
    const mg_request_info *req_info = mg_get_request_info(conn);

    string ip = "<ip unknown>";

    if(req_info != NULL)
        ip = req_info->remote_addr;

    string miner_id;
    CivetServer::getParam(query.c_str(), query.size(), "id", miner_id);

    string miner_name;
    CivetServer::getParam(query.c_str(), query.size(), "worker", miner_name);

    string cblocks_hashrate;
    CivetServer::getParam(query.c_str(), query.size(), "hashrate", cblocks_hashrate);

    string gblocks_hashrate;
    CivetServer::getParam(query.c_str(), query.size(), "hrgpu", gblocks_hashrate);
    if(gblocks_hashrate.empty())
        CivetServer::getParam(query.c_str(), query.size(), "gpuhr", gblocks_hashrate);

    string response = __server.get_proxy().process_info_request(ip, miner_id, miner_name, atof(cblocks_hashrate.c_str()), atof(gblocks_hashrate.c_str()), payload);

    mg_printf(conn, response.c_str());

    return true;
}

bool proxy_mine_handler::__handleSubmit(CivetServer *server, struct mg_connection *conn, const string &payload) {
    const mg_request_info *req_info = mg_get_request_info(conn);

    string ip = "<ip unknown>";

    if(req_info != NULL)
        ip = req_info->remote_addr;

    if(payload.empty())
        return false;

    string decodedPayload;
    CivetServer::urlDecode(payload, decodedPayload);

    string miner_id;
    CivetServer::getParam(decodedPayload.c_str(), decodedPayload.size(), "id", miner_id);
    string miner_name;
    CivetServer::getParam(decodedPayload.c_str(), decodedPayload.size(), "worker", miner_name);
    string argon;
    CivetServer::getParam(decodedPayload.c_str(), decodedPayload.size(), "argon", argon);
    for(int i=0;i<argon.size();i++) { // bugfix for decoding + to ' '
        if(argon[i] == ' ')
            argon[i] = '+';
    }
    string nonce;
    CivetServer::getParam(decodedPayload.c_str(), decodedPayload.size(), "nonce", nonce);
    for(int i=0;i<nonce.size();i++) { // bugfix for decoding + to ' '
        if(nonce[i] == ' ')
            nonce[i] = '+';
    }
    string private_key;
    CivetServer::getParam(decodedPayload.c_str(), decodedPayload.size(), "private_key", private_key);
    string public_key;
    CivetServer::getParam(decodedPayload.c_str(), decodedPayload.size(), "public_key", public_key);
    string address;
    CivetServer::getParam(decodedPayload.c_str(), decodedPayload.size(), "address", address);

    string response = __server.get_proxy().process_submit_request(ip, miner_id, miner_name, argon, nonce, public_key);

    mg_printf(conn, response.c_str());

    return true;
}

proxy_api_handler::proxy_api_handler(proxy_server &server) : __server(server) {

}

bool proxy_api_handler::handleGet(CivetServer *server, struct mg_connection *conn) {
    mg_printf(conn,
              "HTTP/1.1 200 OK\r\nContent-Type: "
              "application/json\r\nConnection: close\r\n\r\n");

    string query;
    CivetServer::getParam(conn, "q", query);

    json::JSON response = json::JSON::Make(json::JSON::Class::Object);

    if(query == "getWorkers") {
        map<string, string> workers = __server.get_proxy().get_workers();
        response = json::JSON::Make(json::JSON::Class::Array);
        for(map<string, string>::iterator iter = workers.begin(); iter != workers.end(); iter++) {
            json::JSON entry = json::JSON::Make(json::JSON::Class::Object);
            entry["worker_id"] = iter->first;
            entry["worker_name"] = iter->second;
            response.append(entry);
        }
    }
    else if(query == "getStatus") {
        string context;
        CivetServer::getParam(conn, "context", context);
        if(context == "global") {
            global_status status = __server.get_proxy().get_global_status();

            response["cblocks_hashrate"] = status.cblocks_hashrate;
            response["gblocks_hashrate"] = status.gblocks_hashrate;
            response["uptime"] = status.uptime;
            response["cblocks_shares"] = status.cblocks_shares;
            response["gblocks_shares"] = status.gblocks_shares;
            response["cblocks_rejects"] = status.cblocks_rejects;
            response["gblocks_rejects"] = status.gblocks_rejects;
            response["workers_count"] = status.workers_count;
            response["current_block"] = status.current_block;
            response["cblocks_dl"] = status.cblocks_dl;
            response["gblocks_dl"] = status.gblocks_dl;
            response["blocks"] = status.blocks;
            response["best_dl"] = status.best_dl;
        }
        else if(!context.empty()) {
            miner_status status = __server.get_proxy().get_worker_status(context);

            response["cblocks_hashrate"] = status.cblocks_hashrate;
            response["gblocks_hashrate"] = status.gblocks_hashrate;
            response["uptime"] = status.uptime;
            response["cblocks_shares"] = status.cblocks_shares;
            response["gblocks_shares"] = status.gblocks_shares;
            response["cblocks_rejects"] = status.cblocks_rejects;
            response["gblocks_rejects"] = status.gblocks_rejects;
            response["devices_count"] = status.devices_count;
            response["blocks"] = status.blocks;
        }
    }
    else if(query == "getBalance") {
        account_balance balance = __server.get_proxy().get_account_balance();
        response["balance"] = balance.amount;
        response["last24"] = balance.last24;
    }
    else if(query == "getGlobalHashrateHistory") {
        response = json::JSON::Make(json::JSON::Class::Array);
        list<miner_hashrate> history;
        __server.get_proxy().get_global_hashrate_history(history);
        for(list<miner_hashrate>::iterator iter = history.begin(); iter != history.end(); iter++) {
            json::JSON entry = json::JSON::Make(json::JSON::Class::Object);
            entry["cblocks_hashrate"] = iter->cblocks_hashrate;
            entry["gblocks_hashrate"] = iter->gblocks_hashrate;
            entry["timestamp"] = iter->timestamp;
            response.append(entry);
        }
    }
    else if(query == "getWorkersList") {
        response = json::JSON::Make(json::JSON::Class::Array);
        vector<miner_list_item> workers;
        __server.get_proxy().get_workers_list(workers);
        for(vector<miner_list_item>::iterator iter = workers.begin(); iter != workers.end(); iter++) {
            json::JSON entry = json::JSON::Make(json::JSON::Class::Object);
            entry["worker_name"] = iter->worker_name;
            entry["cblocks_hashrate"] = iter->cblocks_hashrate;
            entry["gblocks_hashrate"] = iter->gblocks_hashrate;
            entry["uptime"] = iter->uptime;
            response.append(entry);
        }
    }
    else if(query == "getWorkerDevices") {
        string worker_id;
        CivetServer::getParam(conn, "workerId", worker_id);

        if(!worker_id.empty()) {
            response = json::JSON::Make(json::JSON::Class::Array);
            vector<device_details> devices;

            __server.get_proxy().get_worker_devices(worker_id, devices);
            for(vector<device_details>::iterator iter = devices.begin(); iter != devices.end(); iter++) {
                json::JSON entry = json::JSON::Make(json::JSON::Class::Object);
                entry["hasher_name"] = iter->hasher_name;
                entry["device_name"] = iter->device_name;
                entry["cblocks_hashrate"] = iter->cblocks_hashrate;
                entry["gblocks_hashrate"] = iter->gblocks_hashrate;

                response.append(entry);
            }
        }
    }
    else if(query == "getWorkerHashrateHistory") {
        string worker_id;
        CivetServer::getParam(conn, "workerId", worker_id);

        if(!worker_id.empty()) {
            response = json::JSON::Make(json::JSON::Class::Array);
            list<miner_hashrate> history;
            __server.get_proxy().get_worker_hashrate_history(worker_id, history);
            for (list<miner_hashrate>::iterator iter = history.begin(); iter != history.end(); iter++) {
                json::JSON entry = json::JSON::Make(json::JSON::Class::Object);
                entry["cblocks_hashrate"] = iter->cblocks_hashrate;
                entry["gblocks_hashrate"] = iter->gblocks_hashrate;
                entry["timestamp"] = iter->timestamp;
                response.append(entry);
            }
        }
    }

    if(response.IsNull())
        return false;

    mg_printf(conn, response.dump().c_str());

    return true;
}

bool proxy_api_handler::handlePost(CivetServer *server, struct mg_connection *conn) {
    mg_printf(conn,
              "HTTP/1.1 200 OK\r\nContent-Type: "
              "text/html\r\nConnection: close\r\n\r\n");

    string status = "Test data api post";
    mg_printf(conn, status.c_str());

    return true;
}

proxy_base_handler::proxy_base_handler(proxy_server &server) : __server(server) {

}

static map<string, string> mime_types = {
        {"txt", "text/plain"},
        {"js", "text/javascript"},
        {"json", "application/json"},
        {"html", "text/html"},
        {"css", "text/css"},
        {"ico", "image/x-icon"},
        {"jpg", "image/jpeg"},
        {"jpeg", "image/jpeg"},
        {"png", "image/png"},
        {"svg", "image/svg+xml"},
        {"ttf", "application/x-font-ttf"},
        {"eot", "application/vnd.ms-fontobject"},
        {"woff", "application/font-woff"},
        {"map", "application/json"},
        {"scss", "text/x-scss"}
};

bool proxy_base_handler::handleGet(CivetServer *server, struct mg_connection *conn) {
    const mg_request_info *req_info = mg_get_request_info(conn);
    if(req_info == NULL)
        return false;

    string request_uri = req_info->local_uri;

    if(request_uri == "/")
        request_uri = "/index.html";

    request_uri = arguments::get_app_folder() + "/reporting" + request_uri;

    string file;
    size_t last_separator_pos = request_uri.find_last_of("\\/");
    if(last_separator_pos != string::npos) {
        file = request_uri.substr(last_separator_pos + 1);
    }
    else {
        file = request_uri;
    }

    string extension;
    size_t dot_pos = file.find_last_of(".");
    if(dot_pos != string::npos) {
        extension = file.substr(dot_pos + 1);
    }
    else {
        request_uri = arguments::get_app_folder() + "/reporting/index.html";
        extension = "html";
    }

    string content_type = "text/html";

    if(mime_types.find(extension) != mime_types.end()) {
        content_type = mime_types[extension];
    }

    mg_printf(conn,
              "HTTP/1.1 200 OK\r\nContent-Type: "
              "%s\r\nConnection: close\r\n\r\n", content_type.c_str());


    FILE *fl = fopen(request_uri.c_str(), "r");

    if(!fl)
        return false;

    fseek(fl, 0, SEEK_END);
    long len = ftell(fl);
    void *data = malloc(len);
    fseek(fl, 0, SEEK_SET);
    fread(data, 1, len, fl);
    fclose(fl);

    mg_write(conn, data, len);

    return true;
}
