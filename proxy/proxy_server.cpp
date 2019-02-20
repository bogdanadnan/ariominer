//
// Created by Haifa Bogdan Adnan on 20/02/2019.
//

#include "../common/common.h"

#include "proxy_server.h"
#include "index_html.h"

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

    string query;
    CivetServer::getParam(conn, "q", query);

    if(query == "info") {
        return __handleMining(server, conn);
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

    if(query == "info") {
        return __handleMining(server, conn);
    }
    else if(query == "submitNonce") {
        return __handleSubmit(server, conn);
    }

    return false;
}

bool proxy_mine_handler::__handleMining(CivetServer *server, struct mg_connection *conn) {
    const mg_request_info *req_info = mg_get_request_info(conn);

    string ip = "<ip unknown>";

    if(req_info != NULL)
        ip = req_info->remote_addr;

    string miner_id;
    CivetServer::getParam(conn, "id", miner_id);

    string miner_name;
    CivetServer::getParam(conn, "worker", miner_name);

    string cblocks_hashrate;
    CivetServer::getParam(conn, "hashrate", cblocks_hashrate);

    string gblocks_hashrate;
    CivetServer::getParam(conn, "hrgpu", gblocks_hashrate);
    if(gblocks_hashrate.empty())
        CivetServer::getParam(conn, "gpuhr", gblocks_hashrate);

    string response = __server.get_proxy().process_info_request(ip, miner_id, miner_name, atof(cblocks_hashrate.c_str()), atof(gblocks_hashrate.c_str()));

    mg_printf(conn, response.c_str());

    return true;
}

bool proxy_mine_handler::__handleSubmit(CivetServer *server, struct mg_connection *conn) {
    const mg_request_info *req_info = mg_get_request_info(conn);

    string ip = "<ip unknown>";

    if(req_info != NULL)
        ip = req_info->remote_addr;

    string payload = "";
    while(true) {
        char buffer[1024]; buffer[0] = 0;
        int dlen = mg_read(conn, buffer, sizeof(buffer) - 1);
        if(dlen <= 0)
            break;
        buffer[dlen] = 0;
        payload += buffer;
    }

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
    string nonce;
    CivetServer::getParam(decodedPayload.c_str(), decodedPayload.size(), "nonce", nonce);
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
              "text/html\r\nConnection: close\r\n\r\n");

    string status = "Test data api get";
    mg_printf(conn, status.c_str());

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

bool proxy_base_handler::handleGet(CivetServer *server, struct mg_connection *conn) {
    mg_printf(conn,
              "HTTP/1.1 200 OK\r\nContent-Type: "
              "text/html\r\nConnection: close\r\n\r\n");

    mg_printf(conn, index_html.c_str());

    return true;
}
