//
// Created by Haifa Bogdan Adnan on 20/02/2019.
//

#ifndef ARIOMINER_PROXY_SERVER_H
#define ARIOMINER_PROXY_SERVER_H

#include "../http/civetweb/CivetServer.h"
#include "../app/arguments.h"
#include "proxy.h"

class proxy_server;

class proxy_mine_handler : public CivetHandler {
public:
    proxy_mine_handler(proxy_server &server);
    bool handleGet(CivetServer *server, struct mg_connection *conn);
    bool handlePost(CivetServer *server, struct mg_connection *conn);
private:
    bool __handleMining(const string &query, CivetServer *server, struct mg_connection *conn, const string &payload);
    bool __handleSubmit(CivetServer *server, struct mg_connection *conn, const string &payload);
    proxy_server &__server;
};

class proxy_api_handler : public CivetHandler {
public:
    proxy_api_handler(proxy_server &server);
    bool handleGet(CivetServer *server, struct mg_connection *conn);
    bool handlePost(CivetServer *server, struct mg_connection *conn);
private:
    proxy_server &__server;
};

class proxy_base_handler : public CivetHandler {
public:
    proxy_base_handler(proxy_server &server);
    bool handleGet(CivetServer *server, struct mg_connection *conn);
private:
    proxy_server &__server;
};

class proxy_server : public CivetServer {
public:
    proxy_server(vector<string> &options, proxy &prx, arguments &args);
    ~proxy_server();

    proxy &get_proxy();
private:
    proxy &__proxy;
    arguments &__args;

    proxy_mine_handler __proxy_mine_handler;
    proxy_api_handler __proxy_api_handler;
    proxy_base_handler __proxy_base_handler;
};

#endif //ARIOMINER_PROXY_SERVER_H
