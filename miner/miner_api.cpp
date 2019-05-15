//
// Created by Haifa Bogdan Adnan on 15/02/2019.
//

#include "miner_api.h"

miner_api::miner_api(arguments & args, miner &miner) : __args(args), __miner(miner) {
    if(__args.enable_api_port() > 0) {
        vector<string> options;
        options.push_back("listening_ports");
        options.push_back(to_string(__args.enable_api_port()));
        __server = new CivetServer(options);
        __server->addHandler("/status", *this);
    }
    else {
        __server = NULL;
    }
}

miner_api::~miner_api() {
    if(__server != NULL) {
        delete __server;
    }
}

bool miner_api::handleGet(CivetServer *server, struct mg_connection *conn) {
    mg_printf(conn,
              "HTTP/1.1 200 OK\r\nContent-Type: "
              "application/json\r\nConnection: close\r\n\r\n");

    string status = __miner.get_status();
    mg_printf(conn, status.c_str());

    return true;
}

