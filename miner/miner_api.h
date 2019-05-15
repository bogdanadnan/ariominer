//
// Created by Haifa Bogdan Adnan on 15/02/2019.
//

#ifndef ARIOMINER_MINER_API_H
#define ARIOMINER_MINER_API_H

#include "../http/civetweb/CivetServer.h"
#include "../common/common.h"
#include "../app/arguments.h"

#include "miner.h"

class miner_api : public CivetHandler {
public:
    miner_api(arguments &args, miner &miner);
    ~miner_api();

    bool handleGet(CivetServer *server, struct mg_connection *conn);

private:
    CivetServer *__server;
    arguments &__args;
    miner &__miner;
};


#endif //ARIOMINER_MINER_API_H
