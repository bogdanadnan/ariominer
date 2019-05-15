//
// Created by Haifa Bogdan Adnan on 24/02/2019.
//

#ifndef ARIOMINER_NODE_API_H
#define ARIOMINER_NODE_API_H

#include "../common/common.h"
#include "http.h"

struct account_balance {
    double amount;
    double last24;
};

class node_api : public http {
public:
    node_api(string wallet);

    account_balance get_account_balance();

private:
    string __get_peer();

    string __wallet;

    time_t __last_peer_update;
    mutex __peers_lock;
    vector<string> __peers;
};


#endif //ARIOMINER_NODE_API_H
