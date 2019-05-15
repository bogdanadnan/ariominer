//
// Created by Haifa Bogdan Adnan on 24/02/2019.
//

#include "node_api.h"
#include "simplejson/json.h"

node_api::node_api(string wallet) {
    __wallet = wallet;
    __last_peer_update = 0;
}

account_balance node_api::get_account_balance() {
    account_balance balance;
    balance.amount = 0;
    balance.last24 = 0;

    string peer_url = __get_peer();

    string response = _http_get(peer_url + "/api.php?q=getBalance&account=" + __wallet);

    if(!response.empty()) {
        json::JSON data = json::JSON::Load(response);
        if(data.JSONType() == json::JSON::Class::Object &&
           data.hasKey("status") &&
           data["status"].ToString() == "ok" &&
           data.hasKey("data")) {
            balance.amount = atof(data["data"].ToString().c_str());
        }
    }
    else {
        return balance;
    }

    time_t timestamp = time(NULL);
    response = _http_get(peer_url + "/api.php?q=getTransactions&account=" + __wallet);

    if(!response.empty()) {
        json::JSON data = json::JSON::Load(response);
        if(data.JSONType() == json::JSON::Class::Object &&
            data.hasKey("status") &&
            data["status"].ToString() == "ok" &&
            data.hasKey("data")) {
            json::JSON info = data["data"];
            if(info.JSONType() == json::JSON::Class::Array) {
                for(int i=0; i < info.length();i++) {
                    json::JSON entry = info[i];
                    if(entry.JSONType() == json::JSON::Class::Object &&
                            entry.hasKey("date") &&
                            entry.hasKey("type") &&
                            entry.hasKey("val")) {
                        time_t date = entry["date"].ToInt();
                        if (timestamp - date < 86400) {
                            string type = entry["type"].ToString();
                            if (type == "mining" || type == "credit") {
                                double amount = atof(entry["val"].ToString().c_str());
                                balance.last24 += amount;
                            }
                        }
                    }
                }
            }
        }
    }

    return balance;
}

string node_api::__get_peer() {
    if(time(NULL) - __last_peer_update > 3600) {
        string result = _http_get("http://api.arionum.com/peers.txt");
        if (!result.empty() && result.find("http://") != string::npos) {
            vector<string> peers;
            stringstream ss(result);
            string to;

            while (getline(ss, to, '\n')) {
                peers.push_back(to);
            }

            __peers_lock.lock();
            __peers = peers;
            __peers_lock.unlock();
        }

        __last_peer_update = time(NULL);
    }

    string peer_url = "";
    __peers_lock.lock();
    if (__peers.size() > 0) {
        int selected_index = rand() % __peers.size();
        peer_url = __peers[selected_index];
    }
    __peers_lock.unlock();

    return peer_url;
}
