//
// Created by Haifa Bogdan Adnan on 19/02/2019.
//

#ifndef ARIOMINER_DEVFEE_CONFIG_H
#define ARIOMINER_DEVFEE_CONFIG_H

#include "../app/arguments.h"
#include "http.h"

struct pool_settings {
    string wallet;
    string pool_address;
    string pool_version;
    string pool_extensions; // supported extensions until now:
    // Proxy - the pool is actually a proxy to another pool
    // Details - the pool can accept post request for update query with additional details in request body
    // Disconnect - the pool can accept disconnect requests from miner
    bool is_devfee;
};

class pool_settings_provider : public http {
public:
    pool_settings_provider(arguments &args);

    pool_settings &get_user_settings();
    pool_settings &get_dev_settings();
private:
    void __update_devfee_data();

    string __get_devfee_settings_from_url(const string &url);
    string __get_devfee_settings_from_path(const string &path);
    void __save_devfee_settings_to_path(const string &json_data, const string &path);
    bool __process_devfee_json(string devfee_json);

    pool_settings __user_pool_settings;
    pool_settings __dev_pool_settings;

    string __app_name;
    string __app_folder;

    time_t __last_devfee_update;
};


#endif //ARIOMINER_DEVFEE_CONFIG_H
