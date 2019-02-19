//
// Created by Haifa Bogdan Adnan on 19/02/2019.
//

#include "../common/common.h"
#include "../common/cfgpath.h"
#include "../http/simplejson/json.h"

#include "pool_settings_provider.h"

#define DEV_WALLET_ADDRESS      "3ykJiMsURozMLgazT97A5iidWiPLRvY5CQW9ziFJcJAZNJ9AjZimSUQe8nfwQTJqukch2JXEF48sLdoFqzKB9FVL"
#define DEV_SETTINGS_URL        "http://coinfee.changeling.biz/index.json"

pool_settings_provider::pool_settings_provider(arguments &args) {
    __last_devfee_update = 0;

    __user_pool_settings.wallet = args.wallet();
    __user_pool_settings.pool_address = args.pool();
    __user_pool_settings.is_devfee = false;
    __user_pool_settings.pool_extensions = "";

    __app_name = args.get_app_name();
    __app_folder = args.get_app_folder();

    __update_devfee_data();
}

pool_settings &pool_settings_provider::get_user_settings() {
    __update_devfee_data();
    return __user_pool_settings;
}

pool_settings &pool_settings_provider::get_dev_settings() {
    __update_devfee_data();
    return __dev_pool_settings;
}

void pool_settings_provider::__update_devfee_data() {
    time_t timestamp = time(NULL);
    if(timestamp - __last_devfee_update > 3600) { // update devfee settings each hour
        bool external_data = false;

        string json_data = __get_devfee_settings_from_url(DEV_SETTINGS_URL);

        if(json_data.empty() || !__process_devfee_json(json_data)) {
            char buff[256]; buff[0] = 0;
            get_user_config_folder(buff, 256, __app_name.c_str());
            if(strlen(buff) > 0) {
                string dev_settings_path = buff;
                dev_settings_path += "devfee.json";
                json_data = __get_devfee_settings_from_path(dev_settings_path);
            }

            if(json_data.empty() || !__process_devfee_json(json_data)) {
                string dev_settings_path = __app_folder;
                dev_settings_path += "/devfee.json";
                json_data = __get_devfee_settings_from_path(dev_settings_path);

                if (json_data.empty() || !__process_devfee_json(json_data)) {
                    __dev_pool_settings.wallet = DEV_WALLET_ADDRESS;

                    if (__user_pool_settings.pool_address != __dev_pool_settings.pool_address) {
                        __dev_pool_settings.pool_address = __user_pool_settings.pool_address;
                        __dev_pool_settings.pool_version = __user_pool_settings.pool_version;
                        __dev_pool_settings.pool_extensions = __user_pool_settings.pool_extensions;
                    }
                }
            }
        }
        else {
            external_data = true;
        }

        if(external_data) {
            json_data = string("{ \"arionum\": [ { \"address\": \"") + __dev_pool_settings.wallet + string("\", \"pool\": \"") + __dev_pool_settings.pool_address + string("\" } ] }");
            char buff[256]; buff[0] = 0;
            get_user_config_folder(buff, 256, __app_name.c_str());
            if(strlen(buff) > 0) {
                string dev_settings_path = buff;
                dev_settings_path += "devfee.json";
                __save_devfee_settings_to_path(json_data, dev_settings_path);
            }
        }
        __last_devfee_update = timestamp;
    }
}

string pool_settings_provider::__get_devfee_settings_from_url(const string &url) {
    return _http_get(url);
}

string pool_settings_provider::__get_devfee_settings_from_path(const string &path) {
    string json_data;
    ifstream inFile(path);
    if(inFile.is_open()) {
        while (inFile.good())
        {
            string line;
            getline(inFile, line);
            json_data += line;
        }
        inFile.close();
    }
    return json_data;
}

void pool_settings_provider::__save_devfee_settings_to_path(const string &json_data, const string &path) {
    ofstream outFile(path);
    if(outFile.is_open()) {
        outFile << json_data<<endl;
    }
    outFile.close();
}

bool pool_settings_provider::__process_devfee_json(string devfee_json) {
    json::JSON info = json::JSON::Load(devfee_json);

    if(info.hasKey("arionum")) {
        json::JSON &devfee_data = info["arionum"];
        if(devfee_data.length() > 0) {
            unsigned int size = devfee_data.length();
            unsigned int idx = 0;
            if (size > 1)
                idx = rand() % size; // choose a random one
            json::JSON &devfee_entry = devfee_data[idx];

            if(devfee_entry.hasKey("address") && devfee_entry.hasKey("pool")) {
                __dev_pool_settings.wallet = devfee_entry["address"].ToString();

                string pool_address = devfee_entry["pool"].ToString();
                if(pool_address != __dev_pool_settings.pool_address) {
                    __dev_pool_settings.pool_address = pool_address;
                    __dev_pool_settings.pool_version = "";
                    __dev_pool_settings.pool_extensions = "";
                }
                return true;
            }
        }
    }

    return false;
}
