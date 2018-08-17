//
// Created by Haifa Bogdan Adnan on 04/08/2018.
//

#ifndef ARIOMINER_ARGUMENTS_H
#define ARIOMINER_ARGUMENTS_H

class arguments {
public:
    arguments(int argc, char *argv[]);

    bool valid(string &error);

    bool is_help();
    bool is_verbose();
    bool is_miner();
    bool is_proxy();

    string pool();
    string wallet();
    string name();
    int cpu_intensity();
    int gpu_intensity_cblocks();
    int gpu_intensity_gblocks();
    vector<string> gpu_filter();
    int proxy_port();

    int update_interval();
    int report_interval();

    string optimization();

    string get_help();

    static string get_app_folder();

private:
    void __init();
    vector<string> __parse_filter(const string &filter);

    string __error_message;
    bool __error_flag;

    int __help_flag;
    int __verbose_flag;
    int __miner_flag;
    int __proxy_flag;

    string __pool;
    string __wallet;
    string __name;
    int __cpu_intensity;
    int __gpu_intensity_cblocks;
    int __gpu_intensity_gblocks;
    vector<string> __gpu_filter;
    int __proxy_port;
    int __update_interval;
    int __report_interval;

    string __optimization;

    static string __argv_0;
};

#endif //ARIOMINER_ARGUMENTS_H
