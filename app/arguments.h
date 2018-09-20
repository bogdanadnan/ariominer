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
    bool is_autotune();
    bool is_proxy();

    string pool();
    string wallet();
    string name();
    double cpu_intensity();
    vector<double> &gpu_intensity_cblocks();
    vector<double> &gpu_intensity_gblocks();
    vector<string> gpu_filter();
    int proxy_port();
    string argon2_profile();

    double gpu_intensity_start();
    double gpu_intensity_stop();
    double gpu_intensity_step();
    int autotune_step_time();

    int update_interval();
    int report_interval();

	string cpu_optimization();
	string gpu_optimization();

    string get_help();

    static string get_app_folder();

private:
    void __init();
    vector<string> __parse_multiarg(const string &arg);

    string __error_message;
    bool __error_flag;

    int __help_flag;
    int __verbose_flag;
    int __miner_flag;
    int __proxy_flag;
    int __autotune_flag;

    string __pool;
    string __wallet;
    string __name;
    double __cpu_intensity;
    vector<double> __gpu_intensity_cblocks;
    vector<double> __gpu_intensity_gblocks;
    vector<string> __gpu_filter;
    int __proxy_port;
    int __update_interval;
    int __report_interval;

    double __gpu_intensity_start;
    double __gpu_intensity_stop;
    double __gpu_intensity_step;
    int __autotune_step_time;

    string __argon2profile;

	string __cpu_optimization;
	string __gpu_optimization;

    static string __argv_0;
};

#endif //ARIOMINER_ARGUMENTS_H
