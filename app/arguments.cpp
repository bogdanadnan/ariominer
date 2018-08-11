//
// Created by Haifa Bogdan Adnan on 04/08/2018.
//

#include "../common/common.h"

#include "arguments.h"

arguments::arguments(int argc, char **argv) {
    __argv_0 = argv[0];

    __init();

    int c = 0;
    char buff[50];

    if(argc < 2) {
        __help_flag = true;
        return;
    }

    while (c != -1)
    {
        static struct option options[] =
        {
            {"help", no_argument,  NULL, 'h'},
            {"verbose", no_argument, NULL, 'v'},
            {"mode", required_argument, NULL, 'm'},
            {"pool", required_argument, NULL, 'a'},
            {"port", required_argument, NULL, 'p'},
            {"wallet", required_argument, NULL, 'w'},
            {"name", required_argument, NULL, 'n'},
            {"cpu-intensity", required_argument, NULL, 'c'},
            {"gpu-intensity", required_argument, NULL, 'g'},
            {"force-cpu-optimization", required_argument, NULL, 'o'},
            {"update-interval", required_argument, NULL, 'u'},
            {"report-interval", required_argument, NULL, 'r'},
            {0, 0, 0, 0}
        };

        int option_index = 0;

        c = getopt_long (argc, argv, "hvm:a:p:w:n:c:g:o:u:r:",
                         options, &option_index);

        switch (c)
        {
            case -1:
            case 0:
                break;
            case 1:
                sprintf(buff, "%s: invalid arguments",
                                  argv[0]);
                __error_message = buff;
                __error_flag = true;
                c = -1;
                break;
            case 'h':
                __help_flag = 1;
                break;
            case 'v':
                __verbose_flag = 1;
                break;
            case 'm':
                if(strcmp(optarg, "-h") == 0 || strcmp(optarg, "--help") == 0) {
                    __help_flag = 1;
                }
                else {
                    if(strcmp(optarg, "miner") == 0)
                        __miner_flag = 1;
                    else if(strcmp(optarg, "proxy") == 0)
                        __proxy_flag = 1;
                    else {
                        sprintf(buff, "%s: invalid arguments",
                                argv[0]);
                        __error_message = buff;
                        __error_flag = true;
                    }
                }
                break;
            case 'a':
                if(strcmp(optarg, "-h") == 0 || strcmp(optarg, "--help") == 0) {
                    __help_flag = 1;
                }
                else {
                    __pool = optarg;
                }
                break;
            case 'p':
                if(strcmp(optarg, "-h") == 0 || strcmp(optarg, "--help") == 0) {
                    __help_flag = 1;
                }
                else {
                    __proxy_port = atoi(optarg);
                }
                break;
            case 'w':
                if(strcmp(optarg, "-h") == 0 || strcmp(optarg, "--help") == 0) {
                    __help_flag = 1;
                }
                else {
                    __wallet = optarg;
                }
                break;
            case 'n':
                if(strcmp(optarg, "-h") == 0 || strcmp(optarg, "--help") == 0) {
                    __help_flag = 1;
                }
                else {
                    __name = optarg;
                }
                break;
            case 'c':
                if(strcmp(optarg, "-h") == 0 || strcmp(optarg, "--help") == 0) {
                    __help_flag = 1;
                }
                else {
                    __cpu_intensity = atoi(optarg);
                }
                break;
            case 'g':
                if(strcmp(optarg, "-h") == 0 || strcmp(optarg, "--help") == 0) {
                    __help_flag = 1;
                }
                else {
                    __gpu_intensity = atoi(optarg);
                }
                break;
            case 'o':
                if(strcmp(optarg, "-h") == 0 || strcmp(optarg, "--help") == 0) {
                    __help_flag = 1;
                }
                else {
                    if(strcmp(optarg, "REF") == 0)
                        __optimization = "REF";
                    else if(strcmp(optarg, "SSE2") == 0)
                        __optimization = "SSE2";
                    else if(strcmp(optarg, "SSSE3") == 0)
                        __optimization = "SSSE3";
                    else if(strcmp(optarg, "AVX2") == 0)
                        __optimization = "AVX2";
                    else if(strcmp(optarg, "AVX512F") == 0)
                        __optimization = "AVX512F";
                    else {
                        sprintf(buff, "%s: invalid arguments",
                                argv[0]);
                        __error_message = buff;
                        __error_flag = true;
                    }
                }
                break;
            case 'u':
                if(strcmp(optarg, "-h") == 0 || strcmp(optarg, "--help") == 0) {
                    __help_flag = 1;
                }
                else {
                    __update_interval = 1000000 * atoi(optarg);
                }
                break;
            case 'r':
                if(strcmp(optarg, "-h") == 0 || strcmp(optarg, "--help") == 0) {
                    __help_flag = 1;
                }
                else {
                    __report_interval = 1000000 * atoi(optarg);
                }
                break;
            case ':':
                __error_flag = true;
                break;
            default:
                __error_flag = true;
                break;
        }
    }

    if (optind < argc)
    {
        sprintf(buff, "%s: invalid arguments",
                          argv[0]);
        __error_message = buff;
        __error_flag = true;
    }
}

bool arguments::valid(string &error) {
    error = __error_message;

    if(__error_flag)
        return false;

    if(__miner_flag == false) {
        error = "Only miner mode supported for the moment";
        return false;
    }

    if(__pool.empty()) {
        error = "Pool address is mandatory.";
        return false;
    }

    if(__wallet.empty()) {
        error = "Wallet is mandatory.";
        return false;
    }

    if(__name.empty()) {
        error = "Worker name is mandatory.";
        return false;
    }

    if(__cpu_intensity < 0 || __cpu_intensity > 100) {
        error = "CPU intensity must be between 0 - disabled and 100 - full load.";
        return false;
    }

    if(__gpu_intensity < 0 || __gpu_intensity > 100) {
        error = "GPU intensity must be between 0 - disabled and 100 - full load.";
        return false;
    }

    if(__update_interval < 2000000) {
        error = "Pool update interval must be at least 2 sec.";
        return false;
    }

    if(__report_interval < 1000000) {
        error = "Reporting interval must be at least 1 sec.";
        return false;
    }

    return true;
}

bool arguments::is_help() {
    return __help_flag == 1;
}

bool arguments::is_verbose() {
    return __verbose_flag == 1;
}

bool arguments::is_miner() {
    return __miner_flag == 1;
}

bool arguments::is_proxy() {
    return __proxy_flag == 1;
}

int arguments::proxy_port() {
    return __proxy_port;
}

string arguments::pool() {
    return __pool;
}

string arguments::wallet() {
    return __wallet;
}

string arguments::name() {
    return __name;
}

int arguments::cpu_intensity() {
    return __cpu_intensity;
}

int arguments::gpu_intensity() {
    return __gpu_intensity;
}

string arguments::optimization() {
    return __optimization;
}

int arguments::update_interval() {
    return __update_interval;
}

int arguments::report_interval() {
    return __report_interval;
}

string arguments::get_help() {
    return
            "\nArionum CPU/GPU Miner v." ArioMiner_VERSION_MAJOR "." ArioMiner_VERSION_MINOR "\n"
            "Copyright (C) 2018 Haifa Bogdan Adnan\n"
            "\n"
            "Usage:\n"
            "   - starting in miner mode:\n"
            "       ariominer --mode miner --pool <pool / proxy address> --wallet <wallet address> --name <worker name> --cpu-intensity <intensity> --gpu-intensity <intensity>"
            "   - starting in proxy mode:\n"
            "       ariominer --mode proxy --port <proxy port> --pool <pool address> --wallet <wallet address> --name <proxy name>\n"
            "\n"
            "Parameters:\n"
            "   --help: show this help text\n"
            "   --verbose: print more informative text during run\n"
            "   --mode <mode>: start in specific mode - arguments: miner / proxy\n"
            "           - miner: this instance will mine for arionum\n"
            "           - proxy: this instance will act as a hub for multiple miners,\n"
            "                    useful to aggregate multiple miners into a single instance\n"
            "                    reducing the load on the pool\n"
            "   --pool <pool address>: pool/proxy address to connect to (eg. http://aropool.com:80)\n"
            "   --wallet <wallet address>: wallet address\n"
            "                    this is optional if in miner mode and you are connecting to a proxy\n"
            "   --name <worker identifier>: worker identifier\n"
            "                    this is optional if in miner mode and you are connecting to a proxy\n"
            "   --port <proxy port>: proxy specific option, port on which to listen for clients\n"
            "                    this is optional, defaults to 8088\n"
            "   --cpu-intensity: miner specific option, mining intensity on CPU\n"
            "                    value from 0 (disabled) to 100 (full load)\n"
            "                    this is optional, defaults to 100 (*)\n"
            "   --gpu-intensity: miner specific option, mining intensity on GPU\n"
            "                    value from 0 (disabled) to 100 (full load)\n"
            "                    this is optional, defaults to 80 (*)\n"
            "   --force-cpu-optimization: miner specific option, what type of CPU optimization to use\n"
            "                    values: REF, SSE2, SSSE3, AVX2, AVX512F\n"
            "                    this is optional, defaults to autodetect, change only if autodetected one crashes\n"
            "   --update-interval: how often should we update mining settings from pool, in seconds\n"
            "                    increasing it will lower the load on pool but will increase rejection rate\n"
            "                    this is optional, defaults to 2 sec and can't be set lower than that\n"
            "   --report-interval: how often should we display mining reports, in seconds\n"
            "                    this is optional, defaults to 10 sec\n"
            "\n"
            "(*) Mining intensity depends on the number of CPU/GPU cores and available memory. Full load (100) is dynamically calculated by the application.\n"
            ;
}

void arguments::__init() {
    __help_flag = 0;
    __verbose_flag = 0;
    __miner_flag = 0;
    __proxy_flag = 0;

    __pool = "";
    __wallet = "";
    __name = "";
    __cpu_intensity = 100;
    __gpu_intensity = 80;
    __proxy_port = 8088;
    __update_interval = 2000000;
    __report_interval = 10000000;

    __optimization = "";

    __error_flag = false;
}

string arguments::__argv_0 = "./";

string arguments::get_app_folder() {
    size_t last_slash = __argv_0.find_last_of("/\\");
    string app_folder = __argv_0.substr(0, last_slash);
    if(app_folder.empty()) {
        app_folder = ".";
    }
    return app_folder;
}
