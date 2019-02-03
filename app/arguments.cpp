//
// Created by Haifa Bogdan Adnan on 04/08/2018.
//

#include "../common/common.h"

#include <getopt.h>

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
            {"force-cpu-optimization", required_argument, NULL, 'o'},
            {"update-interval", required_argument, NULL, 'u'},
            {"report-interval", required_argument, NULL, 'r'},
            {"block-type", required_argument, NULL, 'b'},
            {0, 0, 0, 0}
        };

        int option_index = 0;

        c = getopt_long (argc, argv, "hvm:a:p:w:n:c:o:u:r:b:",
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
                    __cpu_intensity = atof(optarg);
                }
                break;
            case 'b':
                if(strcmp(optarg, "-h") == 0 || strcmp(optarg, "--help") == 0) {
                    __help_flag = 1;
                }
                else {
                    if(strcmp(optarg, "CPU") == 0)
                        __argon2profile = "1_1_524288";
                    else if(strcmp(optarg, "GPU") == 0)
                        __argon2profile = "4_4_16384";
                    else {
                        sprintf(buff, "%s: invalid arguments",
                                argv[0]);
                        __error_message = buff;
                        __error_flag = true;
                    }
                }
                break;
            case 'o':
                if(strcmp(optarg, "-h") == 0 || strcmp(optarg, "--help") == 0) {
                    __help_flag = 1;
                }
                else {
                    if(strcmp(optarg, "REF") == 0)
                        __optimization = "REF";
#if defined(__x86_64__) || defined(_WIN64)
                    else if(strcmp(optarg, "SSE2") == 0)
                        __optimization = "SSE2";
                    else if(strcmp(optarg, "SSSE3") == 0)
                        __optimization = "SSSE3";
                    else if(strcmp(optarg, "AVX2") == 0)
                        __optimization = "AVX2";
                    else if(strcmp(optarg, "AVX512F") == 0)
                        __optimization = "AVX512F";
#elif defined(__arm__)
                    else if(strcmp(optarg, "NEON") == 0)
                        __optimization = "NEON";
#endif
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

    if(__miner_flag == 1) {
        if (__pool.empty()) {
            error = "Pool address is mandatory.";
            return false;
        }

        if (__pool.find("https://") == 0) {
            error = "Only HTTP protocol is allowed for pool connection, HTTPS is not supported.";
            return false;
        }

        if (__wallet.empty()) {
            error = "Wallet is mandatory.";
            return false;
        }

        if (__name.empty()) {
            error = "Worker name is mandatory.";
            return false;
        }

        if (__cpu_intensity < 0 || __cpu_intensity > 100) {
            error = "CPU intensity must be between 0 - disabled and 100 - full load.";
            return false;
        }

        if (__update_interval < 2000000) {
            error = "Pool update interval must be at least 2 sec.";
            return false;
        }

        if (__report_interval < 1000000) {
            error = "Reporting interval must be at least 1 sec.";
            return false;
        }
    }
    else {
        error = "Only miner mode is supported for the moment";
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

double arguments::cpu_intensity() {
    return __cpu_intensity;
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

string arguments::argon2_profile() {
    return __argon2profile;
}

string arguments::get_help() {
    return
            "\nArionum CPU Miner v." ArioMiner_VERSION_MAJOR "." ArioMiner_VERSION_MINOR "." ArioMiner_VERSION_REVISION " - static linux\n"
            "Copyright (C) 2018 Haifa Bogdan Adnan\n"
            "\n"
            "Usage:\n"
            "   - starting in miner mode:\n"
            "       ariominer --mode miner --pool <pool / proxy address> --wallet <wallet address> --name <worker name> --cpu-intensity <intensity>\n"
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
            "   --force-cpu-optimization: miner specific option, what type of CPU optimization to use\n"
#if defined(__x86_64__) || defined(_WIN64)
            "                    values: REF, SSE2, SSSE3, AVX2, AVX512F\n"
#elif defined(__arm__)
            "                    values: REF, NEON\n"
#else
            "                    values: REF\n"
#endif
            "                    this is optional, defaults to autodetect, change only if autodetected one crashes\n"
            "   --block-type: miner specific option, override block type sent by pool\n"
            "                    useful for tunning intensity; values: CPU, GPU\n"
            "                    don't use for regular mining, shares submitted during opposite block type will be rejected\n"
            "   --update-interval: how often should we update mining settings from pool, in seconds\n"
            "                    increasing it will lower the load on pool but will increase rejection rate\n"
            "                    this is optional, defaults to 2 sec and can't be set lower than that\n"
            "   --report-interval: how often should we display mining reports, in seconds\n"
            "                    this is optional, defaults to 10 sec\n"
            "\n"
            "(*) Mining intensity depends on the number of CPU cores and available memory. Full load (100) is dynamically calculated by the application. You can use fractional numbers for better tunning.\n"
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
    __proxy_port = 8088;
    __update_interval = 2000000;
    __report_interval = 10000000;

    __optimization = "";
    __argon2profile = "";

    __error_flag = false;
}

string arguments::__argv_0 = "./";

string arguments::get_app_folder() {
    size_t last_slash = __argv_0.find_last_of("/\\");
	if (last_slash == string::npos)
		return ".";
    string app_folder = __argv_0.substr(0, last_slash);
    if(app_folder.empty()) {
        app_folder = ".";
    }
    return app_folder;
}

vector<string> arguments::__parse_multiarg(const string &arg) {
    string::size_type pos, lastPos = 0, length = arg.length();
    vector<string> tokens;

    while(lastPos < length + 1)
    {
        pos = arg.find_first_of(",", lastPos);
        if(pos == std::string::npos)
        {
            pos = length;
        }

        if(pos != lastPos)
            tokens.push_back(string(arg.c_str()+lastPos,
                                        pos-lastPos ));

        lastPos = pos + 1;
    }

    return tokens;
}
