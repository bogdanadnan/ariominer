//
// Created by Haifa Bogdan Adnan on 04/08/2018.
//

#ifndef ARIOMINER_COMMON_H
#define ARIOMINER_COMMON_H

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <string>
#include <vector>
#include <queue>
#include <list>
#include <map>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <regex>
#include <random>

#include <thread>
#include <mutex>
#include <chrono>

#include <cmath>
#include <signal.h>

#include <dlfcn.h>
#include "dllimport.h"

#ifndef _WIN64
#include <unistd.h>
#include <sys/time.h>

#include<sys/socket.h>
#include<netdb.h>
#include<arpa/inet.h>
#include <fcntl.h>
#else
#include <win64.h>
#endif

#include <config.h>

using namespace std;

#define LOG(msg) cout<<msg<<endl<<flush

DLLEXPORT uint64_t microseconds();
DLLEXPORT vector<string> get_files(string folder);
DLLEXPORT bool is_number(const string &s);
DLLEXPORT string generate_uid(size_t length);

#endif //ARIOMINER_COMMON_H
