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
#include <iostream>
#include <sstream>
#include <iomanip>
#include <regex>
#include <random>

#include <thread>
#include <mutex>
#include <chrono>

#include <cmath>

#ifndef _WIN64
#include <unistd.h>
#include <sys/time.h>

#include<sys/socket.h>
#include<netdb.h>
#include<arpa/inet.h>
#include <fcntl.h>
#else
#include <WinSock2.h>
#include <WS2tcpip.h>
#include <win64_compatibility_layer.h>

#define close closesocket
#endif

#include <config.h>

using namespace std;

#define LOG(msg) cout<<msg<<endl

uint64_t microseconds();



#endif //ARIOMINER_COMMON_H
