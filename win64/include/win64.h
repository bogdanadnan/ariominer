//
// Created by Haifa Bogdan Adnan on 13/08/2018.
//

#ifndef ARIOMINER_WIN64_COMPATIBILITY_LAYER_H
#define ARIOMINER_WIN64_COMPATIBILITY_LAYER_H

//#include <windows.h>
#include <WinSock2.h>
#include <WS2tcpip.h>

#ifdef __cplusplus 
extern "C" {
#endif
struct timezone
{
	int  tz_minuteswest; /* minutes W of Greenwich */
	int  tz_dsttime;     /* type of dst correction */
};

int gettimeofday(struct timeval *tv, struct timezone *tz);
#ifdef __cplusplus
}
#endif

#endif //ARIOMINER_WIN32_COMPATIBILITY_LAYER_H
