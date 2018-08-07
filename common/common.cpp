//
// Created by Haifa Bogdan Adnan on 05/08/2018.
//

#include "common.h"

uint64_t microseconds() {
    struct timeval time;
    gettimeofday(&time, NULL);
    return time.tv_sec * 1000000 + time.tv_usec;
}
