//
// Created by Haifa Bogdan Adnan on 05/08/2018.
//

#include "common.h"

uint64_t microseconds() {
    struct timeval time;
    gettimeofday(&time, NULL);
    return (uint64_t)time.tv_sec * 1000000 + (uint64_t)time.tv_usec;
}
