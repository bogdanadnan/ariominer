//
// Created by Haifa Bogdan Adnan on 17/08/2018.
//

#include "../common/dllexport.h"
#include "../common/common.h"

#include "random_generator.h"

random_generator::random_generator() : __mt19937Gen(__randomDevice()), __mt19937Distr(0, 255) {

}

random_generator &random_generator::instance() {
    return __instance;
}

void random_generator::get_random_data(char *buffer, int length) {
//    __thread_lock.lock();
    for(int i=0;i<length;i++) {
        buffer[i] = (char)__mt19937Distr(__mt19937Gen);
    }
//    __thread_lock.unlock();
}


random_generator random_generator::__instance;