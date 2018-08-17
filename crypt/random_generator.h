//
// Created by Haifa Bogdan Adnan on 17/08/2018.
//

#ifndef ARIOMINER_RANDOM_GENERATOR_H
#define ARIOMINER_RANDOM_GENERATOR_H


class random_generator {
public:
    random_generator();
    static random_generator &instance();

    void get_random_data(char *buffer, int length);

private:
    random_device __randomDevice;
    mt19937 __mt19937Gen;
    uniform_int_distribution<> __mt19937Distr;
    mutex __thread_lock;

    static random_generator __instance;
};


#endif //ARIOMINER_RANDOM_GENERATOR_H
