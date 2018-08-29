//
// Created by Haifa Bogdan Adnan on 29/08/2018.
//

#ifndef ARIOMINER_AUTOTUNE_H
#define ARIOMINER_AUTOTUNE_H


class autotune {
public:
    autotune(arguments &args);
    ~autotune();

    void run();
private:
    arguments &__args;
};


#endif //ARIOMINER_AUTOTUNE_H
