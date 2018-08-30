//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#ifndef PROJECT_PROXY_H
#define PROJECT_PROXY_H

#include "../app/runner.h"

class proxy : public runner {
public:
    proxy(arguments &args);
    ~proxy();

    virtual void run();
    virtual void stop();
};


#endif //PROJECT_PROXY_H
