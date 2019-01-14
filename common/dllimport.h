//
// Created by Haifa Bogdan Adnan on 04.11.2018.
//

#ifndef ARIOMINER_DLLIMPORT_H
#define ARIOMINER_DLLIMPORT_H

#ifndef DLLEXPORT
    #ifndef _WIN64
        #define DLLEXPORT
    #else
        #define DLLEXPORT __declspec(dllimport)
    #endif
#endif

#endif //ARIOMINER_DLLIMPORT_H
