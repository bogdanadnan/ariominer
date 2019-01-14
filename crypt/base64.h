//
// Created by Haifa Bogdan Adnan on 17/08/2018.
//

#ifndef ARIOMINER_BASE64_H
#define ARIOMINER_BASE64_H

class DLLEXPORT base64 {
public:
    static void encode(const char *input, int input_size, char *output);
};

#endif //ARIOMINER_BASE64_H
