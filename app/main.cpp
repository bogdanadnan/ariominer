//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#include "../common/common.h"
#include "arguments.h"
#include "../miner/miner.h"

int main(int argc, char *argv[]) {
    srand((uint32_t)time(NULL));

    arguments args(argc, argv);

    if(args.is_help()) {
        cout << args.get_help() << endl;
        return 0;
    }

    string args_err;
    if(!args.valid(args_err)) {
        cout << args_err << endl;
        cout << "Type ariominer --help for usage information." << endl;
        return 0;
    }

    if(args.is_miner()) {
        miner m(args);
        m.run();
    }

    return 0;
}