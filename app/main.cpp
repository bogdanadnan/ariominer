//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#include "../common/common.h"
#include "arguments.h"
#include "../miner/miner.h"

int main(int argc, char *argv[]) {
    srand(time(NULL));

    arguments args(argc, argv);

    string args_err;
    if(!args.valid(args_err) || args.is_help()) {
        if(!args_err.empty())
            cout << args_err << endl;
        cout << args.get_help() << endl;
        return 0;
    }

    if(args.is_miner()) {
        miner m(args);
        m.run();
    }

    return 0;
}