//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#include "../common/common.h"
#include "arguments.h"
#include "runner.h"
#include "../miner/miner.h"
#include "../proxy/proxy.h"

runner *main_app = NULL;

void shutdown(int s){
    if(main_app != NULL) {
        main_app->stop();
    }
}

int main(int argc, char *argv[]) {
    srand((uint32_t)time(NULL));
    struct sigaction sigIntHandler;

    sigIntHandler.sa_handler = shutdown;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;

    sigaction(SIGINT, &sigIntHandler, NULL);

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
        main_app = &m;
        m.run();
    }
    else if(args.is_proxy()) {
        proxy p(args);
        main_app = &p;
        p.run();
    }

    return 0;
}
