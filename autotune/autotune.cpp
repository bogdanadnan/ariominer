//
// Created by Haifa Bogdan Adnan on 29/08/2018.
//

#include "../common/common.h"

#include "../app/arguments.h"
#include "../hash/hasher.h"

#include "autotune.h"

autotune::autotune(arguments &args) : __args(args) {
    __running = false;
}

autotune::~autotune() { }

void autotune::run() {
    vector<hasher*> all_hashers = hasher::get_hashers();
    vector<hasher*> hashers;
    for(vector<hasher*>::iterator it = all_hashers.begin();it != all_hashers.end();++it) {
        if((*it)->get_type() == "GPU") {
            if((*it)->initialize()) {
                (*it)->configure(__args);
                (*it)->set_input("test_public_key", "test_blk", "test_difficulty", __args.argon2_profile(), "mine");
                LOG("Compute unit: " + (*it)->get_type());
                LOG((*it)->get_info());
                hashers.push_back(*it);
            }
        }
    }

    double best_intensity = 0;
    double best_hashrate = 0;

    __running = true;

    for(double intensity = __args.gpu_intensity_start(); intensity <= __args.gpu_intensity_stop(); intensity += __args.gpu_intensity_step()) {
        if(!__running) {
            break;
        }

        cout << fixed << setprecision(2) <<"Intensity " << intensity << ": " << flush;
        if(__args.argon2_profile() == "1_1_524288") {
            __args.gpu_intensity_cblocks().clear();
            __args.gpu_intensity_cblocks().push_back(intensity);
        }
        else {
            __args.gpu_intensity_gblocks().clear();
            __args.gpu_intensity_gblocks().push_back(intensity);
        }

        for(vector<hasher*>::iterator it = hashers.begin();it != hashers.end();++it) {
            (*it)->cleanup();
            (*it)->initialize();
            (*it)->configure(__args);
        }

        this_thread::sleep_for(chrono::milliseconds(__args.autotune_step_time() * 1000));

        double hashrate = 0;
        for(vector<hasher*>::iterator it = hashers.begin();it != hashers.end();++it) {
            hashrate += (*it)->get_current_hash_rate();
        }

        if(hashrate > best_hashrate) {
            best_hashrate = hashrate;
            best_intensity = intensity;
        }

        cout << fixed << setprecision(2) << hashrate << " h/s" <<endl << flush;
    }

    for(vector<hasher*>::iterator it = hashers.begin();it != hashers.end();++it) {
        (*it)->cleanup();
    }

    cout << fixed << setprecision(2) << "Best intensity is " << best_intensity << ", running at " << best_hashrate << " h/s." << endl;
}

void autotune::stop() {
    cout << endl << "Received termination request, please wait for cleanup ... " << endl;
    __running = false;
}
