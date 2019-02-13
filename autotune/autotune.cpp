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
	hasher *selected_hasher = NULL;
	string gpu_optimization;
	if(__args.gpu_optimization().size() > 0)
	    gpu_optimization = __args.gpu_optimization()[0];

	for (vector<hasher*>::iterator it = all_hashers.begin(); it != all_hashers.end(); ++it) {
		if ((*it)->get_type() == "GPU") {
			if ((*it)->initialize()) {
				if (selected_hasher == NULL || selected_hasher->get_priority() < (*it)->get_priority()) {
					selected_hasher = *it;
				}
				if ((*it)->get_subtype() == gpu_optimization) {
					selected_hasher = *it;
					break;
				}
			}
		}
	}
	if (selected_hasher != NULL) {
		selected_hasher->configure(__args);
		selected_hasher->set_input("test_public_key", "test_blk", "test_difficulty", __args.argon2_profile(), "mine");
		LOG("Compute unit: " + selected_hasher->get_type() + " - " + selected_hasher->get_subtype());
		LOG(selected_hasher->get_info());
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

		__args.set_cards_count(0);
		selected_hasher->cleanup();
		selected_hasher->initialize();
		selected_hasher->configure(__args);

        this_thread::sleep_for(chrono::milliseconds(__args.autotune_step_time() * 1000));

        double hashrate = selected_hasher->get_current_hash_rate();

        if(hashrate > best_hashrate) {
            best_hashrate = hashrate;
            best_intensity = intensity;
        }

        cout << fixed << setprecision(2) << hashrate << " h/s" <<endl << flush;
    }

	selected_hasher->cleanup();

    cout << fixed << setprecision(2) << "Best intensity is " << best_intensity << ", running at " << best_hashrate << " h/s." << endl;
}

void autotune::stop() {
    cout << endl << "Received termination request, please wait for cleanup ... " << endl;
    __running = false;
}
