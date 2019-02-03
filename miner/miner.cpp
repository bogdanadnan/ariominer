//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#include "../common/common.h"
#include "../app/arguments.h"
#include "../hash/hasher.h"

#include "../crypt/sha512.h"
#include "mini-gmp/mini-gmp.h"

#include "miner.h"

miner::miner(arguments &args) : __args(args), __client(args) {
    __nonce = "";
    __blk = "";
    __difficulty = "";
    __limit = 0;
    __public_key = "";
    __height = 0;
    __found = 0;
    __confirmed = 0;
    __rejected = 0;
    __begin_time = time(NULL);
    __running = false;

    vector<hasher*> hashers = hasher::get_hashers();
    for(vector<hasher*>::iterator it = hashers.begin();it != hashers.end();++it) {
        if((*it)->get_type() == "CPU") {
            (*it)->configure(__args);
        }
        else {
            (*it)->configure(__args);
        }

        LOG("Compute unit: " + (*it)->get_type());
        LOG((*it)->get_info());
    }
    LOG("\n");
}

miner::~miner() {

}

void miner::run() {
    uint64_t last_update, last_report;
    last_update = last_report = 0;

    vector<hasher *> hashers = hasher::get_active_hashers();

    __running = true;

    while (__running) {
        for (vector<hasher *>::iterator it = hashers.begin(); it != hashers.end(); ++it) {
            vector<hash_data> hashes = (*it)->get_hashes();

            for (vector<hash_data>::iterator hash = hashes.begin(); hash != hashes.end(); hash++) {
                if (hash->block != __blk) //the block expired
                    continue;
//                LOG(hash->hash);
                string duration = __calc_duration(hash->base, hash->hash);
                uint64_t result = __calc_compare(duration);
                if (result > 0 && result <= __limit) {
                    if (__args.is_verbose())
                        LOG("--> Submitting nonce: " + hash->nonce + " / " + hash->hash.substr(30));
                    ariopool_submit_result reply = __client.submit(hash->hash, hash->nonce, __public_key);
                    if (reply.success) {
                        if (result <= GOLD_RESULT) {
                            if (__args.is_verbose()) LOG("--> Block found.");
                            __found++;
                        } else {
                            if (__args.is_verbose()) LOG("--> Nonce confirmed.");
                            __confirmed++;
                        }
                    } else {
                        if (__args.is_verbose()) {
                            LOG("--> The nonce did not confirm.");
                            LOG("--> Pool response: ");
                            LOG(reply.pool_response);
                        }
                        __rejected++;
                        if (hash->realloc_flag != NULL)
                            *(hash->realloc_flag) = true;
                    }
                }
            }
        }

        if (microseconds() - last_update > __args.update_interval()) {
            if (__update_pool_data()) {
                for (vector<hasher *>::iterator it = hashers.begin(); it != hashers.end(); ++it) {
                    (*it)->set_input(__public_key, __blk, __difficulty, __argon2profile, __recommendation);
                }
            }
            last_update = microseconds();
        }

        if (microseconds() - last_report > __args.report_interval()) {
            __display_report();
            last_report = microseconds();
        }

        this_thread::sleep_for(chrono::milliseconds(100));
    }

    for (vector<hasher *>::iterator it = hashers.begin(); it != hashers.end(); ++it) {
        (*it)->cleanup();
    }
}

string miner::__calc_duration(const string &base, const string &hash) {
    string combined = base + hash;

    unsigned char *sha512_hash = SHA512::hash((unsigned char*)combined.c_str(), combined.length());
    for (int i = 0; i < 5; i++) {
        unsigned char *tmp = SHA512::hash(sha512_hash, SHA512::DIGEST_SIZE);
        free(sha512_hash);
        sha512_hash = tmp;
    }

    string duration = to_string((int)sha512_hash[10]) + to_string((int)sha512_hash[15]) + to_string((int)sha512_hash[20]) + to_string((int)sha512_hash[23]) +
                      to_string((int)sha512_hash[31]) + to_string((int)sha512_hash[40]) + to_string((int)sha512_hash[45]) + to_string((int)sha512_hash[55]);

    free(sha512_hash);

    for(string::iterator it = duration.begin() ; it != duration.end() ; )
    {
        if( *it == '0' ) it = duration.erase(it) ;
        else break;
    }

    return duration;
}

uint64_t miner::__calc_compare(const string &duration) {
    if(__difficulty.empty()) {
        return -1;
    }

    mpz_t mpzDiff, mpzDuration;
    mpz_t mpzResult;
    mpz_init(mpzResult);
    mpz_init_set_str(mpzDiff, __difficulty.c_str(), 10);
    mpz_init_set_str(mpzDuration, duration.c_str(), 10);

    mpz_tdiv_q(mpzResult, mpzDuration, mpzDiff);

    uint64_t result = (uint64_t)mpz_get_ui(mpzResult);

    mpz_clear (mpzResult);
    mpz_clear (mpzDiff);
    mpz_clear (mpzDuration);

    return result;
}

bool miner::__update_pool_data() {
    vector<hasher*> hashers = hasher::get_active_hashers();

    double hash_rate_cblocks = 0;
    double hash_rate_gblocks = 0;
    for(vector<hasher*>::iterator it = hashers.begin();it != hashers.end();++it) {
        hash_rate_cblocks += (*it)->get_avg_hash_rate_cblocks();
        hash_rate_gblocks += (*it)->get_avg_hash_rate_gblocks();
    }

    ariopool_update_result new_settings = __client.update(hash_rate_cblocks, hash_rate_gblocks);
    if (new_settings.success &&
        (new_settings.block != __blk ||
        new_settings.difficulty != __difficulty ||
        new_settings.limit != __limit ||
        new_settings.public_key != __public_key ||
        new_settings.height != __height)) {
        __blk = new_settings.block;
        __difficulty = new_settings.difficulty;
        __limit = new_settings.limit;
        __public_key = new_settings.public_key;
        __height = new_settings.height;
        __argon2profile = new_settings.argon2profile;
        __recommendation = new_settings.recommendation;

        if(__args.is_verbose()) {
            stringstream ss;
            ss << "-----------------------------------------------------------------------------------------------------------------------------------------" << endl;
            ss << "--> Pool data updated   Block: " << __blk << endl;
            ss << "--> " << ((new_settings.argon2profile == "1_1_524288") ? "CPU round" : (new_settings.recommendation == "pause" ? "Masternode round" : "GPU round"));
            ss << "  Height: " << __height << "  Limit: " << __limit << "  Difficulty: " << __difficulty << endl;
            ss << "-----------------------------------------------------------------------------------------------------------------------------------------";

            LOG(ss.str());
        }
        return true;
    }

    return false;
}

void miner::__display_report() {
    vector<hasher*> hashers = hasher::get_active_hashers();
    stringstream ss;

    double hash_rate = 0;
    double avg_hash_rate_cblocks = 0;
    double avg_hash_rate_gblocks = 0;
    uint32_t hash_count_cblocks = 0;
    uint32_t hash_count_gblocks = 0;

    time_t total_time = time(NULL) - __begin_time;

    if(!__args.is_verbose()) {
        for (vector<hasher *>::iterator it = hashers.begin(); it != hashers.end(); ++it) {
            hash_rate += (*it)->get_current_hash_rate();
            avg_hash_rate_cblocks += (*it)->get_avg_hash_rate_cblocks();
            hash_count_cblocks += (*it)->get_hash_count_cblocks();
            avg_hash_rate_gblocks += (*it)->get_avg_hash_rate_gblocks();
            hash_count_gblocks += (*it)->get_hash_count_gblocks();
        }

        ss << fixed << setprecision(2) << "--> Hash Rate: " << setw(6) << hash_rate << " H/s   " <<
           "Avg. (Cblocks): " << setw(6) << avg_hash_rate_cblocks << " H/s  " <<
           "Avg. (Gblocks): " << setw(6) << avg_hash_rate_gblocks << " H/s  " <<
           "Count: " << setw(4) << (hash_count_cblocks + hash_count_gblocks) << "  " <<
           "Time: " << setw(4) << total_time << "  " <<
           "Shares: " << setw(3) << __confirmed << " " <<
           "Finds: " << setw(3) << __found << " " <<
           "Rejected: " << setw(3) << __rejected;
    }
    else {
        ss << fixed << setprecision(2) << "--> Time: " << setw(4) << total_time << "  " <<
           "Shares: " << setw(3) << __confirmed << " " <<
           "Finds: " << setw(3) << __found << " " <<
           "Rejected: " << setw(3) << __rejected << endl;
        for (vector<hasher *>::iterator it = hashers.begin(); it != hashers.end(); ++it) {
            if((*it)->get_intensity() == 0) continue;
            hash_rate += (*it)->get_current_hash_rate();
            avg_hash_rate_cblocks += (*it)->get_avg_hash_rate_cblocks();
            hash_count_cblocks += (*it)->get_hash_count_cblocks();
            avg_hash_rate_gblocks += (*it)->get_avg_hash_rate_gblocks();
            hash_count_gblocks += (*it)->get_hash_count_gblocks();

            ss << fixed << setprecision(2) << "--> " << (*it)->get_type() << "  " <<
               "Hash rate: " << setw(6)<< (*it)->get_current_hash_rate() << " H/s   " <<
               "Avg. (Cblocks): " << setw(6) << (*it)->get_avg_hash_rate_cblocks() << " H/s  " <<
               "Avg. (Gblocks): " << setw(6) << (*it)->get_avg_hash_rate_gblocks() << "  " <<
               "Count: " << setw(4) << ((*it)->get_hash_count_cblocks() + (*it)->get_hash_count_gblocks());

            if(hashers.size() > 1)
                ss << endl;
        }
        if(hashers.size() > 1) {
            ss << fixed << setprecision(2) << "--> ALL  " <<
               "Hash rate: " << setw(6) << hash_rate << " H/s   " <<
               "Avg. (Cblocks): " << setw(6) << avg_hash_rate_cblocks << " H/s  " <<
               "Avg. (Gblocks): " << setw(6) << avg_hash_rate_gblocks << "  " <<
               "Count: " << setw(4) << (hash_count_cblocks + hash_count_gblocks);
        }
    }

    LOG(ss.str());
}

void miner::stop() {
    cout << endl << "Received termination request, please wait for cleanup ... " << endl;
    __running = false;
}
