//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#if defined(__x86_64__) || defined(__i386__) || defined(_WIN64)
    #include <cpuinfo_x86.h>
#endif
#if defined(__arm__)
    #include <cpuinfo_arm.h>
#endif


#include "../../common/common.h"
#include "../../app/arguments.h"

#include "../hasher.h"
#include "../argon2/argon2.h"

#include "cpu_hasher.h"

cpu_hasher::cpu_hasher() : hasher() {
    _type = "CPU";
    __optimization = "REF";
    __available_processing_thr = 1;
    __available_memory_thr = 1;
    __threads_count = 0;
    __running = false;
    __argon2_blocks_filler_ptr = NULL;
    _description = __detect_features_and_make_description();
}

cpu_hasher::~cpu_hasher() {
    this->cleanup();
}

bool cpu_hasher::configure(arguments &args) {
    double intensity = args.cpu_intensity();
    if(args.optimization() != "") {
        _description += "Overiding detected optimization feature with " + args.optimization() + ".\n";
        __optimization = args.optimization();
    }

    __load_argon2_block_filler();

    if(__argon2_blocks_filler_ptr == NULL) {
        _intensity = 0;
        _description += "Status: DISABLED - argon2 hashing module not found.";
        return false;
    }

    __threads_count = min(__available_processing_thr, __available_memory_thr);

    if (__threads_count == 0) {
        _intensity = 0;
        _description += "Status: DISABLED - not enough resources.";
        return false;
    }

    if (intensity == 0) {
        _intensity = 0;
        _description += "Status: DISABLED - by user.";
        return false;
    }

    _intensity = intensity;

    __threads_count = __threads_count * _intensity / 100;
    if (__threads_count == 0)
        __threads_count = 1;

    __running = true;
    for(int i=0;i<__threads_count;i++) {
		__runners.push_back(new thread([&]() { this->__run(); }));
    }

    _description += "Status: ENABLED - with " + to_string(__threads_count) + " threads.";

    return true;
}

string cpu_hasher::__detect_features_and_make_description() {
    stringstream ss;
#if defined(__x86_64__) || defined(__i386__) || defined(_WIN64)
    char brand_string[49];
    cpu_features::FillX86BrandString(brand_string);

    ss << brand_string << endl;

    cpu_features::X86Features features = cpu_features::GetX86Info().features;
    ss << "Optimization features: ";

#if defined(__x86_64__) || defined(_WIN64)
    ss << "SSE2 ";
    __optimization = "SSE2";
#else
    __optimization = "REF";
#endif

    if(features.ssse3 || features.avx2 || features.avx512f) {
        if (features.ssse3) {
            ss << "SSSE3 ";
            __optimization = "SSSE3";
        }
        if (features.avx2) {
            ss << "AVX2 ";
            __optimization = "AVX2";
        }
        if (features.avx512f) {
            ss << "AVX512F ";
            __optimization = "AVX512F";
        }
    }
    else {
        ss << "none";
    }
    ss << endl;
#endif
#if defined(__arm__)
    cpu_features::ArmFeatures features = cpu_features::GetArmInfo().features;
    ss << "ARM processor" << endl;
    ss << "Optimization features: ";

    __optimization = "REF";

    if(features.neon) {
        ss << "NEON";
        __optimization = "NEON";
    }
    else {
        ss << "none";
    }
    ss << endl;
#endif
    ss << "Selecting " << __optimization << " as candidate for hashing algorithm." << endl;

    __available_processing_thr = thread::hardware_concurrency();
    ss << "Parallelism: " << __available_processing_thr << " concurent threads supported." << endl;

    //check available memory
    vector<void *> memory_test;
    for(__available_memory_thr = 0;__available_memory_thr < __available_processing_thr;__available_memory_thr++) {
        void *memory = malloc(argon2profile_default->memsize + 64); //64 bytes for alignament - to work on AVX512F optimisations
        if(memory == NULL)
            break;
        memory_test.push_back(memory);
    }
    for(vector<void*>::iterator it=memory_test.begin();it != memory_test.end();++it) {
        free(*it);
    }
    ss << "Memory: there is enough memory for " << __available_memory_thr << " concurent threads." << endl;

    return ss.str();
}

void cpu_hasher::__run() {
    void *buffer = NULL;
    void *mem = __allocate_memory(buffer);
    if(mem == NULL) {
        LOG("Error allocating memory");
        return;
    }

    argon2 hash_factory(__argon2_blocks_filler_ptr, mem, NULL);

    bool should_realloc = false;

    while(__running) {
        if(should_pause()) {
            this_thread::sleep_for(chrono::milliseconds(100));
            continue;
        }

        if(should_realloc) {
            void *new_buffer;
            mem = __allocate_memory(new_buffer);
            if(mem == NULL) {
                free(buffer);
                LOG("Error allocating memory");
                return;
            }
            hash_factory.set_seed_memory((uint8_t *)mem);
            free(buffer);
            buffer = new_buffer;
            should_realloc = false;
        }

        hash_data input = get_input();
        argon2profile *profile = get_argon2profile();

        if(!input.base.empty()) {
            hash_factory.set_seed_memory_offset(profile->memsize);
            hash_factory.set_threads((int)(argon2profile_default->memsize / profile->memsize));

            vector<string> hashes = hash_factory.generate_hashes(*profile, input.base, input.salt);
            for(vector<string>::iterator it = hashes.begin(); it != hashes.end(); ++it) {
                input.hash = *it;
                input.realloc_flag = &should_realloc;
                _store_hash(input);
            }
        }
    }

    free(buffer);
}

void *cpu_hasher::__allocate_memory(void *&buffer) {
    size_t mem_size = argon2profile_default->memsize + 64;
    void *mem = malloc(mem_size);
    buffer = mem;
    return align(64, argon2profile_default->memsize, mem, mem_size);
}

extern "C" {
#if defined(__x86_64__) || defined(_WIN64)
void *fill_memory_blocks_REF(void *memory, int threads, argon2profile *profile, void *user_data);
void *fill_memory_blocks_SSE2(void *memory, int threads, argon2profile *profile, void *user_data);
void *fill_memory_blocks_SSSE3(void *memory, int threads, argon2profile *profile, void *user_data);
void *fill_memory_blocks_AVX2(void *memory, int threads, argon2profile *profile, void *user_data);
void *fill_memory_blocks_AVX512F(void *memory, int threads, argon2profile *profile, void *user_data);
#elif defined(__arm__)
void *fill_memory_blocks_NEON(void *memory, int threads, argon2profile *profile, void *user_data);
#endif
}

void cpu_hasher::__load_argon2_block_filler() {
    if(__optimization == "REF")
       __argon2_blocks_filler_ptr = fill_memory_blocks_REF;
#if defined(__x86_64__) || defined(_WIN64)
    else if(__optimization == "SSE2")
       __argon2_blocks_filler_ptr = fill_memory_blocks_SSE2;
    else if(__optimization == "SSSE3")
       __argon2_blocks_filler_ptr = fill_memory_blocks_SSSE3;
    else if(__optimization == "AVX2")
       __argon2_blocks_filler_ptr = fill_memory_blocks_AVX2;
    else if(__optimization == "AVX512F")
       __argon2_blocks_filler_ptr = fill_memory_blocks_AVX512F;
#elif defined(__arm__)
    else if(__optimization == "NEON")
       __argon2_blocks_filler_ptr = fill_memory_blocks_NEON;
#endif
}

void cpu_hasher::cleanup() {
    __running = false;
    for(vector<thread*>::iterator it = __runners.begin();it != __runners.end();++it) {
        (*it)->join();
        delete *it;
    }
    __runners.clear();
}

REGISTER_HASHER(cpu_hasher);
