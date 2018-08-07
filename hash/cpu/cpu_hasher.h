//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#ifndef ARIOMINER_CPU_HASHER_H
#define ARIOMINER_CPU_HASHER_H

class cpu_hasher : public hasher {
public:
    cpu_hasher();
    ~cpu_hasher();

    virtual bool configure(int intensity);

private:
    string __detect_features_and_make_description();

    void __load_argon2_block_filler();
    void __run();

    string __optimization;
    int __available_processing_thr;
    int __available_memory_thr;
    int __threads_count;
    vector<thread*> __runners;
    bool __running;
    argon2_blocks_filler_ptr __argon2_blocks_filler_ptr;
};

#endif //ARIOMINER_CPU_HASHER_H
