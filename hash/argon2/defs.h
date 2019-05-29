//
// Created by Haifa Bogdan Adnan on 06/08/2018.
//

#ifndef ARIOMINER_DEFS_H
#define ARIOMINER_DEFS_H

#define ARGON2_RAW_LENGTH               32
#define ARGON2_TYPE_VALUE               1
#define ARGON2_VERSION                  0x13

#define ARGON2_BLOCK_SIZE               1024
#define ARGON2_QWORDS_IN_BLOCK          ARGON2_BLOCK_SIZE / 8
#define ARGON2_OWORDS_IN_BLOCK          ARGON2_BLOCK_SIZE / 16
#define ARGON2_HWORDS_IN_BLOCK          ARGON2_BLOCK_SIZE / 32
#define ARGON2_512BIT_WORDS_IN_BLOCK    ARGON2_BLOCK_SIZE / 64
#define ARGON2_PREHASH_DIGEST_LENGTH    64
#define ARGON2_PREHASH_SEED_LENGTH      72

#ifdef __cplusplus 
extern "C" {
#endif

typedef struct block_ { uint64_t v[ARGON2_QWORDS_IN_BLOCK]; } block;

typedef struct Argon2Profile {
    uint32_t mem_cost;
    uint32_t thr_cost;
    uint32_t tm_cost;
    size_t memsize;
    int32_t *block_refs;
    size_t block_refs_size;
    char profile_name[15];
    int32_t *segments; // { start, stop (excluding), with_xor }
} argon2profile;

extern DLLEXPORT argon2profile argon2profile_4_4_16384;
extern DLLEXPORT argon2profile argon2profile_1_1_524288;
extern DLLEXPORT argon2profile *argon2profile_default;

#ifdef __cplusplus
}
#endif

#endif //ARIOMINER_DEFS_H
