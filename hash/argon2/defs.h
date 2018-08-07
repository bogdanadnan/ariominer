//
// Created by Haifa Bogdan Adnan on 06/08/2018.
//

#ifndef ARIOMINER_DEFS_H
#define ARIOMINER_DEFS_H

#define ARGON2_RAW_LENGTH               32
#define ARGON2_MEMORY_BLOCKS            524288
#define ARGON2_TYPE_VALUE               1
#define ARGON2_VERSION                  0x13
#define ARGON2_MEMORY_SIZE              99322880

typedef enum Argon2_CoreConstants {
    ARGON2_BLOCK_SIZE = 1024,
    ARGON2_QWORDS_IN_BLOCK = ARGON2_BLOCK_SIZE / 8,
    ARGON2_OWORDS_IN_BLOCK = ARGON2_BLOCK_SIZE / 16,
    ARGON2_HWORDS_IN_BLOCK = ARGON2_BLOCK_SIZE / 32,
    ARGON2_512BIT_WORDS_IN_BLOCK = ARGON2_BLOCK_SIZE / 64,
    ARGON2_PREHASH_DIGEST_LENGTH = 64,
    ARGON2_PREHASH_SEED_LENGTH = 72
} argon2_core_constants;

typedef struct block_ { uint64_t v[ARGON2_QWORDS_IN_BLOCK]; } block;

#define BLOCKS_ADDRESSES_SIZE           1048576
extern int32_t blocks_addresses[];

#endif //ARIOMINER_DEFS_H
