//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#ifndef ARIOMINER_HASHER_H
#define ARIOMINER_HASHER_H

struct hash_data {
    string nonce;
    string base;
    string hash;
};

#define REGISTER_HASHER(x)          x __##x

class hasher {
public:
    hasher();
    virtual ~hasher();

    virtual bool configure(int intensity) = 0;

    string get_type();
    string get_info();
    void set_input(const string &nonce, const string &base);
    string get_base();
    int get_intensity();

    double get_current_hash_rate();
    double get_avg_hash_rate();
    uint get_hash_count();

    vector<hash_data> get_hashes();

    static vector<hasher*> get_hashers();
    static vector<hasher*> get_active_hashers();

protected:
    int _intensity;
    string _type;
    string _description;

    void _store_hash(const string &hash);

private:
    static vector<hasher*> *__registered_hashers;

    double __hash_rate;
    double __avg_hash_rate;
    uint __hash_count;

    mutex __input_mutex;
    string __nonce;
    string __base;

    mutex __hashes_mutex;
    vector<hash_data> __hashes;

    uint64_t __begin_time;
    uint64_t __hashrate_time;

};

#endif //ARIOMINER_HASHER_H
