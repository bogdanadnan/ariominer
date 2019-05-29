//
// Created by Haifa Bogdan Adnan on 19/02/2019.
//

#include "json.h"

namespace  json {
    JSON Array() {
        return std::move( JSON::Make( JSON::Class::Array ) );
    }

    JSON Object() {
        return std::move( JSON::Make( JSON::Class::Object ) );
    }

    std::ostream& operator<<( std::ostream &os, const JSON &json ) {
        os << json.dump();
        return os;
    }

    JSON JSON::Load(const string &str) {
        size_t offset = 0;
        return std::move(parse_next(str, offset));
    }
}
