//
// Created by Haifa Bogdan Adnan on 04/08/2018.
//

#ifndef ARIOMINER_HTTP_H
#define ARIOMINER_HTTP_H


class http {
public:
    http();
    virtual ~http();

protected:
    string _encode(const string &src);
    string _http_get(const string &url);
    string _http_post(const string &url, const string &post_data, const string &content_type);

private:
    vector<string> __resolve_host(const string &hostname);
    string __get_response(const string &url, const string &post_data, const string &content_type);
    static int __socketlib_reference;
};


#endif //ARIOMINER_HTTP_H
