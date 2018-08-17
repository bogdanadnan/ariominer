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
    string _http_post(const string &url, const string &post_data);

    void _http_server(int port);
    void _http_server_stop();

private:
    vector<string> __resolve_host(const string &hostname);
    string __get_response(const string &url, const string &post_data);

};


#endif //ARIOMINER_HTTP_H
