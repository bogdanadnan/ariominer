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
    string __http_get(const string &url);
    string __http_post(const string &url, const string &post_data);

    void __http_server(int port);
    void __http_server_stop();

    void __poll_until(bool &flag);

    string __encode(const string &src);

private:
    void *__internal_data;
    bool __poll_running;
};


#endif //ARIOMINER_HTTP_H
