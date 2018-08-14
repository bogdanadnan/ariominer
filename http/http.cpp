//
// Created by Haifa Bogdan Adnan on 04/08/2018.
//

#include "mongoose/mongoose.h"

#include "../common/common.h"

#include "http.h"

static void mg_ev_handler(struct mg_connection *c, int ev, void *p) {
    if (ev == MG_EV_HTTP_REPLY) {
        http_message *hm = (http_message *)p;
        c->flags |= MG_F_CLOSE_IMMEDIATELY;
        strncpy((char *)c->user_data, hm->body.p, hm->body.len);
    } else if (ev == MG_EV_CLOSE) {
        c->user_data = 0;
    };
}

http::http() {
    __internal_data = new mg_mgr();
    __poll_running = false;
    mg_mgr_init((mg_mgr*)__internal_data, NULL);

}

http::~http() {
    if(__poll_running)
        __http_server_stop();

    mg_mgr_free((mg_mgr*)__internal_data);
    delete (mg_mgr*)__internal_data;
}

string http::__http_get(const string &url) {
    return __http_post(url, "");
}

// TODO modify memory allocation

string http::__http_post(const string &url, const string &post_data) {
    mg_mgr *mgr = (mg_mgr*)__internal_data;

    mg_connection *conn = mg_connect_http(mgr, mg_ev_handler, url.c_str(),
            post_data.empty() ? NULL : "Content-Type: application/x-www-form-urlencoded\r\n",
            post_data.empty() ? NULL : post_data.c_str());

    char *result = new char[1000];
    conn->user_data = (void *)result;

    time_t initial_timestamp = time(NULL);
    while(conn->user_data != 0) {
        if(time(NULL) - initial_timestamp > 10) { //10 sec timeout
            return string("");
        }
        if(!__poll_running) {
            mg_mgr_poll(mgr, 1000);
        }
        else {
            this_thread::sleep_for(chrono::milliseconds(100));
        }
    }

    string t = result;
    if(time(NULL) - initial_timestamp <= 10) {
        delete[] result;
    }
    return t;
}

void http::__http_server(int port) {
    __poll_running = true;
    __poll_until(__poll_running);
}

void http::__http_server_stop() {
    mg_mgr *mgr = (mg_mgr*)__internal_data;
    __poll_running = false;
    while(mgr->user_data != 0) {
        this_thread::sleep_for(chrono::milliseconds(100));
    }
}

void http::__poll_until(bool &flag) {
    mg_mgr *mgr = (mg_mgr*)__internal_data;
    mgr->user_data = (void *)1;
    while(flag) {
        mg_mgr_poll(mgr, 1000);
    }
    mgr->user_data = (void *)0;
}

string http::__encode(const string &src) {
    mg_str input = mg_mk_str(src.c_str());
    mg_str output = mg_url_encode(input);
    string result = output.p;
    free((void *) output.p);
    return result;
}

