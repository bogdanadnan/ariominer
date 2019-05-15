//
// Created by Haifa Bogdan Adnan on 05/08/2018.
//

#include "dllexport.h"
#include "common.h"
#include <dirent.h>

uint64_t microseconds() {
    struct timeval time;
    gettimeofday(&time, NULL);
    return (uint64_t)time.tv_sec * 1000000 + (uint64_t)time.tv_usec;
}

vector<string> get_files(string folder) {
	vector<string> result;
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir (folder.c_str())) != NULL) {
		while ((ent = readdir (dir)) != NULL) {
			if(ent->d_type == DT_REG)
    			result.push_back(ent->d_name);
		}
		closedir (dir);
	}
	return result;
}

bool is_number(const string &s) {
	return !s.empty() && all_of(s.begin(), s.end(), ::isdigit);
}

string generate_uid(size_t length) {
	static char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
	string randomString;

	for (int n = 0; n < length; n++) {
		int key = rand() % (int) (sizeof(charset) - 1);
		randomString += charset[key];
	}

	return randomString;
}

string format_seconds(uint64_t seconds) {
	uint64_t hours = seconds / 3600;
	uint64_t minutes = (seconds - hours * 3600) / 60;
	uint64_t reminder = seconds - hours * 3600 - minutes * 60;
	stringstream ss;
	ss << std::setw(2) << std::setfill('0') << hours << ":" << std::setw(2) << std::setfill('0') << minutes << ":" << std::setw(2) << std::setfill('0') << reminder;
	return ss.str();
}
