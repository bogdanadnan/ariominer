//
// Created by Haifa Bogdan Adnan on 05/08/2018.
//

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
