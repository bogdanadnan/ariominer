//
// Created by Haifa Bogdan Adnan on 04.11.2018.
//

#ifndef ARIOMINER_DLLEXPORT_H
#define ARIOMINER_DLLEXPORT_H

#undef DLLEXPORT

#ifndef _WIN64
	#define DLLEXPORT
#else
	#define DLLEXPORT __declspec(dllexport)
#endif

#endif //ARIOMINER_DLLEXPORT_H
