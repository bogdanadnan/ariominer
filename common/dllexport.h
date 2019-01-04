//
// Created by Haifa Bogdan Adnan on 04.11.2018.
//

#ifndef ARIOMINER_DLLEXPORT_H
#define ARIOMINER_DLLEXPORT_H

#ifndef _WIN64
	#define DLLEXPORT
#else
	#ifdef  EXPORT_SYMBOLS
		#define DLLEXPORT __declspec(dllexport)
	#else
		#define DLLEXPORT __declspec(dllimport)
	#endif
#endif

#endif //ARIOMINER_DLLEXPORT_H
