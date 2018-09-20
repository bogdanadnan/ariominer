//
// Created by Haifa Bogdan Adnan on 18/09/2018.
//

#ifndef ARIOMINER_CUDA_HASHER_H
#define ARIOMINER_CUDA_HASHER_H

class cuda_hasher : public hasher {
public:
	cuda_hasher();
	~cuda_hasher();

	virtual bool initialize();
	virtual bool configure(arguments &args);
	virtual void cleanup();

private:
	bool __running;
};

#endif //ARIOMINER_CUDA_HASHER_H