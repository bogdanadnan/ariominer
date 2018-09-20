__global__ void gpu_fill_block(float* A, float* B, float* C)
{
	int  i = blockIdx.x*blockDim.x + threadIdx.x;
	int  j = blockIdx.y*blockDim.y + threadIdx.y;
	int index = i + j;
	int k;

	//  if ((i<N) && (j<N)) {
	for (k = 0; k < 10; k++)
		C[index] += A[i + k * 10] * B[k + j * 10];

	// C[index] = sqrt(sin(cos(A[index]))) + sqrt(sin(cos(B[index])));
}

void host_fill_block() {

}