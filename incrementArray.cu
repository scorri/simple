/*
Simple non visual example to attempt creating a makefile
This example will increment the contents of an array
*/

#include <stdio.h>
#include <assert.h>
#include <cuda.h>

// increment array on Host
void hostIncrement(int* a, int N)
{
	for(int i=0;i<N;i++)
	{
		a[i]=a[i]+1;
	}

	return;
}

// increment array on Device
__global__ void deviceIncrement(int* a, int N)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(idx <N)
	{
		a[idx] = a[idx]+1;
	}
}

// main function
int main(void)
{
	int N=16;
	size_t size=N*sizeof(int);

	// Allocate memory for arrays on host	
	int* a_h = (int*)malloc(size);
	int* b_h = (int*)malloc(size);

	// Allocate memory on device
	int* a_d;	
	cudaMalloc((void**) &a_d, size);

	// Initialise array data
	for(int i=0;i<N;i++)
	{
		a_h[i] = i;
	}

	// Copy data from host to device
	cudaMemcpy(a_d, a_h, sizeof(int)*N, cudaMemcpyHostToDevice);
	
	// Do calculation on host
	hostIncrement(a_h, N);

	// Do calculation on device
	int blockSize = 4;
	int nBlocks = N/blockSize + (N%blockSize == 0?0:1);
	
	deviceIncrement <<< nBlocks, blockSize >>> (a_d, N);

	// Retrieve results from device and store in b_h
	cudaMemcpy(b_h, a_d, sizeof(int)*N, cudaMemcpyDeviceToHost);

	for(int i=0;i<N;i++)
	{
		assert(a_h[i] == b_h[i]);
		printf("host - %d, device - %d\n", a_h[i], b_h[i]);
	}

	// Release memory
	free(a_h);
	free(b_h);
	cudaFree(a_d);
	
	return 0;
}
