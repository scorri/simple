/*
Voronoi Example
Uses original methodology from the HandsOnLabs
example for DirectCompute Samples to generate
a voronoi diagram.
In this example the points are evenly distributed
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include "FreeImage.h"

// Simulation parameters
const int width = 512;
const int height = 512;
const int ThreadsX = 16;
const int ThreadsY = 16;
const int uNumVoronoiPts = 16;

// Description of Voronoi Buf
struct VoronoiBuf
{
	int x;
	int y;
	int r;
	int g;
	int b;
};

// Globals for graphics
unsigned char* image_data;
unsigned char* output;
VoronoiBuf* Voronoi_d;

// Create Voronoi kernel
__global__ void create_voronoi( unsigned char* image_data, VoronoiBuf * v)
{
    // map from thread to pixel position
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int offset = y * width + x;

	// if in image
	if(x < width && y < height)
	{
		int minDist = 99999;
		int minDistPoint = 0;

		for(int i = 0; i < uNumVoronoiPts; i++)
		{
			int diff_x = (v[i].x - x);
			int diff_y = (v[i].y - y);
			int dist = (diff_x*diff_x + diff_y*diff_y);

			if(dist < minDist)
			{
				minDist = dist;
				minDistPoint = i;
			}
		}

		if(minDist < 25)
		{
			// now calculate the value at that position
			image_data[offset*4 + 0] = v[minDistPoint].r/2;
			image_data[offset*4 + 1] = v[minDistPoint].g/2;
			image_data[offset*4 + 2] = v[minDistPoint].b/2;
			image_data[offset*4 + 3] = 255;
		}
		else
		{
			// now calculate the value at that position
			image_data[offset*4 + 0] = v[minDistPoint].r;
			image_data[offset*4 + 1] = v[minDistPoint].g;
			image_data[offset*4 + 2] = v[minDistPoint].b;
			image_data[offset*4 + 3] = 255;
		}
	}
}

// Save image using FreeImage library
bool saveImage(std::string file, unsigned char* in_buffer)
{
    FREE_IMAGE_FORMAT format = FreeImage_GetFIFFromFilename(file.c_str());
    FIBITMAP *image = FreeImage_ConvertFromRawBits((BYTE*)in_buffer, width,
                        height, width * 4, 32,
                        0xFF000000, 0x00FF0000, 0x0000FF00);
	if(!FreeImage_Save(format, image, file.c_str()))
		return false;

	return true;
}

// Check for a Cuda Error and output error info
bool cudaCheckAPIError(cudaError_t err)
{
	if(err != cudaSuccess)
	{
		std::cerr << "Error : " << cudaGetErrorString(err) << std::endl;
		system("pause");
		return false;
	}

	return true;
}

// Round up to nearest multiple
size_t roundUp(int groupSize, int globalSize)
{
    int r = globalSize % groupSize;
    if(r == 0)
    {
        return globalSize;
    }
    else
    {
        return globalSize + groupSize - r;
    }
}

// Cleanup
void cleanup()
{
	// Free host memory
	free( output );

	// Free device memory
	cudaCheckAPIError( cudaFree( Voronoi_d ) );
	cudaCheckAPIError( cudaFree( image_data ) );

	// Exit Application
	exit(EXIT_SUCCESS);
}

// Read results buffer and save to file
void saveVoronoi(const char* file)
{
	// Allocate host memory
	unsigned char* output;
	output = (unsigned char*)malloc(width*height*4);

	// Transfer results from device to host
	cudaCheckAPIError( cudaMemcpy(output, image_data, sizeof(int)*width*height, cudaMemcpyDeviceToHost) );

	// use free image to save output image to check correctness
	if( saveImage(file, output) )
		printf("Image saved\n\n");

	// free memory
	free(output);
}

// Execute voronoi kernel
void executeVoronoiBM()
{
	dim3 grid( width/ThreadsX, height/ThreadsY);
	dim3 block( ThreadsX, ThreadsY );

	cudaFuncSetCacheConfig(create_voronoi, cudaFuncCachePreferL1);

	create_voronoi <<< grid, block >>> (image_data, Voronoi_d);
}

int main(int argc, char** argv)
{
	printf("Voronoi Benchmark CUDA\n\n");

	const int image_size = width * height * 4;	
	const int voronoi_size = roundUp(16, sizeof(VoronoiBuf)*uNumVoronoiPts);

	VoronoiBuf* Voronoi_h = (VoronoiBuf*)malloc(voronoi_size);
	
	cudaCheckAPIError( cudaMalloc( (void**)&image_data, image_size) );
	cudaCheckAPIError( cudaMalloc( (void**)&Voronoi_d, voronoi_size) );

	// Generate Voronoi Points
	printf("Program Data\n");
	printf("Number of Voronoi Points :\t%d\n", uNumVoronoiPts);
	int k = 0;
	int dim = sqrt((float)uNumVoronoiPts);
	int spacing_x = width/ dim;
	int spacing_y = height/ dim;
	printf("%d %d %d\n", dim, spacing_x, spacing_y); 
	for(int i = 0; i < dim; i++)
	{
		for(int j = 0; j < dim; j++)
		{
			Voronoi_h[k].x = spacing_x/2 + spacing_x*i;
			Voronoi_h[k].y = spacing_y/2 + spacing_y*j;
			Voronoi_h[k].r = 25 + 204 * (rand()%256)/255;
			Voronoi_h[k].g = 25 + 204 * (rand()%256)/255;
			Voronoi_h[k].b = 25 + 204 * (rand()%256)/255;	
			printf("%d %d\n", Voronoi_h[k].x, Voronoi_h[k].y);
			k++;
		}
	}
	
	printf("Image size :\t%d %d\n", width, height);
	printf("Grid Size :\t%d %d\n", width/ThreadsX, height/ThreadsY);
	printf("Block Size :\t%d %d\n", ThreadsX, ThreadsY);

	// copy data from host to device
	cudaCheckAPIError( cudaMemcpy( Voronoi_d, Voronoi_h, voronoi_size, cudaMemcpyHostToDevice) );
	free( Voronoi_h );

	executeVoronoiBM();
	saveVoronoi("voronoi.png");

	cleanup();
	return 0;
}
