/*
  Use OpenCL libraries to increment array
  as in CUDA example
*/

// opencl includes
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include "CL/cl.h"
#endif

// includes
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <sstream>
#include <fstream>

using namespace std;

inline void checkOpenCLError(cl_int err, const char* name)
{
	if(err != CL_SUCCESS)
	{
		std::cerr << err << " error code returned\n" << name << std::endl;
		exit(EXIT_FAILURE);
	}
}

cl_context createContext()
{
	cl_int errNum;
	cl_platform_id platform;
	cl_uint numPlatforms;

	// get number of platforms
	errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
	checkOpenCLError(errNum, "clGetPlatforms");
	printf("Number of platforms available is %d\n", numPlatforms);

	// Just use first platform
	errNum = clGetPlatformIDs(1, &platform, NULL);
	checkOpenCLError(errNum, "clGetPlatforms");

	// create GPU context
	cl_context_properties contextProperties[] =
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)platform,
		0
	};
	cl_context context = clCreateContextFromType(contextProperties,
					CL_DEVICE_TYPE_GPU,
					NULL, NULL, &errNum);
	checkOpenCLError(errNum, "clCreateContext");

	return context;
}

cl_command_queue createCommandQueue(cl_context context, cl_device_id* device)
{
	cl_int errNum;
	cl_device_id *devices;
	size_t bufferSize = -1;

	// Get the size of the devices buffer
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0,
				NULL, &bufferSize);
	checkOpenCLError(errNum, "clGetContextInfo");

	// allocate temp memory for devices buffer
	devices = new cl_device_id[bufferSize/ sizeof(cl_device_id)];
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES,
				bufferSize, devices, NULL);
	checkOpenCLError(errNum, "clGetContextInfo");

	// create a command queue for the first device
	cl_command_queue commandQueue = clCreateCommandQueue(context,
					 devices[0], 0, &errNum);
	checkOpenCLError(errNum, "clCreateCommandQueue");

	*device = devices[0];
	delete [] devices;

	return commandQueue;
}

cl_program createProgram(cl_context context, cl_device_id device, const char* file)
{
	cl_int errNum;
	cl_program program;

	// read in kernel file
	ifstream kernelFile(file, ios::in);
	if(!kernelFile.is_open())
	{
		std::cerr << "Failed to open file " << file << std::endl;
		return NULL;
	}

	// convert
	ostringstream oss;
	oss << kernelFile.rdbuf();
	string srcStdStr = oss.str();
	const char* srcStr = srcStdStr.c_str();

	// create program using source
	program = clCreateProgramWithSource(context, 1, 
			(const char**)&srcStr, NULL, &errNum);
	checkOpenCLError(errNum, "clCreateProgram");

	// build program
	errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if(errNum != CL_SUCCESS)
	{
		// Determine the reason for build error
		char buildLog[16384];
		clGetProgramBuildInfo(program, device, 
				CL_PROGRAM_BUILD_LOG,
				sizeof(buildLog),
				buildLog, NULL);
		std::cerr << "Error in kernel : " << std::endl;
		std::cerr << buildLog;
		clReleaseProgram(program);
		exit(EXIT_FAILURE);
	}

	return program;
}

void hostIncrement(int* a, int N)
{
	for(int i=0;i<N;i++)
	{
		a[i]=a[i]+1;
	}

	return;
}

int main(int argc, char** argv)
{
	int N;
	int blockSize;
	cl_device_id device = 0;

	// Check command line arguments
	if(argc == 1)
	{
		// none given use default
		N = 16;
		blockSize = 4;
	}
	else
	{
		std::istringstream(argv[1]) >> N;
		std::istringstream(argv[2]) >> blockSize;
	}

	// create context
	cl_context context = createContext();

	// create command queue
	cl_command_queue commandQueue = createCommandQueue(context, &device);

	// create opencl program from kernel source
	cl_program program = createProgram(context, device, "increment.cl");

	// create opencl kernel
	cl_kernel kernel = clCreateKernel(program, "increment", NULL);

	// allocate memory for host arrays
	size_t size = N*sizeof(int);
	int* a_h = (int*)malloc(size);
	int* b_h = (int*)malloc(size);

	// initialise array data
	for(int i=0;i<N;i++)
	{
		a_h[i] = i;
	}

	cl_mem memObjects[2] = {0,0};
	// input
	memObjects[0] = clCreateBuffer(context,
			CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			size, a_h, NULL);
	// output
	memObjects[1] = clCreateBuffer(context,
			CL_MEM_READ_WRITE,
			size, NULL, NULL);

	// set kernel arguments
	cl_int errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem),
				&memObjects[0]);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem),
				&memObjects[1]);
	checkOpenCLError(errNum, "clSetKernelArg");

	// set work sizes
	size_t gws[1] = { N };
	size_t lws[1] = { blockSize };

	// queue the kernel for execution
	errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
				gws, lws, 0, NULL, NULL);
	checkOpenCLError(errNum, "clEnqueueNDRange");

	// wait for kernel to finish
	clFinish(commandQueue);

	// read back the output to the host
	errNum = clEnqueueReadBuffer(commandQueue, memObjects[1],
			CL_TRUE, 0, size, b_h, 0, NULL, NULL);
	checkOpenCLError(errNum, "clEnqueueReadBuffer");

	// perform calculation on host
	hostIncrement(a_h, N);

	for(int i=0;i<N;i++)
	{
		assert(a_h[i] == b_h[i]);
		printf("host - %d, device - %d\n", a_h[i], b_h[i]);
	}

	// release host memory
	free(a_h);
	free(b_h);

	// release device memory
	clReleaseMemObject(memObjects[0]);
	clReleaseMemObject(memObjects[1]);
	
	// cleanup
	clReleaseCommandQueue(commandQueue);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseContext(context);

	return 0;
}
