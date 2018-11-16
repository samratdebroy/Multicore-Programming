#include "OpenCLManager.h"
#include "SimulationConstants.h"

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>

void OpenCLManager::init()
{

	// Get platform and device information
	cl_uint platformIdCount = 0;
	ciErrNum = clGetPlatformIDs(0, NULL, &platformIdCount);
	std::vector<cl_platform_id > platform_ids(platformIdCount);
	ciErrNum = clGetPlatformIDs(platformIdCount, platform_ids.data(), NULL);

	// Get GPU devices
	cl_uint deviceIdCount = 0;
	ciErrNum = clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_GPU, 0, NULL, &deviceIdCount);
	std::vector<cl_device_id > gpu_device_ids(deviceIdCount);
	ciErrNum = clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_GPU, deviceIdCount, gpu_device_ids.data(), &deviceIdCount);
	// Get CPU devices
	ciErrNum = clGetDeviceIDs(platform_ids[1], CL_DEVICE_TYPE_CPU, 0, NULL, &deviceIdCount);
	std::vector<cl_device_id > cpu_device_ids(deviceIdCount);
	ciErrNum = clGetDeviceIDs(platform_ids[1], CL_DEVICE_TYPE_CPU, deviceIdCount, cpu_device_ids.data(), &deviceIdCount);
	// TODO: Should check both types of devices and handle dynamically instead of assuming one GPU and one CPU device
	cl_device_id device_id = gpu_device_ids[0];

	// Print info
	char device_string[1024];
	clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
	printf("%s\n", device_string);
	char cOCLVersion[32];
	clGetDeviceInfo(device_id, CL_DEVICE_VERSION, sizeof(cOCLVersion), &cOCLVersion, 0);
	printf("%s\n", cOCLVersion);

	// Create the OpenCL context on a GPU device
	cxMainContext = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ciErrNum);

	// Create a command-queue
	cqCommandQue = clCreateCommandQueue(cxMainContext, device_id, 0, NULL);

	// Allocate all buffer memory objects
	memPos = clCreateBuffer(cxMainContext, CL_MEM_READ_WRITE, sizeof(cl_float2) * NUM_NODES, NULL, &ciErrNum);
	memVel = clCreateBuffer(cxMainContext, CL_MEM_READ_WRITE, sizeof(cl_float2) * NUM_PARTICLES, NULL, &ciErrNum);
	memAcc = clCreateBuffer(cxMainContext, CL_MEM_READ_WRITE, sizeof(cl_float2) * NUM_PARTICLES, NULL, &ciErrNum);
	memMass = clCreateBuffer(cxMainContext, CL_MEM_READ_WRITE, sizeof(float) * NUM_NODES, NULL, &ciErrNum);
	memChild = clCreateBuffer(cxMainContext, CL_MEM_READ_WRITE, sizeof(int) * NUM_NODES * 4, NULL, &ciErrNum);
	memMinmax = clCreateBuffer(cxMainContext, CL_MEM_READ_WRITE, sizeof(cl_float4), NULL, &ciErrNum);

	// Allocate constant buffer memory objects
	int num_particles[1] = { NUM_PARTICLES };
	int num_nodes[1] = { NUM_NODES };
	float  time_step[1] = { TIME_STEP };
	memNumNodes = clCreateBuffer(cxMainContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &ciErrNum);
	memNumParticles = clCreateBuffer(cxMainContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &ciErrNum);
	memTimeStep = clCreateBuffer(cxMainContext, CL_MEM_READ_WRITE, sizeof(float), NULL, &ciErrNum);
	ciErrNum = clEnqueueWriteBuffer(cqCommandQue, memNumNodes, CL_TRUE, 0, sizeof(int), &num_nodes, 0, NULL, NULL);
	ciErrNum = clEnqueueWriteBuffer(cqCommandQue, memNumParticles, CL_TRUE, 0, sizeof(int), &num_particles, 0, NULL, NULL);
	ciErrNum = clEnqueueWriteBuffer(cqCommandQue, memTimeStep, CL_TRUE, 0, sizeof(float), &time_step, 0, NULL, NULL);

	// Create the programs for each kernel

	/*std::vector<cl_mem*> computeForcesArgs = { &memPos,&memAcc, &memMass, &memChild, &memMinmax, &memNumParticles };
	computeForces = Kernel(cxMainContext, &device_id, "computeForcesKernel.cl", "compute_force_from_nodes_kernel", computeForcesArgs);
	
	std::vector<cl_mem*> resetQuadtreeArgs = { &memPos, &memMass, &memChild, &memNumParticles, &memNumNodes };
	resetQuadtreeFields = Kernel(cxMainContext, &device_id, "resetQuadtreeKernel.cl", "reset_quadtree_kernel", resetQuadtreeArgs);*/

	std::vector<cl_mem*> integrateArgs = { &memPos,&memVel, &memAcc, &memNumParticles , &memTimeStep};
	integrate.reset(new Kernel(cxMainContext, &device_id, "integrateKernel.cl", "integrate_kernel", integrateArgs));

	std::vector<cl_mem*> bruteForceArgs = { &memPos, &memAcc, &memMass, &memNumParticles };
	bruteForce.reset(new Kernel(cxMainContext, &device_id, "bruteForceKernel.cl", "brute_force_kernel", bruteForceArgs));

	// set work-item dimensions
	szGlobalWorkSize[0] = NUM_PARTICLES;
	szLocalWorkSize[0] = BLOCKSIZE;
}

void OpenCLManager::computeForcesAndIntegrate(cl_float2 * pos, cl_float2 * vel, cl_float2 * acc, float * mass, int * child, cl_float4 * min_max_extents)
{

	// Write input
	ciErrNum = clEnqueueWriteBuffer(cqCommandQue, memPos, CL_TRUE, 0, NUM_NODES * sizeof(cl_float2), pos, 0, NULL, NULL);
	ciErrNum = clEnqueueWriteBuffer(cqCommandQue, memVel, CL_TRUE, 0, NUM_PARTICLES * sizeof(cl_float2), vel, 0, NULL, NULL);
	ciErrNum = clEnqueueWriteBuffer(cqCommandQue, memAcc, CL_TRUE, 0, NUM_PARTICLES * sizeof(cl_float2), acc, 0, NULL, NULL);
	ciErrNum = clEnqueueWriteBuffer(cqCommandQue, memMass, CL_TRUE, 0, NUM_NODES * sizeof(float), mass, 0, NULL, NULL);
	ciErrNum = clEnqueueWriteBuffer(cqCommandQue, memChild, CL_TRUE, 0, NUM_NODES * 4 * sizeof(int), child, 0, NULL, NULL);
	ciErrNum = clEnqueueWriteBuffer(cqCommandQue, memMinmax, CL_TRUE, 0, sizeof(cl_float4), min_max_extents, 0, NULL, NULL);
	// Execute kernel
	cl_event computeForcesEvent, integrateEvent;
	ciErrNum = clEnqueueNDRangeKernel(cqCommandQue, computeForces->kernel, 1, NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, &computeForcesEvent);
	ciErrNum = clEnqueueNDRangeKernel(cqCommandQue, integrate->kernel, 1, NULL, szGlobalWorkSize, szLocalWorkSize, 1, &computeForcesEvent, &integrateEvent);
	// Read output
	ciErrNum = clEnqueueReadBuffer(cqCommandQue, memPos, CL_TRUE, 0, NUM_NODES * sizeof(cl_float2), pos, 1, &integrateEvent, NULL);
	ciErrNum = clEnqueueReadBuffer(cqCommandQue, memVel, CL_TRUE, 0, NUM_PARTICLES * sizeof(cl_float2), vel, 1, &integrateEvent, NULL);
	ciErrNum = clEnqueueReadBuffer(cqCommandQue, memAcc, CL_TRUE, 0, NUM_PARTICLES * sizeof(cl_float2), acc, 1, &integrateEvent, NULL);
}

void OpenCLManager::computeForcesAndIntegrate(cl_float2 * pos, cl_float2 * vel, cl_float2 * acc, float * mass)
{
	// Write input
	ciErrNum = clEnqueueWriteBuffer(cqCommandQue, memPos, CL_TRUE, 0, NUM_NODES * sizeof(cl_float2), pos, 0, NULL, NULL);
	ciErrNum = clEnqueueWriteBuffer(cqCommandQue, memVel, CL_TRUE, 0, NUM_PARTICLES * sizeof(cl_float2), vel, 0, NULL, NULL);
	ciErrNum = clEnqueueWriteBuffer(cqCommandQue, memAcc, CL_TRUE, 0, NUM_PARTICLES * sizeof(cl_float2), acc, 0, NULL, NULL);
	ciErrNum = clEnqueueWriteBuffer(cqCommandQue, memMass, CL_TRUE, 0, NUM_NODES * sizeof(float), mass, 0, NULL, NULL);
	// Execute kernel
	cl_event event;
	ciErrNum = clEnqueueNDRangeKernel(cqCommandQue, bruteForce->kernel, 1, NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, &event);
	cl_event event2;
	ciErrNum = clEnqueueNDRangeKernel(cqCommandQue, integrate->kernel, 1, NULL, szGlobalWorkSize, szLocalWorkSize, 1, &event, &event2);
	// Read output
	ciErrNum = clEnqueueReadBuffer(cqCommandQue, memPos, CL_TRUE, 0, NUM_NODES * sizeof(cl_float2), pos, 1, &event2, NULL);
	ciErrNum = clEnqueueReadBuffer(cqCommandQue, memVel, CL_TRUE, 0, NUM_PARTICLES * sizeof(cl_float2), vel, 1, &event2, NULL);
	ciErrNum = clEnqueueReadBuffer(cqCommandQue, memAcc, CL_TRUE, 0, NUM_PARTICLES * sizeof(cl_float2), acc, 1, &event2, NULL);
}

void OpenCLManager::resetQuadtree(cl_float2 * pos, float * mass, int * child)
{
	cl_event event[3];

	// Write input
	ciErrNum = clEnqueueWriteBuffer(cqCommandQue, memPos, CL_TRUE, 0, NUM_NODES * sizeof(cl_float2), pos, 0, NULL, &event[0]);
	ciErrNum = clEnqueueWriteBuffer(cqCommandQue, memChild, CL_TRUE, 0, NUM_NODES * 4 * sizeof(int), child, 0, NULL, &event[1]);
	ciErrNum = clEnqueueWriteBuffer(cqCommandQue, memMass, CL_TRUE, 0, NUM_NODES * sizeof(float), mass, 0, NULL, &event[2]);
	// Execute kernel
	cl_event kernel_event;
	ciErrNum = clEnqueueNDRangeKernel(cqCommandQue, resetQuadtreeFields->kernel, 1, NULL, szGlobalWorkSize, szLocalWorkSize, 3, event, &kernel_event);
	// Read output
	ciErrNum = clEnqueueReadBuffer(cqCommandQue, memPos, CL_TRUE, 0, NUM_NODES * sizeof(cl_float2), pos, 1, &kernel_event, NULL);
	ciErrNum = clEnqueueReadBuffer(cqCommandQue, memChild, CL_TRUE, 0, NUM_NODES * 4 * sizeof(int), child, 1, &kernel_event, NULL);
	ciErrNum = clEnqueueReadBuffer(cqCommandQue, memMass, CL_TRUE, 0, NUM_NODES * sizeof(float), mass, 1, &kernel_event, NULL);
}

OpenCLManager::OpenCLManager()
{
}


OpenCLManager::~OpenCLManager()
{
	// release kernel, program, and memory objects
	clReleaseMemObject(memPos);
	clReleaseMemObject(memVel);
	clReleaseMemObject(memAcc);
	clReleaseMemObject(memMass);
	clReleaseMemObject(memChild);
	clReleaseMemObject(memMinmax);
	clReleaseCommandQueue(cqCommandQue);
	clReleaseContext(cxMainContext);
}

OpenCLManager::Kernel::Kernel()
{
}

OpenCLManager::Kernel::Kernel(cl_context& context, cl_device_id* device_id, const char * kernel_path, const char * name, std::vector<cl_mem*> args)
{ 
	file = getKernel(kernel_path);
	const char* file_c_str = file.c_str();

	// Build the program
	program = clCreateProgramWithSource(context, 1, &file_c_str, NULL, &errNumber); // Assumes only one string
	errNumber = clBuildProgram(program, 1, device_id, NULL, NULL, NULL); // Assumes only one device

	// Catch syntax errors in the Kernel code
	if (errNumber != 0) {
		// Determine the size of the log
		size_t log_size;
		clGetProgramBuildInfo(program, *device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

		// Allocate memory for the log
		char *log = (char *)malloc(log_size);

		// Get the log
		clGetProgramBuildInfo(program, *device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

		// Print the log
		printf("%s\n", log);
		
		free(log);
	}

	// Create the kernel
	kernel = clCreateKernel(program, name, &errNumber);
	// Set the kernel argument values
	for (int i = 0; i < args.size(); ++i)
	{
		errNumber |= clSetKernelArg(kernel, i, sizeof(cl_mem), (void*)args[i]);
	}
}

OpenCLManager::Kernel::~Kernel()
{
	clReleaseKernel(kernel);
	clReleaseProgram(program);
}

std::string OpenCLManager::Kernel::getKernel(const char* kernel_path)
{
	std::ifstream file(kernel_path);
	std::stringstream buffer;
	buffer << file.rdbuf();
	return buffer.str();
}