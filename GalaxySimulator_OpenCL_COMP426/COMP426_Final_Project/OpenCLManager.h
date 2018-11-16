#pragma once

#include <CL/cl.h>
#include <string>
#include <vector>
#include <memory>

class OpenCLManager
{
private:

	class Kernel
	{
	private:
		std::string file;
		cl_int errNumber = 0;			// Error code var
		cl_program program;				// OpenCL program

		
	public:
		cl_kernel kernel;				// OpenCL kernel
		Kernel();
		Kernel(cl_context& context, cl_device_id* device_id, const char* kernel_path, const char* name, std::vector<cl_mem*> args);
		~Kernel();
		static std::string getKernel(const char* kernel_path);
	};

	cl_context cxMainContext;		// OpenCL context
	cl_command_queue cqCommandQue;	// OpenCL command queue
	//cl_device_id* device_id;		// OpenCL device list
	cl_int ciErrNum = 0;			// Error code var
	size_t szGlobalWorkSize[1];		// Global # of work items
	size_t szLocalWorkSize[1];

	// OpenCL memory buffer objects
	cl_mem memPos, memVel, memAcc, memMass, memChild, memMinmax;
	// OpenCL constants
	cl_mem memNumNodes, memNumParticles, memTimeStep;

	// OpenCL Kernels
	std::unique_ptr<Kernel> computeForces, integrate, resetQuadtreeFields, bruteForce;

	// # of Work Items in Work Group
	size_t szParmDataBytes;
	
	// byte length of parameter storage
	size_t szKernelLength;// byte Length of kernel code
	int iTestN = 10000; // Length of demo test vectors

public:
	void init();
	void computeForcesAndIntegrate(cl_float2* pos, cl_float2* vel, cl_float2* acc, float* mass, int* child, cl_float4* min_max_extents);
	void computeForcesAndIntegrate(cl_float2* pos, cl_float2* vel, cl_float2* acc, float* mass);
	void resetQuadtree(cl_float2* pos, float* mass, int* child);
	OpenCLManager();
	~OpenCLManager();
};

