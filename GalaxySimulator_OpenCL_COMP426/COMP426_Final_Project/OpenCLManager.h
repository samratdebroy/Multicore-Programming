#pragma once

#include <CL/cl.h>
#include <string>
#include <vector>
#include <memory>
#include <windows.h> // For OpenGL Context stuff

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
		std::string getKernel(const char* kernel_path);

		template <class T>
		void setKernelScalarArg(int param_index, T& arg)
		{
			errNumber |= clSetKernelArg(kernel, param_index, sizeof(T), (void*)&arg);
		}
		void setKernelArg(int param_index, cl_mem* args)
		{
			errNumber |= clSetKernelArg(kernel, param_index, sizeof(cl_mem), (void*)args);
		}
	};

	cl_context cxMainContext;		// OpenCL context
	cl_command_queue cqCommandQue;	// OpenCL command queue
	cl_device_id device_id;		// OpenCL device list
	cl_int ciErrNum = 0;			// Error code var
	size_t szGlobalWorkSize[1];		// Global # of work items
	size_t szLocalWorkSize[1];

	// OpenCL memory buffer objects
	cl_mem memPos, memVel, memAcc, memMass, memChild, memMinmax;
	// OpenCL constants
	cl_mem memNumNodes, memNumParticles, memTimeStep;
	// OpenGL/OpenCL Interop
	std::vector< std::unique_ptr<cl_mem>> memVBO;

	// OpenCL Kernels
	std::unique_ptr<Kernel> computeForces, integrate, resetQuadtreeFields, bruteForce, updateVBOValues;

	// # of Work Items in Work Group
	size_t szParmDataBytes;
	
	// byte length of parameter storage
	size_t szKernelLength;// byte Length of kernel code
	int iTestN = 10000; // Length of demo test vectors
	bool OpenGLInteropSupported = false;

public:

	void init(HGLRC& openGLContext, HDC& hdc);
	void initVBO(const std::vector<cl_GLuint>& vbos);
	void updateVBO(const std::vector<std::pair<int, int>>& offsetsAndSize, int extent);
	void computeForcesAndIntegrate(cl_float2* pos, cl_float2* vel, cl_float2* acc, float* mass, int* child, cl_float4* min_max_extents);
	void computeForcesAndIntegrate(cl_float2* pos, cl_float2* vel, cl_float2* acc, float* mass);
	void resetQuadtree(cl_float2* pos, float* mass, int* child);

	bool isOpenGLInteropSupported() { return OpenGLInteropSupported; };

	OpenCLManager();
	~OpenCLManager();
};

