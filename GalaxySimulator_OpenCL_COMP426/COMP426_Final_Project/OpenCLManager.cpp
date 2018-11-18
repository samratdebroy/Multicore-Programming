#include "OpenCLManager.h"
#include "SimulationConstants.h"

#include "CL/cl_gl.h"
#include "glad/glad.h"

#include <stdio.h>
#include <stdlib.h>
#include<iostream>
#include <fstream>
#include <sstream>

void OpenCLManager::init(HGLRC& openGLContext, HDC& hdc)
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
	device_id = cpu_device_ids[0];

#ifdef OPENGL_INTEROP
	// Check if any of the GPU devices support context sharing with OpenGL
	OpenGLInteropSupported = false;
	for (auto& gpu_device : gpu_device_ids)
	{
		// Get the number of extensions supported by this device
		size_t extensionSize;
		ciErrNum = clGetDeviceInfo(gpu_device, CL_DEVICE_EXTENSIONS, 0, NULL, &extensionSize);
		if (extensionSize > 0)
		{
			// Get the list of extensions
			char* extensions = (char*)malloc(extensionSize);
			ciErrNum = clGetDeviceInfo(gpu_device, CL_DEVICE_EXTENSIONS, extensionSize, extensions, &extensionSize);
			std::string stdDevString(extensions);
			free(extensions);

			auto match_found = stdDevString.find("cl_khr_gl_sharing");
			if (match_found != std::string::npos)
			{
				// This device supports OpenGL interop, set it as device_id
				OpenGLInteropSupported = true;
				device_id = gpu_device;
				break;
			}
			else
			{
				std::cout << "No device found that supports OpenCL/OpenGL Interrop context sharing" << std::endl;
			}
		}
	}
#endif
	// Print info
	char device_string[1024];
	clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
	std::cout << device_string << std::endl;
	char cOCLVersion[32];
	clGetDeviceInfo(device_id, CL_DEVICE_VERSION, sizeof(cOCLVersion), &cOCLVersion, 0);
	std::cout << cOCLVersion << std::endl;

	// If OpenGL context sharing is supported, then get the appropriate context properties
	if (OpenGLInteropSupported)
	{
		cl_context_properties props[] =
		{ CL_CONTEXT_PLATFORM, (cl_context_properties)platform_ids[0] /*OpenCL GPU platform*/,
		  CL_GL_CONTEXT_KHR,   (cl_context_properties)openGLContext	/*OpenGL context*/,
		  CL_WGL_HDC_KHR,     (cl_context_properties)hdc	/*HDC used to create the OpenGL context*/,
		  NULL
		};

		// Create the OpenCL context on a GPU device
		cxMainContext = clCreateContext(props, 1, &device_id, NULL, NULL, &ciErrNum);
	}
	else
	{
		// Create the OpenCL context on a GPU device
		cxMainContext = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ciErrNum);
	}


	// Create a command-queue
	cqCommandQue = clCreateCommandQueue(cxMainContext, device_id, 0, NULL);

	// Allocate all buffer memory objects
	memPos = clCreateBuffer(cxMainContext, CL_MEM_READ_WRITE, sizeof(cl_float2) * NUM_NODES, NULL, &ciErrNum);
	memVel = clCreateBuffer(cxMainContext, CL_MEM_READ_WRITE, sizeof(cl_float2) * NUM_PARTICLES, NULL, &ciErrNum);
	memAcc = clCreateBuffer(cxMainContext, CL_MEM_READ_WRITE, sizeof(cl_float2) * NUM_PARTICLES, NULL, &ciErrNum);
	memMass = clCreateBuffer(cxMainContext, CL_MEM_READ_WRITE, sizeof(float) * NUM_NODES, NULL, &ciErrNum);
	memChild = clCreateBuffer(cxMainContext, CL_MEM_READ_WRITE, sizeof(int) * NUM_NODES * 4, NULL, &ciErrNum);
	memMinmax = clCreateBuffer(cxMainContext, CL_MEM_READ_WRITE, sizeof(cl_float4), NULL, &ciErrNum);

	// Create constant scalar objects to be used as scalar arguments in the kernel
	static const cl_int num_particles = NUM_PARTICLES;
	static const cl_int num_nodes = NUM_NODES;
	static const cl_float  time_step = TIME_STEP;

	// Create the programs/kernel for each kernel and set the arguments
	// Compute Forces Kernel
#ifdef BRUTE_FORCE
	std::vector<cl_mem*> bruteForceArgs = { &memPos, &memAcc, &memMass };
	bruteForce.reset(new Kernel(cxMainContext, &device_id, "bruteForceKernel.cl", "brute_force_kernel", bruteForceArgs));
	bruteForce->setKernelScalarArg(3, num_particles);
#else
	std::vector<cl_mem*> computeForcesArgs = { &memPos,&memAcc, &memMass, &memChild, &memMinmax};
	computeForces.reset(new Kernel(cxMainContext, &device_id, "computeForcesKernel.cl", "compute_force_from_nodes_kernel", computeForcesArgs));
	computeForces->setKernelScalarArg(5, num_particles);
#endif

	// Reset Quadtree Fields Kernel
	std::vector<cl_mem*> resetQuadtreeArgs = { &memPos, &memMass, &memChild};
	resetQuadtreeFields.reset(new Kernel(cxMainContext, &device_id, "resetQuadtreeKernel.cl", "reset_quadtree_kernel", resetQuadtreeArgs));
	resetQuadtreeFields->setKernelScalarArg(3, num_particles);
	resetQuadtreeFields->setKernelScalarArg(4, num_nodes);

	// Integrate Kernel
	std::vector<cl_mem*> integrateArgs = { &memPos,&memVel, &memAcc};
	integrate.reset(new Kernel(cxMainContext, &device_id, "integrateKernel.cl", "integrate_kernel", integrateArgs));
	integrate->setKernelScalarArg(3, num_particles);
	integrate->setKernelScalarArg(4, time_step);

	// VBO update Kernel
	if (OpenGLInteropSupported) 
	{
		// Initialize the Kernel
		std::vector<cl_mem*> updateVBOArgs = { &memPos };
		updateVBOValues.reset(new Kernel(cxMainContext, &device_id, "vboKernel.cl", "vbo_kernel", updateVBOArgs));
		updateVBOValues->setKernelScalarArg(2, num_particles);
	}

	// set work-item dimensions
	szGlobalWorkSize[0] = NUM_PARTICLES;
	szLocalWorkSize[0] = BLOCKSIZE;
}

void OpenCLManager::initVBO(const std::vector<cl_GLuint>& vbos)
{
	if (OpenGLInteropSupported)
	{
		memVBO.reserve(vbos.size());
		for (const auto& vbo : vbos)
		{
			// OpenGL/OpenCL Interop VBO
			if (OpenGLInteropSupported)
			{
				memVBO.push_back(std::make_unique<cl_mem>(clCreateFromGLBuffer(cxMainContext, CL_MEM_WRITE_ONLY, vbo, &ciErrNum)));
			}
		}
	}
}

void OpenCLManager::updateVBO(const std::vector<std::pair<int,int>>& offsetsAndSize, int maxExtentRadius)
{
	if (OpenGLInteropSupported)
	{
		cl_float extent = maxExtentRadius;
		updateVBOValues->setKernelScalarArg(4, extent);

		glFinish();
		std::vector<cl_event> events(offsetsAndSize.size());
		std::vector<cl_event> more_events(offsetsAndSize.size() + 1);
		for (int i = 0; i < offsetsAndSize.size(); ++i)
		{
			// Map OpenGL buffer for writing from OpenCL
			ciErrNum = clEnqueueAcquireGLObjects(cqCommandQue, 1, memVBO[i].get(), 0, 0, &events[i]);
		}
		for (int i = 0; i < offsetsAndSize.size(); ++i)
		{
			// Execute kernels to update VBOs
			size_t globalWorkSize[1] = { ((offsetsAndSize[i].second+szLocalWorkSize[0]-1)/(szLocalWorkSize[0]))*(szLocalWorkSize[0]) };
			updateVBOValues->setKernelArg(1, memVBO[i].get());
			cl_int offset = offsetsAndSize[i].first;
			updateVBOValues->setKernelScalarArg(3, offset);
			ciErrNum |= clEnqueueNDRangeKernel(cqCommandQue, updateVBOValues->kernel, 1, NULL, globalWorkSize, szLocalWorkSize, events.size(), events.data(), &more_events[0]);
		}
		for (int i = 0; i < offsetsAndSize.size(); ++i)
		{
			// Unmap buffer objects
			ciErrNum = clEnqueueReleaseGLObjects(cqCommandQue, 1, memVBO[i].get(), 1, &more_events[i], &more_events[i+1]);
		}
		ciErrNum = clFinish(cqCommandQue);

	}
}

void OpenCLManager::computeForcesAndIntegrate(cl_float2 * pos, cl_float2 * vel, cl_float2 * acc, float * mass, int * child, cl_float4 * min_max_extents)
{
	glFinish();
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
	for (auto& memvbo : memVBO)
	{
		clReleaseMemObject(*memvbo.release());
	}
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
		std::cout << log << std::endl;
		
		free(log);
	}

	// Create the kernel
	kernel = clCreateKernel(program, name, &errNumber);
	// Set the kernel argument values
	for (int i = 0; i < args.size(); ++i)
	{
		setKernelArg(i, args[i]);
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