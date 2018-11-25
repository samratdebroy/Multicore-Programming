#include "OpenCLManager.h"
#include "SimulationConstants.h"
#include "OpenCLHelper.h"

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
	errorNum_ = clGetPlatformIDs(0, NULL, &platformIdCount);
	oclHelper::clCheckError(errorNum_);
	std::vector<cl_platform_id > platform_ids(platformIdCount);
	errorNum_ = clGetPlatformIDs(platformIdCount, platform_ids.data(), NULL);
	oclHelper::clCheckError(errorNum_);

	// Get GPU devices
	cl_uint deviceIdCount = 0;
	errorNum_ = clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_GPU, 0, NULL, &deviceIdCount);
	oclHelper::clCheckError(errorNum_);

	std::vector<cl_device_id > gpu_device_ids(deviceIdCount);
	errorNum_ = clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_GPU, deviceIdCount, gpu_device_ids.data(), &deviceIdCount);
	oclHelper::clCheckError(errorNum_);

	// Get CPU devices
	errorNum_ = clGetDeviceIDs(platform_ids[1], CL_DEVICE_TYPE_CPU, 0, NULL, &deviceIdCount);
	oclHelper::clCheckError(errorNum_);

	std::vector<cl_device_id > cpu_device_ids(deviceIdCount);
	errorNum_ = clGetDeviceIDs(platform_ids[1], CL_DEVICE_TYPE_CPU, deviceIdCount, cpu_device_ids.data(), &deviceIdCount);
	oclHelper::clCheckError(errorNum_);

	// TODO: Should check both types of devices and handle dynamically instead of assuming one GPU and one CPU device
	cpuDeviceID_ = cpu_device_ids[0];

#ifdef OPENGL_INTEROP
	// Check if any of the GPU devices support context sharing with OpenGL
	OpenGLInteropSupported = false;
	for (auto& gpu_device : gpu_device_ids)
	{
		// Get the number of extensions supported by this device
		size_t extensionSize;
		errorNum_ = clGetDeviceInfo(gpu_device, CL_DEVICE_EXTENSIONS, 0, NULL, &extensionSize);
		oclHelper::clCheckError(errorNum_);

		if (extensionSize > 0)
		{
			// Get the list of extensions
			char* extensions = (char*)malloc(extensionSize);
			errorNum_ = clGetDeviceInfo(gpu_device, CL_DEVICE_EXTENSIONS, extensionSize, extensions, &extensionSize);
			oclHelper::clCheckError(errorNum_);

			std::string stdDevString(extensions);
			free(extensions);

			auto match_found = stdDevString.find("cl_khr_gl_sharing");
			if (match_found != std::string::npos)
			{
				// This device supports OpenGL interop, set it as device_id
				OpenGLInteropSupported = true;
				gpuDeviceID_ = gpu_device;
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
	std::vector<decltype(gpuDeviceID_)> deviceIDs = { cpuDeviceID_, gpuDeviceID_ };
	for (auto& device_id : deviceIDs)
	{
		char device_string[1024];
		clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
		std::cout << device_string << std::endl;
		char openCLVersion[32];
		clGetDeviceInfo(device_id, CL_DEVICE_VERSION, sizeof(openCLVersion), &openCLVersion, 0);
		std::cout << openCLVersion << std::endl;
	}

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
		gpuContext_ = clCreateContext(props, 1, &gpuDeviceID_, NULL, NULL, &errorNum_);
		oclHelper::clCheckError(errorNum_);
	}
	else
	{
		// Create the OpenCL context on a GPU device
		gpuContext_ = clCreateContext(NULL, 1, &gpuDeviceID_, NULL, NULL, &errorNum_);
		oclHelper::clCheckError(errorNum_);
	}

	// Create the OpenCL context on a CPU device
	cpuContext_ = clCreateContext(NULL, 1, &cpuDeviceID_, NULL, NULL, &errorNum_);

	// Create the command-queues
	gpu_command_queue_ = clCreateCommandQueue(gpuContext_, gpuDeviceID_, 0, &errorNum_);
	oclHelper::clCheckError(errorNum_);

	cpu_command_queue_ = clCreateCommandQueue(cpuContext_, cpuDeviceID_, 0, &errorNum_);
	oclHelper::clCheckError(errorNum_);

	// Allocate all buffer memory objects
	// GPU
	memPos = clCreateBuffer(gpuContext_, CL_MEM_READ_WRITE, sizeof(cl_float2) * NUM_NODES, NULL, &errorNum_);
	oclHelper::clCheckError(errorNum_);
	memVel = clCreateBuffer(gpuContext_, CL_MEM_READ_WRITE, sizeof(cl_float2) * NUM_PARTICLES, NULL, &errorNum_);
	oclHelper::clCheckError(errorNum_);
	memAcc = clCreateBuffer(gpuContext_, CL_MEM_READ_WRITE, sizeof(cl_float2) * NUM_PARTICLES, NULL, &errorNum_);
	oclHelper::clCheckError(errorNum_);
	memMass = clCreateBuffer(gpuContext_, CL_MEM_READ_WRITE, sizeof(float) * NUM_NODES, NULL, &errorNum_);
	oclHelper::clCheckError(errorNum_);
	memChild = clCreateBuffer(gpuContext_, CL_MEM_READ_WRITE, sizeof(int) * NUM_NODES * 4, NULL, &errorNum_);
	oclHelper::clCheckError(errorNum_);
	memMinmax = clCreateBuffer(gpuContext_, CL_MEM_READ_WRITE, sizeof(cl_float4), NULL, &errorNum_);
	oclHelper::clCheckError(errorNum_);

	// Create constant scalar objects to be used as scalar arguments in the kernel
	static const cl_int num_particles = NUM_PARTICLES;
	static const cl_int num_nodes = NUM_NODES;
	static const cl_float  time_step = TIME_STEP;

	// Create the programs/kernel for each kernel and set the arguments
	// Compute Forces Kernel
#ifdef BRUTE_FORCE
	std::vector<cl_mem*> bruteForceArgs = { &memPos, &memAcc, &memMass };
	bruteForce.reset(new Kernel(mainContext_, &gpuDeviceID_, "bruteForceKernel.cl", "brute_force_kernel", bruteForceArgs));
	bruteForce->setKernelScalarArg(3, num_particles);
#else
	std::vector<cl_mem*> computeForcesArgs = { &memPos,&memAcc, &memMass, &memChild, &memMinmax};
	computeForces.reset(new Kernel(gpuContext_, &gpuDeviceID_, "computeForcesKernel.cl", "compute_force_from_nodes_kernel", computeForcesArgs));
	computeForces->setKernelScalarArg(5, num_particles);
#endif

	// Integrate Kernel
	std::vector<cl_mem*> integrateArgs = { &memPos,&memVel, &memAcc};
	integrate.reset(new Kernel(gpuContext_, &gpuDeviceID_, "integrateKernel.cl", "integrate_kernel", integrateArgs));
	integrate->setKernelScalarArg(3, num_particles);
	integrate->setKernelScalarArg(4, time_step);

	// VBO update Kernel
	if (OpenGLInteropSupported) 
	{
		// Initialize the Kernel
		std::vector<cl_mem*> updateVBOArgs = { &memPos };
		updateVBOValues.reset(new Kernel(gpuContext_, &gpuDeviceID_, "vboKernel.cl", "vbo_kernel", updateVBOArgs));
		updateVBOValues->setKernelScalarArg(2, num_particles);
	}

	// set work-item dimensions
	globalWorkSize_[0] = NUM_PARTICLES;
	gpuLocalWorkSize_[0] = BLOCKSIZE;
	cpuLocalWorkSize_[0] = BLOCKSIZE;
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
				memVBO.push_back(std::make_unique<cl_mem>(clCreateFromGLBuffer(gpuContext_, CL_MEM_WRITE_ONLY, vbo, &errorNum_)));
				oclHelper::clCheckError(errorNum_);
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
			errorNum_ = clEnqueueAcquireGLObjects(gpu_command_queue_, 1, memVBO[i].get(), 0, 0, &events[i]);
			oclHelper::clCheckError(errorNum_);
		}
		for (int i = 0; i < offsetsAndSize.size(); ++i)
		{
			// Execute kernels to update VBOs
			size_t globalWorkSize[1] = { ((offsetsAndSize[i].second+gpuLocalWorkSize_[0]-1)/(gpuLocalWorkSize_[0]))*(gpuLocalWorkSize_[0]) };
			updateVBOValues->setKernelArg(1, memVBO[i].get());
			cl_int offset = offsetsAndSize[i].first;
			updateVBOValues->setKernelScalarArg(3, offset);
			errorNum_ |= clEnqueueNDRangeKernel(gpu_command_queue_, updateVBOValues->kernel, 1, NULL, globalWorkSize, gpuLocalWorkSize_, events.size(), events.data(), &more_events[0]);
			oclHelper::clCheckError(errorNum_);
		}
		for (int i = 0; i < offsetsAndSize.size(); ++i)
		{
			// Unmap buffer objects
			errorNum_ = clEnqueueReleaseGLObjects(gpu_command_queue_, 1, memVBO[i].get(), 1, &more_events[i], &more_events[i+1]);
			oclHelper::clCheckError(errorNum_);
		}
		errorNum_ = clFinish(gpu_command_queue_);
		oclHelper::clCheckError(errorNum_);
	}
}

void OpenCLManager::computeForcesAndIntegrate(cl_float2 * pos, cl_float2 * vel, cl_float2 * acc, float * mass, int * child, cl_float4 * min_max_extents)
{
	glFinish();
	// Write input
	errorNum_ = clEnqueueWriteBuffer(gpu_command_queue_, memPos, CL_TRUE, 0, NUM_NODES * sizeof(cl_float2), pos, 0, NULL, NULL);
	errorNum_ |= clEnqueueWriteBuffer(gpu_command_queue_, memVel, CL_TRUE, 0, NUM_PARTICLES * sizeof(cl_float2), vel, 0, NULL, NULL);
	errorNum_ |= clEnqueueWriteBuffer(gpu_command_queue_, memAcc, CL_TRUE, 0, NUM_PARTICLES * sizeof(cl_float2), acc, 0, NULL, NULL);
	errorNum_ |= clEnqueueWriteBuffer(gpu_command_queue_, memMass, CL_TRUE, 0, NUM_NODES * sizeof(float), mass, 0, NULL, NULL);
	errorNum_ |= clEnqueueWriteBuffer(gpu_command_queue_, memChild, CL_TRUE, 0, NUM_NODES * 4 * sizeof(int), child, 0, NULL, NULL);
	errorNum_ |= clEnqueueWriteBuffer(gpu_command_queue_, memMinmax, CL_TRUE, 0, sizeof(cl_float4), min_max_extents, 0, NULL, NULL);
	oclHelper::clCheckError(errorNum_);

	// Execute kernel
	cl_event computeForcesEvent, integrateEvent;
	errorNum_ = clEnqueueNDRangeKernel(gpu_command_queue_, computeForces->kernel, 1, NULL, globalWorkSize_, gpuLocalWorkSize_, 0, NULL, &computeForcesEvent);
	errorNum_ |= clEnqueueNDRangeKernel(gpu_command_queue_, integrate->kernel, 1, NULL, globalWorkSize_, gpuLocalWorkSize_, 1, &computeForcesEvent, &integrateEvent);
	oclHelper::clCheckError(errorNum_);

	// Read output
	errorNum_ = clEnqueueReadBuffer(gpu_command_queue_, memPos, CL_TRUE, 0, NUM_NODES * sizeof(cl_float2), pos, 1, &integrateEvent, NULL);
	errorNum_ |= clEnqueueReadBuffer(gpu_command_queue_, memVel, CL_TRUE, 0, NUM_PARTICLES * sizeof(cl_float2), vel, 1, &integrateEvent, NULL);
	errorNum_ |= clEnqueueReadBuffer(gpu_command_queue_, memAcc, CL_TRUE, 0, NUM_PARTICLES * sizeof(cl_float2), acc, 1, &integrateEvent, NULL);
	oclHelper::clCheckError(errorNum_);
}

void OpenCLManager::computeForcesAndIntegrate(cl_float2 * pos, cl_float2 * vel, cl_float2 * acc, float * mass)
{
	// Write input
	errorNum_ = clEnqueueWriteBuffer(gpu_command_queue_, memPos, CL_TRUE, 0, NUM_NODES * sizeof(cl_float2), pos, 0, NULL, NULL);
	errorNum_ |= clEnqueueWriteBuffer(gpu_command_queue_, memVel, CL_TRUE, 0, NUM_PARTICLES * sizeof(cl_float2), vel, 0, NULL, NULL);
	errorNum_ |= clEnqueueWriteBuffer(gpu_command_queue_, memAcc, CL_TRUE, 0, NUM_PARTICLES * sizeof(cl_float2), acc, 0, NULL, NULL);
	errorNum_ |= clEnqueueWriteBuffer(gpu_command_queue_, memMass, CL_TRUE, 0, NUM_NODES * sizeof(float), mass, 0, NULL, NULL);
	oclHelper::clCheckError(errorNum_);

	// Execute kernel
	cl_event event;
	errorNum_ = clEnqueueNDRangeKernel(gpu_command_queue_, bruteForce->kernel, 1, NULL, globalWorkSize_, gpuLocalWorkSize_, 0, NULL, &event);
	cl_event event2;
	errorNum_ |= clEnqueueNDRangeKernel(gpu_command_queue_, integrate->kernel, 1, NULL, globalWorkSize_, gpuLocalWorkSize_, 1, &event, &event2);
	oclHelper::clCheckError(errorNum_);

	// Read output
	errorNum_ = clEnqueueReadBuffer(gpu_command_queue_, memPos, CL_TRUE, 0, NUM_NODES * sizeof(cl_float2), pos, 1, &event2, NULL);
	errorNum_ |= clEnqueueReadBuffer(gpu_command_queue_, memVel, CL_TRUE, 0, NUM_PARTICLES * sizeof(cl_float2), vel, 1, &event2, NULL);
	errorNum_ |= clEnqueueReadBuffer(gpu_command_queue_, memAcc, CL_TRUE, 0, NUM_PARTICLES * sizeof(cl_float2), acc, 1, &event2, NULL);
	oclHelper::clCheckError(errorNum_);
}

void OpenCLManager::initCPUKernels(cl_float2 * pos, float * mass, int * child, cl_float4* minmax)
{
	// CPU memory elems
	cpu_memPos = clCreateBuffer(cpuContext_, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(cl_float2) * NUM_NODES, pos, &errorNum_);
	oclHelper::clCheckError(errorNum_);
	cpu_memMass = clCreateBuffer(cpuContext_, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * NUM_NODES, mass, &errorNum_);
	oclHelper::clCheckError(errorNum_);
	cpu_memChild = clCreateBuffer(cpuContext_, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(int) * NUM_NODES * 4, child, &errorNum_);
	oclHelper::clCheckError(errorNum_);
	cpu_memMinMax = clCreateBuffer(cpuContext_, CL_MEM_READ_WRITE, sizeof(cl_float4), NULL, &errorNum_);
	oclHelper::clCheckError(errorNum_);

	// Reset Quadtree Fields Kernel
	std::vector<cl_mem*> resetQuadtreeArgs = { &cpu_memPos, &cpu_memMass, &cpu_memChild };
	resetQuadtreeFields.reset(new Kernel(cpuContext_, &cpuDeviceID_, "resetQuadtreeKernel.cl", "reset_quadtree_kernel", resetQuadtreeArgs));
	static const cl_int num_particles = NUM_PARTICLES;
	static const cl_int num_nodes = NUM_NODES;
	resetQuadtreeFields->setKernelScalarArg(3, num_particles);
	resetQuadtreeFields->setKernelScalarArg(4, num_nodes);

	// MinMax Kernel
	std::vector<cl_mem*> minMaxArgs = { &cpu_memPos, &cpu_memMinMax};
	minMax.reset(new Kernel(cpuContext_, &cpuDeviceID_, "minmaxKernel.cl", "minmax_kernel", minMaxArgs));
	minMax->setKernelScalarArg(2, num_particles);
}


void OpenCLManager::resetQuadtree(cl_float2 * pos, float * mass, int * child)
{

	// Execute kernel
	cl_event kernel_event;
	errorNum_ = clEnqueueNDRangeKernel(cpu_command_queue_, resetQuadtreeFields->kernel, 1, NULL, globalWorkSize_, cpuLocalWorkSize_, 0, NULL, &kernel_event);
	oclHelper::clCheckError(errorNum_);

	// Read output
	clEnqueueMapBuffer(cpu_command_queue_, cpu_memPos, CL_TRUE, CL_MAP_READ, 0, NUM_NODES * sizeof(cl_float2), 1, &kernel_event, NULL, &errorNum_);
	clEnqueueMapBuffer(cpu_command_queue_, cpu_memChild, CL_TRUE, CL_MAP_READ, 0, NUM_NODES * 4 * sizeof(int), 1, &kernel_event, NULL, &errorNum_);
	clEnqueueMapBuffer(cpu_command_queue_, cpu_memMass, CL_TRUE, CL_MAP_READ, 0, NUM_NODES * sizeof(float), 1, &kernel_event, NULL, &errorNum_);
	oclHelper::clCheckError(errorNum_);
}

void OpenCLManager::getMinMax(cl_float2 * pos, cl_float4* minmax)
{
	// Write input
	errorNum_ = clEnqueueWriteBuffer(cpu_command_queue_, cpu_memPos, CL_TRUE, 0, NUM_NODES * sizeof(cl_float2), pos, 0, NULL, NULL);
	errorNum_ = clEnqueueWriteBuffer(cpu_command_queue_, cpu_memMinMax, CL_TRUE, 0, sizeof(cl_float4), minmax, 0, NULL, NULL);
	oclHelper::clCheckError(errorNum_);

	// Execute kernel
	cl_event kernel_event;
	static const size_t size[1] = { 1 };
	errorNum_ = clEnqueueNDRangeKernel(cpu_command_queue_, minMax->kernel, 1, NULL, size, size, 0, NULL, &kernel_event);
	oclHelper::clCheckError(errorNum_);

	// Read output
	errorNum_ = clEnqueueReadBuffer(cpu_command_queue_, cpu_memMinMax, CL_TRUE, 0, sizeof(cl_float4), minmax, 1, &kernel_event, NULL);
	errorNum_ = clEnqueueReadBuffer(cpu_command_queue_, cpu_memPos, CL_TRUE, 0, NUM_NODES * sizeof(cl_float2), pos, 1, &kernel_event, NULL);
	oclHelper::clCheckError(errorNum_);
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
	clReleaseMemObject(cpu_memPos);
	clReleaseMemObject(cpu_memMass);
	clReleaseMemObject(cpu_memChild);
	clReleaseMemObject(cpu_memMinMax);
	for (auto& memvbo : memVBO)
	{
		clReleaseMemObject(*memvbo.release());
	}
	clReleaseCommandQueue(gpu_command_queue_);
	clReleaseCommandQueue(cpu_command_queue_);
	clReleaseContext(gpuContext_);
	clReleaseContext(cpuContext_);
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