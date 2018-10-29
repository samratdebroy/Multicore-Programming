#include "cuda_computation.cuh"

#include <stdio.h>

__global__ void integrate_kernel(float dt, float2* pos, float2* vel, const float2* acc)
{
	int i = threadIdx.x;
	vel[i] += acc[i] * dt;
	pos[i] += vel[i] * dt;
}

// Helper function for using CUDA to integrate particles in parallel
cudaError_t integrate_with_cuda(float dt, float2* pos, float2* vel, const float2* acc, unsigned int size)
{
	float2 *dev_pos = 0;
	float2 *dev_vel = 0;
	float2 *dev_acc = 0;
	cudaError_t cudaStatus;

	// Allocate GPU buffers for three vectors (two input/output, one pure input)    .
	cudaStatus = cudaMalloc((void**)&dev_acc, size * sizeof(float2));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_pos, size * sizeof(float2));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_vel, size * sizeof(float2));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_pos, pos, size * sizeof(float2), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_vel, vel, size * sizeof(float2), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_acc, acc, size * sizeof(float2), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	int BLOCKSIZE = 32;
	int NUM_BLOCKS = (size + BLOCKSIZE - 1) / BLOCKSIZE;
	integrate_kernel<<<NUM_BLOCKS, BLOCKSIZE >>>(dt, dev_pos, dev_vel, dev_acc);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(pos, dev_pos, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(vel, dev_vel, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_acc);
	cudaFree(dev_pos);
	cudaFree(dev_vel);

	return cudaStatus;
}