//#include "cuda_computation.cuh"
//#include "SimulationConstants.h"
//#include <stdio.h>
//
//// Helper function to reset array values for quadtree
//cudaError_t reset_quadtree_with_cuda(float2* pos, float* mass, int* child)
//{
//	float2 *dev_pos = 0;
//	float *dev_mass = 0;
//	int *dev_child = 0;
//
//	cudaError_t cudaStatus;
//
//	// Allocate GPU buffers for three vectors (three input/output)    .
//	cudaStatus = cudaMalloc((void**)&dev_pos, NUM_NODES * sizeof(float2));
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc failed!");
//		goto Error;
//	}
//
//	cudaStatus = cudaMalloc((void**)&dev_mass, NUM_NODES * sizeof(float));
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc failed!");
//		goto Error;
//	}
//
//	cudaStatus = cudaMalloc((void**)&dev_child, NUM_NODES * 4 * sizeof(int));
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc failed!");
//		goto Error;
//	}
//
//	// Copy input vectors and variabels from host memory to GPU buffers.
//	cudaStatus = cudaMemcpy(dev_pos, pos, NUM_NODES * sizeof(float2), cudaMemcpyHostToDevice);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//
//	cudaStatus = cudaMemcpy(dev_mass, mass, NUM_NODES * sizeof(float), cudaMemcpyHostToDevice);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//
//	cudaStatus = cudaMemcpy(dev_child, child, NUM_NODES * 4 * sizeof(int), cudaMemcpyHostToDevice);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//
//	// Launch a kernel on the GPU with one thread for each element.
//	int NUM_BLOCKS = (NUM_NODES + BLOCKSIZE - 1) / BLOCKSIZE;
//	reset_quadtree_kernel <<<NUM_BLOCKS, BLOCKSIZE >>>(dev_pos, dev_mass, dev_child, NUM_PARTICLES, NUM_NODES);
//
//	// Check for any errors launching the kernel
//	cudaStatus = cudaGetLastError();
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "reset_quadtree_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//		goto Error;
//	}
//
//	// cudaDeviceSynchronize waits for the kernel to finish, and returns
//	// any errors encountered during the launch.
//	cudaStatus = cudaDeviceSynchronize();
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching reset_quadtree_kernel!\n", cudaStatus);
//		goto Error;
//	}
//
//	// Copy output vector from GPU buffer to host memory.
//	cudaStatus = cudaMemcpy(pos, dev_pos, NUM_NODES * sizeof(float2), cudaMemcpyDeviceToHost);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//
//	// Copy output vector from GPU buffer to host memory.
//	cudaStatus = cudaMemcpy(mass, dev_mass, NUM_NODES * sizeof(float), cudaMemcpyDeviceToHost);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//
//	// Copy output vector from GPU buffer to host memory.
//	cudaStatus = cudaMemcpy(child, dev_child, NUM_NODES * 4 * sizeof(int), cudaMemcpyDeviceToHost);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//
//Error:
//	cudaFree(dev_pos);
//	cudaFree(dev_mass);
//	cudaFree(dev_child);
//
//	return cudaStatus;
//}
//
//// Helper function for using CUDA to integrate particles in parallel
//cudaError_t compute_forces_and_integrate_with_cuda(float dt, float2* pos, float2* vel, const float2* acc, float* mass, int* child, float4* min_max_extents)
//{
//	float2 *dev_pos = 0;
//	float2 *dev_vel = 0;
//	float2 *dev_acc = 0;
//	float *dev_mass = 0;
//	int *dev_child = 0;
//	float4 *dev_min_max_extents = 0;
//
//	cudaError_t cudaStatus;
//
//	// Allocate GPU buffers for three vectors (two input/output, one pure input)    .
//	cudaStatus = cudaMalloc((void**)&dev_acc, NUM_PARTICLES * sizeof(float2));
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc failed!");
//		goto Error;
//	}
//
//	cudaStatus = cudaMalloc((void**)&dev_pos, NUM_NODES * sizeof(float2));
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc failed!");
//		goto Error;
//	}
//
//	cudaStatus = cudaMalloc((void**)&dev_vel, NUM_PARTICLES * sizeof(float2));
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc failed!");
//		goto Error;
//	}
//
//	cudaStatus = cudaMalloc((void**)&dev_mass, NUM_NODES * sizeof(float));
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc failed!");
//		goto Error;
//	}
//
//	cudaStatus = cudaMalloc((void**)&dev_child, NUM_NODES * 4 * sizeof(int));
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc failed!");
//		goto Error;
//	}
//
//	cudaStatus = cudaMalloc((void**)&dev_min_max_extents, sizeof(float4));
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc failed!");
//		goto Error;
//	}
//
//	// Copy input vectors and variabels from host memory to GPU buffers.
//	cudaStatus = cudaMemcpy(dev_pos, pos, NUM_NODES * sizeof(float2), cudaMemcpyHostToDevice);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//
//	cudaStatus = cudaMemcpy(dev_vel, vel, NUM_PARTICLES * sizeof(float2), cudaMemcpyHostToDevice);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//
//	cudaStatus = cudaMemcpy(dev_acc, acc, NUM_PARTICLES * sizeof(float2), cudaMemcpyHostToDevice);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//
//	cudaStatus = cudaMemcpy(dev_mass, mass, NUM_NODES * sizeof(float), cudaMemcpyHostToDevice);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//
//	cudaStatus = cudaMemcpy(dev_child, child, NUM_NODES * 4 * sizeof(int), cudaMemcpyHostToDevice);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//
//	cudaStatus = cudaMemcpy(dev_min_max_extents, min_max_extents, sizeof(float4), cudaMemcpyHostToDevice);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//
//	// COMPUTE FORCES FROM EACH NODE ON PARTICLES
//	// Launch a kernel on the GPU with one thread for each element.
//	int NUM_BLOCKS = (NUM_PARTICLES + BLOCKSIZE - 1) / BLOCKSIZE;
//	compute_force_from_nodes_kernel <<<NUM_BLOCKS, BLOCKSIZE >>>(dev_pos, dev_acc, dev_mass, dev_child, dev_min_max_extents, NUM_PARTICLES);
//	
//	// Check for any errors launching the kernel
//	cudaStatus = cudaGetLastError();
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "compute_force_from_nodes_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//		goto Error;
//	}
//
//	// cudaDeviceSynchronize waits for the kernel to finish, and returns
//	// any errors encountered during the launch.
//	cudaStatus = cudaDeviceSynchronize();
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching compute_force kernel!\n", cudaStatus);
//		goto Error;
//	}
//
//	// INTEGRATE POSITIONS AND VELOCITIES
//	// Launch a kernel on the GPU with one thread for each element.
//	NUM_BLOCKS = (NUM_PARTICLES + BLOCKSIZE - 1) / BLOCKSIZE;
//	integrate_kernel<<<NUM_BLOCKS, BLOCKSIZE >>>(dt, dev_pos, dev_vel, dev_acc, NUM_PARTICLES);
//
//	// Check for any errors launching the kernel
//	cudaStatus = cudaGetLastError();
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "integrate_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//		goto Error;
//	}
//
//	// cudaDeviceSynchronize waits for the kernel to finish, and returns
//	// any errors encountered during the launch.
//	cudaStatus = cudaDeviceSynchronize();
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching integrate_kernel!\n", cudaStatus);
//		goto Error;
//	}
//
//	// Copy output vector from GPU buffer to host memory.
//	cudaStatus = cudaMemcpy(pos, dev_pos, NUM_NODES * sizeof(float2), cudaMemcpyDeviceToHost);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//
//	// Copy output vector from GPU buffer to host memory.
//	cudaStatus = cudaMemcpy(vel, dev_vel, NUM_PARTICLES * sizeof(float2), cudaMemcpyDeviceToHost);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//
//Error:
//	cudaFree(dev_acc);
//	cudaFree(dev_pos);
//	cudaFree(dev_vel);
//	cudaFree(dev_mass);
//	cudaFree(dev_child);
//	cudaFree(dev_min_max_extents);
//
//	return cudaStatus;
//}