#include "cuda_computation.cuh"
#include "SimulationConstants.h"
#include <stdio.h>

const int BLOCKSIZE_HOST = 64;
__device__ const int BLOCKSIZE = 64;
__device__ const int MAX_DEPTH = 64;
__device__ const int WARP_SIZE = 32;
//__device__ const float THETA = 1.0f;
//__device__ const float GRAVITATIONAL_CONSTANT = 6.67408e-11;

/**
__global__ void build_tree_kernel(float2* pos, float* mass, int* count, int* start, int* child, int* index, float2 bottom_left, float2 top_right, int num_particles, int num_nodes)
{
	int particleID = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x*gridDim.x;
	bool particleIsInTree = false;

	// build quadtree
	float minX;
	float maxX;
	float minY;
	float maxY;
	int childPath;
	int nodeID;

	// Add every particle assigned to this block
	while ((particleID) < num_particles) {

		// If the particle isn't already in the tree, then add it
		if (!particleIsInTree) {
			particleIsInTree = true;

			minX = bottom_left.x;
			maxX = top_right.x;
			minY = bottom_left.y;
			maxY = top_right.y;

			// Check which quadrant the particle belongs to and get the appropriate path
			nodeID = 0;
			childPath = 0;
			if (pos[particleID].x < 0.5*(minX + maxX)) {
				childPath += 1;
				maxX = 0.5*(minX + maxX);
			}
			else {
				minX = 0.5*(minX + maxX);
			}
			if (pos[particleID].y < 0.5*(minY + maxY)) {
				childPath += 2;
				maxY = 0.5*(maxY + minY);
			}
			else {
				minY = 0.5*(maxY + minY);
			}
		}
		int childIndex = child[nodeID * 4 + childPath];

		// traverse tree until we hit leaf node
		while (childIndex >= num_particles) {

			// Again, at each iteration find the appropriate quadrant/childpath to traverse
			nodeID = childIndex;
			childPath = 0;
			if (pos[particleID].x < 0.5*(minX + maxX)) {
				childPath += 1;
				maxX = 0.5*(minX + maxX);
			}
			else {
				minX = 0.5*(minX + maxX);
			}
			if (pos[particleID].y < 0.5*(minY + maxY)) {
				childPath += 2;
				maxY = 0.5*(maxY + minY);
			}
			else {
				minY = 0.5*(maxY + minY);
			}

			atomicAdd(&pos[nodeID].x, mass[particleID] * pos[particleID].x);
			atomicAdd(&pos[nodeID].y, mass[particleID] * pos[particleID].y);
			atomicAdd(&mass[nodeID], mass[particleID]);
			atomicAdd(&count[nodeID], 1);
			childIndex = child[nodeID * 4 + childPath];
		}


		if (childIndex != -2) {
			int locked = nodeID * 4 + childPath;
			// TODO: atomicCAS is deprecated, replace
			if (atomicCAS(&child[locked], childIndex, -2) == childIndex) {
				if (childIndex == -1) {
					child[locked] = particleID;
				}
				else {
					int patch = 4 * num_particles; //TODO: rename
					while (childIndex >= 0 && childIndex < num_particles) {

						int node = atomicAdd(index, 1);
						patch = min(patch, node);
						if (patch != node) {
							child[4 * nodeID + childPath] = node;
						}

						// insert old particle
						childPath = 0;
						if (pos[childIndex].x < 0.5*(minX + maxX)) {
							childPath += 1;
						}
						if (pos[childIndex].y < 0.5*(minY + maxY)) {
							childPath += 2;
						}

						//if (DEBUG) {
						//
						//	if (node >= num_nodes) {
						//		printf("%s\n", "error node index is too large!!");
						//		printf("node: %d\n", node);
						//	}
						//}

						pos[node] += mass[childIndex] * pos[childIndex];
						mass[node] += mass[childIndex];
						count[node] += count[childIndex];
						child[4 * node + childPath] = childIndex;

						start[node] = -1;


						// insert new particle
						nodeID = node;
						childPath = 0;
						if (pos[particleID].x < 0.5*(minX + maxX)) {
							childPath += 1;
							maxX = 0.5*(minX + maxX);
						}
						else {
							minX = 0.5*(minX + maxX);
						}
						if (pos[particleID].y < 0.5*(minY + maxY)) {
							childPath += 2;
							maxY = 0.5*(maxY + minY);
						}
						else {
							minY = 0.5*(maxY + minY);
						}
						pos[node] += mass[particleID] * pos[particleID];
						mass[node] += mass[particleID];
						count[node] += count[particleID];
						childIndex = child[4 * nodeID + childPath];
					}

					child[4 * nodeID + childPath] = particleID;

					__threadfence();  // we have been writing to global memory arrays (child, x, y, mass) thus need to fence

					child[locked] = patch;
				}

				// __threadfence(); // we have been writing to global memory arrays (child, x, y, mass) thus need to fence

				particleID += stride;
				particleIsInTree = false;
			}

		}

		__syncthreads(); // not strictly needed 
	}
}

**/

// Reset the QuadTree for a new set of computations
__global__ void reset_quadtree_kernel(float2* pos, float* mass, int* child, int num_particles, int num_nodes)
{
	int nodeID = threadIdx.x + blockIdx.x*blockDim.x;
	const int stride = blockDim.x*gridDim.x;

	while (nodeID < num_nodes)
	{
		// Reset all child node indices to -1 (Null)
		for (int i = 0; i < 4; ++i)
		{
			child[nodeID * 4 + i] = -1;
		}
		
		// Only reset mass and position for internal nodes
		if (nodeID >= num_particles)
		{
			pos[nodeID] = make_float2(0.0f, 0.0f);
			mass[nodeID] = 0.0f;
		}
		nodeID += stride;
	}
}

// Calculate acceleration on target_p due to force from this node's subtree on target_p
__global__ void compute_force_from_nodes_kernel(float2* pos, float2* acc, float* mass, int* child, float4* min_max_extents, int num_particles)
{
	int particleID = threadIdx.x + blockIdx.x*blockDim.x;
	const int stride = blockDim.x*gridDim.x;
	const int warp_groupID = threadIdx.x / WARP_SIZE;	// Which warp group this thread is part of in the block
	const int warpID = threadIdx.x % WARP_SIZE;			// Local ID of the thread within its warp
	const int stackID = MAX_DEPTH * warp_groupID;		// Index in the stack at which this warp's data starts

	// Create two stacks that keeps track of the size of a quadrant at each depth
	// and that tracks children to visit
	__shared__ float quadrant_size[MAX_DEPTH * BLOCKSIZE / WARP_SIZE];
	__shared__ int stack[MAX_DEPTH * BLOCKSIZE / WARP_SIZE];

	// Max radius of the top level quadrant that encompasses all the particles
	const float quadrant_radius = 0.5*(min_max_extents->x - min_max_extents->z);

	int stack_offset = -1;
	for (int i = 0; i < 4; ++i)
	{
		int root_nodeID = num_particles * 4;
		if (child[i + root_nodeID] != -1)
		{
			++stack_offset;
		}
	}

	// Compute acceleration for every particle assigned to this block
	while (particleID < num_particles) {

		int sortedIndex = particleID; // TODO: this should be returned from a sorted ID list

		float2 particle_pos = pos[sortedIndex];
		float2 particle_acc = make_float2(0.0f, 0.0f);

		// Initialize the stack using the first thread of the warp
		int stack_top = stackID + stack_offset; // Keep track of where the stack's top pointer is for this warp
		if (warpID == 0)
		{
			int childID = 0;
			for (int i = 0; i < 4; ++i)
			{
				// Init the stacks for the root node's children
				int root_nodeID = num_particles * 4;
				if (child[i + root_nodeID] != -1)
				{
					stack[stackID + childID] = child[i + root_nodeID];
					quadrant_size[stackID + childID] = quadrant_radius * quadrant_radius / THETA;
					++childID;
				}
			}
		}

		// Sync threads so that all threads in the block have the same stack
		__syncthreads();

		// While the stack is not empty for this warp
		while (stack_top >= stackID)
		{
			// Get a node from the top of the stack
			int nodeID = stack[stack_top];
			// The size of a quadrant in the next depth level will be 1/4th of current one
			float next_quadrant_size = 0.25*quadrant_size[stack_top];

			// Compute acceleration from all four child nodes of the current node
			for (int i = 0; i < 4; ++i)
			{
				int childID = child[nodeID * 4 + i];

				// Make sure child is not null
				if (childID >= 0)
				{
					float2 difference_vector = pos[childID] - particle_pos;
					float squared_dist = dot(difference_vector, difference_vector) + SOFTENER; // dx*dx + dy*dy + softener

					// Compute acceleration only if the child is a particle (ie. a leaf node) or if it meets the cutoff criterion
					if (childID < num_particles || __all(next_quadrant_size <= squared_dist))
					{
						float inv_dist = rsqrtf(squared_dist); // 1/sqrt(squared_dist)

						// The particle is far enough to approximate the node as a single point
						const float g = GRAVITATIONAL_CONSTANT * mass[childID] * inv_dist * inv_dist *inv_dist;
						particle_acc += difference_vector*g;
					}
					else
					{
						// If this is the first thread of the warp, update stacks for next depth
						if (warpID == 0)
						{
							stack[stack_top] = childID;
							quadrant_size[stack_top] = next_quadrant_size;
						}
						stack_top++;
						//__syncthreads(); // TODO:  Is this necessary?
					}
				}
				else
				{
					// TODO: If the remaining nodes will also be null if this child is null then you can early-exit
					// stack_top = max(stackID, stack_top -1); 
				}
			}
			--stack_top;
		}

		// Update the particle's acceleration
		acc[sortedIndex] = particle_acc;
		particleID += stride;

		__syncthreads();
	}
}

__global__ void integrate_kernel(float dt, float2* pos, float2* vel, const float2* acc, int num_particles)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < num_particles)
	{
		vel[i] += acc[i] * dt;
		pos[i] += vel[i] * dt;
	}
}

// Helper function to reset array values for quadtree
cudaError_t reset_quadtree_with_cuda(float2* pos, float* mass, int* child)
{
	float2 *dev_pos = 0;
	float *dev_mass = 0;
	int *dev_child = 0;
	//int *dev_num_particles = 0;
	//int *dev_num_nodes = 0;

	cudaError_t cudaStatus;

	// Allocate GPU buffers for three vectors (three input/output)    .
	cudaStatus = cudaMalloc((void**)&dev_pos, NUM_NODES * sizeof(float2));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_mass, NUM_NODES * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_child, NUM_NODES * 4 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//cudaStatus = cudaMalloc((void**) &dev_num_particles, sizeof(int));
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMalloc failed!");
	//	goto Error;
	//}

	//cudaStatus = cudaMalloc((void**) &dev_num_nodes, sizeof(int));
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMalloc failed!");
	//	goto Error;
	//}

	// Copy input vectors and variabels from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_pos, pos, NUM_NODES * sizeof(float2), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_mass, mass, NUM_NODES * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_child, child, NUM_NODES * 4 * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	//cudaStatus = cudaMemset(dev_num_particles, NUM_PARTICLES, sizeof(int));
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMemSet failed!");
	//	goto Error;
	//}

	//cudaStatus = cudaMemset(dev_num_nodes, NUM_NODES, sizeof(int));
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMemSet failed!");
	//	goto Error;
	//}

	// Launch a kernel on the GPU with one thread for each element.
	int NUM_BLOCKS = (NUM_NODES + BLOCKSIZE_HOST - 1) / BLOCKSIZE_HOST;
	reset_quadtree_kernel <<<NUM_BLOCKS, BLOCKSIZE_HOST >>>(dev_pos, dev_mass, dev_child, NUM_PARTICLES, NUM_NODES);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "reset_quadtree_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching reset_quadtree_kernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(pos, dev_pos, NUM_NODES * sizeof(float2), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(mass, dev_mass, NUM_NODES * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(child, dev_child, NUM_NODES * 4 * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_pos);
	cudaFree(dev_mass);
	cudaFree(dev_child);
	//cudaFree(dev_num_particles);
	//cudaFree(dev_num_nodes);

	return cudaStatus;
}

// Helper function for using CUDA to integrate particles in parallel
cudaError_t compute_forces_and_integrate_with_cuda(float dt, float2* pos, float2* vel, const float2* acc, float* mass, int* child, float4* min_max_extents)
{
	float2 *dev_pos = 0;
	float2 *dev_vel = 0;
	float2 *dev_acc = 0;
	float *dev_mass = 0;
	int *dev_child = 0;
	//int *dev_num_particles = 0;
	//int *dev_num_nodes = 0;
	float4 *dev_min_max_extents = 0;

	cudaError_t cudaStatus;

	// Allocate GPU buffers for three vectors (two input/output, one pure input)    .
	cudaStatus = cudaMalloc((void**)&dev_acc, NUM_PARTICLES * sizeof(float2));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_pos, NUM_NODES * sizeof(float2));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_vel, NUM_PARTICLES * sizeof(float2));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_mass, NUM_NODES * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_child, NUM_NODES * 4 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//cudaStatus = cudaMalloc((void**)&dev_num_particles, sizeof(int));
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMalloc failed!");
	//	goto Error;
	//}

	//cudaStatus = cudaMalloc((void**)&dev_num_nodes, sizeof(int));
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMalloc failed!");
	//	goto Error;
	//}

	cudaStatus = cudaMalloc((void**)&dev_min_max_extents, sizeof(float4));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors and variabels from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_pos, pos, NUM_NODES * sizeof(float2), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_vel, vel, NUM_PARTICLES * sizeof(float2), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_acc, acc, NUM_PARTICLES * sizeof(float2), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_mass, mass, NUM_NODES * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_child, child, NUM_NODES * 4 * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	//cudaStatus = cudaMemset(dev_num_particles, NUM_PARTICLES, sizeof(int));
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMemSet failed!");
	//	goto Error;
	//}

	//cudaStatus = cudaMemset(dev_num_nodes, NUM_NODES, sizeof(int));
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMemSet failed!");
	//	goto Error;
	//}

	cudaStatus = cudaMemcpy(dev_min_max_extents, min_max_extents, sizeof(float4), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// COMPUTE FORCES FROM EACH NODE ON PARTICLES
	// Launch a kernel on the GPU with one thread for each element.
	int NUM_BLOCKS = (NUM_PARTICLES + BLOCKSIZE_HOST - 1) / BLOCKSIZE_HOST;
	compute_force_from_nodes_kernel <<<NUM_BLOCKS, BLOCKSIZE_HOST >>>(dev_pos, dev_acc, dev_mass, dev_child, dev_min_max_extents, NUM_PARTICLES);
	
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "compute_force_from_nodes_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching compute_force kernel!\n", cudaStatus);
		goto Error;
	}

	// INTEGRATE POSITIONS AND VELOCITIES
	// Launch a kernel on the GPU with one thread for each element.
	NUM_BLOCKS = (NUM_PARTICLES + BLOCKSIZE_HOST - 1) / BLOCKSIZE_HOST;
	integrate_kernel<<<NUM_BLOCKS, BLOCKSIZE_HOST >>>(dt, dev_pos, dev_vel, dev_acc, NUM_PARTICLES);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "integrate_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching integrate_kernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(pos, dev_pos, NUM_NODES * sizeof(float2), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(vel, dev_vel, NUM_PARTICLES * sizeof(float2), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_acc);
	cudaFree(dev_pos);
	cudaFree(dev_vel);
	cudaFree(dev_mass);
	cudaFree(dev_child);
	//cudaFree(dev_num_particles);
	//cudaFree(dev_num_nodes);
	cudaFree(dev_min_max_extents);

	return cudaStatus;
}