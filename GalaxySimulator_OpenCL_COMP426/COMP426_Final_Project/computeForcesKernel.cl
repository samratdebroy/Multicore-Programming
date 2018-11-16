#define WARP_SIZE 32
#define MAX_DEPTH 64
#define BLOCKSIZE 64
#define GRAVITATIONAL_CONSTANT 6.67408e-11
#define THETA 1.0f
#define SOFTENER 10000000.0f

// Calculate acceleration on target_p due to force from this node's subtree on target_p
__kernel void compute_force_from_nodes_kernel(__global float2* pos, __global float2* acc, __global float* mass, __global int* child, __global float4* min_max_extents, __global int* num_particles)
{
	int particleID = get_global_id(0);
	const int stride = get_global_size(0);
	const int warp_groupID = get_local_id(0) / WARP_SIZE;	// Which warp group this thread is part of in the block
	const int warpID = get_local_id(0) % WARP_SIZE;			// Local ID of the thread within its warp
	const int stack_startIdx = MAX_DEPTH * warp_groupID;		// Index in the stack at which this warp's data starts

	// Create two stacks that keeps track of the size of a quadrant at each depth
	// and that tracks children to visit
	__local volatile float quadrant_size[MAX_DEPTH * BLOCKSIZE / WARP_SIZE];
	__local volatile int stack[MAX_DEPTH * BLOCKSIZE / WARP_SIZE];

	// Max radius of the top level quadrant that encompasses all the particles
	const float quadrant_radius = 0.5*(min_max_extents->x - min_max_extents->z);

	// The stack is initialized with the valid children of the root node
	// Every thread needs to be aware of how many valid children the root has
	// so that they can offset the stack's top pointer appropriately
	int stack_offset = -1;
	for (int i = 0; i < 4; ++i)
	{
		int root_nodeID = num_particles[0] * 4;
		if (child[i + root_nodeID] != -1)
		{
			++stack_offset;
		}
	}

	// Compute acceleration for every particle assigned to this block
	while (particleID < num_particles[0]) {

		// TODO: this should be returned from a sorted ID list
		// Ensuring that particles computed in the same warp are close to each other
		// will reduce warp divergence which results in a serial execution of the threads
		// could get close to a 32x speedup if the particles are properly sorted beforehand!
		int sortedIndex = particleID;

		float2 particle_pos = pos[sortedIndex];
		float2 particle_acc = (float2)(0.0f, 0.0f);

		// Initialize the stack using the first thread of the warp
		if (warpID == 0)
		{
			int childID = 0;
			for (int i = 0; i < 4; ++i)
			{
				// Init the stacks for the root node's children
				int root_nodeID = num_particles[0] * 4;
				if (child[i + root_nodeID] != -1)
				{
					stack[stack_startIdx + childID] = child[i + root_nodeID];
					quadrant_size[stack_startIdx + childID] = quadrant_radius * quadrant_radius / THETA;
					++childID;
				}
			}
		}

		// Sync threads so that all threads in the block have the same stack
		barrier(CLK_LOCAL_MEM_FENCE);

		// While the stack is not empty for this warp
		int stack_top = stack_startIdx + stack_offset; // Keep track of where the stack's top pointer is for this 
		while (stack_top >= stack_startIdx)
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
					if (childID < num_particles[0] || work_group_all(next_quadrant_size <= squared_dist))
					{
						float inv_dist = rsqrt(squared_dist); // 1/sqrt(squared_dist)

															   // The particle is far enough to approximate the node as a single point
						const float g = GRAVITATIONAL_CONSTANT * mass[childID] * inv_dist * inv_dist *inv_dist;
						particle_acc += difference_vector * g;
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
					}
				}
				else
				{
					/**
					The article "An Efficient CUDA Implementation of the Tree - Based Barnes Hut n-Body Algorithm" (Martin Burtscher, Keshav Pingali)
					suggests that: If the remaining nodes will also be null if this child is null then you can early-exit using:
					stack_top = max(stack_startIdx, stack_top -1);
					but in this architecture we have no guarantee that if the first child is null then the second will also be null, so this isn't implemented
					*/

				}
			}
			--stack_top;
		}

		// Update the particle's acceleration
		acc[sortedIndex] = particle_acc;
		particleID += stride;

		barrier(CLK_LOCAL_MEM_FENCE);
	}
}