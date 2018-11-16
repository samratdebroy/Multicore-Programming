// Reset the QuadTree for a new set of computations
__kernel void reset_quadtree_kernel(__global float2* pos, __global float* mass, __global int* child, __global int* num_particles, __global int* num_nodes)
{
	int nodeID = get_global_id(0);
	int stride = get_global_size(0);

	while (nodeID < num_nodes[0])
	{
		// Reset all child node indices to -1 (Null)
		for (int i = 0; i < 4; ++i)
		{
			child[nodeID * 4 + i] = -1;
		}

		// Only reset mass and position for internal nodes
		if (nodeID >= num_particles[0])
		{
			pos[nodeID] = (float2)(0.0f, 0.0f);
			mass[nodeID] = 0.0f;
		}
		nodeID += stride;
	}
}