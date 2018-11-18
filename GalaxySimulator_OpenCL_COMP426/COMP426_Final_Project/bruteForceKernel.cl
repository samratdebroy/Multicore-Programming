// Calculate acceleration on target_p bruteforce
__kernel void brute_force_kernel(__global float2* pos, __global float2* acc, __global float* mass, int num_particles)
{
	int particleID = get_global_id(0);
	float2 thisPos = pos[particleID];
	float2 particle_acc = (float2)(0.0f, 0.0f);

	for (int i = 0; i < num_particles; ++i)
	{
		float2 difference_vector = pos[i] - thisPos;
		float squared_dist = dot(difference_vector, difference_vector) + 10000000.0f; // dx*dx + dy*dy + softener

		// Compute acceleration only if the child is a particle (ie. a leaf node) or if it meets the cutoff criterion
		if (i < num_particles )
		{
			float inv_dist = rsqrt(squared_dist); // 1/sqrt(squared_dist)

			// The particle is far enough to approximate the node as a single point
			const float g = 6.67408e-11 * mass[i] * inv_dist * inv_dist *inv_dist;
			particle_acc += difference_vector * g;
		}
	}
	acc[particleID] = particle_acc;
}