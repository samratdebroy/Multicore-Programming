__kernel void integrate_kernel(__global float2* pos, __global float2* vel, __global float2* acc, int num_particles, const float dt)
{
	const int i = get_global_id(0);
	if (i < num_particles)
	{
		vel[i] += acc[i] * dt;
		pos[i] += vel[i] * dt;
	}
}