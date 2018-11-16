__kernel void integrate_kernel(__global float2* pos, __global float2* vel, __global float2* acc, __global int* num_particles, __global float* dt)
{
	const int i = get_global_id(0);
	if (i < num_particles[0])
	{
		vel[i] += acc[i] * dt[0];
		pos[i] += vel[i] * dt[0];
	}
}