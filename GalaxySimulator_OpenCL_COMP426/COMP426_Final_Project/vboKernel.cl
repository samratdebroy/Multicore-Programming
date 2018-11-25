__kernel void vbo_kernel(__global float2* pos, __global float2* vbo, const int num_particles, const int offset, const float extent)
{
	const int i = get_global_id(0) + offset;
	if (i < num_particles)
	{
		vbo[i - offset] = pos[i] / extent;
	}
}