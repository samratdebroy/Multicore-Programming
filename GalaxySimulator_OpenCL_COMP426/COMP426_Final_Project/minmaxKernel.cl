__kernel void minmax_kernel(__global float2* pos, __global float4* minmax, int num_particles)
{
	for (int i =0; i < num_particles; ++i)
	{
		if (pos[i].x < minmax[0].x)
			minmax[0].x = pos[i].x;
		if (pos[i].y < minmax[0].y)
			minmax[0].y = pos[i].y;
		if (pos[i].x > minmax[0].z)
			minmax[0].z = pos[i].x;
		if (pos[i].y > minmax[0].w)
			minmax[0].w = pos[i].y;
	}
}