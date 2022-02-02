// #pragma OPENCL EXTENSION cl_nvi_printf : enable

#define M_PI 3.1415926535897932384626433832795f

 inline void atomicAdd_g_f(volatile __global float *addr, float val)
{
	union 
	{
		unsigned int u32;
		float f32;
	}
	next, expected, current;
	current.f32 = *addr;
	do 
	{
		expected.f32 = current.f32;
		next.f32 = expected.f32 + val;
		current.u32 = atomic_cmpxchg( (volatile __global unsigned int *)addr,
		expected.u32, next.u32);
	}
	while( current.u32 != expected.u32 );
}

float2 complex_mull(  __global const float2 *a, const float2 *b)
{
	float2 c;
	c.x = a[0].x*b[0].x - a[0].y*b[0].y;
	c.y = a[0].x*b[0].y + a[0].y*b[0].x;
	return c;
}


float2 complex_mull_2(const float2 *a, const float2 *b)
{
	float2 c;
	c.x = a[0].x*b[0].x - a[0].y*b[0].y;
	c.y = a[0].x*b[0].y + a[0].y*b[0].x;
	return c;
}


float2 conj(__global const float2 *a)
{
	float2 c;
	c.x = a[0].x;
	c.y = -a[0].y;
	return c;
}


__kernel void ambig_calculation(
						__global const float2 *sig1,
						__global const float2 *sig2,
						__global const uint *id1,
						__global const uint *id2,
						__global float *conv,
						const uint dist_0_idx,
						const int min_dopp,
						const uint sig_size,
						const uint size_dist,
						const uint size_dopp
						)
{
	const int dopp =  min_dopp + get_global_id(2);

	const uint dopp_idx = get_global_id(2);

	const uint conv_offset = size_dist * dopp_idx;

	const uint i = get_global_id(0);
	const uint j = get_global_id(1);
	const uint idx_sig1 = id1[i];
	const uint idx_sig2 = id2[j];
	const int id_dist = idx_sig2 - idx_sig1 + dist_0_idx;

	if(id_dist > size_dist)
	{
		return;
	}

	const uint idx = id_dist + conv_offset;
	
	
	const float2 conj_sig2 = conj(&sig2[j]);

	const float2 val = complex_mull(&sig1[i], &conj_sig2);

	const float2 expon = {cos(2 * M_1_PI / sig_size * dopp * idx_sig1), - sin(2 * M_1_PI / sig_size * dopp * idx_sig1)};

	float2 end_val = complex_mull_2(&val, &expon);


	atomicAdd_g_f(&conv[2*idx], end_val.x);
	atomicAdd_g_f(&conv[2*idx + 1], end_val.y);
	// printf("%f\n", conv[2*idx]);
}
