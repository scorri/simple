__kernel void increment(__global const int* a,
			__global int* b)
{
	int idx = get_global_id(0);

	b[idx] = a[idx]+1;
}
