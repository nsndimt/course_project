#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"

#include <stdio.h>
#include <stdlib.h>

#include "cuda_code.h"


__global__ void gpu_add_block_sums(uint32_t* d_in, uint32_t* d_block_sums, uint32_t allnumElems)
{
	const uint32_t numElems = 2 * MAX_BLOCK_SIZE;
	uint32_t thid = threadIdx.x;
	uint32_t block_sum = d_block_sums[blockIdx.x];
	uint32_t block_shift = blockIdx.x * numElems;
	if (block_shift + 2 * thid < allnumElems)
		d_in[block_shift + 2 * thid] += block_sum;
	if (block_shift + 2 * thid + 1 < allnumElems)
		d_in[block_shift + 2 * thid + 1] += block_sum;
}

__global__ void gpu_block_scan(uint32_t* d_in, uint32_t* d_block_sums, uint32_t allnumElems)
{
	__shared__ uint32_t s_out[CONFLICT_FREE_OFFSET(2 * MAX_BLOCK_SIZE)];
	const uint32_t numElems = 2 * MAX_BLOCK_SIZE;
	const uint32_t padnumElems = CONFLICT_FREE_OFFSET(2 * MAX_BLOCK_SIZE);


	uint32_t thid = threadIdx.x;
	uint32_t dthid = 2 * thid;
	uint32_t dthid1 = 2 * thid + 1;
	uint32_t dthid2 = 2 * thid + 2;


	s_out[dthid] = 0;
	s_out[dthid1] = 0;
	if (thid + numElems < padnumElems)
		s_out[thid + numElems] = 0;

	__syncthreads();

	uint32_t block_shift = blockIdx.x * numElems;
	if (block_shift + dthid < allnumElems)
		s_out[CONFLICT_FREE_OFFSET(dthid)] = d_in[block_shift + dthid];

	if (block_shift + dthid1 < allnumElems)
		s_out[CONFLICT_FREE_OFFSET(dthid1)] = d_in[block_shift + dthid1];

	int offset = 0;
	for (int d = numElems >> 1; d > 0; d >>= 1)
	{
		__syncthreads();

		if (thid < d)
		{
			int ai = (dthid1 << offset) - 1;
			int bi = (dthid2 << offset) - 1;
			s_out[CONFLICT_FREE_OFFSET(bi)] += s_out[CONFLICT_FREE_OFFSET(ai)];
		}
		offset++;
	}
	
	if (thid == 0)
	{
		d_block_sums[blockIdx.x] = s_out[CONFLICT_FREE_OFFSET(numElems - 1)];
		s_out[CONFLICT_FREE_OFFSET(numElems - 1)] = 0;
	}
	
	for (int d = 1; d < numElems; d <<= 1)
	{
		offset--;
		__syncthreads();

		if (thid < d)
		{
			int ai = (dthid1 << offset) - 1;
			int bi = (dthid2 << offset) - 1;
			uint32_t temp = s_out[CONFLICT_FREE_OFFSET(ai)];
			s_out[CONFLICT_FREE_OFFSET(ai)] = s_out[CONFLICT_FREE_OFFSET(bi)];
			s_out[CONFLICT_FREE_OFFSET(bi)] += temp;
		}
	}
	__syncthreads();
	
	if (block_shift + dthid < allnumElems)
		d_in[block_shift + dthid] = s_out[CONFLICT_FREE_OFFSET(dthid)];

	if (block_shift + dthid1 < allnumElems)
		d_in[block_shift + dthid1] = s_out[CONFLICT_FREE_OFFSET(dthid1)];
}

__host__ void dump(char* s, uint32_t* d_array, uint32_t numElems) {
	uint32_t* h_array = (uint32_t*)malloc(numElems * sizeof(uint32_t));
	memset(h_array, 0, numElems * sizeof(uint32_t));
	checkCudaErrors(cudaMemcpy(h_array, d_array, numElems * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	printf("%s:\n",s);
	for (int i = 0; i < numElems && i < 32; ++i)
	{
		printf("%d:\t%d\n", i, h_array[i]);
	}
	printf("\n\n\n");
	free(h_array);
}
__host__ void gpu_scan(uint32_t* d_in, uint32_t numElems)
{	
	dim3 dimBlock(MAX_BLOCK_SIZE);
	uint32_t slice_size = 2 * MAX_BLOCK_SIZE;
	dim3 dimGrid(numElems % slice_size == 0 ? numElems / slice_size : numElems / slice_size + 1);

	uint32_t* d_block_sums;
	checkCudaErrors(cudaMalloc(&d_block_sums, sizeof(uint32_t) * dimGrid.x));
	checkCudaErrors(cudaMemset(d_block_sums, 0, sizeof(uint32_t) * dimGrid.x));
	gpu_block_scan<<<dimGrid, dimBlock>>>(d_in, d_block_sums, numElems);
	//dump("d_block_sums", d_block_sums, dimGrid.x);
	//dump("d_in", d_in, numElems);
	if (numElems > slice_size)
		gpu_scan(d_block_sums, dimGrid.x);
	else return;
	//dump("d_block_sums", d_block_sums, dimGrid.x);
	gpu_add_block_sums<<<dimGrid, dimBlock >>>(d_in, d_block_sums, numElems);
	//dump("d_in", d_in, numElems);
	checkCudaErrors(cudaFree(d_block_sums));
}

