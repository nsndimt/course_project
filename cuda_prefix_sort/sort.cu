#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda_code.h"

typedef struct share_mem_layout{
	uint32_t data[2 * MAX_BLOCK_SIZE];
	uint32_t suffix_match[CONFLICT_FREE_OFFSET(2 * MAX_BLOCK_SIZE)];
	uint32_t suffix_scan_res[2 * MAX_BLOCK_SIZE];
	uint32_t suffix_counter[4];
	uint32_t counter_scan_res[4];
} shm_layout;

__global__ void gpu_block_radix_sort(uint32_t* d_in, uint32_t* d_out, uint32_t* d_prefix_sums, uint32_t* d_block_sums, uint32_t shift, size_t allnumElems)
{
	__shared__ share_mem_layout shm;
	const uint32_t numElems = 2 * MAX_BLOCK_SIZE;
	const uint32_t padnumElems = CONFLICT_FREE_OFFSET(2 * MAX_BLOCK_SIZE);

	uint32_t thid = threadIdx.x;
	uint32_t dthid = 2 * thid;
	uint32_t dthid1 = 2 * thid + 1;
	uint32_t dthid2 = 2 * thid + 2;
	uint32_t src = numElems * blockIdx.x + dthid;
	uint32_t src1 = numElems * blockIdx.x + dthid1;

	
	if (src < allnumElems)
		shm.data[dthid] = d_in[src];
	/*else
		shm.data[thid] = 0;*/

	if (src1 < allnumElems)
		shm.data[dthid1] = d_in[src1];
	/*else
		shm.data[thid] = 0;*/

	__syncthreads();

	uint32_t t_data = shm.data[dthid];
	uint32_t t_2bit_extract = (t_data >> shift) & 3;
	uint32_t t_data1 = shm.data[dthid1];
	uint32_t t_2bit_extract1 = (t_data1 >> shift) & 3;

	for (int i = 0; i < 4; ++i)
	{
		shm.suffix_match[CONFLICT_FREE_OFFSET(dthid)] = 0;
		shm.suffix_match[CONFLICT_FREE_OFFSET(dthid1)] = 0;
		if (thid + numElems < padnumElems)
			shm.suffix_match[thid + numElems] = 0;
		__syncthreads();
		
		uint32_t suffix_match = t_2bit_extract == i ? 1 : 0;
		if (src < allnumElems)
			shm.suffix_match[CONFLICT_FREE_OFFSET(dthid)] = suffix_match;
		uint32_t suffix_match1 = t_2bit_extract1 == i ? 1 : 0;
		if (src1 < allnumElems)
			shm.suffix_match[CONFLICT_FREE_OFFSET(dthid1)] = suffix_match1;
		__syncthreads();

		int offset = 0;
		for (int d = numElems >> 1; d > 0; d >>= 1)
		{
			__syncthreads();

			if (thid < d)
			{
				int ai = (dthid1 << offset) - 1;
				int bi = (dthid2 << offset) - 1;
				shm.suffix_match[CONFLICT_FREE_OFFSET(bi)] += shm.suffix_match[CONFLICT_FREE_OFFSET(ai)];
			}
			offset++;
		}

		if (thid == 0)
		{
			uint32_t total_sum = shm.suffix_match[CONFLICT_FREE_OFFSET(numElems - 1)];
			d_block_sums[i * gridDim.x + blockIdx.x] = total_sum;
			shm.suffix_counter[i] = total_sum;
			shm.suffix_match[CONFLICT_FREE_OFFSET(numElems - 1)] = 0;
		}

		for (int d = 1; d < numElems; d <<= 1)
		{
			offset--;
			__syncthreads();

			if (thid < d)
			{
				int ai = (dthid1 << offset) - 1;
				int bi = (dthid2 << offset) - 1;
				uint32_t temp = shm.suffix_match[CONFLICT_FREE_OFFSET(ai)];
				shm.suffix_match[CONFLICT_FREE_OFFSET(ai)] = shm.suffix_match[CONFLICT_FREE_OFFSET(bi)];
				shm.suffix_match[CONFLICT_FREE_OFFSET(bi)] += temp;
			}
		}
		__syncthreads();
		
		if (suffix_match && (src < allnumElems))
			shm.suffix_scan_res[dthid] = shm.suffix_match[CONFLICT_FREE_OFFSET(dthid)];
		if (suffix_match1 && (src1 < allnumElems))
			shm.suffix_scan_res[dthid1] = shm.suffix_match[CONFLICT_FREE_OFFSET(dthid1)];

		__syncthreads();
	}
	//comment start when you do not need global memory coalescing
	if (thid == 0)
	{
		uint32_t run_sum = 0;
		shm.counter_scan_res[0] = run_sum;
		run_sum += shm.suffix_counter[0];
		shm.counter_scan_res[1] = run_sum;
		run_sum += shm.suffix_counter[1];
		shm.counter_scan_res[2] = run_sum;
		run_sum += shm.suffix_counter[2];
		shm.counter_scan_res[3] = run_sum;
	}
	__syncthreads();


	uint32_t t_prefix_sum = shm.suffix_scan_res[dthid];
	uint32_t new_pos = t_prefix_sum + shm.counter_scan_res[t_2bit_extract];
	uint32_t t_prefix_sum1 = shm.suffix_scan_res[dthid1];
	uint32_t new_pos1 = t_prefix_sum1 + shm.counter_scan_res[t_2bit_extract1];

	__syncthreads();
	if (src < allnumElems)
	{
		shm.data[new_pos] = t_data;
		shm.suffix_scan_res[new_pos] = t_prefix_sum;
	}
	if (src1 < allnumElems)
	{
		shm.data[new_pos1] = t_data1;
		shm.suffix_scan_res[new_pos1] = t_prefix_sum1;
	}
	__syncthreads();
	//comment end when you do not need global memory coalescing

	if (src < allnumElems)
	{
		d_prefix_sums[src] = shm.suffix_scan_res[dthid];
		d_out[src] = shm.data[dthid];
	}
	if (src1 < allnumElems)
	{
		d_prefix_sums[src1] = shm.suffix_scan_res[dthid1];
		d_out[src1] = shm.data[dthid1];
	}
}

__global__ void gpu_shuffle(uint32_t* d_in, uint32_t* d_out, uint32_t* d_prefix_sums, uint32_t* d_block_sums,uint32_t shift, uint32_t allnumElems)
{
	const uint32_t numElems = 2 * MAX_BLOCK_SIZE;

	uint32_t thid = threadIdx.x;
	uint32_t dthid = 2 * thid;
	uint32_t dthid1 = 2 * thid + 1;
	uint32_t src = numElems * blockIdx.x + dthid;
	uint32_t src1 = numElems * blockIdx.x + dthid1;

	uint32_t t_data = d_out[src];
	uint32_t t_2bit_extract = (t_data >> shift) & 3;
	uint32_t t_prefix_sum = d_prefix_sums[src];
	uint32_t data_glbl_pos = d_block_sums[t_2bit_extract * gridDim.x + blockIdx.x] + t_prefix_sum;
	uint32_t t_data1 = d_out[src1];
	uint32_t t_2bit_extract1 = (t_data1 >> shift) & 3;
	uint32_t t_prefix_sum1 = d_prefix_sums[src1];
	uint32_t data_glbl_pos1 = d_block_sums[t_2bit_extract1 * gridDim.x + blockIdx.x] + t_prefix_sum1;
	
	__syncthreads();
	if (src < allnumElems)
		d_in[data_glbl_pos] = t_data;

	if (src1 < allnumElems)
		d_in[data_glbl_pos1] = t_data1;
}

__host__ void gpu_radix_sort(uint32_t* d_in, uint32_t numElems)
{
	dim3 dimBlock(MAX_BLOCK_SIZE);
	uint32_t slice_size = 2 * MAX_BLOCK_SIZE;
	dim3 dimGrid(numElems % slice_size == 0 ? numElems / slice_size : numElems / slice_size + 1);
	
	uint32_t* d_out;
	checkCudaErrors(cudaMalloc(&d_out, sizeof(uint32_t) * numElems));
	checkCudaErrors(cudaMemset(d_out, 0, sizeof(uint32_t) * numElems));

	uint32_t* d_prefix_sums;
	checkCudaErrors(cudaMalloc(&d_prefix_sums, sizeof(uint32_t) * numElems));
	checkCudaErrors(cudaMemset(d_prefix_sums, 0, sizeof(uint32_t) * numElems));

	uint32_t* d_block_sums;
	checkCudaErrors(cudaMalloc(&d_block_sums, sizeof(uint32_t) * 4 * dimGrid.x));
	checkCudaErrors(cudaMemset(d_block_sums, 0, sizeof(uint32_t) * 4 * dimGrid.x));

	for (uint32_t shift_width = 0; shift_width <= RAND_STEP-2; shift_width += 2)
	{
		gpu_block_radix_sort<<<dimGrid, dimBlock>>>(d_in, d_out, d_prefix_sums, d_block_sums, shift_width, numElems);
		//dump("d_out", d_out, numElems);
		//dump("d_prefix_sums", d_prefix_sums, numElems);
		//dump("d_block_sums", d_block_sums, 4 * dimGrid.x);
		gpu_scan(d_block_sums, 4 * dimGrid.x);
		//dump("d_block_sums", d_block_sums, 4 * dimGrid.x);
		gpu_shuffle<<<dimGrid, dimBlock>>>(d_in, d_out, d_prefix_sums, d_block_sums, shift_width, numElems);
		//dump("d_in", d_in, numElems);
	}

	checkCudaErrors(cudaFree(d_block_sums));
	checkCudaErrors(cudaFree(d_prefix_sums));
}