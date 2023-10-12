#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

#include "ipps.h"
#include "ippcore.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <windows.h>
#include <algorithm>

#include "cuda_code.h"

__host__ void cpu_scan(uint32_t* h_out, uint32_t* h_in, uint32_t numElems)
{
	uint32_t run_sum = 0;
	for (int i = 0; i < numElems; ++i)
	{
		h_out[i] = run_sum;
		run_sum = run_sum + h_in[i];
	}
}
int compare(const void* a, const void* b)
{
	return (*(int*)a - *(int*)b);
}
void cpu_sort(uint32_t* h_out, uint32_t numElems) {
	qsort(h_out, numElems, sizeof(uint32_t), compare);
	//std::sort(h_out, h_out + numElems);
}
void one_round_radix_sort(uint32_t* h_out, uint32_t* h_in, uint32_t numElems, uint32_t shift)
{
	int count[4] = { 0 };

	for (int i = 0; i < numElems; i++)
		count[(h_in[i] >> shift) & 3]++;

	int prefix_sum = 0, temp;
	for (int i = 0; i < 4; i++) {
		temp = count[i];
		count[i] = prefix_sum;
		prefix_sum += temp;
	}

	for (int i = 0; i < numElems; i++)
	{
		int id = (h_in[i] >> shift) & 3;
		assert(0 <= id && id <= 3);
		h_out[count[id]] = h_in[i];
		count[id]++;
	}
}

void cpu_radix_sort(uint32_t* h_out, uint32_t* h_in, uint32_t numElems)
{
	uint32_t* temp = (uint32_t*)malloc(numElems * sizeof(uint32_t));
	for (int i = 0; i < numElems; i++)
	{
		temp[i] = h_in[i];
	}
	for (uint32_t shift_width = 0; shift_width <= RAND_STEP - 2; shift_width += 2) 
	{
		one_round_radix_sort(h_out, temp, numElems, shift_width);
		for (int i = 0; i < numElems; i++)
		{
			temp[i] = h_out[i];
		}
	}
}

__host__ void debug(uint32_t* h_in, uint32_t* h_out_cpu, uint32_t* h_out_gpu, uint32_t numElems) {
	int print_count = 32;
	for (int i = 0; i < numElems; ++i)
	{
		if (h_out_cpu[i] != h_out_gpu[i] && print_count > 0){
			printf("%d:\t%d\t%d\t%d\n", i, h_in[i], h_out_cpu[i], h_out_gpu[i]);
			print_count--;
		}
	}
}

__host__ void check(uint32_t* h_out_cpu, uint32_t* h_out_gpu, uint32_t numElems) {
	for (int i = 0; i < numElems; ++i)
	{
		if (h_out_cpu[i] != h_out_gpu[i])
		{
			printf("error\n");
			break;
		}
	}
}

__host__ uint64_t microseconds()
{
	LARGE_INTEGER fq, t;
	QueryPerformanceFrequency(&fq);
	QueryPerformanceCounter(&t);
	return (1000000 * t.QuadPart) / fq.QuadPart;
}

//void thrust_sort(uint32_t* h_out, uint32_t numElems, double* duration) {
//	uint64_t t0, t1;
//
//	thrust::host_vector<uint32_t > h_vec(numElems);
//	h_vec.assign(h_out, h_out + numElems);
//	thrust::device_vector<uint32_t> d_vec = h_vec;
//	t0 = microseconds();
//	thrust::sort(d_vec.begin(), d_vec.end());
//	t1 = microseconds();
//	*duration = t1 - t0;
//}
__host__ void scan_main()
{
	srand(1);
	uint64_t t0, t1;
	double duration;

	uint32_t h_in_len = 0;
	for (int k = 0; k < 1; ++k)
	{
		uint32_t h_in_len = (1 << 5) + (rand() % RAND_MAX);
		uint32_t h_in_size = h_in_len * sizeof(uint32_t);

		uint32_t* h_in = (uint32_t*)malloc(h_in_size);
		uint32_t* h_out_cpu = (uint32_t*)malloc(h_in_size);
		uint32_t* h_out_gpu = (uint32_t*)malloc(h_in_size);

		for (int i = 0; i < h_in_len; ++i)
		{
			h_in[i] = (rand() % RAND_MAX) + 1;
		}
		memset(h_out_cpu, 0, h_in_size);
		memset(h_out_gpu, 0, h_in_size);
		//no need to flush cache again since the array_size is much bigger than L3, we finish the flush process calling memset

		uint32_t* d_in;
		checkCudaErrors(cudaMalloc(&d_in, h_in_size));
		checkCudaErrors(cudaMemcpy(d_in, h_in, h_in_size, cudaMemcpyHostToDevice));

		t0 = microseconds();
		cpu_scan(h_out_cpu, h_in, h_in_len);
		t1 = microseconds();
		printf("CPU time: %lld\n", t1 - t0);

		t0 = microseconds();
		gpu_scan(d_in, h_in_len);
		t1 = microseconds();
		printf("GPU time: %lld\n", t1 - t0);

		checkCudaErrors(cudaMemcpy(h_out_gpu, d_in, h_in_size, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(d_in));

		debug(h_in, h_out_cpu, h_out_gpu, h_in_len);
		//check(h_out_cpu, h_out_gpu, h_in_len);

		free(h_in);
		free(h_out_cpu);
		free(h_out_gpu);
	}
}

__host__ void sort_main()
{
	srand(1);
	uint64_t t0,t1;
	double duration1=0, duration2=0, duration3=0, duration4=0;

	uint32_t h_in_len = 0;
	for (int k = 0; k < 20; ++k)
	{
		//uint32_t h_in_len = (1 << 25) + (rand() % MAX_BLOCK_SIZE);
		uint32_t h_in_len = (1 << 25);
		uint32_t h_in_size = h_in_len * sizeof(uint32_t);

		uint32_t* h_in = (uint32_t*)malloc(h_in_size);
		uint32_t* h_out_cpu = (uint32_t*)malloc(h_in_size);
		uint32_t* h_out_ipp = (uint32_t*)malloc(h_in_size);
		uint32_t* h_out_gpu = (uint32_t*)malloc(h_in_size);


		for (int i = 0; i < h_in_len; ++i)
		{
			//h_in[i] = (rand() % (RAND_MAX - 1)) + 1;
			h_in[i] = rand() % RAND_MAX;
		}
		memset(h_out_gpu, 0, h_in_size);
		//no need to flush cache again since the array_size is much bigger than L3, we finish the flush process calling memset

		uint32_t* d_in;
		checkCudaErrors(cudaMalloc(&d_in, h_in_size));
		checkCudaErrors(cudaMemcpy(d_in, h_in, h_in_size, cudaMemcpyHostToDevice));
		
		for (int i = 0; i < h_in_len; ++i)
		{
			h_out_cpu[i] = h_in[i];
		}

		int ipp_buffer_size;
		ippsSortRadixGetBufferSize(h_in_len, ipp32u, &ipp_buffer_size);
		Ipp8u* ipp_buffer = (Ipp8u*)malloc(ipp_buffer_size);
		for (int i = 0; i < h_in_len; i++)
		{
			h_out_ipp[i] = h_in[i];
		}

		t0 = microseconds();
		//cpu_radix_sort(h_out_cpu, h_in, h_in_len);
		cpu_sort(h_out_cpu, h_in_len);
		t1 = microseconds();
		duration1 += t1 - t0;
		//printf("qsort time: %lf\n", duration1);



		t0 = microseconds();
		IppStatus status = ippsSortRadixAscend_32u_I(h_out_ipp, h_in_len, ipp_buffer);
		//printf(" %d : %s\n", status, ippGetStatusString(status));
		t1 = microseconds();
		duration2 += t1 - t0;
		//printf("ippsort time: %lf\n", duration2);

		t0 = microseconds();
		gpu_radix_sort(d_in, h_in_len);
		t1 = microseconds();
		duration3 += t1 - t0;
		//printf("GPU time: %lf\n", duration3);

		//thrust_sort(h_in, h_in_size, &duration4);
		//printf("thrust time: %lf\n", duration4);

		checkCudaErrors(cudaMemcpy(h_out_gpu, d_in, h_in_size, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(d_in));
		
		//debug(h_in, h_out_cpu, h_out_gpu, h_in_len);
		check(h_out_cpu, h_out_ipp, h_in_len);
		check(h_out_cpu, h_out_gpu, h_in_len);

		free(ipp_buffer);

		free(h_in);
		free(h_out_cpu);
		free(h_out_ipp);
		free(h_out_gpu);
	}
	printf("ippsort vs qsort speedup: %lf\n", duration1 / duration2);
	printf("GPU vs qsort speedup: %lf\n", duration1 / duration3);
	//printf("thrust vs qsort speedup: %lf\n", duration1 / duration4);
}

__host__ int main()
{
	//scan_main ();
	sort_main();
	return 0;
}