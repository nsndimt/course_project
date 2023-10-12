#define RAND_MAX 256
#define RAND_STEP 8
#define MAX_BLOCK_SIZE 1024

#define BANK_SHIFT 4
#define CONFLICT_FREE_OFFSET(n) ( n + ((n) >> BANK_SHIFT))

#include <stdint.h>
void gpu_scan(uint32_t* d_in, uint32_t numElems);
void gpu_radix_sort(uint32_t* d_in, uint32_t numElems);
void dump(char*s , uint32_t* d_array, uint32_t numElems);
