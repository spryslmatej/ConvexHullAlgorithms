#include "partition.h"

__device__ bool partitionNotRemoved::operator()(const FLAG_TYPE x) { return x == 1; }

__device__ bool partitionByFlag::operator()(const FLAG_TYPE x) { return x == 0; }
