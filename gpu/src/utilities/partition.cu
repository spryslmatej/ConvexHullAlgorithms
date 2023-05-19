#include "partition.h"

#include <thrust/partition.h>
#include <thrust/execution_policy.h>

__host__ __device__ bool partitionRemoved::operator()(const Point &x)
{
    return x.removed == 0;
}

__host__ __device__ bool partitionInHull::operator()(const Point &x)
{
    return x.inHull == 1;
}
