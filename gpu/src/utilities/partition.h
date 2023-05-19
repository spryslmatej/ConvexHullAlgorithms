#pragma once

#include "../model/point.h"

struct partitionRemoved
{
    __host__ __device__ bool operator()(const Point &x);
};

struct partitionInHull
{
    __host__ __device__ bool operator()(const Point &x);
};
