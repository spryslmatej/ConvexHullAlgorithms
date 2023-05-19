#pragma once

#include "qhData.h"

// used for sorting points in first split
struct cPoint
{
    COORDS_TYPE x, y, ptIndex;

    __device__ __host__ cPoint();
    __device__ __host__ cPoint(COORDS_TYPE a, COORDS_TYPE b, INDEXES_TYPE i);
};

struct sortPointsAscending
{
    __device__ bool operator()(const cPoint &lhs, const cPoint &rhs);
};

struct sortPointsDescending
{
    __device__ bool operator()(const cPoint &lhs, const cPoint &rhs);
};
