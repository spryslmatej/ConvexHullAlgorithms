
#include "cPoint.h"

#include "qhData.h"

__device__ __host__ cPoint::cPoint() {}
__device__ __host__ cPoint::cPoint(COORDS_TYPE a, COORDS_TYPE b, INDEXES_TYPE i) : x(a),
                                                                                   y(b),
                                                                                   ptIndex(i) {}

__device__ bool sortPointsAscending::operator()(const cPoint &lhs, const cPoint &rhs)
{
    return lhs.x < rhs.x || (lhs.x == rhs.x && lhs.y < rhs.y);
}

__device__ bool sortPointsDescending::operator()(const cPoint &lhs, const cPoint &rhs)
{
    return lhs.x > rhs.x || (lhs.x == rhs.x && lhs.y > rhs.y);
}
