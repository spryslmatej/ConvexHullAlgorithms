#pragma once

#include "point.h"

enum Turn
{
    COL = 0,
    CLW = 1,
    CCW = 2
};

__host__ __device__ Turn crossProduct(const Point x, const Point y, const Point z);
