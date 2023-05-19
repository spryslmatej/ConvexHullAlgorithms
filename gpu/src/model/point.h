#pragma once

#define POINT_DATATYPE int

struct Point
{
    // has to have a sufficient data type, otherwise overflow will ensue
    POINT_DATATYPE X, Y;
    int removed, inHull;

    __device__ __host__ Point();
    __device__ __host__ Point(POINT_DATATYPE x, POINT_DATATYPE y);
    __device__ __host__ Point(const Point &x);

    __device__ __host__ friend bool operator<(const Point l, const Point r);

    __device__ __host__ friend bool operator==(const Point l, const Point r);

    __device__ __host__ friend bool operator!=(const Point l, const Point r);
};

struct sortPointsByYDescending
{
    __host__ __device__ bool operator()(const Point l, const Point r);
};

struct sortPointsByXAscending
{
    __host__ __device__ bool operator()(const Point l, const Point r);
};