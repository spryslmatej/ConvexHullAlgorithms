#include "point.h"

Point::Point() {}
Point::Point(POINT_DATATYPE x, POINT_DATATYPE y) : X(x), Y(y), removed(false), inHull(false) {}
Point::Point(const Point &x) : X(x.X), Y(x.Y), removed(x.removed), inHull(x.inHull) {}

bool operator<(const Point l, const Point r)
{
    if (l.X < r.X)
        return true;
    else if (l.X == r.X)
        return l.Y < r.Y;
    else
        return false;
}

bool operator==(const Point l, const Point r)
{
    return l.X == r.X && l.Y == r.Y;
}

bool operator!=(const Point l, const Point r)
{
    return !(l == r);
}

__host__ __device__ bool sortPointsByYDescending::operator()(const Point l, const Point r)
{
    return l.Y > r.Y || (l.Y == r.Y && l.X > r.Y);
}

__host__ __device__ bool sortPointsByXAscending::operator()(const Point l, const Point r)
{
    return l.X < r.X || (l.X == r.X && l.Y < r.Y);
}