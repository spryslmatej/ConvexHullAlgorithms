#pragma once

#include <ostream>
#define POINT_DATATYPE long long int

using namespace std;

struct Point
{
    // has to have a sufficient data type, otherwise overflow will ensue
    POINT_DATATYPE X, Y;
    bool removed, inHull;

    Point() {}
    Point(POINT_DATATYPE x, POINT_DATATYPE y) : X(x), Y(y), removed(false), inHull(false) {}
    Point(const Point &x) : X(x.X), Y(x.Y), removed(x.removed), inHull(x.inHull) {}

};
