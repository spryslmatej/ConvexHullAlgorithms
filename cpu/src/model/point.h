#pragma once

#include <ostream>
#define POINT_DATATYPE long long int

struct Point
{
    // has to have a sufficient data type, otherwise overflow will ensue
    POINT_DATATYPE X, Y;
    bool removed = false,
         inHull = false;

    Point();
    Point(POINT_DATATYPE x, POINT_DATATYPE y);
    Point(const Point &x);

    friend bool operator<(const Point l, const Point r);
    friend bool operator==(const Point l, const Point r);
    friend bool operator!=(const Point l, const Point r);
};

std::ostream &operator<<(std::ostream &out, const Point &p);
