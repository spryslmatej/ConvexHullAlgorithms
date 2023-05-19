#pragma once

#include "point.h"

struct Triangle
{
    Point x, y, z;

    Triangle(Point p1, Point p2, Point p3) : x(p1), y(p2), z(p3) {}
};