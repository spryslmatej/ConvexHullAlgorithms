#pragma once

#include "point.h"

class CSolver
{
public:
    virtual void solve(const Point *points, const unsigned n,
                       Point **hull, unsigned *hullSize) = 0;
};
