#pragma once

#include <vector>

#include "point.h"

class CSolver
{
public:
    virtual void solve(std::vector<Point> &q, std::vector<Point> &hull) = 0;
};
