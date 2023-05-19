#pragma once

#include <vector>
#include <iostream>

#include "../model/point.h"
#include "../model/solver.h"
#include "../crawlers/crawlerSegment.h"

using namespace std;

class CConcurrentHull : public CSolver
{
    unsigned gridDim;

    void chanConc(vector<vector<Point>> &segments, vector<Point> &hull);

public:
    CConcurrentHull(unsigned n) : gridDim(n) {}

    void solve(vector<Point> &points, vector<Point> &hull);
};