#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <stack>
#include <limits>
#include <cassert>

#include "omp.h"

#include "../model/point.h"
#include "../model/segment.h"
#include "../utilities/pointPosition.h"

#include "../graham/graham.h"

using namespace std;

class CChan : public CSolver
{
    unsigned gridDim;

    void createSegments(vector<Segment> &segments,
                        const unsigned pointCount);

public:
    CChan(unsigned n) : gridDim(n) {}

    void jarvisMarchCrossProduct(const vector<vector<Point>> &hulls,
                                 vector<Point> &hull);
    void solve(std::vector<Point> &q, std::vector<Point> &hull) override;
};
