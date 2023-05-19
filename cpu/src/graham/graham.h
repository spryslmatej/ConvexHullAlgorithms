#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#include "../model/point.h"
#include "../model/segment.h"
#include "../model/solver.h"
#include "../utilities/pointPosition.h"

#include "dualstack.cpp"

using namespace std;

class CGrahamScan : public CSolver
{
    Point getPointHighestY(const vector<Point> &points,
                            const Segment s);

    void runThrough(const vector<Point> &points,
                    const Segment segment,
                    vector<Point> &hull);

    void sortPointsByAngle(vector<Point> &points,
                           const Segment segment,
                           const Point x);

public:
    void solve(std::vector<Point> &q, std::vector<Point> &hull) override;
    void solveOnSegment(vector<Point> &points, Segment segment, vector<Point> &hull);
};
