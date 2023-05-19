#include <vector>
#include <iostream>
#include <algorithm>

#include "../model/point.h"
#include "../model/segment.h"
#include "../model/line.cpp"
#include "../model/triangle.cpp"
#include "../model/solver.h"

class CQuickHullWithCrawlers : public CSolver
{
    unsigned gridDim;

public:
    CQuickHullWithCrawlers(unsigned n) : gridDim(n) {}

    void solve(std::vector<Point> &q, std::vector<Point> &hull) override;
};
