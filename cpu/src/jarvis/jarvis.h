#pragma once

#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <cmath>
#include <stack>

#include "../model/point.h"
#include "../model/segment.h"
#include "../model/solver.h"
#include "../utilities/pointPosition.h"

using namespace std;

class CJarvisMarch : public CSolver
{
    unsigned getMinXIndex(vector<Point> &points);

public:
    void solve(std::vector<Point> &q, std::vector<Point> &hull) override;
};
