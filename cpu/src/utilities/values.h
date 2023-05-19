#pragma once

#include <vector>
#include "../model/point.h"
#include "../model/stats.cpp"

using namespace std;

void findExtremeValues(const vector<Point> &points,
                       Stats &stats);
                