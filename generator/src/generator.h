#pragma once

#include <cstdlib>
#include <iostream>
#include <ctime>
#include <cmath>
#include <vector>

#include "point.cpp"

using namespace std;

class CGenerator
{
public:
    virtual void generate(vector<Point> &points,
                          const unsigned count,
                          const unsigned maxVal) = 0;
};
