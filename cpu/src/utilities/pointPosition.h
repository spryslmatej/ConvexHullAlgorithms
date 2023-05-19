#pragma once

#include "../model/point.h"

enum Turn
{
    COL = 0,
    CLW = 1,
    CCW = 2
};

Turn crossProduct(const Point x, const Point y, const Point z);
