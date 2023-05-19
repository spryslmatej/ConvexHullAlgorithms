#pragma once

#include "../model/point.h"
#include "../model/solver.h"

#include "concData.h"

class CConcurrentHull : public CSolver
{
    unsigned gridDim;

    void findExtremeValues(const Point *points, const unsigned n,
                           Stats &stats);

    void createGrid(const Point *points, const unsigned n,
                    concData &data);

    void outputHull(concData &data, Point **hull, unsigned *hullSize);

public:
    CConcurrentHull(unsigned n) : gridDim(n) {}

    void solve(const Point *points, const unsigned n,
               Point **hull, unsigned *hullSize) override;
};
