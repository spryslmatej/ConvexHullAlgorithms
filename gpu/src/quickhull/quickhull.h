#pragma once

#include "../model/point.h"
#include "../model/solver.h"

#include "qhData.h"

class CQuickHull : public CSolver
{
    bool checkToContinue(qhData &data);
    void checkPreconditions();

public:
    void solve(const Point *points, const unsigned n,
               Point **hull, unsigned *hullSize) override;
};
