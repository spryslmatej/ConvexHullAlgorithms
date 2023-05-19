#include "../model/point.h"
#include "../model/solver.h"

class CQuickHullWithCrawlers : public CSolver
{
    unsigned gridDim;

public:
    CQuickHullWithCrawlers(unsigned n) : gridDim(n) {}

    void solve(const Point *points, const unsigned n,
               Point **hull, unsigned *hullSize) override;
};
