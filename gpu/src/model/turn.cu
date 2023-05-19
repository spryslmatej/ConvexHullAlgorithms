#include "turn.h"

__host__ __device__ Turn crossProduct(const Point x, const Point y, const Point z)
{
    double val = ((double)(y.X - x.X) * (double)(z.Y - x.Y) -
                  (double)(y.Y - x.Y) * (double)(z.X - x.X));

    if (val > 0)
        return Turn::CCW;
    else if (val == 0)
        return Turn::COL;
    else
        return Turn::CLW;
}
