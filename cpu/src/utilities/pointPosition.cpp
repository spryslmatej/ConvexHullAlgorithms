#include "pointPosition.h"

Turn crossProduct(const Point x, const Point y, const Point z)
{
    double val = ((y.X - x.X) * (z.Y - x.Y) -
               (y.Y - x.Y) * (z.X - x.X));

    if (val > 0)
        return Turn::CCW;
    else if (val == 0)
        return Turn::COL;
    else
        return Turn::CLW;
}
