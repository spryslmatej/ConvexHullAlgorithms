#include "values.h"

void findExtremeValues(const vector<Point> &points,
                       Stats &stats)
{
    stats.xMin = points.at(0).X;
    stats.xMax = points.at(0).X;
    stats.yMin = points.at(0).Y;
    stats.yMax = points.at(0).Y;

    for (auto item : points)
    {
        if (item.X < stats.xMin)
            stats.xMin = item.X;
        else if (item.X > stats.xMax)
            stats.xMax = item.X;
        if (item.Y < stats.yMin)
            stats.yMin = item.Y;
        else if (item.Y > stats.yMax)
            stats.yMax = item.Y;
    }
}
