#include "jarvis.h"

unsigned CJarvisMarch::getMinXIndex(vector<Point> &points)
{
    int min = 0;
    for (unsigned i = 0; i < points.size(); i++)
        if (points[i].X < points[min].X)
            min = i;

    return min;
}

void CJarvisMarch::solve(vector<Point> &points, vector<Point> &hull)
{
    // if point count is less or equal to 3, add all points to hull
    if (points.size() <= 3)
    {
        for (auto it : points)
            hull.push_back(it);
    }

    unsigned min = getMinXIndex(points);

    unsigned curIndex = min, nextIndex;

    do
    {
        hull.push_back(points[curIndex]);

        nextIndex = (curIndex + 1) % points.size();
        for (unsigned i = 0; i < points.size(); i++)
        {
            if (crossProduct(points[curIndex], points[i], points[nextIndex]) ==
                Turn::CCW)
                nextIndex = i;
        }
        curIndex = nextIndex;
    } while (curIndex != min);
}
