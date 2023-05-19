#include "graham.h"

Point CGrahamScan::getPointHighestY(const vector<Point> &points,
                                     const Segment s)
{
    auto item = std::min_element(
        points.begin() + s.begin,
        points.begin() + s.end, [](const Point &l, const Point &r)
        { return l.Y > r.Y || (l.Y == r.Y && l.X > r.X); });

    return *item;
}

void CGrahamScan::runThrough(const vector<Point> &points,
                             const Segment segment,
                             vector<Point> &hull)
{
    DualStack s;
    s.push(points.at(segment.begin));
    s.push(points.at(segment.begin + 1));

    // iterate
    for (unsigned i = segment.begin + 2; i < segment.end; i++)
    {
        Point curPoint = points.at(i);
        while (s.size() > 2 &&
               crossProduct(s.nextToTop(),
                            s.top(),
                            curPoint) !=
                   Turn::CLW)
        {
            s.pop();
        }
        s.push(curPoint);
    }

    // fill hull
    while (!s.empty())
    {
        hull.push_back(s.top());
        s.pop();
    }
}

void CGrahamScan::sortPointsByAngle(vector<Point> &points,
                                    const Segment segment,
                                    const Point x)
{
    // std::sort is alot quicker than a manual map of angles
    std::sort(points.begin() + segment.begin,
              points.begin() + segment.end,
              [x](Point a, Point b)
              {
                  double atana = atan2(a.Y - x.Y, a.X - x.X),
                         atanb = atan2(b.Y - x.Y, b.X - x.X);
                  return atana > atanb ||
                         (atana == atanb && a.X < b.X);
              });
}

void CGrahamScan::solveOnSegment(vector<Point> &points, Segment segment, vector<Point> &hull)
{
    // if point count is less or equal to 3, add all points to hull
    if (segment.size() <= 3)
    {
        for (unsigned i = segment.begin; i < segment.end; i++)
            hull.push_back(points.at(i));
        return;
    }

    // find point x with lowest y coordinate
    Point x = getPointHighestY(points, segment);

    // sort points by polar angle from x
    // map of angles is actually slower

    sortPointsByAngle(points, segment, x);

    // go through all points, removing ones that create a right turn
    runThrough(points, segment, hull);
}

void CGrahamScan::solve(vector<Point> &points, vector<Point> &hull)
{
    solveOnSegment(points, Segment(0, points.size()), hull);
}
