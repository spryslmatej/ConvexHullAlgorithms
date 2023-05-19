#pragma once

#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

#include "../model/point.h"
#include "../model/segment.h"
#include "../model/line.cpp"
#include "../model/triangle.cpp"
#include "../model/solver.h"

using namespace std;

class CQuickHull : public CSolver
{
    void findExtremaIndexes(vector<Point> &q, unsigned &a, unsigned &b);

    //  places extremas according to scheme [x, y, {rest}]
    void placeExtremas(vector<Point> &q, unsigned &iX, unsigned &iY);

    //  sorts the vector to scheme: [x, y, {s>0}, {s<0}]
    void sortBySide(vector<Point> &q,
                    const Segment s,
                    const Point x,
                    const Point y, unsigned &splitter);

    unsigned getIndexOfFarthestPointFromLine(const vector<Point> &q,
                                             const Line line,
                                             const Segment s);

    Segment removePointsInTriangle(vector<Point> &q,
                                   const Segment s,
                                   const Triangle triangle);

    void sortPointsToSidesOfLine(vector<Point> &q,
                                 const Segment s,
                                 const Line line,
                                 Segment &greater, Segment &lesser);

    void recurse(vector<Point> &q,
                 Segment s,
                 const Point x,
                 const Point y);

public:
    void solve(std::vector<Point> &q, std::vector<Point> &hull) override;
};
